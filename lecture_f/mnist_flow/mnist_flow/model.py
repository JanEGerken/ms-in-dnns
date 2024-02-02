import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import wandb

# Code heavily inspired by
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html


class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)


class LayerNormChannels(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        """
        This module applies layer norm across channels in an image.
        Inputs:
            c_in - Number of channels of the input
            eps - Small constant to stabilize std
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        y = y * self.gamma + self.beta
        return y


class GatedConv(nn.Module):
    def __init__(self, c_in, c_hidden):
        """
        This module applies a two-layer convolutional ResNet block with input gate
        Inputs:
            c_in - Number of channels of the input
            c_hidden - Number of hidden dimensions we want to model (usually similar to c_in)
        """
        super().__init__()
        self.net = nn.Sequential(
            ConcatELU(),
            nn.Conv2d(2 * c_in, c_hidden, kernel_size=3, padding=1),
            ConcatELU(),
            nn.Conv2d(2 * c_hidden, 2 * c_in, kernel_size=1),
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class GatedConvNet(nn.Module):
    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=3):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = []
        layers += [nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1)]
        for layer_index in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden), LayerNormChannels(c_hidden)]
        layers += [ConcatELU(), nn.Conv2d(2 * c_hidden, c_out, kernel_size=3, padding=1)]
        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)


class CouplingLayer(nn.Module):
    def __init__(self, network, mask, c_in):
        """
        Coupling layer inside a normalizing flow.
        Inputs:
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask - Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in - Number of input channels
        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer("mask", mask)

    def forward(self, z, ldj, reverse=False):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
        """
        # Apply network to masked input
        z_in = z * self.mask
        nn_out = self.network(z_in)
        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1, 2, 3])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1, 2, 3])

        return z, ldj


class SqueezeFlow(nn.Module):
    def forward(self, z, ldj, reverse=False):
        B, C, H, W = z.shape
        if not reverse:
            # Reverse direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, C, H // 2, 2, W // 2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(B, 4 * C, H // 2, W // 2)
        else:
            # Forward direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, C // 4, 2, 2, H, W)
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(B, C // 4, H * 2, W * 2)
        return z, ldj


class SplitFlow(nn.Module):
    def __init__(self, z_dist):
        super().__init__()
        self.z_dist = z_dist

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z, z_split = z.chunk(2, dim=1)
            ldj += self.z_dist.log_prob(z_split).sum(dim=[1, 2, 3])
        else:
            z_split = self.z_dist.sample(sample_shape=z.shape).to(z.device)
            z = torch.cat([z, z_split], dim=1)
            ldj -= self.z_dist.log_prob(z_split).sum(dim=[1, 2, 3])
        return z, ldj


class MNISTFlow(nn.Module):
    def __init__(self, res_blocks, z_dist):
        super().__init__()
        layer_list = []

        for i in range(2):
            layer_list.append(
                CouplingLayer(
                    GatedConvNet(c_in=1, c_hidden=32, num_layers=3),
                    mask=self.build_checkerboard_mask(28, i % 2),
                    c_in=1,
                )
            )
        layer_list += [SqueezeFlow()]
        for i in range(2):
            layer_list.append(
                CouplingLayer(
                    GatedConvNet(c_in=4, c_hidden=48, num_layers=3),
                    mask=self.build_channel_mask(4, i % 2),
                    c_in=4,
                )
            )
        layer_list += [SplitFlow(z_dist), SqueezeFlow()]
        for i in range(4):
            layer_list.append(
                CouplingLayer(
                    GatedConvNet(c_in=8, c_hidden=64, num_layers=3),
                    mask=self.build_channel_mask(8, i % 2),
                    c_in=8,
                )
            )

        self.layers = nn.ModuleList(layer_list)

    def build_checkerboard_mask(self, size, config=1.0):
        """Builds a binary checkerboard mask.

        (Only for constructing masks for checkerboard coupling layers.)

        Args:
            size: height/width of features.
            config: mask configuration that determines which pixels to mask up.
                    if 1:        if 0:
                        1 0         0 1
                        0 1         1 0
        Returns:
            a binary mask (1: pixel on, 0: pixel off).

        """
        mask = torch.arange(size).reshape(-1, 1) + torch.arange(size)
        mask = torch.remainder(config + mask, 2)
        mask = mask.reshape(-1, 1, size, size)
        return mask.to(torch.float)

    def build_channel_mask(self, c_in, config=1.0):
        mask = torch.cat(
            [
                torch.ones(c_in // 2, dtype=torch.float32),
                torch.zeros(c_in - c_in // 2, dtype=torch.float32),
            ]
        )
        mask = mask.view(1, c_in, 1, 1)
        mask = config * (1 - mask) + (1 - config) * mask
        return mask

    def forward(self, imgs, reverse):
        z, ldj = imgs, torch.zeros(imgs.shape[0], device=imgs.device)
        for layer in reversed(self.layers) if reverse else self.layers:
            z, ldj = layer(z, ldj, reverse=reverse)

        return z, ldj


def logit_transform(x, constraint=0.9, reverse=False):
    """Transforms data from [0, 1] into unbounded space.

    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).

    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    """
    if reverse:
        x = 1.0 / (torch.exp(-x) + 1.0)  # [0.05, 0.95]
        x *= 2.0  # [0.1, 1.9]
        x -= 1.0  # [-0.9, 0.9]
        x /= constraint  # [-1, 1]
        x += 1.0  # [0, 2]
        x /= 2.0  # [0, 1]
        return x, 0
    else:
        [B, C, H, W] = list(x.size())

        # dequantization
        noise = torch.distributions.Uniform(0.0, 1.0).sample((B, C, H, W)).to(x.device)
        x = (x * 255.0 + noise) / 256.0

        # restrict data
        x *= 2.0  # [0, 2]
        x -= 1.0  # [-1, 1]
        x *= constraint  # [-0.9, 0.9]
        x += 1.0  # [0.1, 1.9]
        x /= 2.0  # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1.0 - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(np.log(constraint) - np.log(1.0 - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))


class MNISTFlowModule(L.LightningModule):
    def __init__(self, lr=1e-3, res_blocks=3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.z_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        self.model = MNISTFlow(res_blocks=res_blocks, z_dist=self.z_dist)

    def forward(self, x, reverse):
        return self.model(x, reverse=reverse)

    def nll(self, imgs_z, ldj, mean=True):
        # the z_dist is one-dimensional and applied element-wise
        # the joint is the product (elements are independent), so sum the logs
        log_qz = self.z_dist.log_prob(imgs_z).sum(dim=[1, 2, 3])
        log_qx = ldj + log_qz
        nll = -log_qx
        nll = nll.mean() if mean else nll
        return nll

    def log_samples(self, n_samples, stage="pred"):
        imgs_z = self.z_dist.sample(sample_shape=(n_samples, 8, 7, 7)).to(self.device)
        imgs_x, ldj = self(imgs_z, reverse=True)
        imgs_x, _ = logit_transform(imgs_x, reverse=True)
        if stage == "pred":
            table = wandb.Table(columns=["image", "epoch"])
        elif stage == "val":
            table = self.val_samples_table
        else:
            print("unknown stage, not generating samples")
            return
        for i in range(n_samples):
            table.add_data(
                wandb.Image(imgs_x[i].cpu().numpy().transpose(1, 2, 0)),
                self.current_epoch + 1,
            )
        if stage == "pred":
            wandb.log({"best/samples": table})
        return imgs_x

    def shared_step(self, batch, metric_prefix):
        imgs_x, labels = batch
        x, ldj_logit = logit_transform(imgs_x)
        imgs_z, ldj = self(x, reverse=False)
        loss = self.nll(imgs_z, ldj + ldj_logit)
        self.log(f"{metric_prefix}/loss", loss, on_epoch=True, on_step=False)

        if metric_prefix in ["train", "val"]:
            self.log("step", float(self.current_epoch + 1), on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        if ((self.current_epoch + 1) % 5 == 0) and batch_idx == 0:
            self.log_samples(5, "val")
        return loss

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "best")

    def on_fit_start(self):
        self.val_samples_table = wandb.Table(columns=["image", "epoch"])

    def on_fit_end(self):
        wandb.log({"val/samples": self.val_samples_table})

    def predict_step(self, batch, batch_idx):
        return self.log_samples(batch[0].shape[0], "pred")

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
