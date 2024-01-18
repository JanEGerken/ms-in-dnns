# Assignment E: PyTorch Lighntning (60 Points)

## 1. CIFAR in Lightning (40 Points)
This exercise builds on the script you wrote in the previous assignment to train a VGG-16 CNN on CIFAR-10.

### 1.a Port to lightning (10 Points)
Make the `cifar10_net.py` script into a `cifar10_net` package using Lightning, as discussed in the lecture. You can start from the `income_net` package developed in the lecture which is available in this directory. Log the training- and validation accuracy as well as -loss during training and separately for the best epoch (according to validation accuracy).

### 1.b Confusion matrix (10 Points) 
Furthermore, implement logging of the confusion matrix on the best epoch, similar to what you did using pure PyTorch in the previous assignment. In particular
- Use torchmetrics to compute the confusion matrix during the test step.
- Using a hook at the end of the test epoch, log the confusion matrix using a direct call to `wandb` (i.e. not `self.log`).
- Have a look at the last lines of the implementation of `wandb.plot.confusion_matrix` [here](https://github.com/wandb/wandb/blob/6a211b19f02ee7c6b87b82eafd5789c4ba3739ec/wandb/plot/confusion_matrix.py#L82) to see how to log the computed confusion matrix without having wandb computing it from predictions and targets.

### 1.c Finetune pretrained VGG-16 (20 Points)
Torchvision includes many model architectures for computer vision and has trained weights for these models available to download, have a look [here](https://pytorch.org/vision/0.14/models.html) for the documentation.

The VGG-16 model on which our CIFAR-classifier is based, is available in a version with batch norm [here](https://pytorch.org/vision/0.14/models/generated/torchvision.models.vgg16_bn.html). However, the model was designed for and trained on the ImageNet dataset which features larger images in 1000 classes. Therefore, we need to adapt the original VGG-16 to fit our purpose.

Write a new `torch.nn.Module` model which uses the VGG-16 model with batch norm mentioned above for feature extraction. You can see the general structure of this model [here](https://github.com/pytorch/vision/blob/71b27a00eefc1b169d1469434c656dd4c0a5b18d/torchvision/models/vgg.py#L35). To account for the smaller images in CIFAR, replace the `avgpool` layer with an identity layer and to account for the smaller number of classses, replace the fully-connected classifier with the classifier we have been using so far in `CIFAR10Net`. Train this model from scratch as a baseline for 60 epochs, without dropout but with data augmentation.

Then, load the weights trained on ImageNet which are available in torchvision. You should set the torch cache directory to the `torch_cache` directory in the root of this repository and a smilir directory on Google Cloud using `torch.hub.set_dir`. Train the model again using the pretrained weights for 60 epochs (without dropout but with data augmentation again) and compare the performances. Discuss your results in a WandB report.

## 2. Adversarial Attacks (20 Points)

As discussed in Lecture 4, it is not hard to find images which fool a well-trained classifier into giving the wrong prediction, even with no perceptible difference to a correctly classified image. This is known as adversarial vulnerability. In this execise, you will compute adversarial examples for the VGG-16 model you trained on CIFAR-10.

Write a script `adv_attacks.py` which is part of the `cifar10_net` package and logs to a new WandB project. For data loading, you can import the data module from the package and you can load a training checkpoint by using the `load_from_checkpoint` classmethod of the `LightningModule` class.

Your script should have the following `argparse` arguments:
- `--data-root` for the directory where to find the CIFAR10 data
- `--run-name` for the name of the run
- `--ckpt-path` for the path to the checkpoint to be used
- `--source-class` the class of the samples to be optimized
- `--n-samples` the number of samples to start the attack from (default: `5`)
- `--max-iter` the maximum number of optimization iterations (default: `100`)
- `--lr` learning rate of the optimizer (default: `1e-3`)
- `--prob-threshold` the predicted probability of the target class at which to stop the optimization (default: `0.99`)

First, find `--n-samples` samples in the validation data which lie in the source class. Then, for each class as target class, optimize the input to the network using the Adam optimizer until the maximum number of iterations is reached or the predicted probability for the target class surpasses the probability threshold. It is easiest to do this in pure PyTorch and not Lightning. You should end up with `n_samples*10` new images. Log your results into a WandB table with columns source image, ground truth class, target class, adversary, difference to original and final target probability.

As the source class, pick a class with high accuracy according to the confusion matrix you computed in the previous exercise and run the attack with the defaults given above. Discuss your results in a WandB report.
