from datetime import datetime
import os
import sys

import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar, ModelCheckpoint

from mnist_flow.data import MNISTDataModule
from mnist_flow.model import MNISTFlowModule
from mnist_flow.utils import get_wandb_key, args_to_flat_dict

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


def main(args):

    seed_everything(0xDEADBEEF, workers=True)

    if "LOG_PATH" in os.environ:
        wandb_save_dir = os.path.dirname(os.environ["LOG_PATH"])
    else:
        wandb_save_dir = "."
    wandb.login(key=get_wandb_key())
    args.trainer.logger = WandbLogger(
        project="ms-in-dnns-mnist-flow", name=args.run_name, save_dir=wandb_save_dir
    )
    args.trainer.logger.experiment.config.update(args_to_flat_dict(args))

    dm = MNISTDataModule(**vars(args.data))
    model = MNISTFlowModule(**vars(args.model))

    args.trainer.callbacks = [
        RichModelSummary(max_depth=2),
        RichProgressBar(),
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_last=True,
            filename="epoch={epoch}-val_loss={val/loss:.2f}",
            auto_insert_metric_name=False,
        ),
    ]

    trainer = Trainer(**vars(args.trainer))
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)
    # trainer.test(model, datamodule=dm, ckpt_path="best")
    # trainer.predict(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(Trainer, "trainer")
    parser.set_defaults({"trainer.max_epochs": 2, "trainer.num_sanity_val_steps": 2})

    parser.add_lightning_class_args(MNISTFlowModule, "model")

    parser.add_lightning_class_args(MNISTDataModule, "data")
    if "LOG_PATH" in os.environ:
        parser.set_defaults({"data.data_root": "/gcs/msindnn_staging/mnist_data"})
    else:
        parser.set_defaults({"data.data_root": "../../../data/mnist"})

    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)

    args = parser.parse_args()
    main(args)
