# Path: experiments/cifar/experiment.py

from ...data.loaders.cifar10 import CifarLightningDataModule
from ...models.resnet import ResNet
from ...models.resnet.resnet import BasicBlock
from ...trainers.base_classification import LightningClassificationModule
import torch
import pytorch_lightning
import pytorch_lightning.callbacks
from aim.pytorch_lightning import AimLogger


data = CifarLightningDataModule(
    location="{data_dir}",
    batch_size=128,
    image_size=[256, 256],
)

model = ResNet(
    block=BasicBlock,
    layers=[3, 4, 6, 3],
    num_classes=1000,
    in_channels=3,
    zero_init_residual=False,
    groups=1,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

pl_module = LightningClassificationModule(
    model=model,
    optimizers={
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    },
)


trainer = pytorch_lightning.Trainer(
    gpus=1,
    max_epochs=10,
    callbacks=[
        pytorch_lightning.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath="{save_dir}/checkpoints",
            save_top_k=1,
            verbose=True,
            save_last=True,
        )
    ],
    logger=AimLogger(
        experiment="cifar",
        train_metric_prefix="train_",
        val_metric_prefix="val_",
    ),
)

commands = {
    "train": trainer.fit(pl_module, datamodule=data),
    "test": trainer.test(pl_module, datamodule=data),
}
