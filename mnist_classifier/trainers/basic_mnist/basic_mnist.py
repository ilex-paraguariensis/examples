import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from pytorch_lightning import Trainer
from mate.trainer import LightningTrainer
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchvision import transforms
import ipdb

# imports DataLoader


class LitMNIST(LightningModule):
    def __init__(
        self,
        classifier: nn.Module,
        data_dir: str = ".",
        hidden_size: int = 64,
        learning_rate: float = 2e-4,
    ):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.model = classifier

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        ipdb.set_trace()
        return optimizer


class BasicMnist(LightningTrainer):
    def __init__(
        self,
        classifier: nn.Module,
        lightning_trainer: Trainer,
        data_dir: str = ".",
        hidden_size: int = 3,
        learning_rate: float = 0.001,
    ):
        super().__init__(lightning_trainer)
        self._module = LitMNIST(
            classifier, data_dir, hidden_size, learning_rate
        )
