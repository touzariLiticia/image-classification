import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix
import pytorch_lightning as pl
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_ftr_extractor(pretrained=True, model_name="densenet201"):
    """This function return pretrained model_name features"""

    if model_name == "densenet121":
        ftrs = models.densenet121(pretrained=True).features
        n = 1024
    elif model_name == "mobilenet_v2":
        ftrs = models.mobilenet_v2(pretrained=True).features
        n = 1280
    elif model_name == "resnet152":
        m = models.resnet152(pretrained=True)
        ftrs = nn.Sequential(*list(m.children())[:-2])
        n = 2048
    else:
        m = models.shufflenet_v2_x1_0(pretrained=False)
        ftrs = nn.Sequential(*list(m.children())[:-2])
        n = 464
    return ftrs, n


class Net(pl.LightningModule):
    """
    Parameters
    ----------
    backbone_name : str
        Backbone name f
    num_classes : int
        The number of training classes
    dropout_fts : float
        Dropout
    pretrained : bool
        Use a pretrained model as backbone or not
    lr : float
        Learning rate
    epochs : int
        Number of epochs
    """

    def __init__(
        self,
        backbone_name,
        num_classes,
        dropout_fts,
        pretrained,
        lr,
        epochs,

    ):
        super(Net, self).__init__()
        self.lr = lr
        self.epochs = epochs
        self.accuracy = Accuracy(task="binary")
        self.confusion_matrix = ConfusionMatrix(task="binary")
        self.num_classes = num_classes
        self.dropout_fts = dropout_fts
        self.save_hyperparameters()
        self.ftrs, self.num_fts = get_ftr_extractor(
            pretrained=pretrained, model_name=backbone_name
        )
        self.drop = nn.Dropout(dropout_fts)
        self.classifier = nn.Linear(self.num_fts, num_classes)

    def forward(self, x):
        x_ftrs = self.ftrs(x).mean(3).mean(2)
        x_ftrs = self.drop(x_ftrs)
        outputs = self.classifier(x_ftrs)
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        # log loss and accuracy
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        batch_dictionary = {"loss": loss,  "train_acc": acc}
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        # log loss and accuracy
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        batch_dictionary = {"loss": loss,  "val_acc": acc,
                            'predictions': preds, 'targets': y}
        return batch_dictionary

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # calculating average accuracy
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "Accuracy/Train", avg_acc, self.current_epoch)
        print('TRAIN: ', 'loss : ', avg_loss, 'acc : ', avg_acc)

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # calculating correect and total predictions
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        # prediction
        preds = torch.cat([x['predictions'] for x in outputs], dim=0)
        # targets
        targets = torch.cat([x['targets'] for x in outputs], dim=0)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Validation", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "Accuracy/Validation", avg_acc, self.current_epoch)
        print('VALIDATION: ', 'loss : ', avg_loss, 'acc : ', avg_acc)

        # log the confusion matrix
        confusion_matrix = self.confusion_matrix(
            preds, targets).detach().cpu().numpy()
        confusion_matrix = confusion_matrix.astype(
            'float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(confusion_matrix, index=range(
            self.num_classes), columns=range(self.num_classes))
        plt.figure(figsize=(10, 10))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.close(fig_)
        self.logger.experiment.add_figure(
            "Confusion matrix", fig_, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_dict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.lr / 100
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_scheduler",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}
