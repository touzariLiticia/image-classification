from argparse import ArgumentParser, Namespace
from pathlib import Path
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from image_classifier.models.network import Net
from image_classifier.data_models.data_utils import get_train_dataloader, get_test_dataloader


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_data_path", type=str,
                        default="data/db_train.raw")
    parser.add_argument("--train_labels_path", type=str,
                        default="data/label_train.txt")
    parser.add_argument("--test_data_path", type=str, default="data/db_test.raw")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--backbone_name", type=str, default="densenet121")
    parser.add_argument("--logger_path", type=str, default="../tb_logs")
    parser.add_argument("--checkpoints_path", type=str,
                        default="../checkpoints")

    return parser.parse_args()


def run(args):
    """Run model training"""
    logger = TensorBoardLogger(
        Path(args.logger_path), name="face_classification")

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        mode='min',
        dirpath=Path(args.checkpoints_path),
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}'
    )

    model = Net(
        backbone_name=args.backbone_name,
        num_classes=args.num_classes,
        dropout_fts=args.dropout,
        pretrained=args.pretrained,
        lr=args.lr,
        epochs=args.epochs,
    )

    trainer = Trainer(
        gpus=args.n_gpus if torch.cuda.is_available() else 0,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    train_dataloader = get_train_dataloader(
        Path(args.train_data_path),
        Path(args.train_labels_path),
        sample=True,
        augment=True,
        batch_size=args.batch_size
    )
    test_dataloader = get_test_dataloader(
        Path(args.test_data_path),
        batch_size=args.batch_size
    )

    trainer.fit(model, train_dataloader)
    # predict(checkpoint_path, test_dataloader)


def predict(checkpoint_path, test_dataloader):
    """save test predictions"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net.load_from_checkpoint(checkpoint_path).to(device)
    model = model.to(device)
    model.eval()
    predictions = []
    for x in test_dataloader:
        logits = model(x.to(device))
        preds = torch.argmax(logits, dim=1)
        predictions += preds.detach().cpu().numpy().tolist()
    predictions = pd.DataFrame({'predictions': predictions})
    predictions.to_csv(r'label_test.txt', header=None,
                       index=None, sep=' ', mode='a')


if __name__ == "__main__":
    args = parse_args()
    run(args)
