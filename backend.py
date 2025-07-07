import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torchmetrics import Accuracy

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.progress import TQDMProgressBar

def get_dataloaders(dataset: Dataset, batch_size: int, train_frac: float=0.8, test_frac: float=0.8) -> list[DataLoader]:
    '''
    Splits dataset into train, validation and test datasets

    dataset: PyTorch dataset to split.
    batch_size: The minibatch size to use.
    train_frac: The fraction of the total dataset to devote to training.
    test_frac: The fraction of the non-training section to devote to testing (rest to validation). i.e. p_test = (1 - train_frac)*test_frac
    return: List of the training, validation and testing dataloaders
    '''
    len_train_set = int(train_frac*len(dataset))
    len_eval_set = len(dataset) - len_train_set
    len_test_set = int(test_frac*len_eval_set)
    len_valid_set = len_eval_set - len_test_set

    print(f'Train: {len_train_set}\nValid: {len_valid_set}\nTest: {len_test_set}')

    # Split into train, validation and test datasets
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_eval_set])
    test_dataset, valid_dataset = torch.utils.data.random_split(eval_dataset, [len_test_set, len_valid_set])

    # Return dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader

class GNNClassificationModule(pl.LightningModule):
    '''
    PyTorch Lightning module to expedite training and evaluation for classification tasks.

    model: PyTorch model to use.
    dataloaders: List of train, validation and test loaders.
    num_classes: Number of classes.
    optim: PyTorch optimizer to use.
    optim_kwargs: Optimizer kwargs.
    '''
    def __init__(self, model: nn.Module, dataloaders: list[DataLoader], num_classes: int, optim: torch.optim.Optimizer, **optim_kwargs):
        super().__init__()

        self.train_loader, self.valid_loader, self.test_loader = dataloaders

        # Metrics for benchmarking
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Configure model and optimizer
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim(self.parameters(), **optim_kwargs)
            
    def forward(self, data):
        return self.model(data)

    def forward_step(self, batch, batch_idx):
        y = batch.y

        logits = self(batch)
        loss = self.criterion(logits, y)

        preds = logits.argmax(1)

        return y, preds, loss
    
    ### Functions evaluated at each step of training, validation, testing and inference ###

    def training_step(self, batch, batch_idx):
        # Obtain predictions
        y, preds, loss = self.forward_step(batch, batch_idx)

        self.train_acc.update(preds, y)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        y, preds, loss = self.forward_step(batch, batch_idx)

        self.val_acc.update(preds, y)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        y, preds, loss = self.forward_step(batch, batch_idx)

        self.test_acc.update(preds, y)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)

        return preds

    def predict_step(self, batch, batch_idx):
        y = batch.y

        pred = self(batch)
        
        return (pred, y, x)

    ### Data-related hooks ###

    def configure_optimizers(self):
        return self.optim

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader