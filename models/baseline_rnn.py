
import torch
from torch import nn
import pytorch_lightning as pl


class BaselineRNN(pl.LightningModule):

    def __init__(self, embed_size): 
        super().__init__()
        
        #Parameters & Metrics
        self.lr = 0.01 
        self.loss = nn.BCELoss()
        self.accuracy = pl.metrics.Accuracy()
        
        #Model
        self.rnn1 = nn.GRU(embed_size, 512, batch_first=True)
        self.rnn2 = nn.GRU(512, 64, batch_first=True)
        self.dense1 = nn.Linear(64, 16)
        self.relu = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h, _ = self.rnn1(x)
        _, h = self.rnn2(h)
        h = torch.squeeze(h)
        h = self.dense1(h)
        h = self.relu(h)
        y_hat = self.sig(self.dense2(h))
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x.float())
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y.int())
        print("Training Loss: ", loss)
        return {'loss': loss, 'acc': acc}
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x.float())
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y.int())
        print("Training Loss: ", loss)
        return {'loss': loss, 'acc': acc}
    
    def _evaluate(self, batch, batch_idx, stage=None):
        x, y, _ = batch
        y_hat = self.forward(x.float())
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y.int())

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

        return loss, acc

    def validation_step(self, batch, batch_idx): 
        loss, acc = self._evaluate(batch, batch_idx, 'val')
        return {'val_loss': loss, 'val_acc': acc}
        
    def test_step(self, batch, batch_idx):
        loss, acc = self._evaluate(batch, batch_idx, 'test')
        return {'test_loss': loss, 'test_acc': acc}

    def validation_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()
        print('Validation Loss: ', val_loss_mean)
        print('Validation Accuracy: ', val_acc_mean)
        return {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}

    def test_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc_mean = torch.stack([x['test_acc'] for x in outputs]).mean()
        print('Test Loss: ', test_loss_mean)
        print('Test Accuracy: ', test_acc_mean)
        return {'test_loss': test_loss_mean, 'test_acc': test_acc_mean}
           
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    



