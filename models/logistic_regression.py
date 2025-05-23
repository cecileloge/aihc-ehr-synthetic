import torch
from torch import nn
import pytorch_lightning as pl

class LogRegressionModel(pl.LightningModule):
    
    def __init__(self, input_dim, weight_decay=0, lr=0.01):
        super().__init__()
        
        # Hyperparameters
        self.lr = 0.01
        self.embed_size = input_dim
        self.weight_decay = weight_decay
        
        # Metrics
        self.loss = nn.BCELoss()
        
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        
        self.train_auroc = pl.metrics.AUROC('macro')
        self.val_auroc = pl.metrics.AUROC('macro')
        self.test_auroc = pl.metrics.AUROC('macro')
        
        self.train_results = None
        self.val_results = None
        self.test_results = None
        
        # Model
        self.dense = nn.Linear(self.embed_size, 1)
        self.sig = nn.Sigmoid()
        
    # X is of dimension: (n_patients, n_visits, n_concepts)
    def forward(self, x):
        x = torch.stack(x)
        x = x.float() 
        y_hat = self.sig(self.dense(x))
        return y_hat
    
    def _evaluate(self, batch, batch_idx, acc, auroc, stage=None):
        x, y, _ = batch

        preds = self.forward(x)
        loss = self.loss(preds, y)
        
        self.log(f'{stage}_loss', loss, prog_bar=True)

        acc.update(preds, y.int())
        auroc.update(preds, y.int())
        
        return preds, loss, y
    
    def training_step(self, batch, batch_idx):
        preds, loss, targets = self._evaluate(batch, batch_idx, self.train_acc, self.train_auroc, 'train')
        return {'loss': loss, 'train_preds': preds, 'train_targets': targets}
    
    def validation_step(self, batch, batch_idx):
        preds, loss, targets = self._evaluate(batch, batch_idx, self.val_acc, self.val_auroc, 'val')
        return {'val_loss': loss, 'val_preds': preds, 'val_targets': targets}
    
    def test_step(self, batch, batch_idx):
        preds, loss, targets = self._evaluate(batch, batch_idx, self.test_acc, self.test_auroc, 'test')
        return {'test_loss': loss, 'test_preds': preds, 'test_targets': targets}
    
    def training_epoch_end(self, outputs):
        print(f"\n----------EPOCH------------")
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        print(f"Train loss: {train_loss_mean}")
        train_auroc = self.train_auroc.compute()
        print(f"Train AUROC: {train_auroc}")
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f"Validation loss: {val_loss_mean}")
        val_auroc = self.val_auroc.compute()
        print(f"Val AUROC: {val_auroc}\n")
        
    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = self.test_acc.compute()
        test_auroc = self.test_auroc.compute()
        test_targets = torch.cat([x['test_targets'] for x in outputs], dim=0)
        test_preds = torch.cat([x['test_preds'] for x in outputs], dim=0)
        self.test_results = {'test_loss': test_loss_mean, 'test_acc': test_acc, 
                             'test_auroc': test_auroc, 'y': test_targets,
                             'y_hat': test_preds}
        print(f"TEST_RESULTS: \nAUROC: {self.test_results['test_auroc']}\nACC: {self.test_results['test_acc']}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    