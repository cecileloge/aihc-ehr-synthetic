import torch
from torch import nn
import pytorch_lightning as pl
from .transformers import VisitTransformer

class TransformerModel(pl.LightningModule):

    def __init__(self, max_days, num_concepts, max_visits, learning_rate=0.01):
        super(TransformerModel, self).__init__()

        #Parameters
        self.lr = learning_rate 
        self.dropout = 0.3
        self.max_days = max_days
        self.num_concepts = num_concepts
        self.max_visits = max_visits
        self.pool_dim = 10
        
        #Loss & Metrics
        self.loss = nn.BCELoss()
        self.auc_test = pl.metrics.AUROC('macro')
        self.test_results = None

        #Model
        self.trans = VisitTransformer(self.max_days, self.num_concepts)
        self.pooler = torch.nn.Linear(self.max_visits, self.pool_dim)
        self.linear = torch.nn.Linear(self.trans.embedding_dim * 10, 1)
        self.dropout = torch.nn.Dropout(self.dropout)
        self.sig = nn.Sigmoid()
    
    def forward(self, x, t):
        #Embedding & Transformer
        z = self.trans(x, t)
        
        #Convolution & Pooling
        pooled = self.pooler(
            z.transpose(1,2)
        ).view(-1, self.pool_dim * self.trans.embedding_dim)
        pooled = self.dropout(pooled)
        y_pred = self.linear(torch.nn.ReLU()(pooled))
        y_pred = self.sig(y_pred)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y, t = batch
        y_hat = self.forward(x, t)
        loss = self.loss(y_hat, y)
        return {'loss': loss }
    
    def _evaluate(self, batch, batch_idx, stage=None):
        x, y, t = batch
        y_hat = self.forward(x, t)
        loss = self.loss(y_hat, y)
        
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            if stage == 'test': self.auc_test.update(y_hat, y.int())    

        return loss, y, y_hat

    def validation_step(self, batch, batch_idx): 
        loss, _, _ = self._evaluate(batch, batch_idx, 'val')
        return {'val_loss': loss}
        
    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self._evaluate(batch, batch_idx, 'test')
        return {'test_loss': loss, 'y': y, 'y_hat': y_hat}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        print("Validation Loss: ", val_loss_mean.item())

    
    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        print("Training Loss: ", loss_mean.item())

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_auc = self.auc_test.compute()
        test_y = torch.cat([x['y'] for x in outputs], dim=0)
        test_y_hat = torch.cat([x['y_hat'] for x in outputs], dim=0)
        self.test_results = {'test_loss': test_loss_mean, 'test_auc': test_auc, 
                             'y': test_y, 'y_hat': test_y_hat}
           
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
