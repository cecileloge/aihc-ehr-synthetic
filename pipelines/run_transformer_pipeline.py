from models.transformer_model import TransformerModel
from dataloader.data import DataModuleTask
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from argparse import ArgumentParser

# -------------------------------------------------------------------
#  (dm) DataModuleTask(task) with task either 'eol' or 'surgical'
#  (model) TransformerModel(embed_size, max_concepts, max_visits) 
# -------------------------------------------------------------------

def main():
    dm = DataModuleTask(task='eol')
    dm.setup()
    model = TransformerModel(max_days=dm.max_days, 
                            num_concepts=dm.num_concepts,
                            max_visits=dm.max_visits,
                            learning_rate=0.001)
    
    trainer = Trainer(gpus=[4], max_epochs=3)
    trainer.fit(model, dm)
    
    trainer.test(datamodule=dm)
    print("Test Loss: ", model.test_results['test_loss'].item())
    print("Test AUROC: ", model.test_results['test_auc'].item())

if __name__ == '__main__':
    main()
    
