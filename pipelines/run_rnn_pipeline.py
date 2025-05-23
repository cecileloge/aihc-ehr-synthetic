from models.baseline_rnn import BaselineRNN
from dataloader.data_rnn import DataModuleEoL, DataModuleSurg
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from argparse import ArgumentParser

# -----------------------------------------------------------
#  (dm) DataModuleEoL or DataModuleSurg
#  (model) BaselineRNN(embed_size) 
# -----------------------------------------------------------

def main(hparams):
    dm = DataModuleEoL()
    dm.setup()
    model = BaselineRNN(embed_size=dm.embed_size)
    
    trainer = Trainer(gpus=hparams.gpus)
    trainer.fit(model, dm)
    
    result = trainer.test(datamodule=dm)
    print("Test Results: ", result)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()

    main(args)
    
