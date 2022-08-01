import os
import sys
sys.path.insert(0, os.getcwd())
import click
import dotenv
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger

from model import FeedbackModel
from src.config import CFG
from src.data.data_processing_utils import make_tokenizer_dataset_loader
from utils import CheckpointToMlflow

dotenv.load_dotenv('.env')


@click.command()
@click.argument('train_df_path', type=click.Path(exists=True))
@click.argument('valid_df_path', type=click.Path(exists=True))
def train(train_df_path, valid_df_path):
    train_df = pd.read_csv(train_df_path)
    valid_df = pd.read_csv(valid_df_path)
    train_tokenizer, train_dataset, train_loader = make_tokenizer_dataset_loader(CFG, train_df, 'train')
    _, valid_dataset, valid_loader = make_tokenizer_dataset_loader(CFG, valid_df, 'train', train_tokenizer)

    mlf_logger = MLFlowLogger(experiment_name=CFG.exp_name,
                              tracking_uri=os.environ.get('MLFLOW_TRACKING_URI'))

    trainer = Trainer(logger=mlf_logger, max_epochs=1, accelerator=CFG.accelerator, log_every_n_steps=1,
                      callbacks=[TQDMProgressBar(), CheckpointToMlflow("../../models/", monitor='valid_acc_epoch')])
    model = FeedbackModel(CFG, pretrained=True)
    trainer.fit(model, train_loader, valid_loader)
    return model, train_tokenizer


if __name__ == '__main__':
    train()
