import os

import dotenv
import mlflow
from pytorch_lightning.callbacks import ModelCheckpoint

from model import FeedbackModel
from src.config import CFG

dotenv.load_dotenv('../../.env')


class CheckpointToMlflow(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(CheckpointToMlflow, self).__init__(*args, **kwargs)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super(CheckpointToMlflow, self).on_train_end(trainer, pl_module)
        os.environ['MLFLOW_RUN_ID'] = trainer.logger.run_id  # Hack to force MLFlow to 'know' about this run
        model_to_save = FeedbackModel.load_from_checkpoint(self.best_model_path, cfg=CFG)
        mlflow.pytorch.log_model(model_to_save.base_model, 'feedback_model',
                                 registered_model_name="Feedback_model")
