import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from torchmetrics import Accuracy


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class TransformerCover(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        self.pooler = MeanPooling()

        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self.output = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.cfg.target_size)
        )

    def forward(self, ids, mask, token_type_ids=None):
        if token_type_ids:
            transformer_out = self.model(ids, mask, token_type_ids)
        else:
            transformer_out = self.model(ids, mask)

        sequence_output = self.pooler(transformer_out.last_hidden_state, mask)

        # Main task
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        return logits

    def freeze(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False


class FeedbackModel(pl.LightningModule):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.base_model = TransformerCover(cfg, config_path, pretrained)
        self.base_model.freeze()
        self.trainAcc = Accuracy()
        self.validAcc = Accuracy()
        self.save_hyperparameters(
            {key: value for key, value in cfg.__dict__.items() if not key.startswith('__') and not callable(key)})

    def training_step(self, batch, batch_idx):
        out = self.base_model(batch['input_ids'], batch['attention_mask'])
        loss = nn.functional.cross_entropy(out, batch['target'])
        self.log("train_loss", loss)
        self.trainAcc(out, batch['target'])
        self.log('train_acc', self.trainAcc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.base_model(batch['input_ids'], batch['attention_mask'])
        loss = nn.functional.cross_entropy(out, batch['target'])
        self.validAcc(out, batch['target'])
        self.log('valid_acc', self.validAcc, on_step=True, on_epoch=True)
        self.log("validation_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        out = self.base_model(batch['input_ids'], batch['attention_mask'])
        return torch.nn.functional.softmax(out)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.base_model.parameters(), lr=0.001)
