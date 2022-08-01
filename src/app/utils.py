import os
import sys

import mlflow.pyfunc
import numpy as np
import torch
from torch.nn.functional import softmax

from src.data.utils import resolve_encodings_and_normalize

sys.path.insert(0, os.path.join(os.getcwd(), 'src/models'))
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
mlflow.set_registry_uri(os.environ.get('MLFLOW_TRACKING_URI'))


def get_model(model_name, model_version):
    print('loading_model')
    model = mlflow.pytorch.load_model(model_uri=f'models:/{model_name}/Production')  # Uri через models:/работает не всегда
    print('loading done')
    return model


def preprocess_input_data(data):
    data['discourse_text'] = data['discourse_text'].apply(lambda x: resolve_encodings_and_normalize(x))
    data['essay_text'] = data['essay_text'].apply(lambda x: resolve_encodings_and_normalize(x))
    data['text'] = data['discourse_type'] + ' ' + data['discourse_text'] + '[SEP]' + data['essay_text']

    return data


def infer_model(data_loader, model):
    preds = []
    model.eval()
    for data in data_loader:
        ids = data['input_ids']
        mask = data['attention_mask']
        with torch.no_grad():
            y_preds = model(ids, mask)
        y_preds = torch.argmax(softmax(y_preds), dim=1).numpy()
        preds.append(y_preds)
    predictions = np.concatenate(preds)
    return predictions


def decode_infer(preds, label_encoder):
    res = label_encoder.inverse_transform(preds)
    return res
