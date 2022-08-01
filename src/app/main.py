import os
import pickle

import pandas as pd
from fastapi import FastAPI, UploadFile, File

from src.config import CFG
from src.data.data_processing_utils import make_tokenizer_dataset_loader
from .utils import get_model, preprocess_input_data, infer_model, decode_infer

app = FastAPI()
model = get_model('feedback')

with open(os.path.join(os.getcwd(), 'models/label_encoder.sk'), 'rb') as f:
    le = pickle.load(f)


@app.post('/predict')
async def get_predict(df: UploadFile = File(...)):
    if df.filename.endswith('.csv'):
        with open(df.filename, 'wb') as f:
            f.write(df.file.read())
    data = pd.read_csv(df.filename)

    data_clean = preprocess_input_data(data)
    tokenizer, dataset, loader = make_tokenizer_dataset_loader(CFG, data_clean, 'test')
    infer = infer_model(loader, model)
    predict = decode_infer(infer, le)
    return predict.tolist()
