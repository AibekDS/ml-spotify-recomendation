from xgboost import XGBClassifier
import requests
import pickle
import pandas as pd
import numpy as np

def from_pkl(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        obj = pickle.loads(response.content)
        print("Объект успешно загружен из удалённого файла!")
        return obj
    else:
        print(f"Ошибка при скачивании: {response.status_code}")

@transformer
def transform(X, *args, **kwargs):
    # best model url
    model_xgb_url = "https://drive.google.com/uc?export=download&id=1GTuephEZVzwWC3STGrGbVTw64MgM8-19"
    # Download
    model_xgb = from_pkl(model_xgb_url)

    threshold = 0.3

    y_pred_proba = model_xgb.predict_proba(X)
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
    print(y_pred)
    return y_pred
