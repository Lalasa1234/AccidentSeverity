import numpy as np
import joblib
from xgboost import XGBClassifier

def model_prediction(model, data):
    return model.predict(data)
    