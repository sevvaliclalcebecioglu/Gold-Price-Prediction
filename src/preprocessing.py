from src.config import TARGET
import pandas as pd

def preprocess(df):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X = pd.get_dummies(X, drop_first=True)
    return X, y
