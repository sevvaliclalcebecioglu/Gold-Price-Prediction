from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

def evaluate(y_test, predictions):
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return r2, rmse

def create_comparison_df(y_test, predictions):
    df = pd.DataFrame({
        "gercek_teslimat_zamanı": y_test.reset_index(drop=True),
        "tahmin_teslimat_zamanı": predictions
    })
    return df
