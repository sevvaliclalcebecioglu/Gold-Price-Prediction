import joblib

def save_model(model, columns):
    joblib.dump(model, "best_model.pkl")
    joblib.dump(columns, "columns.pkl")

