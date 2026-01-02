# ===============================
# 1️⃣ GEREKLİ IMPORTLAR
# ===============================

import pandas as pd

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.train import split_data
from src.evaluation import evaluate
from src.save_model import save_model

from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import GradientBoostingModel


# ===============================
# 2️⃣ VERİYİ YÜKLE
# ===============================

df = load_data()


# ===============================
# 3️⃣ PREPROCESSING
# ===============================

X, y = preprocess(df)


# ===============================
# 4️⃣ TRAIN - TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = split_data(X, y)


# ===============================
# 5️⃣ MODELLERİ TANIMLA
# ===============================

models = {
    "Linear Regression": LinearRegressionModel(),
    "Random Forest": RandomForestModel(),
    "Gradient Boosting": GradientBoostingModel()
}


# ===============================
# 6️⃣ EĞİT – TAHMİN – DEĞERLENDİR
# ===============================

results = []

for model_name, model in models.items():
    print(f"\n🚀 Model eğitiliyor: {model_name}")

    model.train(X_train, y_train)
    predictions = model.predict(X_test)

    r2, rmse = evaluate(y_test, predictions)

    results.append({
        "Model": model_name,
        "R2_Score": r2,
        "RMSE": rmse
    })


# ===============================
# 7️⃣ SONUÇLARI TABLO HALİNE GETİR
# ===============================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2_Score", ascending=False)

print("\n📊 Model Karşılaştırma Sonuçları:")
print(results_df)


# ===============================
# 8️⃣ EN İYİ MODELİ SEÇ
# ===============================

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

print(f"\n🏆 En iyi model: {best_model_name}")


# ===============================
# 9️⃣ MODEL VE KOLONLARI KAYDET
# ===============================

save_model(best_model.model, X.columns)

print("\n✅ Model ve kolonlar başarıyla kaydedildi.")
