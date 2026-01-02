# Gold Price Prediction Using Machine Learning

This project focuses on predicting gold prices using machine learning techniques in Python. Historical gold price data in multiple currencies is analyzed, enriched with feature engineering, and used to train regression models for accurate price prediction.

---

## 📊 Dataset
- Source: Public gold price dataset
- Total observations: **4,689**
- Features: **18 numerical features**
- Currencies: USD, GBP, EURO (AM & PM prices)

---

## 🛠 Feature Engineering
To improve model performance, several feature engineering steps were applied:

- Date-based features (Year, Month, Day, DayOfWeek)
- Intraday price changes (AM → PM differences)
- Daily returns (percentage change)
- Rolling averages (7-day and 30-day moving averages)
- Missing PM values were filled using time-series–aware methods

After feature extraction, the original date column was removed, leaving a fully numerical dataset suitable for machine learning.

---

## 🤖 Models Used
The following regression models were trained and evaluated:

- **Linear Regression** (Baseline model)
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

---

## 📈 Model Performance

| Model              | R² Score | RMSE |
|-------------------|---------|------|
| Linear Regression | 1.0000  | ~0.0 |
| Gradient Boosting | 0.9999  | 3.90 |
| Random Forest     | 0.9999  | 3.94 |

🏆 **Best Model:** Linear Regression

---

## 📌 Key Insights
- Strong linear relationships exist between engineered features and gold prices.
- Feature engineering significantly improved predictive performance.
- Even simple models like Linear Regression can achieve excellent results with well-prepared data.

---

## 🚀 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit (for deployment)

