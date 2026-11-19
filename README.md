# 🚗 Used Car Price Prediction & Valuation Analytics

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **Statistically rigorous vehicle valuation model — outputs power a Power BI inventory pricing dashboard for dealership operations.**

---

## 📌 Business Overview

In the pre-owned automotive industry, pricing efficiency is the #1 profitability lever. This project builds a valuation model to predict used car Selling_Price, enabling dealerships to optimize procurement bids, set data-driven listing prices, and reduce Days to Sell with dynamic pricing recommendations.

The model output feeds a **Power BI Inventory Dashboard** showing real-time price deviation (Actual vs Model Price) by make, fuel type, and year cohort.

---

## 📊 Technical Approach

| Step | Detail |
|------|--------|
| **Data Source** | car_data.csv — Year, Present Price, Kms Driven, Fuel Type, Transmission |
| **Baseline** | Multiple Linear Regression (OLS via Statsmodels) |
| **Primary Model** | Polynomial Regression (degree=2) — captures non-linear depreciation |
| **AutoML Benchmark** | LazyPredict — rapid comparison of 20+ regressors |
| **Diagnostics** | VIF (multicollinearity), Q-Q Plot, Residuals vs Fitted |
| **Evaluation** | R², Adjusted R², RMSE, MAE |

---

## 📈 Key Findings

- 📉 **Depreciation is Non-Linear** — Polynomial Regression beat linear by 11% RMSE; cars depreciate fastest in years 1-3
- 🏆 **Top Predictor: Present Price** — Original showroom price explains 74% of resale value variance
- ⛽ **Diesel Premium** — Diesel vehicles retained 18% higher resale value vs petrol
- 🔢 **VIF Cleanup** — Removed Kms_Driven squared after VIF > 10, improving stability
- 🤖 **LazyPredict Winner** — GradientBoostingRegressor led with R²=0.96

---

## 🧠 BI Integration Pipeline

```
Python Model → predicted_prices.csv → Power BI (via ADF refresh) → Pricing Dashboard
```

Dashboard shows: underpriced listings (quick flip opportunities), overpriced listings (aging risk), segment-level price curves by fuel type and year.

---

## 🛠 Tools & Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Modeling | Scikit-Learn, Statsmodels, LazyPredict |
| Diagnostics | VIF, Residual plots, Q-Q plots |
| Visualization | Seaborn, Matplotlib |
| BI Output | CSV → Power BI Pricing Dashboard |

---

## 📂 Project Structure

```
Used-Car-Price-Prediction/
├── car_price_prediction.py
├── data/car_data.csv      # download - see data/README.md
├── notebooks/Car_Price_Prediction.ipynb
├── outputs/predicted_prices.csv   # created on run
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

First download the public dataset as described in [`data/README.md`](data/README.md).

```bash
git clone https://github.com/omkarpallerla/Used-Car-Price-Prediction.git
cd Used-Car-Price-Prediction
pip install -r requirements.txt
jupyter notebook notebooks/Car_Price_Prediction.ipynb
```

---

## 📊 Model Results

| Model | R² | RMSE | Notes |
|-------|-----|------|-------|
| **Polynomial Regression** | **0.93** | **1.21** | Best interpretable |
| Gradient Boosting | 0.96 | 0.89 | Best raw accuracy |
| Linear Regression | 0.82 | 2.04 | Baseline |

---


## 🧪 Data & Reproducibility

`car_data.csv` is a public third-party dataset and is **not redistributed** here - see `data/README.md`. The Power BI / ADF refresh in the pipeline diagram is the intended integration design; this repo ships the modelling code and CSV hand-off only. Exact metrics depend on library versions and the train/test split.

---

<div align="center">
  <sub><a href="https://github.com/omkarpallerla">Omkar Pallerla</a> · MS Business Analytics, ASU · Business Intelligence Analyst · Google Cloud | Azure | Databricks | Snowflake Certified</sub>
</div>