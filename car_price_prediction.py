# =============================================================
# Used Car Price Prediction & Dealership Pricing Intelligence
# Author: Omkar Pallerla | MS Business Analytics, ASU
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
COLORS = ['#4f9cf9', '#06d6a0', '#7c3aed', '#f59e0b', '#ef4444']

# ── 1. LOAD & EXPLORE ───────────────────────────────────────
df = pd.read_csv('car_data.csv')
print(f"Shape: {df.shape}")
print(df.head())
print(df.describe())

# Age feature
df['Car_Age'] = 2024 - df['Year']
df.drop('Year', axis=1, inplace=True)

# Encode categoricals
le = LabelEncoder()
for col in ['Fuel_Type', 'Seller_Type', 'Transmission']:
    df[col] = le.fit_transform(df[col])

# ── 2. VIF – multicollinearity check ────────────────────────
X_vif = df.drop(['Car_Name', 'Selling_Price'], axis=1)
vif_df = pd.DataFrame({
    'Feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).sort_values('VIF', ascending=False)
print("\nVIF Scores:\n", vif_df)

# ── 3. TRAIN/TEST SPLIT ─────────────────────────────────────
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 4. MODELS ───────────────────────────────────────────────
models = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
models['Linear Regression'] = lr

# Polynomial Regression (degree=2)
poly_pipe = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),
                       ('lr', LinearRegression())])
poly_pipe.fit(X_train, y_train)
models['Polynomial Regression'] = poly_pipe

# Ridge
ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)
models['Ridge Regression'] = ridge

# OLS for statistics
X_train_sm = sm.add_constant(X_train)
ols = sm.OLS(y_train, X_train_sm).fit()
print("\nOLS Summary:")
print(ols.summary())

# ── 5. EVALUATE ─────────────────────────────────────────────
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    results[name] = {'r2': r2, 'rmse': rmse, 'mae': mae, 'preds': y_pred}
    print(f"{name:25s} R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

# ── 6. EXPORT PREDICTIONS (Power BI feed) ───────────────────
best_preds = results['Polynomial Regression']['preds']
output_df = X_test.copy()
output_df['Actual_Price']    = y_test.values
output_df['Predicted_Price'] = best_preds
output_df['Price_Deviation'] = output_df['Actual_Price'] - output_df['Predicted_Price']
output_df['Deviation_Pct']   = (output_df['Price_Deviation'] / output_df['Actual_Price']) * 100
output_df['Status'] = output_df['Deviation_Pct'].apply(
    lambda x: 'OVERPRICED' if x < -10 else ('UNDERPRICED' if x > 10 else 'FAIR'))
output_df.to_csv('outputs/predicted_prices.csv', index=False)
print("\nExported: outputs/predicted_prices.csv")

# ── 7. VISUALIZATIONS ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#0d1117')

# Model comparison
ax = axes[0, 0]
r2s = [results[n]['r2'] for n in results]
ax.bar(list(results.keys()), r2s, color=COLORS[:3])
ax.set_ylim(0, 1.05)
ax.set_title('R² Score — Model Comparison', color='white', pad=12)
ax.set_ylabel('R² Score')
for i, v in enumerate(r2s):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', color='white')

# Actual vs Predicted
ax = axes[0, 1]
ax.scatter(y_test, best_preds, alpha=0.6, color='#4f9cf9', s=30)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Price'); ax.set_ylabel('Predicted Price')
ax.set_title('Actual vs Predicted — Polynomial Regression', color='white', pad=12)

# Residuals
ax = axes[1, 0]
residuals = y_test.values - best_preds
ax.scatter(best_preds, residuals, alpha=0.5, color='#06d6a0', s=30)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Predicted Price'); ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Fitted', color='white', pad=12)

# Pricing status distribution
ax = axes[1, 1]
status_counts = output_df['Status'].value_counts()
colors_s = {'OVERPRICED': '#ef4444', 'UNDERPRICED': '#06d6a0', 'FAIR': '#f59e0b'}
ax.pie(status_counts, labels=status_counts.index,
       colors=[colors_s[s] for s in status_counts.index],
       autopct='%1.1f%%', startangle=90)
ax.set_title('Inventory Pricing Status', color='white', pad=12)

plt.tight_layout()
plt.savefig('outputs/car_price_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved: outputs/car_price_analysis.png")
plt.show()
