import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scipy import stats

warnings.filterwarnings("ignore")

# data files should be in the same folder as this script
base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(base_dir, "Energy2017_2019.xlsx")
test_file = os.path.join(base_dir, "Energy2020.xlsx")
weather_file = os.path.join(base_dir, "WeatherData.csv")

# load

print("Loading data...")

expected_col = "Auth Date"

def load_file(path):
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df.columns = [str(c).strip() for c in df.columns]
        if expected_col in df.columns:
            return df
    raise ValueError(f"Could not find '{expected_col}' column in any sheet of {path}")

train_raw = load_file(train_file)
test_raw  = load_file(test_file)
raw = pd.concat([train_raw, test_raw], ignore_index=True)

print(f"  Training rows: {len(train_raw):,}")
print(f"  Test rows: {len(test_raw):,}")
print(f"  Combined rows: {len(raw):,}")

# clean

print("\nCleaning data...")

raw["Auth Date"] = pd.to_datetime(raw["Auth Date"], errors="coerce")
before = len(raw)
raw = raw.dropna(subset=["Auth Date"])
print(f"  Dropped {before - len(raw):,} rows with unparseable Auth Date")

raw["Energy Season"] = pd.to_numeric(raw["Energy Season"], errors="coerce")
before = len(raw)
raw = raw.dropna(subset=["Energy Season"])
raw["Energy Season"] = raw["Energy Season"].astype(int)
print(f"  Dropped {before - len(raw):,} rows with missing Energy Season")

# coerce numeric columns
for col in ["Price Per Gallon ($)", "Rate of Consumption (gal/day)", "Ben Level",
            "Amount 1($)", "Amount 2($)", "Inv Dol($)", "Inv Gal($)"]:
    if col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

for date_col in ["Inv.Date", "Bill Received", "Del Date"]:
    if date_col in raw.columns:
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")

raw = raw.dropna(how="all")

print(f"  Rows after cleaning: {len(raw):,}")

# week of season assignments

print("\nAssigning week_of_season...")

season_min_date = raw.groupby("Energy Season")["Auth Date"].min().rename("season_start")
raw = raw.join(season_min_date, on="Energy Season")
raw["week_of_season"] = ((raw["Auth Date"] - raw["season_start"]).dt.days // 7).astype(int)

raw["inv_date_lag"] = (raw["Inv.Date"] - raw["Auth Date"]).dt.days
raw["bill_received_lag"] = (raw["Bill Received"] - raw["Auth Date"]).dt.days
raw["del_date_lag"] = (raw["Del Date"] - raw["Auth Date"]).dt.days

weekly_demand = (
    raw.groupby(["Energy Season", "week_of_season"])
    .size()
    .reset_index()
    .rename(columns={0: "demand"})
)

# predictor variables

print("Developing predictor features...")

raw["is_crisis"] = (
    raw["Auth Type1"].astype(str).str.lower().str.contains("crisis") |
    raw["Auth Type2"].astype(str).str.lower().str.contains("crisis")
)

weekly_features = (
    raw.groupby(["Energy Season", "week_of_season"])
    .agg(
        week_total_records=("App#", "count"),
        avg_price=("Price Per Gallon ($)", "mean"),
        avg_consumption=("Rate of Consumption (gal/day)", "mean"),
        avg_ben_level=("Ben Level", "mean"),
        crisis_count=("is_crisis", "sum"),
    )
    .reset_index()
)

weekly_features = weekly_features.sort_values(["Energy Season", "week_of_season"])
weekly_features["prior_week_demand"] = weekly_features.groupby("Energy Season")["week_total_records"].shift(1)
weekly_features["prior_week_avg_price"] = weekly_features.groupby("Energy Season")["avg_price"].shift(1)
weekly_features["prior_week_avg_consumption"] = weekly_features.groupby("Energy Season")["avg_consumption"].shift(1)
weekly_features["prior_week_avg_ben_level"] = weekly_features.groupby("Energy Season")["avg_ben_level"].shift(1)
weekly_features["prior_week_crisis_count"] = weekly_features.groupby("Energy Season")["crisis_count"].shift(1)

# week-over-week change
weekly_features["demand_change"] = weekly_features.groupby("Energy Season")["week_total_records"].diff()
weekly_features["prior_week_demand_change"] = weekly_features.groupby("Energy Season")["demand_change"].shift(1)

# rolling averages
weekly_features["rolling_demand_2w"] = (
    weekly_features.groupby("Energy Season")["week_total_records"]
    .transform(lambda x: x.shift(1).rolling(2).mean())
)
weekly_features["rolling_demand_3w"] = (
    weekly_features.groupby("Energy Season")["week_total_records"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
)

# more predictor variables

weekly_features_ext = (
    raw.groupby(["Energy Season", "week_of_season"])
    .agg(
        avg_amount1=("Amount 1($)", "mean"),
        avg_amount2=("Amount 2($)", "mean"),
        avg_inv_dol=("Inv Dol($)", "mean"),
        avg_inv_gal=("Inv Gal($)", "mean"),
        avg_inv_date_lag=("inv_date_lag", "mean"),
        avg_bill_received_lag=("bill_received_lag", "mean"),
        avg_del_date_lag=("del_date_lag", "mean"),
        unique_vendors=("Vendor", "nunique"),
        unique_pay_for=("Pay For", "nunique"),
    )
    .reset_index()
)

weekly_features_ext = weekly_features_ext.sort_values(["Energy Season", "week_of_season"])
weekly_features_ext["prior_week_avg_amount1"] = weekly_features_ext.groupby("Energy Season")["avg_amount1"].shift(1)
weekly_features_ext["prior_week_avg_amount2"] = weekly_features_ext.groupby("Energy Season")["avg_amount2"].shift(1)
weekly_features_ext["prior_week_avg_inv_dol"] = weekly_features_ext.groupby("Energy Season")["avg_inv_dol"].shift(1)
weekly_features_ext["prior_week_avg_inv_gal"] = weekly_features_ext.groupby("Energy Season")["avg_inv_gal"].shift(1)
weekly_features_ext["prior_week_avg_inv_date_lag"] = weekly_features_ext.groupby("Energy Season")["avg_inv_date_lag"].shift(1)
weekly_features_ext["prior_week_avg_bill_received_lag"] = weekly_features_ext.groupby("Energy Season")["avg_bill_received_lag"].shift(1)
weekly_features_ext["prior_week_avg_del_date_lag"] = weekly_features_ext.groupby("Energy Season")["avg_del_date_lag"].shift(1)
weekly_features_ext["prior_week_unique_vendors"] = weekly_features_ext.groupby("Energy Season")["unique_vendors"].shift(1)
weekly_features_ext["prior_week_unique_pay_for"] = weekly_features_ext.groupby("Energy Season")["unique_pay_for"].shift(1)

# weekly HDD from weather data

print("Computing weekly HDD from weather data...")

weather = pd.read_csv(weather_file)
weather["DATE"] = pd.to_datetime(weather["DATE"])
weather["TMAX"] = pd.to_numeric(weather["TMAX"], errors="coerce")
weather["TMIN"] = pd.to_numeric(weather["TMIN"], errors="coerce")

# average across all reporting stations per day, then compute heating degree days (base 65F)
daily_avg = weather.groupby("DATE")[["TMAX", "TMIN"]].mean().reset_index()
daily_avg["hdd"] = (65 - (daily_avg["TMAX"] + daily_avg["TMIN"]) / 2).clip(lower=0)

sorted_starts = season_min_date.sort_values()

def assign_season_week(date):
    season, start = None, None
    for s, d in sorted_starts.items():
        if date >= d:
            season, start = s, d
    if season is None:
        return pd.Series([None, None])
    return pd.Series([season, (date - start).days // 7])

daily_avg[["Energy Season", "week_of_season"]] = daily_avg["DATE"].apply(assign_season_week)
daily_avg = daily_avg.dropna(subset=["Energy Season"])
daily_avg["Energy Season"] = daily_avg["Energy Season"].astype(int)
daily_avg["week_of_season"] = daily_avg["week_of_season"].astype(int)

weekly_hdd = (
    daily_avg.groupby(["Energy Season", "week_of_season"])["hdd"]
    .sum()
    .reset_index()
    .rename(columns={"hdd": "week_hdd"})
)
weekly_hdd = weekly_hdd.sort_values(["Energy Season", "week_of_season"])
weekly_hdd["prior_week_hdd"] = weekly_hdd.groupby("Energy Season")["week_hdd"].shift(1)

# model dataset

ext_cols = ["Energy Season", "week_of_season",
            "prior_week_avg_amount1", "prior_week_avg_amount2",
            "prior_week_avg_inv_dol", "prior_week_avg_inv_gal",
            "prior_week_avg_inv_date_lag", "prior_week_avg_bill_received_lag",
            "prior_week_avg_del_date_lag", "prior_week_unique_vendors",
            "prior_week_unique_pay_for"]

modeling = weekly_demand.merge(
    weekly_features[["Energy Season", "week_of_season",
                      "prior_week_demand", "prior_week_avg_price",
                      "prior_week_avg_consumption", "prior_week_avg_ben_level",
                      "prior_week_crisis_count", "prior_week_demand_change",
                      "rolling_demand_2w", "rolling_demand_3w"]],
    on=["Energy Season", "week_of_season"],
    how="left"
).merge(
    weekly_features_ext[ext_cols],
    on=["Energy Season", "week_of_season"],
    how="left"
).merge(
    weekly_hdd[["Energy Season", "week_of_season", "prior_week_hdd"]],
    on=["Energy Season", "week_of_season"],
    how="left"
)

predictors = [
    "week_of_season",
    "prior_week_demand",
    "prior_week_avg_price",
    "prior_week_avg_consumption",
    "prior_week_avg_ben_level",
    "prior_week_crisis_count",
    "prior_week_avg_amount1",
    "prior_week_avg_amount2",
    "prior_week_avg_inv_dol",
    "prior_week_avg_inv_gal",
    "prior_week_avg_inv_date_lag",
    "prior_week_avg_bill_received_lag",
    "prior_week_avg_del_date_lag",
    "prior_week_unique_vendors",
    "prior_week_unique_pay_for",
    "prior_week_demand_change",
    "prior_week_hdd",
    "rolling_demand_2w",
    "rolling_demand_3w",
]

# LASSO confirmed variables used for LASSO and GBR to reduce overfitting
trimmed_predictors = [
    "week_of_season",
    "prior_week_avg_del_date_lag",
    "prior_week_avg_amount1",
    "prior_week_crisis_count",
    "prior_week_hdd",
    "prior_week_avg_amount2",
    "prior_week_demand_change",
    "prior_week_demand",
    "rolling_demand_2w",
    "rolling_demand_3w",
]

for col in predictors:
    if col in modeling.columns:
        modeling[col] = modeling[col].fillna(0)

modeling = modeling.dropna(subset=predictors + ["demand"])
modeling = modeling.reset_index(drop=True)

print(f"  Modeling dataset rows: {len(modeling):,}")
print(f"  Seasons: {sorted(modeling['Energy Season'].unique())}")

# train & test

train = modeling[modeling["Energy Season"].isin([2017, 2018, 2019])].copy()
test  = modeling[modeling["Energy Season"] == 2020].copy().sort_values("week_of_season")

X_train = train[predictors]
y_train = train["demand"]
X_test = test[predictors]
y_test = test["demand"]

print(f"\nTrain weeks: {len(train)} | Test weeks: {len(test)}")

# metrics

def compute_metrics(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": label, "MAE": mae, "RMSE": rmse, "R2": r2}

print("MODEL 1: Single-Variable")

single_results = []
single_preds = {}

for pred in predictors:
    slope, intercept, r_value, p_value, std_err = stats.linregress(X_train[pred], y_train)
    preds = slope * X_test[pred].values + intercept
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    single_results.append({
        "Predictor": pred,
        "Coeff": round(slope, 4),
        "p-value": round(p_value, 4),
        "R2": round(r2, 4),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "sig": "*" if p_value < 0.05 else "",
    })
    single_preds[pred] = preds

single_df = pd.DataFrame(single_results).sort_values("R2", ascending=False).reset_index(drop=True)
single_df.index += 1
print(single_df.to_string(
    formatters={
        "Coeff": "{:.4f}".format,
        "p-value": "{:.4f}".format,
        "R2": "{:.4f}".format,
        "MAE": "{:.2f}".format,
        "RMSE": "{:.2f}".format,
    }
))
print("  (* = significant at p < 0.05)")

best_predictor = single_df.sort_values("MAE").iloc[0]["Predictor"]
best_single_preds = single_preds[best_predictor]
print(f"\n  Best single predictor (baseline): {best_predictor}")

# LASSO regression

print("MODEL 2: LASSO Regression")

tscv = TimeSeriesSplit(n_splits=3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso = LassoCV(cv=tscv, random_state=37, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
lasso_preds = lasso.predict(X_test_scaled)

print(f"\n  alpha: {lasso.alpha_:.6f}")
print("\n  Retained non-zero coefficients:")
coef_df = pd.DataFrame({"Predictor": predictors, "Coefficient": lasso.coef_})
coef_df = coef_df[coef_df["Coefficient"] != 0].sort_values("Coefficient", key=abs, ascending=False)

col_w = max(len(p) for p in coef_df["Predictor"]) + 2
print(f"  {'Predictor':<{col_w}} {'Coefficient':>12}")
print(f"  {'-'*col_w} {'------------':>12}")
for _, row in coef_df.iterrows():
    print(f"  {row['Predictor']:<{col_w}} {row['Coefficient']:>12.4f}")

# gradient boosting

print("MODEL 3: Gradient Boosting Regressor")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [1, 2, 3],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_samples_leaf": [3, 5, 7],
    "subsample": [0.7, 0.8, 1.0],
}
gbr_search = GridSearchCV(
    GradientBoostingRegressor(random_state=37),
    param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)
gbr_search.fit(X_train[trimmed_predictors], y_train)
gbr_preds = gbr_search.best_estimator_.predict(X_test[trimmed_predictors])

print(f"\n  Best parameters: {gbr_search.best_params_}")

fi_df = pd.DataFrame({
    "Predictor": trimmed_predictors,
    "Importance": gbr_search.best_estimator_.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)
fi_df.index += 1

col_w = 30
print("\nFeature importances:")
print(f"  {'Predictor':<{col_w}} {'Importance':>10}")
print(f"  {'-' * col_w} {'----------':>10}")
for _, row in fi_df.iterrows():
    print(f"  {row['Predictor']:<{col_w}} {row['Importance']:>10.4f}")

# summary

print("Final Model Performance on 2020 Test Data")

m1 = compute_metrics(y_test, best_single_preds, label=f"Single-Variable ({best_predictor})")
m2 = compute_metrics(y_test, lasso_preds, label="LASSO Regression")
m3 = compute_metrics(y_test, gbr_preds, label="Gradient Boosting")

final_metrics = pd.DataFrame([m1, m2, m3])
print(final_metrics.to_string(
    index=False,
    formatters={"MAE": "{:.2f}".format, "RMSE": "{:.2f}".format, "R2": "{:.4f}".format}
))

# observed vs predicted line graph

weeks = test["week_of_season"].values

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(weeks, y_test.values, label="Observed", linewidth=2, marker="o", markersize=4)
ax1.plot(weeks, best_single_preds, label=f"Single-Variable ({best_predictor})", linewidth=1.5, linestyle="--", marker="s", markersize=4)
ax1.plot(weeks, lasso_preds, label="LASSO", linewidth=1.5, linestyle="-.", marker="^", markersize=4)
ax1.plot(weeks, gbr_preds, label="Gradient Boosting", linewidth=1.5, linestyle=":", marker="D", markersize=4)
ax1.set_title("2020 Season: Observed vs Predicted Weekly Authorization Volume", fontsize=13)
ax1.set_xlabel("Week of Season", fontsize=11)
ax1.set_ylabel("Total Authorization Records", fontsize=11)
ax1.legend(fontsize=9)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# MAE & RMSE bar chart

model_names = ["Single-Variable", "LASSO", "Gradient Boosting"]
mae_values = [m1["MAE"], m2["MAE"], m3["MAE"]]
rmse_values = [m1["RMSE"], m2["RMSE"], m3["RMSE"]]

x = np.arange(len(model_names))
bar_width = 0.35

fig2, ax2 = plt.subplots(figsize=(9, 5))
bars_mae = ax2.bar(x - bar_width / 2, mae_values, bar_width, label="MAE", color="steelblue")
bars_rmse = ax2.bar(x + bar_width / 2, rmse_values, bar_width, label="RMSE", color="coral")

for bar in bars_mae:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
for bar in bars_rmse:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

ax2.set_title("Prediction Error by Model (2020 Test Set)", fontsize=13)
ax2.set_xlabel("Model", fontsize=11)
ax2.set_ylabel("Prediction Error (Number of Authorization Records)", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# LASSO coefficients bar chart

fig3, ax3 = plt.subplots(figsize=(9, 5))
colors = ["steelblue" if v > 0 else "coral" for v in coef_df["Coefficient"]]
ax3.bar(coef_df["Predictor"], coef_df["Coefficient"], color=colors)
ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax3.set_title("LASSO Retained Coefficients (Standardized Predictors)", fontsize=13)
ax3.set_xlabel("Predictor", fontsize=11)
ax3.set_ylabel("Coefficient Value", fontsize=11)
ax3.tick_params(axis="x", rotation=20)
ax3.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

