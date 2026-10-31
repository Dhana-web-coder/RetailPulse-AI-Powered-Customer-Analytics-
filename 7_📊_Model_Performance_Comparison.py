import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page configuration
st.set_page_config(
    page_title="Model Performance Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ARIMA vs SARIMA Model Comparison")
st.write("Compare two forecasting models using actual sales data.")

# Project file paths
BASE_DIR = Path(__file__).resolve().parent.parent

ARIMA_FILE = BASE_DIR / "arima_evaluation_results.csv"
SARIMA_FILE = BASE_DIR / "sarima_evaluation_results.csv"


# Load data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    required_columns = [
        "Date",
        "Actual Sales",
        "Predicted Sales"
    ]

    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"Required columns are missing in {file_path.name}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df["Actual Sales"] = pd.to_numeric(
        df["Actual Sales"], errors="coerce"
    )

    df["Predicted Sales"] = pd.to_numeric(
        df["Predicted Sales"], errors="coerce"
    )

    df = df.dropna(subset=required_columns)

    return df.sort_values("Date")


# Check files and load
if not ARIMA_FILE.exists() or not SARIMA_FILE.exists():
    st.error("ARIMA or SARIMA evaluation file was not found.")
    st.stop()

try:
    arima_df = load_data(ARIMA_FILE)
    sarima_df = load_data(SARIMA_FILE)
except Exception as e:
    st.error(f"Unable to load data: {e}")
    st.stop()


# Align models by date
arima_df = arima_df.rename(
    columns={"Predicted Sales": "ARIMA Prediction"}
)

sarima_df = sarima_df.rename(
    columns={
        "Actual Sales": "SARIMA Actual Sales",
        "Predicted Sales": "SARIMA Prediction"
    }
)

comparison = pd.merge(
    arima_df[
        ["Date", "Actual Sales", "ARIMA Prediction"]
    ],
    sarima_df[
        ["Date", "SARIMA Actual Sales", "SARIMA Prediction"]
    ],
    on="Date",
    how="inner"
)

if comparison.empty:
    st.error("No matching dates found in both files.")
    st.stop()

if not np.allclose(
    comparison["Actual Sales"],
    comparison["SARIMA Actual Sales"],
    rtol=0,
    atol=0.01
):
    st.warning(
        "Actual sales differ between files. "
        "The ARIMA file's actual sales are used for comparison."
    )


# Calculate metrics
actual = comparison["Actual Sales"].to_numpy()

arima_pred = comparison["ARIMA Prediction"].to_numpy()
sarima_pred = comparison["SARIMA Prediction"].to_numpy()


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    denominator = np.abs(actual) + np.abs(predicted)

    smape_values = np.divide(
        2 * np.abs(actual - predicted),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0
    )

    smape = np.mean(smape_values) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE (%)": smape
    }


arima_metrics = calculate_metrics(actual, arima_pred)
sarima_metrics = calculate_metrics(actual, sarima_pred)

metrics_df = pd.DataFrame(
    [arima_metrics, sarima_metrics],
    index=["ARIMA", "SARIMA"]
)


# Summary metrics
st.subheader("📌 Evaluation Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Evaluation Days", len(comparison))

col2.metric(
    "Zero-Sales Days",
    int((actual == 0).sum())
)

col3.metric("Models Compared", 2)


# Metrics table
st.subheader("📊 Accuracy Metrics")

st.dataframe(
    metrics_df.round(2),
    width="stretch"
)

st.caption(
    "Lower MAE and RMSE indicate smaller prediction errors. "
    "Lower sMAPE indicates better relative accuracy. "
    "sMAPE is used because actual sales contain zero values."
)


# Metric comparison chart
st.subheader("📉 Error Metrics Comparison")

selected_metric = st.selectbox(
    "Select a metric",
    ["MAE", "RMSE", "sMAPE (%)"]
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    metrics_df.index,
    metrics_df[selected_metric]
)

ax.set_title(f"ARIMA vs SARIMA - {selected_metric}")
ax.set_ylabel(selected_metric)
ax.grid(axis="y", alpha=0.3)

st.pyplot(fig)
plt.close(fig)
# ==========================================
# RESIDUAL ANALYSIS
# ==========================================

st.subheader("📊 Residual Analysis")

# Calculate residuals
comparison["ARIMA Residual"] = (
    comparison["Actual Sales"] -
    comparison["ARIMA Prediction"]
)

comparison["SARIMA Residual"] = (
    comparison["Actual Sales"] -
    comparison["SARIMA Prediction"]
)

# Display average residual
col1, col2 = st.columns(2)

col1.metric(
    "ARIMA Mean Residual",
    f"{comparison['ARIMA Residual'].mean():,.2f}"
)

col2.metric(
    "SARIMA Mean Residual",
    f"{comparison['SARIMA Residual'].mean():,.2f}"
)

# Residual comparison chart
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    comparison["Date"],
    comparison["ARIMA Residual"],
    label="ARIMA Residual"
)

ax.plot(
    comparison["Date"],
    comparison["SARIMA Residual"],
    label="SARIMA Residual"
)

ax.axhline(
    y=0,
    linestyle="--",
    label="Zero Error"
)

ax.set_title("ARIMA vs SARIMA Residual Analysis")
ax.set_xlabel("Date")
ax.set_ylabel("Residual Error")
ax.legend()
ax.grid(alpha=0.3)

plt.xticks(rotation=45)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ==========================================
# ERROR DISTRIBUTION HISTOGRAM
# ==========================================

st.subheader("📊 Error Distribution Histogram")

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    comparison["ARIMA Residual"],
    bins=30,
    alpha=0.6,
    label="ARIMA Error"
)

ax.hist(
    comparison["SARIMA Residual"],
    bins=30,
    alpha=0.6,
    label="SARIMA Error"
)

ax.set_title("ARIMA vs SARIMA Error Distribution")
ax.set_xlabel("Residual Error")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# ==========================================
# MONTHLY PERFORMANCE ANALYSIS
# ==========================================

st.subheader("📅 Monthly Performance Analysis")

monthly_df = comparison.copy()

monthly_df["Month"] = (
    monthly_df["Date"].dt.to_period("M").astype(str)
)

monthly_df["ARIMA Error"] = (
    monthly_df["Actual Sales"] -
    monthly_df["ARIMA Prediction"]
).abs()

monthly_df["SARIMA Error"] = (
    monthly_df["Actual Sales"] -
    monthly_df["SARIMA Prediction"]
).abs()

monthly_performance = monthly_df.groupby("Month").agg(
    ARIMA_MAE=("ARIMA Error", "mean"),
    SARIMA_MAE=("SARIMA Error", "mean")
).reset_index()

st.dataframe(
    monthly_performance.round(2),
    width="stretch"
)

# Monthly comparison chart
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    monthly_performance["Month"],
    monthly_performance["ARIMA_MAE"],
    marker="o",
    label="ARIMA MAE"
)

ax.plot(
    monthly_performance["Month"],
    monthly_performance["SARIMA_MAE"],
    marker="o",
    label="SARIMA MAE"
)

ax.set_title("Monthly Forecast Error Comparison")
ax.set_xlabel("Month")
ax.set_ylabel("Mean Absolute Error (MAE)")
ax.legend()
ax.grid(alpha=0.3)

plt.xticks(rotation=45)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ==========================================
# FORECAST BIAS ANALYSIS
# ==========================================

st.subheader("📊 Forecast Bias Analysis")

arima_bias = comparison["ARIMA Residual"].mean()
sarima_bias = comparison["SARIMA Residual"].mean()

col1, col2 = st.columns(2)

col1.metric("ARIMA Forecast Bias", f"{arima_bias:,.2f}")
col2.metric("SARIMA Forecast Bias", f"{sarima_bias:,.2f}")

def bias_status(value):
    if value < 0:
        return "Over-predicting"
    elif value > 0:
        return "Under-predicting"
    else:
        return "No average bias"

st.write(
    f"**ARIMA:** {bias_status(arima_bias)}"
)

st.write(
    f"**SARIMA:** {bias_status(sarima_bias)}"
)

st.caption(
    "Negative residual = Over-prediction. "
    "Positive residual = Under-prediction. "
    "Bias is calculated as Actual Sales - Predicted Sales."
)
# ==========================================
# ACTUAL VS PREDICTED SCATTER PLOT
# ==========================================

st.subheader("🎯 Actual vs Predicted Scatter Plot")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    comparison["Actual Sales"],
    comparison["ARIMA Prediction"],
    alpha=0.5,
    label="ARIMA"
)

ax.scatter(
    comparison["Actual Sales"],
    comparison["SARIMA Prediction"],
    alpha=0.5,
    label="SARIMA"
)

# Perfect prediction reference line
min_value = min(
    comparison["Actual Sales"].min(),
    comparison["ARIMA Prediction"].min(),
    comparison["SARIMA Prediction"].min()
)

max_value = max(
    comparison["Actual Sales"].max(),
    comparison["ARIMA Prediction"].max(),
    comparison["SARIMA Prediction"].max()
)

ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("Actual vs Predicted Sales Comparison")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
# Actual versus predicted chart
st.subheader("📈 Actual vs Predicted Sales")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    comparison["Date"],
    comparison["Actual Sales"],
    label="Actual Sales",
    marker="o",
    markersize=3
)

ax.plot(
    comparison["Date"],
    comparison["ARIMA Prediction"],
    label="ARIMA Prediction"
)

ax.plot(
    comparison["Date"],
    comparison["SARIMA Prediction"],
    label="SARIMA Prediction"
)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Actual Sales vs Model Predictions")
ax.legend()
ax.grid(alpha=0.3)

plt.xticks(rotation=45)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ==========================================
# MODEL PERFORMANCE SUMMARY
# ==========================================

st.subheader("🏆 Model Performance Summary")

best_smape = metrics_df["sMAPE (%)"].idxmin()

summary_data = {
    "Metric": ["MAE", "RMSE", "sMAPE (%)"],
    "ARIMA": [
        arima_metrics["MAE"],
        arima_metrics["RMSE"],
        arima_metrics["sMAPE (%)"]
    ],
    "SARIMA": [
        sarima_metrics["MAE"],
        sarima_metrics["RMSE"],
        sarima_metrics["sMAPE (%)"]
    ]
}

summary_df = pd.DataFrame(summary_data)

st.dataframe(
    summary_df.round(2),
    width="stretch",
    hide_index=True
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Best MAE Model",
    metrics_df["MAE"].idxmin()
)

col2.metric(
    "Best RMSE Model",
    metrics_df["RMSE"].idxmin()
)

col3.metric(
    "Best sMAPE Model",
    best_smape
)

if (
    metrics_df["MAE"].idxmin()
    == metrics_df["RMSE"].idxmin()
    == best_smape
):
    st.success(
        f"Overall, {best_smape} performs better "
        "across all three error metrics."
    )
else:
    st.info(
        "Different metrics favour different models. "
        "Compare their errors before selecting the final model."
    )
# Best model recommendation
st.subheader("🏆 Best Model Recommendation")

best_mae = metrics_df["MAE"].idxmin()
best_rmse = metrics_df["RMSE"].idxmin()

if best_mae == best_rmse:
    st.success(
        f"{best_mae} has the lowest MAE and RMSE "
        "among these two evaluated models."
    )
else:
    st.info(
        f"MAE favours {best_mae}, while RMSE favours "
        f"{best_rmse}. Review the business cost of prediction errors "
        "before choosing a model."
    )

st.write(f"**ARIMA MAE:** {arima_metrics['MAE']:,.2f}")
st.write(f"**SARIMA MAE:** {sarima_metrics['MAE']:,.2f}")
st.write(f"**ARIMA RMSE:** {arima_metrics['RMSE']:,.2f}")
st.write(f"**SARIMA RMSE:** {sarima_metrics['RMSE']:,.2f}")


# Detailed results
st.subheader("📋 Detailed Forecast Results")

st.dataframe(
    comparison[
        [
            "Date",
            "Actual Sales",
            "ARIMA Prediction",
            "SARIMA Prediction"
        ]
    ].round(2),
    width="stretch"
)


# Download report
report = comparison[
    [
        "Date",
        "Actual Sales",
        "ARIMA Prediction",
        "SARIMA Prediction"
    ]
].copy()

report["ARIMA Absolute Error"] = (
    report["Actual Sales"] - report["ARIMA Prediction"]
).abs()

report["SARIMA Absolute Error"] = (
    report["Actual Sales"] - report["SARIMA Prediction"]
).abs()
# Percentage Error Calculation

report["ARIMA Percentage Error (%)"] = np.where(
    report["Actual Sales"] != 0,
    (
        report["ARIMA Absolute Error"]
        / report["Actual Sales"].abs()
    ) * 100,
    np.nan
)

report["SARIMA Percentage Error (%)"] = np.where(
    report["Actual Sales"] != 0,
    (
        report["SARIMA Absolute Error"]
        / report["Actual Sales"].abs()
    ) * 100,
    np.nan
)

st.download_button(
    label="📥 Download Comparison Report",
    data=report.to_csv(index=False).encode("utf-8"),
    file_name="arima_sarima_performance_comparison.csv",
    mime="text/csv"
)

st.success("Model comparison dashboard loaded successfully!")