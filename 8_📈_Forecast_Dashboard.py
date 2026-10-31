from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RetailPulse Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 RetailPulse - Forecast Dashboard")

st.markdown(
    "### Monthly Sales Analysis and Demand Forecasting"
)

st.divider()


# ==========================================
# 2. LOAD DATASET
# ==========================================

BASE_DIR = Path(r"C:\Users\acer\Desktop\RERAILPULSE")

DATA_PATH = BASE_DIR / "retail_cleaned_final.csv"

# Alternative location if the file is inside data/
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "data" / "retail_cleaned_final.csv"

# Stop if the file does not exist
if not DATA_PATH.exists():
    st.error("Dataset file not found!")

    st.write("Project folder:", str(BASE_DIR))

    st.write("CSV files available in the project:")

    csv_files = list(BASE_DIR.rglob("*.csv"))

    for file in csv_files:
        st.write(str(file))

    st.info(
        "Check the CSV filename above and update DATA_PATH."
    )

    st.stop()

try:
    df = pd.read_csv(DATA_PATH, low_memory=False)

    st.success("Dataset loaded successfully!")

except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ==========================================
# 3. CHECK REQUIRED COLUMNS
# ==========================================

st.write("Dataset location:", str(DATA_PATH))

st.write("Dataset rows and columns:", df.shape)

# Normalize column names
df.columns = df.columns.str.strip()

# Map common alternative column names
column_mapping = {}

for col in df.columns:
    normalized = col.lower().replace("_", "").replace(" ", "")

    if normalized in ["invoicedate", "date", "datetime"]:
        column_mapping[col] = "InvoiceDate"

    elif normalized in ["quantity", "qty"]:
        column_mapping[col] = "Quantity"

    elif normalized in ["unitprice", "price", "unitcost"]:
        column_mapping[col] = "Price"

df = df.rename(columns=column_mapping)

required_columns = [
    "InvoiceDate",
    "Quantity",
    "Price"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"Required columns are missing: {missing_columns}"
    )

    st.write("Available columns:", df.columns.tolist())

    st.stop()


# ==========================================
# 4. DATA CLEANING
# ==========================================

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)

df = df.dropna(
    subset=["InvoiceDate", "Quantity", "Price"]
)

# Exclude cancelled or invalid sales from sales analysis
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
].copy()

if df.empty:
    st.error("No valid sales records found in the dataset.")
    st.stop()


# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================

df["Sales"] = df["Quantity"] * df["Price"]

df["YearMonth"] = (
    df["InvoiceDate"].dt.to_period("M")
)


# ==========================================
# 6. MONTHLY SALES AGGREGATION
# ==========================================

monthly_sales = (
    df.groupby("YearMonth")["Sales"]
    .sum()
    .sort_index()
)

monthly_sales.index = (
    monthly_sales.index.to_timestamp()
)

# Fill missing calendar months with zero
full_months = pd.date_range(
    start=monthly_sales.index.min(),
    end=monthly_sales.index.max(),
    freq="MS"
)

monthly_sales = monthly_sales.reindex(
    full_months,
    fill_value=0
)

monthly_sales.name = "Sales"

if len(monthly_sales) < 6:
    st.warning(
        "At least 6 months of data is recommended "
        "for this baseline forecast."
    )
    st.stop()


# ==========================================
# 7. KPI CARDS
# ==========================================

total_sales = monthly_sales.sum()

average_sales = monthly_sales.mean()

last_month_sales = monthly_sales.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Historical Sales",
    f"{total_sales:,.2f}"
)

col2.metric(
    "Average Monthly Sales",
    f"{average_sales:,.2f}"
)

col3.metric(
    "Last Month Sales",
    f"{last_month_sales:,.2f}"
)

col4.metric(
    "Months of Data",
    len(monthly_sales)
)

st.divider()


# ==========================================
# 8. MONTHLY SALES TREND
# ==========================================

st.subheader("📊 Monthly Historical Sales")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    label="Actual Sales"
)

ax.set_xlabel("Month")
ax.set_ylabel("Sales (£)")
ax.set_title("Monthly Sales Trend")

ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ==========================================
# 9. FORECAST PERIOD SELECTION
# ==========================================

st.divider()

st.subheader("🔮 Demand Forecasting")

forecast_period = st.selectbox(
    "Select forecast period (months)",
    [1, 3, 6],
    index=1
)


# ==========================================
# 10. ARIMA MODEL AND FORECAST
# ==========================================

model = ARIMA(
    monthly_sales,
    order=(1, 1, 1)
)

fitted_model = model.fit()

forecast_result = fitted_model.get_forecast(
    steps=forecast_period
)

forecast_values = forecast_result.predicted_mean
confidence_interval = forecast_result.conf_int()

future_dates = pd.date_range(
    start=monthly_sales.index[-1] + pd.offsets.MonthBegin(1),
    periods=forecast_period,
    freq="MS"
)

forecast_values = pd.Series(
    forecast_values.to_numpy(),
    index=future_dates
).clip(lower=0)

confidence_interval.index = future_dates

lower_bound = confidence_interval.iloc[:, 0].clip(lower=0)
upper_bound = confidence_interval.iloc[:, 1].clip(lower=0)

# Prepare confidence interval
confidence_interval.index = future_dates

lower_bound = confidence_interval.iloc[:, 0].clip(lower=0)
upper_bound = confidence_interval.iloc[:, 1].clip(lower=0)

st.success("Forecast generated successfully!")
        # Next month forecast
st.metric(
            "Predicted Sales for Next Month",
            f"{forecast_values.iloc[0]:,.2f}"
        )

        # ======================================
        # 11. ACTUAL VS FORECAST CHART
        # ======================================

st.subheader("📈 Actual vs Forecast Sales")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
            monthly_sales.index,
            monthly_sales.values,
            marker="o",
            label="Actual Sales"
        )
ax.fill_between(
        future_dates,
        lower_bound,
        upper_bound,
        alpha=0.2,
        label="95% Confidence Interval"
        )

ax.plot(
            forecast_values.index,
            forecast_values.values,
            marker="o",
            linestyle="--",
            label="Forecast Sales"
        )
        # Display confidence interval
ax.fill_between(
        future_dates,
        lower_bound,
        upper_bound,
        alpha=0.2,
        label="95% Confidence Interval"
        )

ax.set_xlabel("Month")
ax.set_ylabel("Sales (£)")
ax.set_title("ARIMA Sales Forecast")

ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

        # ======================================
        # 12. FORECAST RESULTS TABLE
        # ======================================

st.subheader("📋 Forecast Results")

forecast_df = pd.DataFrame({
            "Month": forecast_values.index.strftime("%Y-%m"),
            "Predicted Sales (£)": forecast_values.values.round(2)
        })

st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

        # ======================================
        # 13. DOWNLOAD FORECAST
        # ======================================

csv_data = forecast_df.to_csv(
            index=False
        ).encode("utf-8")

st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_data,
            file_name="retailpulse_forecast.csv",
            mime="text/csv"
        )

        # ======================================
        # 14. BUSINESS INSIGHTS
        # ======================================

st.divider()

st.subheader("💡 Business Insights")

if forecast_values.iloc[0] > last_month_sales:

    st.write(
    "📈 Forecast sales are higher than the last "
    "observed month. Review inventory availability."
     )

elif forecast_values.iloc[0] < last_month_sales:

            st.write(
                "📉 Forecast sales are lower than the last "
                "observed month. Review stock levels carefully."
            )

else:

            st.write(
                "➡️ Forecast sales are approximately equal "
                "to the last observed month."
            )

st.caption(
            "Note: Forecasts are estimates, not guaranteed sales. "
            "Validate the model using a time-based test set before "
            "making inventory decisions."
        )

try:
    # Forecast code
    pass

except Exception as error:
    st.error(f"Forecasting failed: {error}")

# ==========================================

# 15. PRODUCT-WISE DEMAND FORECAST

# ==========================================

st.divider()

st.header("📦 Product-wise Demand Forecast")

st.write(
"Select a product to analyze its historical monthly demand "
"and forecast future demand."
)
# Check required columns

if "StockCode" not in df.columns or "Description" not in df.columns:
    st.error("StockCode or Description column is missing.")
    st.stop()

# Get available products
product_data = df.dropna(
    subset=["StockCode", "Description"]
).copy()

product_data["StockCode"] = (
    product_data["StockCode"].astype(str)
)

product_data["ProductLabel"] = (
    product_data["StockCode"]
    + " - "
    + product_data["Description"].astype(str)
)

product_options = (
    product_data[["ProductLabel", "StockCode"]]
    .drop_duplicates()
    .sort_values("ProductLabel")
)

if product_options.empty:
    st.warning("No products are available.")
    st.stop()

selected_product = st.selectbox(
    "Select a product",
    product_options["ProductLabel"].tolist(),
    key="product_demand_forecast_selector"
)

selected_code = selected_product.split(" - ")[0]

selected_data = product_data[
    product_data["StockCode"] == selected_code
].copy()

product_data["ProductLabel"] = (
    product_data["StockCode"]
    + " - "
    + product_data["Description"].astype(str)
)

product_options = (
    product_data[["ProductLabel", "StockCode"]]
    .drop_duplicates()
    .sort_values("ProductLabel")
)

# Aggregate positive sales quantity by month
selected_data["YearMonth"] = (
    selected_data["InvoiceDate"].dt.to_period("M")
)

monthly_demand = (
    selected_data[selected_data["Quantity"] > 0]
    .groupby("YearMonth")["Quantity"]
    .sum()
    .sort_index()
)

if monthly_demand.empty:
    st.warning("No valid demand data is available for this product.")

else:
    monthly_demand.index = monthly_demand.index.to_timestamp()

    # Fill missing months
    all_months = pd.date_range(
        start=monthly_demand.index.min(),
        end=monthly_demand.index.max(),
        freq="MS"
    )

    monthly_demand = monthly_demand.reindex(
        all_months,
        fill_value=0
    )

    st.subheader("Product Demand Summary")

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Total Quantity Sold",
        f"{monthly_demand.sum():,.0f}"
    )

    k2.metric(
        "Average Monthly Demand",
        f"{monthly_demand.mean():,.1f}"
    )

    k3.metric(
        "Months of Data",
        len(monthly_demand)
    )

    # Historical demand chart
    st.subheader("Monthly Product Demand")

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        monthly_demand.index,
        monthly_demand.values,
        marker="o"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Quantity Sold")
    ax.set_title(f"Monthly Demand: {selected_product}")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Forecast settings
    product_forecast_period = st.selectbox(
        "Forecast demand for how many months?",
        [1, 3, 6],
        index=1,
        key="product_forecast_period_selector"
    )

    if st.button(
        "Generate Product Demand Forecast",
        type="primary",
        key="generate_product_demand_forecast"
    ):

        try:
            if len(monthly_demand) >= 6:
                from statsmodels.tsa.arima.model import ARIMA

                model = ARIMA(
                    monthly_demand.astype(float),
                    order=(1, 1, 1)
                )
                fitted_model = model.fit()

                # Generate forecast with confidence interval
                forecast_result = fitted_model.get_forecast(
                steps=forecast_period
                )

                forecast_values = forecast_result.predicted_mean

                confidence_interval = forecast_result.conf_int()
            
                # Generate product demand forecast
                predictions = fitted_model.forecast(
    steps=product_forecast_period
)

                predictions = predictions.clip(lower=0)

                forecast_dates = pd.date_range(
                    start=monthly_demand.index[-1]
                    + pd.offsets.MonthBegin(1),
                    periods=product_forecast_period,
                    freq="MS"
                )

                product_forecast_df = pd.DataFrame({
                    "Month": forecast_dates.strftime("%Y-%m"),
                    "Predicted_Demand": predictions.to_numpy().round(0).astype(int)
                })

                st.success("Product demand forecast generated!")

                st.metric(
                    "Next Month Predicted Demand",
                    f"{product_forecast_df['Predicted_Demand'].iloc[0]:,} units"
                )

                st.subheader("Product Demand Forecast Results")

                st.dataframe(
                    product_forecast_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Historical vs forecast chart
                fig, ax = plt.subplots(figsize=(12, 4))

                ax.plot(
                    monthly_demand.index,
                    monthly_demand.values,
                    marker="o",
                    label="Historical Demand"
                )

                ax.plot(
                    forecast_dates,
                    predictions.to_numpy(),
                    marker="o",
                    linestyle="--",
                    label="Predicted Demand"
                )

                ax.set_xlabel("Month")
                ax.set_ylabel("Quantity Sold")
                ax.set_title("Historical vs Forecast Product Demand")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                # Download results
                csv_data = product_forecast_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Product Forecast CSV",
                    data=csv_data,
                    file_name="product_demand_forecast.csv",
                    mime="text/csv",
                    key="download_product_demand_forecast"
                )

                st.caption(
                    "Forecasts are estimates based on historical quantity sold. "
                    "They are not guaranteed future demand."
                )

            else:
                st.warning(
                    "At least 6 months of demand data is required "
                    "to generate a forecast for this product."
                )

        except Exception as e:
            st.error(f"Product demand forecasting failed: {e}")
