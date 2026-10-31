import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="💰",
    layout="wide"
)

st.title("💰 RetailPulse - Sales Dashboard")
st.write("Retail Sales Performance Analysis")


@st.cache_data
def load_data():
    df = pd.read_csv(
        "retail_cleaned_final.csv",
        low_memory=False
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
        dayfirst=True
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["Price"] = pd.to_numeric(
        df["Price"], errors="coerce"
    )

    df["TotalSales"] = df["Quantity"] * df["Price"]

    return df
try:
    df = load_data()

    sales_df = df[
        (df["Quantity"] > 0) &
        (df["Price"] > 0)
    ].copy()

    st.write("Dataset loaded successfully!")

except FileNotFoundError:
    st.error("Dataset file not found!")

st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Sales",
    f"{sales_df['TotalSales'].sum():,.2f}"
)

col2.metric(
    "Total Orders",
    f"{sales_df['Invoice'].nunique():,}"
)

col3.metric(
    "Unique Customers",
    f"{sales_df['Customer ID'].nunique():,}"
)

total_quantity = sales_df["Quantity"].sum()

st.metric(
    label="Total Quantity",
    value=f"{total_quantity:,.0f}"
)
st.divider()

st.subheader("📊 Daily Sales Trend")

daily_sales = (
    sales_df.set_index("InvoiceDate")["TotalSales"]
    .resample("D")
    .sum()
)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    daily_sales.index,
    daily_sales.values,
    linewidth=1.5,
    label="Daily Sales"
)

# Display every month on X-axis
ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=1)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

ax.set_xlabel("Month")
ax.set_ylabel("Sales Amount")
ax.set_title("Daily Sales Trend - Month-wise Labels")

ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)
st.divider()

st.subheader("🌍 Top 10 Countries by Sales")

country_sales = (
    sales_df.groupby("Country")["TotalSales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(
    country_sales,
    horizontal=True,
    width="stretch"
)
# ==========================================
# MONTHLY SALES TREND
# ==========================================

st.divider()
st.subheader("📅 Monthly Sales Trend")

monthly_sales = (
    sales_df.set_index("InvoiceDate")["TotalSales"]
    .resample("MS")
    .sum()
)

st.line_chart(
    monthly_sales,
    x_label="Month",
    y_label="Sales Amount",
    width="stretch"
)
# ==========================================
# YEAR-WISE SALES ANALYSIS
# ==========================================

st.divider()
st.subheader("📅 Year-wise Sales Analysis")

# Extract year from InvoiceDate
sales_df["Year"] = sales_df["InvoiceDate"].dt.year

# Calculate total sales for each year
yearly_sales = (
    sales_df.groupby("Year")["TotalSales"]
    .sum()
    .sort_index()
)

# Display yearly sales chart
st.bar_chart(
    yearly_sales,
    x_label="Year",
    y_label="Total Sales",
    width="stretch"
)

# Display yearly sales details
st.write("### Year-wise Sales Details")

st.dataframe(
    yearly_sales.reset_index().rename(
        columns={"TotalSales": "Total Sales"}
    ),
    use_container_width=True
)
# ==========================================
# SALES BY DAY OF WEEK
# ==========================================

st.divider()
st.subheader("📅 Sales by Day of Week")

# Extract day name from InvoiceDate
sales_df["Day"] = sales_df["InvoiceDate"].dt.day_name()

# Set correct weekday order
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

# Calculate total sales for each day
weekday_sales = (
    sales_df.groupby("Day")["TotalSales"]
    .sum()
    .reindex(day_order, fill_value=0)
)

# Display chart
st.bar_chart(
    weekday_sales,
    x_label="Day of Week",
    y_label="Total Sales",
    width="stretch"
)

# Display sales details
st.write("### Day-wise Sales Details")

st.dataframe(
    weekday_sales.rename("Total Sales").reset_index(),
    use_container_width=True
)

# Find highest sales day
best_day = weekday_sales.idxmax()

st.success(f"Highest Sales Day: {best_day}")