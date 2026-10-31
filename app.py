import streamlit as st
import pandas as pd
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================



BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 2. LOAD DATASET
# ==========================================

DATA_FILES = [
    BASE_DIR / "retail_cleaned_final.csv",
    BASE_DIR / "data" / "retail_2009_2010.csv",
    BASE_DIR / "data" / "retail_data_1.csv",
]


@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    if "InvoiceDate" in data.columns:
        data["InvoiceDate"] = pd.to_datetime(
            data["InvoiceDate"], errors="coerce"
        )

    for column in ["Quantity", "Price"]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    return data


data_path = next(
    (file for file in DATA_FILES if file.exists()), None
)

if data_path is None:
    st.error(
        "Dataset not found. Check the CSV file path."
    )
    st.stop()

try:
    df = load_data(str(data_path))
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


required_columns = [
    "Invoice", "StockCode", "Description", "Quantity",
    "Price", "InvoiceDate", "Customer ID", "Country"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()


# ==========================================
# 3. DATA PREPARATION
# ==========================================

df["TotalSales"] = df["Quantity"] * df["Price"]

sales_df = df[
    (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & df["InvoiceDate"].notna()
].copy()

if sales_df.empty:
    st.error("No valid sales records found.")
    st.stop()

# ==========================================
# 3. RETAILPULSE PROFESSIONAL HOME PAGE
# ==========================================

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #242477 0%, #4939a8 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #29265c;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 15px;
        color: #73758a;
        margin-bottom: 25px;
    }

    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.07);
        border: 1px solid #e8eaf2;
        margin-bottom: 15px;
    }

    .kpi-label {
        color: #73758a;
        font-size: 14px;
        font-weight: 600;
    }

    .kpi-value {
        color: #29265c;
        font-size: 26px;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #29265c;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    div.stButton > button {
        width: 100%;
        min-height: 100px;
        border-radius: 12px;
        border: 1px solid #e3e5ef;
        background-color: white;
        color: #29265c;
        font-weight: 650;
        white-space: pre-wrap;
    }

    div.stButton > button:hover {
        border-color: #6655cc;
        color: #4939a8;
        background-color: #f0edff;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("# 🛒 RetailPulse")
    st.caption("AI-POWERED RETAIL ANALYTICS")
    st.divider()

    st.markdown("### NAVIGATION")

    page = st.radio(
    "Select Dashboard",
    [
        "🏠 Home",
        "📊 Executive Dashboard",
        "💰 Sales Analysis",
        "👥 Customer Analytics",
        "🎯 Customer Segmentation",
        "🤖 Churn Prediction",
        "📦 Inventory Analysis",
        "🔮 Demand Forecasting",
        "🌍 Country Analysis",
        "🛍️ Product Analysis",
        "📦 Current Stock Analysis",
        "📈 Model Performance Comparison"
    ],
    label_visibility="collapsed"
)

    st.divider()
    st.caption("RetailPulse | Analytics Platform")


# Main welcome section
st.markdown(
    '<p class="main-title">🏠 Welcome to RetailPulse</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">'
    'AI-Powered Customer Analytics & Demand Forecasting Platform'
    '</p>',
    unsafe_allow_html=True
)

st.info(f"📁 Active Dataset: {data_path.name}")


# KPI calculation
total_sales = sales_df["TotalSales"].sum()
total_orders = sales_df["Invoice"].nunique()
total_customers = sales_df["Customer ID"].nunique()
total_products = sales_df["StockCode"].nunique()
total_countries = sales_df["Country"].nunique()

total_quantity = sales_df["Quantity"].sum()

# Use the actual number of products in each demand category.
if "product_summary" in globals():
    product_count = len
else:
    product_count = total_products

product_summary = (
    sales_df.dropna(subset=["StockCode", "Description"])
    .groupby(["StockCode", "Description"])
    .agg(
        TotalQuantitySold=("Quantity", "sum"),
        TotalSales=("TotalSales", "sum"),
        NumberOfOrders=("Invoice", "nunique")
    )
    .reset_index()
)

low_cutoff = product_summary["TotalQuantitySold"].quantile(0.25)
high_cutoff = product_summary["TotalQuantitySold"].quantile(0.75)

high_demand = int(
    (product_summary["TotalQuantitySold"] >= high_cutoff).sum()
)

low_demand = int(
    (product_summary["TotalQuantitySold"] <= low_cutoff).sum()
)


# Reusable KPI card
def show_kpi(label, value, icon):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# First row: 4 KPI cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    show_kpi("Total Revenue", f"{total_sales:,.2f}", "💰")

with c2:
    show_kpi("Total Orders", f"{total_orders:,}", "🧾")

with c3:
    show_kpi("Total Customers", f"{total_customers:,}", "👥")

with c4:
    show_kpi("Total Products", f"{total_products:,}", "📦")


# Second row: 4 KPI cards
c5, c6, c7, c8 = st.columns(4)

with c5:
    show_kpi("Countries Served", f"{total_countries:,}", "🌍")

with c6:
    show_kpi("Quantity Sold", f"{total_quantity:,.0f}", "🛍️")

with c7:
    show_kpi("High-Demand Products", f"{high_demand:,}", "🔥")

with c8:
    show_kpi("Low-Demand Products", f"{low_demand:,}", "⚠️")


# Quick navigation
st.markdown(
    '<p class="section-title">🚀 Quick Navigation</p>',
    unsafe_allow_html=True
)

st.caption(
    "Choose a section below to explore your retail business insights."
)

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    if st.button("📊 Executive Dashboard"):
        st.session_state["selected_page"] = "📊 Executive Dashboard"
        st.rerun()

with nav2:
    if st.button("💰 Sales Analysis"):
        st.session_state["selected_page"] = "💰 Sales Analysis"
        st.rerun()

with nav3:
    if st.button("👥 Customer Analytics"):
        st.session_state["selected_page"] = "👥 Customer Analytics"
        st.rerun()

with nav4:
    if st.button("🎯 Customer Segmentation"):
        st.session_state["selected_page"] = "🎯 Customer Segmentation"
        st.rerun()

nav5, nav6, nav7, nav8 = st.columns(4)
with nav5:
    if st.button("🤖\nChurn Prediction"):
        st.session_state["selected_page"] = "🤖 Churn Prediction"
        st.rerun()

with nav6:
    if st.button("📦\nInventory Analysis"):
        st.session_state["selected_page"] = "📦 Inventory Analysis"
        st.rerun()

with nav7:
    if st.button("🔮\nDemand Forecasting"):
        st.session_state["selected_page"] = "🔮 Demand Forecasting"
        st.rerun()

with nav8:
    if st.button("🌍\nCountry Analysis"):
        st.session_state["selected_page"] = "🌍 Country Analysis"
        st.rerun()


# Display selected page
selected_page = st.session_state.get("selected_page", page)
st.divider()

if selected_page == "🏠 Home":
    st.success(
        "Welcome! Use the navigation menu to explore RetailPulse."
    )
elif selected_page == "📊 Executive Dashboard":
    st.header("📊 Executive Dashboard")
    st.write("Business overview and key performance indicators.")
    st.bar_chart(
        sales_df.groupby(
            sales_df["InvoiceDate"].dt.to_period("M").astype(str)
        )["TotalSales"].sum()
    )
elif selected_page == "💰 Sales Analysis":
    st.header("💰 Sales Analysis")
    st.line_chart(
        sales_df.groupby(
            sales_df["InvoiceDate"].dt.date
        )["TotalSales"].sum()
    )
elif selected_page == "👥 Customer Analytics":
    st.header("👥 Customer Analytics")
    st.write(f"Unique customers: {total_customers:,}")
    st.write(f"Total orders: {total_orders:,}")
elif selected_page == "🎯 Customer Segmentation":
    st.header("🎯 Customer Segmentation")

    rfm_path = Path(r"C:\Users\acer\Desktop\RERAILPULSE\rfm_results.csv")

    if rfm_path.exists():
        rfm_df = pd.read_csv(rfm_path)

        st.subheader("RFM Customer Segmentation")
        st.metric("Total Customers", f"{len(rfm_df):,}")

        if "Segment" in rfm_df.columns:
            st.subheader("Customer Segments")
            st.dataframe(
                rfm_df["Segment"].value_counts().reset_index(),
                use_container_width=True
            )

        st.subheader("RFM Results")
        st.dataframe(rfm_df, use_container_width=True)
    else:
        st.error("rfm_results.csv file not found.")
elif selected_page == "🤖 Churn Prediction":
    churn_file = Path(__file__).resolve().parent / "pages" / "3_📉_Churn_Prediction.py"

    if churn_file.exists():
        exec(compile(churn_file.read_text(encoding="utf-8"), str(churn_file), "exec"))
    else:
        st.error("Churn Prediction file not found.")
elif selected_page == "📦 Inventory Analysis":
    inventory_file = Path(__file__).resolve().parent / "pages" / "4_📦_Inventory_Dashboard.py"

    if inventory_file.exists():
        exec(compile(inventory_file.read_text(encoding="utf-8"), str(inventory_file), "exec"))
    else:
        st.error("Inventory Dashboard file not found.")

elif selected_page == "🔮 Demand Forecasting":

    forecast_file = (
        BASE_DIR / "pages" / "8_📈_Forecast_Dashboard.py"
    )

    if forecast_file.exists():
        exec(
            compile(
                forecast_file.read_text(encoding="utf-8"),
                str(forecast_file),
                "exec"
            )
        )
    else:
        st.error("Forecast Dashboard file not found.")

elif selected_page == "🌍 Country Analysis":

    country_file = (
        BASE_DIR / "pages" / "9_🌍_Country_Analysis.py"
    )

    if country_file.exists():
        exec(
            compile(
                country_file.read_text(encoding="utf-8"),
                str(country_file),
                "exec"
            )
        )
    else:
        st.error("Country Analysis file not found.")
# ==========================================
# 4. COMPANY OVERVIEW
# ==========================================

st.header("🏢 Company Overview")

total_sales = sales_df["TotalSales"].sum()
total_orders = sales_df["Invoice"].nunique()
total_customers = sales_df["Customer ID"].nunique()
total_quantity = sales_df["Quantity"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"{total_sales:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Unique Customers", f"{total_customers:,}")
col4.metric("Quantity Sold", f"{total_quantity:,.0f}")

st.divider()


# ==========================================
# 5. SALES PERFORMANCE
# ==========================================

st.header("📈 Sales Performance")

daily_sales = (
    sales_df.groupby(sales_df["InvoiceDate"].dt.date)
    ["TotalSales"].sum()
)

st.subheader("Daily Sales Trend")
st.line_chart(daily_sales)

st.subheader("🌍 Sales by Country")

country_sales = (
    sales_df.groupby("Country")["TotalSales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(country_sales)
st.divider()


# ==========================================
# 6. PRODUCT SUMMARY
# ==========================================

st.header("📦 Inventory Movement Analysis")

product_summary = (
    sales_df.dropna(subset=["StockCode", "Description"])
    .groupby(["StockCode", "Description"])
    .agg(
        TotalQuantitySold=("Quantity", "sum"),
        TotalSales=("TotalSales", "sum"),
        NumberOfOrders=("Invoice", "nunique")
    )
    .reset_index()
    .sort_values("TotalQuantitySold", ascending=False)
)

st.subheader("🏆 Top 10 Best-Selling Products")

top_products = product_summary.head(10)

st.bar_chart(
    top_products.set_index("Description")["TotalQuantitySold"]
)
st.dataframe(top_products, use_container_width=True)

st.subheader("🐢 Top 10 Slow-Moving Products")

slow_products = product_summary.sort_values(
    "TotalQuantitySold", ascending=True
).head(10)

st.bar_chart(
    slow_products.set_index("Description")["TotalQuantitySold"]
)
st.dataframe(slow_products, use_container_width=True)

st.divider()


# ==========================================
# 7. DEMAND-BASED REORDER RECOMMENDATIONS
# ==========================================

st.header("🔄 Demand-Based Reorder Recommendations")

st.write(
    "Historical sales quantity is used to classify product demand."
)

reorder_df = product_summary.copy()

if not reorder_df.empty:
    low_cutoff = reorder_df["TotalQuantitySold"].quantile(0.25)
    high_cutoff = reorder_df["TotalQuantitySold"].quantile(0.75)

    def classify_demand(quantity):
        if quantity <= low_cutoff:
            return "Low Demand"
        elif quantity >= high_cutoff:
            return "High Demand"
        return "Medium Demand"

    reorder_df["DemandStatus"] = (
        reorder_df["TotalQuantitySold"].apply(classify_demand)
    )

    def get_recommendation(status):
        if status == "High Demand":
            return "Review stock and consider replenishment"
        elif status == "Low Demand":
            return "Review slow-moving inventory"
        return "Monitor demand"

    reorder_df["Recommendation"] = (
        reorder_df["DemandStatus"].apply(get_recommendation)
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "High-Demand Products",
        int((reorder_df["DemandStatus"] == "High Demand").sum())
    )
    col2.metric(
        "Medium-Demand Products",
        int((reorder_df["DemandStatus"] == "Medium Demand").sum())
    )
    col3.metric(
        "Low-Demand Products",
        int((reorder_df["DemandStatus"] == "Low Demand").sum())
    )

    st.subheader("🔥 High-Demand Products")
    st.dataframe(
        reorder_df[reorder_df["DemandStatus"] == "High Demand"]
        .sort_values("TotalQuantitySold", ascending=False)
        .head(10),
        use_container_width=True
    )

    st.subheader("📊 Medium-Demand Products")
    st.dataframe(
        reorder_df[reorder_df["DemandStatus"] == "Medium Demand"]
        .sort_values("TotalQuantitySold", ascending=False)
        .head(10),
        use_container_width=True
    )

    st.subheader("⚠️ Low-Demand Products")
    st.dataframe(
        reorder_df[reorder_df["DemandStatus"] == "Low Demand"]
        .sort_values("TotalQuantitySold", ascending=True)
        .head(10),
        use_container_width=True
    )

    st.info(
        "Current stock levels are not available in this dataset. "
        "Recommendations are based on historical sales demand."
    )
else:
    st.warning("No product demand data available.")

st.divider()


# ==========================================
# 8. CUSTOMER SEGMENTATION - RFM
# ==========================================

st.header("👥 Customer Segmentation Dashboard")

customer_sales = sales_df.dropna(
    subset=["Customer ID"]
).copy()

if not customer_sales.empty:
    reference_date = (
        customer_sales["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        customer_sales.groupby("Customer ID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda dates: (reference_date - dates.max()).days
            ),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalSales", "sum")
        )
        .reset_index()
    )

    if len(rfm) >= 4:
        rfm["R_Score"] = pd.qcut(
            rfm["Recency"].rank(method="first"),
            4, labels=[4, 3, 2, 1]
        ).astype(int)

        rfm["F_Score"] = pd.qcut(
            rfm["Frequency"].rank(method="first"),
            4, labels=[1, 2, 3, 4]
        ).astype(int)

        rfm["M_Score"] = pd.qcut(
            rfm["Monetary"].rank(method="first"),
            4, labels=[1, 2, 3, 4]
        ).astype(int)

        rfm["RFM_Score"] = (
            rfm["R_Score"].astype(str)
            + rfm["F_Score"].astype(str)
            + rfm["M_Score"].astype(str)
        )

        def segment_customer(row):
            if (
                row["R_Score"] >= 3
                and row["F_Score"] >= 3
                and row["M_Score"] >= 3
            ):
                return "Best Customers"
            elif row["R_Score"] <= 2:
                return "At-Risk Customers"
            elif row["F_Score"] >= 3 and row["M_Score"] >= 3:
                return "Loyal Customers"
            return "Regular Customers"

        rfm["CustomerSegment"] = rfm.apply(
            segment_customer, axis=1
        )

        st.subheader("Customer Segment Distribution")
        st.bar_chart(rfm["CustomerSegment"].value_counts())

        st.subheader("RFM Customer Details")
        st.dataframe(
            rfm.sort_values("Monetary", ascending=False),
            use_container_width=True
        )
    else:
        st.warning("Not enough customers to calculate RFM scores.")
else:
    st.warning("Customer ID data is not available.")

st.divider()


# ==========================================
# 9. DATASET SUMMARY
# ==========================================

st.header("📋 Dataset Summary")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.subheader("Missing Values")
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]
    st.dataframe(missing_values, use_container_width=True)

st.divider()


# ==========================================
# 10. ARIMA vs SARIMA - 30 DAYS FORECAST
# ==========================================

st.header("📈 ARIMA vs SARIMA - Next 30 Days Forecast")

graph_file = BASE_DIR / "arima_sarima_forecast_comparison.png"
csv_file = BASE_DIR / "arima_sarima_forecast_comparison.csv"

if graph_file.exists():
    st.image(
        str(graph_file),
        caption="ARIMA vs SARIMA - 30 Days Sales Forecast",
        use_container_width=True
    )
else:
    st.warning(
        "Comparison graph not found. Run python compare_forecasts.py."
    )

if csv_file.exists():
    forecast_df = pd.read_csv(csv_file)

    st.subheader("30 Days Forecast Values")
    st.dataframe(forecast_df, use_container_width=True)

    st.caption(
        f"Forecast rows available: {len(forecast_df)}"
    )

    if len(forecast_df) != 30:
        st.warning(
            "The comparison file does not contain exactly 30 rows. "
            "Check that both forecast CSV files contain 30 dates."
        )
else:
    st.warning(
        "Forecast comparison CSV not found. "
        "Run python compare_forecasts.py."
    )

st.divider()

st.success("RetailPulse Dashboard loaded successfully.")