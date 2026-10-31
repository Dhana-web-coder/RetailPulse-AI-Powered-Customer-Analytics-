import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Current Stock Analysis", page_icon="📦", layout="wide")
st.title("📦 Current Stock Level Analysis")
st.info("Demo stock data is used for this analysis. Replace it with actual stock data before making business decisions.")

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Load current stock
stock_file = BASE_DIR / "data" / "current_stock.csv"
if not stock_file.exists():
    st.error("current_stock.csv file not found in the data folder.")
    st.stop()

df = pd.read_csv(stock_file)
df.columns = df.columns.str.strip()

required_columns = ["StockCode", "Product", "Current_Stock", "Reorder_Level"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    st.error(f"Missing columns in current_stock.csv: {', '.join(missing)}")
    st.stop()

df["StockCode"] = df["StockCode"].astype(str).str.strip()
df["Product"] = df["Product"].fillna("Unknown Product").astype(str)
df["Current_Stock"] = pd.to_numeric(df["Current_Stock"], errors="coerce")
df["Reorder_Level"] = pd.to_numeric(df["Reorder_Level"], errors="coerce")
df = df.dropna(subset=["StockCode", "Current_Stock", "Reorder_Level"]).copy()
df = df[df["StockCode"] != ""].drop_duplicates(subset=["StockCode"], keep="last")

if df.empty:
    st.warning("No valid stock records found.")
    st.stop()

# 2. Stock status and reorder quantity
df["Stock_Status"] = df.apply(
    lambda row: "Reorder Required"
    if row["Current_Stock"] <= row["Reorder_Level"]
    else "Stock Available",
    axis=1
)
df["Suggested_Reorder_Qty"] = (
    df["Reorder_Level"] - df["Current_Stock"]
).clip(lower=0)

# 3. KPI cards
st.subheader("📊 Inventory Summary")
total_products = df["StockCode"].nunique()
total_stock = df["Current_Stock"].sum()
reorder_count = (df["Stock_Status"] == "Reorder Required").sum()
available_count = (df["Stock_Status"] == "Stock Available").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Products", f"{total_products:,}")
c2.metric("Total Stock Units", f"{total_stock:,.0f}")
c3.metric("Reorder Required", f"{reorder_count:,}")
c4.metric("Stock Available", f"{available_count:,}")

st.divider()

# 4. Stock availability chart
st.subheader("Stock Availability Overview")
status_counts = (
    df["Stock_Status"].value_counts()
    .rename_axis("Stock Status")
    .reset_index(name="Products")
)
fig = px.pie(status_counts, names="Stock Status", values="Products", hole=0.4,
             title="Products by Stock Status")
st.plotly_chart(fig, use_container_width=True)

# 5. Product-wise stock chart
st.subheader("📦 Product-wise Current Stock")
fig = px.bar(
    df.sort_values("Current_Stock"),
    x="Product", y="Current_Stock", color="Stock_Status",
    title="Current Stock by Product", hover_data=["StockCode", "Reorder_Level"]
)
fig.update_layout(height=550, xaxis_tickangle=-45,
                  xaxis_title="Product", yaxis_title="Current Stock Units")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# 6. Product search and filter
st.subheader("🔍 Product Search & Filter")
search_col, filter_col = st.columns(2)
with search_col:
    search_product = st.text_input("Search Product", placeholder="Enter Product Name or StockCode")
with filter_col:
    stock_filter = st.selectbox(
        "Filter by Stock Status",
        ["All", "Stock Available", "Reorder Required"]
    )

filtered_df = df.copy()
if search_product.strip():
    product_match = filtered_df["Product"].str.contains(search_product.strip(), case=False, na=False)
    code_match = filtered_df["StockCode"].str.contains(search_product.strip(), case=False, na=False)
    filtered_df = filtered_df[product_match | code_match]
if stock_filter != "All":
    filtered_df = filtered_df[filtered_df["Stock_Status"] == stock_filter]

st.caption(f"{len(filtered_df):,} matching product record(s)")
if filtered_df.empty:
    st.warning("No matching products found.")
else:
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

st.divider()

# 7. Low stock alert
st.subheader("🚨 Low Stock Alert")
low_stock_df = df[df["Current_Stock"] <= df["Reorder_Level"]].copy()
if low_stock_df.empty:
    st.success("All products are above their reorder level.")
else:
    st.warning(f"{len(low_stock_df):,} product(s) are at or below their reorder level.")
    st.dataframe(
        low_stock_df[["StockCode", "Product", "Current_Stock", "Reorder_Level", "Suggested_Reorder_Qty"]],
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "📥 Download Low Stock Alert Report",
        data=low_stock_df.to_csv(index=False).encode("utf-8"),
        file_name="low_stock_alert.csv", mime="text/csv"
    )

st.divider()

# 8. Smart reorder recommendation
st.subheader("📦 Smart Reorder Recommendation")
recommendation_df = df[df["Suggested_Reorder_Qty"] > 0].copy()
if recommendation_df.empty:
    st.success("No reorder recommendations at the moment.")
else:
    recommendation_df["Recommended_Order_Qty"] = recommendation_df["Suggested_Reorder_Qty"]
    st.caption(
        "This quantity brings stock up to the configured Reorder Level only. "
        "It does not include future demand, supplier lead time, or safety stock."
    )
    st.dataframe(
        recommendation_df[["StockCode", "Product", "Current_Stock", "Reorder_Level", "Recommended_Order_Qty"]],
        use_container_width=True, hide_index=True
    )
    st.metric("Total Recommended Order Quantity", f"{recommendation_df['Recommended_Order_Qty'].sum():,.0f}")
    st.download_button(
        "📥 Download Reorder Recommendation",
        data=recommendation_df.to_csv(index=False).encode("utf-8"),
        file_name="smart_reorder_recommendation.csv", mime="text/csv"
    )

# 9. Find sales dataset for demand/dead stock analysis
st.divider()
st.subheader("📈 Historical Sales-Based Inventory Analysis")

sales_candidates = [
    BASE_DIR / "data" / "retail_customer_clean.csv",
    BASE_DIR / "data" / "retail_cleaned_final.csv",
    BASE_DIR / "retail_cleaned_final.csv"
]
sales_file = next((p for p in sales_candidates if p.exists()), None)
sales_df = None

if sales_file is None:
    st.warning("Sales dataset not found. Demand comparison and Dead Stock Analysis are unavailable.")
else:
    sales_df = pd.read_csv(sales_file)
    sales_df.columns = sales_df.columns.str.strip()
    needed = {"StockCode", "Quantity"}
    if not needed.issubset(sales_df.columns):
        st.warning("Sales dataset must contain StockCode and Quantity columns.")
        sales_df = None
    else:
        sales_df["StockCode"] = sales_df["StockCode"].astype(str).str.strip()
        sales_df["Quantity"] = pd.to_numeric(sales_df["Quantity"], errors="coerce")
        sales_df = sales_df.dropna(subset=["StockCode", "Quantity"]).copy()
        sales_df = sales_df[sales_df["Quantity"] > 0].copy()

# 10. Historical demand vs current stock
st.subheader("📊 Historical Demand vs Current Stock")
if sales_df is None:
    st.info("This section needs a sales dataset with StockCode and Quantity.")
else:
    demand_df = (
        sales_df.groupby("StockCode")["Quantity"]
        .sum().reset_index(name="Historical_Demand")
    )
    comparison_df = df.merge(demand_df, on="StockCode", how="left")
    comparison_df["Historical_Demand"] = comparison_df["Historical_Demand"].fillna(0)
    comparison_df["Historical_Demand_Gap"] = (
        comparison_df["Historical_Demand"] - comparison_df["Current_Stock"]
    )
    comparison_df["Demand_Status"] = comparison_df["Historical_Demand_Gap"].apply(
        lambda value: "Historical Demand Exceeds Stock"
        if value > 0 else "Stock Meets/Exceeds Historical Demand"
    )
    st.caption(
        "This compares total historical sales quantity with the current stock snapshot. "
        "It is not a future demand forecast."
    )
    st.dataframe(
        comparison_df[[
            "StockCode", "Product", "Current_Stock", "Historical_Demand",
            "Historical_Demand_Gap", "Demand_Status"
        ]],
        use_container_width=True, hide_index=True
    )
    chart_df = comparison_df.nlargest(15, "Historical_Demand")
    if not chart_df.empty:
        fig_demand = px.bar(
            chart_df, x="Product", y=["Current_Stock", "Historical_Demand"],
            barmode="group", title="Top 15 Products: Historical Demand vs Current Stock"
        )
        fig_demand.update_layout(height=550, xaxis_tickangle=-45, yaxis_title="Units")
        st.plotly_chart(fig_demand, use_container_width=True)

# 11. Dead stock analysis
st.divider()
st.subheader("🛑 Dead Stock Analysis")
if sales_df is None:
    st.info("Dead Stock Analysis needs a sales dataset.")
elif "InvoiceDate" not in sales_df.columns:
    st.warning("InvoiceDate column not found. Dead Stock Analysis cannot be calculated.")
else:
    sales_df["InvoiceDate"] = pd.to_datetime(
        sales_df["InvoiceDate"], errors="coerce", dayfirst=True
    )
    dated_sales = sales_df.dropna(subset=["StockCode", "InvoiceDate"]).copy()
    if dated_sales.empty:
        st.warning("No valid sales dates found in the sales dataset.")
    else:
        last_sales = (
            dated_sales.groupby("StockCode")["InvoiceDate"]
            .max().reset_index(name="Last_Sale_Date")
        )
        dead_stock_df = df.merge(last_sales, on="StockCode", how="left")
        reference_date = dated_sales["InvoiceDate"].max()
        dead_stock_df["Days_Since_Last_Sale"] = (
            reference_date - dead_stock_df["Last_Sale_Date"]
        ).dt.days
        dead_stock_df["Dead_Stock_Status"] = dead_stock_df["Days_Since_Last_Sale"].apply(
            lambda days: "No Sales History" if pd.isna(days)
            else "Dead Stock" if days > 90 else "Active Stock"
        )
        dead_stock = dead_stock_df[
            dead_stock_df["Dead_Stock_Status"] == "Dead Stock"
        ].copy()
        no_history = dead_stock_df[
            dead_stock_df["Dead_Stock_Status"] == "No Sales History"
        ].copy()

        m1, m2 = st.columns(2)
        m1.metric("Dead Stock Products (over 90 days)", len(dead_stock))
        m2.metric("Products with No Sales History", len(no_history))
        st.caption(
            f"The 90-day threshold is measured from the latest transaction date in the dataset: "
            f"{reference_date:%d-%b-%Y}."
        )
        if dead_stock.empty:
            st.success("No dead stock products found using the 90-day threshold.")
        else:
            st.dataframe(
                dead_stock[[
                    "StockCode", "Product", "Current_Stock",
                    "Last_Sale_Date", "Days_Since_Last_Sale"
                ]],
                use_container_width=True, hide_index=True
            )
            st.download_button(
                "📥 Download Dead Stock Report",
                data=dead_stock.to_csv(index=False).encode("utf-8"),
                file_name="dead_stock_report.csv", mime="text/csv"
            )

st.divider()

# 12. Products requiring reorder
st.subheader("🚨 Products Requiring Reorder")
reorder_df = df[df["Stock_Status"] == "Reorder Required"].copy()
if reorder_df.empty:
    st.success("No products currently require reorder.")
else:
    st.dataframe(reorder_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Download Reorder Report",
        data=reorder_df.to_csv(index=False).encode("utf-8"),
        file_name="stock_reorder_report.csv", mime="text/csv"
    )

st.divider()

# 13. Complete stock details
st.subheader("📋 Complete Stock Details")
st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button(
    "📥 Download Complete Stock Report",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="current_stock_analysis.csv", mime="text/csv"
)

st.caption(
    "This page analyzes the stock snapshot in current_stock.csv. "
    "Historical sales data does not automatically update current stock quantities."
)