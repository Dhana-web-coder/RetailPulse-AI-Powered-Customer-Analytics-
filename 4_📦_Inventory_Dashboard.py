import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Inventory Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 RetailPulse - Inventory Dashboard")
st.markdown("### Product Demand & Inventory Movement Analysis")

# Find dataset
BASE_DIR = Path(r"C:\Users\acer\Desktop\RERAILPULSE")

files = [
    BASE_DIR / "retailpulse_combined_with_stock_status.csv",
    BASE_DIR / "data" / "retailpulse_combined_with_stock_status.csv",
    BASE_DIR / "retail_cleaned_2009_2010.csv",
    BASE_DIR / "retail_cleaned_final.csv",
    BASE_DIR / "retail_2009_2010.csv",
    BASE_DIR / "data" / "retail_data_1.csv"
]

file_path = next((f for f in files if f.exists()), None)

if file_path is None:
    st.error("Dataset not found. Please check the CSV file location.")
    st.write("Checked folder:", str(BASE_DIR))
    st.stop()

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

st.caption(f"Dataset loaded: {file_path.name}")

# Identify columns
quantity_col = next(
    (c for c in df.columns if c.strip().lower() == "quantity"), None
)
stock_col = next(
    (c for c in df.columns
     if c.strip().lower().replace(" ", "") == "stockcode"), None
)
description_col = next(
    (c for c in df.columns if c.strip().lower() == "description"), None
)

if quantity_col is None or stock_col is None:
    st.error("Quantity or StockCode column not found.")
    st.write("Available columns:", list(df.columns))
    st.stop()

# Clean data
df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce")
df = df.dropna(subset=[quantity_col, stock_col])
df = df[df[quantity_col] > 0].copy()

if df.empty:
    st.warning("No positive sales quantities found in this dataset.")
    st.stop()

# Product summary
summary = df.groupby(stock_col).agg(
    Total_Quantity=(quantity_col, "sum"),
    Transactions=(quantity_col, "count")
).reset_index()

if description_col:
    descriptions = (
        df.groupby(stock_col)[description_col]
        .first()
        .reset_index()
    )
    summary = summary.merge(descriptions, on=stock_col, how="left")
    summary["Product"] = summary[description_col].fillna("Unknown")
else:
    summary["Product"] = summary[stock_col].astype(str)

# Demand classification
low_limit = summary["Total_Quantity"].quantile(0.25)
high_limit = summary["Total_Quantity"].quantile(0.75)

summary["Demand_Status"] = "Medium Demand"
summary.loc[
    summary["Total_Quantity"] <= low_limit, "Demand_Status"
] = "Low Demand"
summary.loc[
    summary["Total_Quantity"] >= high_limit, "Demand_Status"
] = "High Demand"

# KPI cards
total_products = summary[stock_col].nunique()
total_quantity = df[quantity_col].sum()
total_transactions = len(df)

high = (summary["Demand_Status"] == "High Demand").sum()
medium = (summary["Demand_Status"] == "Medium Demand").sum()
low = (summary["Demand_Status"] == "Low Demand").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Products", f"{total_products:,}")
c2.metric("Total Quantity Sold", f"{total_quantity:,.0f}")
c3.metric("Sales Records", f"{total_transactions:,}")
c4.metric("Low-Demand Products", f"{low:,}")

st.divider()

# Charts
left, right = st.columns(2)

with left:
    st.subheader("📊 Demand Distribution")

    demand = (
        summary["Demand_Status"]
        .value_counts()
        .reindex(
            ["High Demand", "Medium Demand", "Low Demand"],
            fill_value=0
        )
        .rename_axis("Demand Status")
        .reset_index(name="Products")
    )

    fig = px.pie(
        demand,
        names="Demand Status",
        values="Products",
        hole=0.45
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🏆 Top 10 Products")

    top10 = summary.nlargest(
        10, "Total_Quantity"
    ).sort_values("Total_Quantity")

    fig = px.bar(
        top10,
        x="Total_Quantity",
        y="Product",
        orientation="h",
        hover_data=[stock_col, "Transactions"],
        labels={
            "Total_Quantity": "Total Quantity Sold",
            "Product": "Product"
        },
        title="Top 10 Products by Quantity Sold"
    )
    st.plotly_chart(fig, use_container_width=True)

    # High-Demand Products Analysis

st.divider()

st.subheader("🔥 High-Demand Products Analysis")

high_demand_products = (
    summary[summary["Demand_Status"] == "High Demand"]
    .nlargest(10, "Total_Quantity")
    .sort_values("Total_Quantity", ascending=True)
)

if high_demand_products.empty:
    st.info("No high-demand products found.")
else:
    fig = px.bar(
        high_demand_products,
        x="Total_Quantity",
        y="Product",
        orientation="h",
        hover_data=[stock_col, "Transactions"],
        title="Top 10 High-Demand Products",
        labels={
            "Total_Quantity": "Total Quantity Sold",
            "Product": "Product Name"
        }
    )

    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        high_demand_products[
            [stock_col, "Product", "Total_Quantity",
             "Transactions", "Demand_Status"]
        ],
        width="stretch",
        hide_index=True
    )

st.divider()

# Low-demand products
st.subheader("🐢 Low-Demand Products")

low_products = (
    summary[summary["Demand_Status"] == "Low Demand"]
    .nsmallest(10, "Total_Quantity")
)

if low_products.empty:
    st.info("No low-demand products found.")
else:
    st.dataframe(
        low_products[
            [stock_col, "Product", "Total_Quantity",
             "Transactions", "Demand_Status"]
        ],
        use_container_width=True,
        hide_index=True
    )

st.info(
    "Demand status is based on historical positive sales quantity. "
    "This dataset does not contain current stock-on-hand values, "
    "so actual available inventory cannot be calculated."
)
#Monthly Product Demand Analysis

st.divider()

st.header("📅 Monthly Product Demand Analysis")

#Check required columns

required_columns = ["InvoiceDate", "Description", "Quantity"]

if all(col in df.columns for col in required_columns):
    demand_df = df.copy()

    # Prepare data
    demand_df["InvoiceDate"] = pd.to_datetime(
        demand_df["InvoiceDate"], errors="coerce"
    )

    demand_df["Quantity"] = pd.to_numeric(
        demand_df["Quantity"], errors="coerce"
    )

demand_df["InvoiceDate"] = pd.to_datetime(
    demand_df["InvoiceDate"], errors="coerce"
)

demand_df["Quantity"] = pd.to_numeric(
    demand_df["Quantity"], errors="coerce"
)

demand_df = demand_df.dropna(
    subset=["InvoiceDate", "Description", "Quantity"]
)

# Exclude cancelled/negative quantity records
demand_df = demand_df[demand_df["Quantity"] > 0]

demand_df["Month"] = (
    demand_df["InvoiceDate"].dt.to_period("M").astype(str)
)

# Product dropdown
products = sorted(demand_df["Description"].unique())

selected_product = st.selectbox(
    "Select a Product",
    products,
    key="monthly_demand_product"
)

# Filter selected product
product_df = demand_df[
    demand_df["Description"] == selected_product
]

# Calculate monthly demand
monthly_demand = (
    product_df.groupby("Month")["Quantity"]
    .sum()
    .sort_index()
)

if not monthly_demand.empty:

    # Monthly demand chart
    st.subheader(f"Monthly Demand: {selected_product}")

    st.line_chart(monthly_demand)

    # Highest and lowest demand months
    highest_month = monthly_demand.idxmax()
    lowest_month = monthly_demand.idxmin()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Highest Demand Month",
            highest_month,
            f"{monthly_demand.max():,.0f} units"
        )

    with col2:
        st.metric(
            "Lowest Demand Month",
            lowest_month,
            f"{monthly_demand.min():,.0f} units"
        )

    # Demand trend
    st.subheader("📈 Demand Trend")

    if len(monthly_demand) < 2:
        st.info("Not enough monthly data to identify a demand trend.")

    else:
        first_demand = monthly_demand.iloc[0]
        last_demand = monthly_demand.iloc[-1]

        if last_demand > first_demand:
            st.success("Demand Trend: Increasing 📈")

        elif last_demand < first_demand:
            st.warning("Demand Trend: Decreasing 📉")

        else:
            st.info("Demand Trend: Stable ➡️")

    # Monthly demand table
    st.subheader("📋 Monthly Demand Details")

    monthly_table = monthly_demand.reset_index()
    monthly_table.columns = ["Month", "Total Quantity Sold"]

    st.dataframe(monthly_table, use_container_width=True)

else:
    st.warning("No sales data found for this product.")
    # Product Demand Growth Analysis

st.divider()
st.header("📈 Product Demand Growth Analysis")

if all(col in df.columns for col in ["InvoiceDate", "Description", "Quantity"]):

    growth_df = df.copy()

    growth_df["InvoiceDate"] = pd.to_datetime(
        growth_df["InvoiceDate"], errors="coerce"
    )

    growth_df["Quantity"] = pd.to_numeric(
        growth_df["Quantity"], errors="coerce"
    )

    growth_df = growth_df.dropna(
        subset=["InvoiceDate", "Description", "Quantity"]
    )

    growth_df = growth_df[growth_df["Quantity"] > 0].copy()

    growth_df["Month"] = (
        growth_df["InvoiceDate"].dt.to_period("M").astype(str)
    )

    monthly_sales = (
        growth_df.groupby(["Month", "Description"])["Quantity"]
        .sum()
        .reset_index()
    )

    products = sorted(growth_df["Description"].unique())

    selected_growth_product = st.selectbox(
        "Select Product to Analyze Demand Growth",
        products,
        key="growth_product"
    )

    product_growth = monthly_sales[
        monthly_sales["Description"] == selected_growth_product
    ].sort_values("Month")

    if len(product_growth) >= 2:

        first_month = product_growth.iloc[0]
        last_month = product_growth.iloc[-1]

        first_quantity = first_month["Quantity"]
        last_quantity = last_month["Quantity"]

        growth_percentage = (
            (last_quantity - first_quantity) / first_quantity * 100
            if first_quantity != 0 else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "First Month Demand",
            f"{first_quantity:,.0f}"
        )

        col2.metric(
            "Latest Month Demand",
            f"{last_quantity:,.0f}"
        )

        col3.metric(
            "Demand Growth",
            f"{growth_percentage:.2f}%"
        )

        fig = px.line(
            product_growth,
            x="Month",
            y="Quantity",
            markers=True,
            title=f"Monthly Demand Growth: {selected_growth_product}"
        )

        st.plotly_chart(fig, width="stretch")

        if last_quantity > first_quantity:
            st.success("Demand increased between the first and latest recorded months.")
        elif last_quantity < first_quantity:
            st.warning("Demand decreased between the first and latest recorded months.")
        else:
            st.info("Demand remained the same between the first and latest recorded months.")

    else:
        st.info("Not enough monthly data to compare demand growth.")

else:
    st.warning("Required columns are missing from the dataset.")
    # ABC Inventory Analysis

st.divider()
st.header("📊 ABC Inventory Analysis")

if all(col in df.columns for col in ["Description", "Quantity", "Price"]):

    abc_df = df.copy()

    abc_df["Quantity"] = pd.to_numeric(
        abc_df["Quantity"], errors="coerce"
    )

    abc_df["Price"] = pd.to_numeric(
        abc_df["Price"], errors="coerce"
    )

    abc_df = abc_df.dropna(
        subset=["Description", "Quantity", "Price"]
    )

    # Keep positive sales records
    abc_df = abc_df[
        (abc_df["Quantity"] > 0) &
        (abc_df["Price"] > 0)
    ].copy()

    # Calculate product sales value
    abc_df["Sales Value"] = (
        abc_df["Quantity"] * abc_df["Price"]
    )

    abc_summary = (
        abc_df.groupby("Description")["Sales Value"]
        .sum()
        .reset_index()
        .sort_values("Sales Value", ascending=False)
    )

    total_sales_value = abc_summary["Sales Value"].sum()

    if total_sales_value > 0:

        abc_summary["Sales Contribution (%)"] = (
            abc_summary["Sales Value"] /
            total_sales_value * 100
        )

        abc_summary["Cumulative Contribution (%)"] = (
            abc_summary["Sales Contribution (%)"].cumsum()
        )

        # Assign ABC categories
        abc_summary["ABC Category"] = "C"

        abc_summary.loc[
            abc_summary["Cumulative Contribution (%)"] <= 80,
            "ABC Category"
        ] = "A"

        abc_summary.loc[
            (
                abc_summary["Cumulative Contribution (%)"] > 80
            ) &
            (
                abc_summary["Cumulative Contribution (%)"] <= 95
            ),
            "ABC Category"
        ] = "B"

        # Ensure the product crossing 80% remains in A
        crossing_a = abc_summary[
            abc_summary["Cumulative Contribution (%)"] > 80
        ].head(1).index

        if len(crossing_a) > 0:
            abc_summary.loc[crossing_a, "ABC Category"] = "A"

        # Summary cards
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "A Category Products",
            int((abc_summary["ABC Category"] == "A").sum())
        )

        col2.metric(
            "B Category Products",
            int((abc_summary["ABC Category"] == "B").sum())
        )

        col3.metric(
            "C Category Products",
            int((abc_summary["ABC Category"] == "C").sum())
        )

        # Category chart
        category_summary = (
            abc_summary.groupby("ABC Category")["Sales Value"]
            .sum()
            .reindex(["A", "B", "C"], fill_value=0)
            .reset_index()
        )

        fig = px.pie(
            category_summary,
            names="ABC Category",
            values="Sales Value",
            title="Sales Value Contribution by ABC Category",
            hole=0.4
        )

        st.plotly_chart(fig, width="stretch")

        # Filter products
        abc_filter = st.selectbox(
            "Select ABC Category",
            ["All", "A", "B", "C"],
            key="abc_category_filter"
        )

        if abc_filter == "All":
            abc_display = abc_summary.copy()
        else:
            abc_display = abc_summary[
                abc_summary["ABC Category"] == abc_filter
            ].copy()

        st.dataframe(
            abc_display,
            width="stretch",
            hide_index=True
        )

        
    else:
        st.warning("Total sales value is zero; ABC analysis cannot be calculated.")

else:
    st.warning("Description, Quantity or Price column is missing.")
    # No Recent Sales Analysis

st.divider()
st.header("🕒 No Recent Sales Analysis")

if all(
    col in df.columns
    for col in ["InvoiceDate", "Description", "Quantity"]
):

    recent_df = df.copy()

    recent_df["InvoiceDate"] = pd.to_datetime(
        recent_df["InvoiceDate"],
        errors="coerce"
    )

    recent_df["Quantity"] = pd.to_numeric(
        recent_df["Quantity"],
        errors="coerce"
    )

    recent_df = recent_df.dropna(
        subset=["InvoiceDate", "Description", "Quantity"]
    )

    recent_df = recent_df[
        recent_df["Quantity"] > 0
    ].copy()

    if not recent_df.empty:

        latest_date = recent_df["InvoiceDate"].max()

        cutoff_date = latest_date - pd.Timedelta(days=30)

        recent_sales = recent_df[
            recent_df["InvoiceDate"] > cutoff_date
        ]

        recent_products = set(
            recent_sales["Description"].unique()
        )

        all_products = set(
            recent_df["Description"].unique()
        )

        no_recent_products = all_products - recent_products

        st.metric(
            "Products Without Sales in Latest 30 Days",
            len(no_recent_products)
        )

        if no_recent_products:

            no_sales_df = recent_df[
                recent_df["Description"].isin(no_recent_products)
            ]

            product_summary = (
                no_sales_df.groupby("Description")
                .agg(
                    Last_Sale_Date=("InvoiceDate", "max"),
                    Historical_Quantity=("Quantity", "sum")
                )
                .reset_index()
                .sort_values("Last_Sale_Date", ascending=False)
            )

            st.dataframe(
                product_summary,
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "Download No Recent Sales Report",
                data=product_summary.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="no_recent_sales_report.csv",
                mime="text/csv"
            )

        else:
            st.success(
                "All recorded products have sales within the latest 30 days."
            )

    else:
        st.warning("No valid sales records available.")

else:
    st.warning("Required date, product or quantity columns are missing.")
#Stock Risk Analysis

st.divider()
st.header("⚠️ Stock Risk Analysis")

if all(col in df.columns for col in ["InvoiceDate", "Description", "Quantity"]):

 risk_df = df.copy()

risk_df["InvoiceDate"] = pd.to_datetime(
    risk_df["InvoiceDate"], errors="coerce"
)
risk_df["Quantity"] = pd.to_numeric(
    risk_df["Quantity"], errors="coerce"
)

risk_df = risk_df.dropna(
    subset=["InvoiceDate", "Description", "Quantity"]
)
risk_df = risk_df[risk_df["Quantity"] > 0]

# Compare the latest two months available in the dataset
risk_df["Month"] = risk_df["InvoiceDate"].dt.to_period("M")

available_months = sorted(risk_df["Month"].unique())

if len(available_months) < 2:
    st.warning("At least two months of sales data are required.")
else:
    latest_month = available_months[-1]
    previous_month = available_months[-2]

    monthly_product_sales = (
        risk_df.groupby(["Month", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )

    previous_sales = monthly_product_sales.loc[previous_month]
    latest_sales = monthly_product_sales.loc[latest_month]

    risk_table = pd.DataFrame({
        "Product": monthly_product_sales.columns,
        "Previous Month Sales": previous_sales.reindex(
            monthly_product_sales.columns, fill_value=0
        ).values,
        "Latest Month Sales": latest_sales.reindex(
            monthly_product_sales.columns, fill_value=0
        ).values
    })

    risk_table["Demand Change (%)"] = (
        (
            risk_table["Latest Month Sales"]
            - risk_table["Previous Month Sales"]
        )
        / risk_table["Previous Month Sales"].replace(0, float("nan"))
    ) * 100

    risk_table["Risk Level"] = "Low Risk"

    risk_table.loc[
        risk_table["Demand Change (%)"] <= -30,
        "Risk Level"
    ] = "High Risk"

    risk_table.loc[
        (risk_table["Demand Change (%)"] > -30)
        & (risk_table["Demand Change (%)"] <= -10),
        "Risk Level"
    ] = "Medium Risk"

    risk_table.loc[
        risk_table["Previous Month Sales"] == 0,
        "Risk Level"
    ] = "Review Required"

    st.caption(
        f"Comparing {previous_month} with {latest_month}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "High-Risk Products",
            int((risk_table["Risk Level"] == "High Risk").sum())
        )

    with col2:
        st.metric(
            "Medium-Risk Products",
            int((risk_table["Risk Level"] == "Medium Risk").sum())
        )

    with col3:
        st.metric(
            "Products to Review",
            int((risk_table["Risk Level"] == "Review Required").sum())
        )

    st.subheader("Product Demand Risk Details")

    risk_filter = st.selectbox(
        "Filter by Risk Level",
        [
            "All",
            "High Risk",
            "Medium Risk",
            "Low Risk",
            "Review Required"
        ],
        key="stock_risk_filter"
    )

    if risk_filter != "All":
        display_risk = risk_table[
            risk_table["Risk Level"] == risk_filter
        ]
    else:
        display_risk = risk_table

    st.dataframe(
        display_risk.sort_values("Demand Change (%)", na_position="first"),
        use_container_width=True
    )

    st.download_button(
        "Download Stock Risk Report",
        data=risk_table.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_stock_risk.csv",
        mime="text/csv"
    )
    # Inventory Performance Summary

st.divider()
st.header("📊 Inventory Performance Summary")

if all(
    col in df.columns
    for col in ["Description", "Quantity", "Price"]
):

    performance_df = df.copy()

    performance_df["Quantity"] = pd.to_numeric(
        performance_df["Quantity"], errors="coerce"
    )

    performance_df["Price"] = pd.to_numeric(
        performance_df["Price"], errors="coerce"
    )

    performance_df = performance_df.dropna(
        subset=["Description", "Quantity", "Price"]
    )

    performance_df = performance_df[
        (performance_df["Quantity"] > 0) &
        (performance_df["Price"] > 0)
    ].copy()

    performance_df["Sales Value"] = (
        performance_df["Quantity"] * performance_df["Price"]
    )

    performance_summary = (
        performance_df.groupby("Description")
        .agg(
            Total_Quantity=("Quantity", "sum"),
            Total_Sales_Value=("Sales Value", "sum")
        )
        .reset_index()
    )

    if not performance_summary.empty:

        top_product = performance_summary.loc[
            performance_summary["Total_Quantity"].idxmax()
        ]

        low_product = performance_summary.loc[
            performance_summary["Total_Quantity"].idxmin()
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Products Analyzed",
            f"{len(performance_summary):,}"
        )

        col2.metric(
            "Total Historical Sales Value",
            f"{performance_summary['Total_Sales_Value'].sum():,.2f}"
        )

        col3.metric(
            "Average Quantity per Product",
            f"{performance_summary['Total_Quantity'].mean():,.2f}"
        )

        st.subheader("🏆 Product Performance")

        col4, col5 = st.columns(2)

        col4.write("**Top-Selling Product**")
        col4.write(str(top_product["Description"]))
        col4.metric(
            "Quantity Sold",
            f"{top_product['Total_Quantity']:,.0f}"
        )

        col5.write("**Lowest-Selling Product**")
        col5.write(str(low_product["Description"]))
        col5.metric(
            "Quantity Sold",
            f"{low_product['Total_Quantity']:,.0f}"
        )

        st.subheader("📋 Product Performance Details")

        st.dataframe(
            performance_summary.sort_values(
                "Total_Quantity",
                ascending=False
            ),
            width="stretch",
            hide_index=True
        )

# Top 10 Best-Selling Products Chart
        st.subheader("📊 Top 10 Best-Selling Products")

        top_10_products = performance_summary.nlargest(
            10, "Total_Quantity"
        ).sort_values(
            "Total_Quantity",
            ascending=True
        )

        st.bar_chart(
            top_10_products.set_index("Description")["Total_Quantity"],
            horizontal=True
        )
    else:
        st.warning("No valid product sales data available.")

else:
    st.warning("Description, Quantity or Price column is missing.")
    

#Inventory Recommendations

st.divider()
st.header("💡 Inventory Recommendations")

if all(col in df.columns for col in ["Description", "Quantity"]):

 recommendation_df = df.copy()

recommendation_df["Quantity"] = pd.to_numeric(
    recommendation_df["Quantity"],
    errors="coerce"
)

recommendation_df = recommendation_df.dropna(
    subset=["Description", "Quantity"]
)

recommendation_df = recommendation_df[
    recommendation_df["Quantity"] > 0
]

# Calculate total sales quantity for each product
product_summary = (
    recommendation_df.groupby("Description")["Quantity"]
    .sum()
    .reset_index()
)

product_summary.columns = [
    "Product",
    "Total Quantity Sold"
]

# Classify products based on demand
low_limit = product_summary[
    "Total Quantity Sold"
].quantile(0.25)

high_limit = product_summary[
    "Total Quantity Sold"
].quantile(0.75)

def make_recommendation(quantity):
    if quantity <= low_limit:
        return "Review demand before placing new orders"
    elif quantity >= high_limit:
        return "Prioritize demand monitoring and replenishment planning"
    else:
        return "Monitor sales and review periodically"

product_summary["Recommendation"] = (
    product_summary["Total Quantity Sold"]
    .apply(make_recommendation)
)

# Summary
col1, col2, col3 = st.columns(3)

col1.metric(
    "Products Analyzed",
    product_summary["Product"].nunique()
)

col2.metric(
    "Products Needing Review",
    int(
        (
            product_summary["Total Quantity Sold"] <= low_limit
        ).sum()
    )
)

col3.metric(
    "High-Demand Products",
    int(
        (
            product_summary["Total Quantity Sold"] >= high_limit
        ).sum()
    )
)

# ==============================
# PRODUCT SEARCH
# ==============================

st.divider()
st.header("🔍 Product Search")

if "Description" in df.columns:

    search_text = st.text_input(
        "Enter Product Name",
        placeholder="Example: PAPER",
        key="inventory_product_search"
    )

    if search_text.strip():

        filtered_search = df[
            df["Description"]
            .astype(str)
            .str.contains(
                search_text.strip(),
                case=False,
                na=False
            )
        ].copy()

        if not filtered_search.empty:

            st.success(
                f"Found {len(filtered_search)} matching records."
            )

            st.dataframe(
                filtered_search.head(100),
                use_container_width=True,
                hide_index=True
            )

            search_csv = filtered_search.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Search Results",
                data=search_csv,
                file_name="product_search_results.csv",
                mime="text/csv",
                key="download_product_search_results"
            )

        else:
            st.warning("No matching products found.")

    else:
        st.info("Type a product name to search.")

else:
    st.warning("Product Description column is missing.")
st.subheader("📋 Product-wise Recommendations")

# Filter
recommendation_filter = st.selectbox(
    "Select Product Group",
    [
        "All Products",
        "Low Demand",
        "Medium Demand",
        "High Demand"
    ],
    key="inventory_recommendation_filter"
)

product_summary["Demand Category"] = (
    product_summary["Total Quantity Sold"].apply(
        lambda q: (
            "Low Demand" if q <= low_limit
            else "High Demand" if q >= high_limit
            else "Medium Demand"
        )
    )
)

if recommendation_filter == "All Products":
    filtered_products = product_summary.copy()
else:
    filtered_products = product_summary[
        product_summary["Demand Category"] == recommendation_filter
    ]

st.dataframe(
    filtered_products.sort_values(
        "Total Quantity Sold",
        ascending=False
    ),
    use_container_width=True
)

# Download report
report = product_summary.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Inventory Recommendations",
    data=report,
    file_name="retailpulse_inventory_recommendations.csv",
    mime="text/csv"
)

st.info(
    "Recommendations are based on historical sales demand. "
    "Check current stock levels, supplier lead times and purchase orders "
    "before making actual inventory decisions."
)

from pathlib import Path

# Low Stock Alert and Stock Reorder Recommendation

st.subheader("🚨 Low Stock Alert")

stock_file = (
    Path(r"C:\Users\acer\Desktop\RERAILPULSE")
    / "data"
    / "current_stock.csv"
)

if stock_file.exists():

    stock_df = pd.read_csv(stock_file)

    stock_df["Current_Stock"] = pd.to_numeric(
        stock_df["Current_Stock"], errors="coerce"
    )
    stock_df["Reorder_Level"] = pd.to_numeric(
        stock_df["Reorder_Level"], errors="coerce"
    )

    stock_df = stock_df.dropna(
        subset=["Current_Stock", "Reorder_Level"]
    )

    low_stock = stock_df[
        stock_df["Current_Stock"] <= stock_df["Reorder_Level"]
    ].copy()

    critical_stock = stock_df[
        stock_df["Current_Stock"] <= stock_df["Reorder_Level"] * 0.5
    ].copy()
    # Products to Reorder
    reorder_products = low_stock.copy()

    reorder_products["Suggested_Order"] = (
        reorder_products["Reorder_Level"]
        - reorder_products["Current_Stock"]
    ).clip(lower=0)

    # Stock Status Distribution

    stock_df["Stock_Status"] = "Normal Stock"

    stock_df.loc[
        stock_df["Current_Stock"] <= stock_df["Reorder_Level"],
        "Stock_Status"
    ] = "Low Stock"

    stock_df.loc[
        stock_df["Current_Stock"] <= stock_df["Reorder_Level"] * 0.5,
        "Stock_Status"
    ] = "Critical Stock"

    st.markdown("### 📊 Stock Status Distribution")

    status_counts = stock_df["Stock_Status"].value_counts().reset_index()
    status_counts.columns = ["Stock Status", "Product Count"]

    fig = px.pie(
        status_counts,
        names="Stock Status",
        values="Product Count",
        title="Inventory Stock Status"
    )

    st.plotly_chart(fig, use_container_width=True)

# Products to Reorder
    st.markdown("### 📦 Products to Reorder")

    reorder_products = low_stock.copy()

    reorder_products["Suggested_Order"] = (
        reorder_products["Reorder_Level"]
        - reorder_products["Current_Stock"]
    ).clip(lower=0)

    if not reorder_products.empty:

        st.dataframe(
            reorder_products[
                [
                    "StockCode",
                    "Product",
                    "Current_Stock",
                    "Reorder_Level",
                    "Suggested_Order"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        reorder_csv = reorder_products.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Reorder Recommendation",
            data=reorder_csv,
            file_name="stock_reorder_recommendation.csv",
            mime="text/csv",
            key="download_reorder_recommendation"
        )

    else:
        st.success("No products need reordering.")

    # Reorder Summary
    st.markdown("### 📊 Reorder Summary")

    col1, col2 = st.columns(2)

    col1.metric(
        "Products to Reorder",
        len(reorder_products)
    )

    col2.metric(
        "Total Suggested Units",
        int(reorder_products["Suggested_Order"].sum())
    )

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric("Low Stock Products", len(low_stock))
    col2.metric("Critical Stock Products", len(critical_stock))

    if not low_stock.empty:

        st.warning("⚠️ Some products need stock replenishment.")

        st.dataframe(
            low_stock[
                ["StockCode", "Product", "Current_Stock", "Reorder_Level"]
            ],
            use_container_width=True,
            hide_index=True
        )

        stock_report = low_stock.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Low Stock Report",
            data=stock_report,
            file_name="low_stock_report.csv",
            mime="text/csv",
            key="download_low_stock_report"
        )

    else:
        st.success("All products have sufficient stock.")

else:
    st.warning("Current stock file not found.")