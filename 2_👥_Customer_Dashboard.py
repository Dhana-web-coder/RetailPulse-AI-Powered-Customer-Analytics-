import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Customer Dashboard",
    page_icon="👥",
    layout="wide"
)

st.title("👥 RetailPulse - Customer Dashboard")
st.write("Customer Segmentation using RFM Analysis")

df = pd.read_csv(
    "data/retail_customer_clean.csv",
    low_memory=False
)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")

df = df.dropna(subset=["Customer ID", "InvoiceDate"])

df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
].copy()

df["Sales"] = df["Quantity"] * df["Price"]

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("Sales", "sum")
).reset_index()
st.write("Total customers in DF:", df["Customer ID"].nunique())
st.write("Total customers in RFM:", rfm["Customer ID"].nunique())

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select RFM features
X = rfm[["Recency", "Frequency", "Monetary"]].copy()

# Handle skewed customer spending
X["Monetary"] = X["Monetary"].clip(
    upper=X["Monetary"].quantile(0.99)
)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Customer Segment"] = kmeans.fit_predict(X_scaled)

# Display results
st.subheader("🤖 ML Customer Segmentation")

st.write("Customers are grouped into 4 clusters using K-Means.")

# Customer Segment Filter
st.subheader("🔎 Customer Filters")

segment_options = ["All Segments"] + sorted(
    rfm["Customer Segment"].unique().tolist()
)

selected_segment = st.selectbox(
    "Select Customer Segment",
    segment_options
)

if selected_segment == "All Segments":
    filtered_rfm = rfm.copy()
else:
    filtered_rfm = rfm[
        rfm["Customer Segment"] == selected_segment
    ].copy()

st.write(f"Customers displayed: {len(filtered_rfm):,}")

# Customer ID Search
st.subheader("🔍 Search Customer")

search_id = st.text_input(
    "Enter Customer ID",
    placeholder="Example: 12346"
)

if search_id.strip():
    filtered_rfm = filtered_rfm[
        filtered_rfm["Customer ID"]
        .astype(str)
        .str.contains(search_id.strip(), regex=False)
    ].copy()

st.write(f"Matching customers: {len(filtered_rfm):,}")
# Display chart
st.bar_chart(
    filtered_rfm["Customer Segment"].value_counts().sort_index()
)

# Display filtered customer table
st.dataframe(
    filtered_rfm[
        [
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary",
            "Customer Segment"
        ]
    ],
    use_container_width=True
)


# Customer Spending Filter
st.subheader("💰 Customer Spending Filter")

min_spending = float(rfm["Monetary"].min())
max_spending = float(rfm["Monetary"].max())

col1, col2 = st.columns(2)

with col1:
    spending_min = st.number_input(
        "Minimum Spending ",
        min_value=0.0,
        max_value=max_spending,
        value=0.0,
        key="minimum_spending_filter"
    )

with col2:
    spending_max = st.number_input(
        "Maximum Spending ",
        min_value=0.0,
        max_value=max_spending,
        value=float(max_spending),
        key="maximum_spending_filter"
    )

if spending_min <= spending_max:
    filtered_rfm = filtered_rfm[
        (filtered_rfm["Monetary"] >= spending_min) &
        (filtered_rfm["Monetary"] <= spending_max)
    ].copy()
else:
    st.warning("Minimum spending must be less than maximum spending.")

st.write(f"Customers after filters: {len(filtered_rfm):,}")

# Display chart
st.bar_chart(
    filtered_rfm["Customer Segment"].value_counts().sort_index()
)

# Display filtered customer table
st.dataframe(
    filtered_rfm[
        [
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary",
            "Customer Segment"
        ]
    ],
    use_container_width=True
)
# Download Filtered Customer Data
st.subheader("📥 Download Customer Report")

csv_data = filtered_rfm.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Customers (CSV)",
    data=csv_data,
    file_name="retailpulse_filtered_customers.csv",
    mime="text/csv",
    key="download_filtered_customers"
)
st.subheader("Customer KPIs")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{rfm['Customer ID'].nunique():,}")
col2.metric("Total Orders", f"{rfm['Frequency'].sum():,}")
col3.metric("Total Customer Sales", f"{rfm['Monetary'].sum():,.2f}")

st.subheader("Top 10 Customers")

top_customers = rfm.sort_values(
    "Monetary", ascending=False
).head(10)

st.dataframe(top_customers, use_container_width=True)

st.subheader("Customer Purchase Frequency")

frequency_data = (
    rfm["Frequency"]
    .value_counts()
    .sort_index()
    .head(20)
)

st.bar_chart(frequency_data)

st.subheader("Top 20 Customers by Spending")

spending_data = (
    rfm.nlargest(20, "Monetary")
    .set_index("Customer ID")["Monetary"]
)

st.bar_chart(spending_data)

# RFM Customer Analysis
st.header("📊 RFM Customer Analysis")

rfm_analysis = rfm.copy()

# Create RFM scores
rfm_analysis["R_Score"] = pd.qcut(
    rfm_analysis["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm_analysis["F_Score"] = pd.qcut(
    rfm_analysis["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm_analysis["M_Score"] = pd.qcut(
    rfm_analysis["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Calculate total RFM score
rfm_analysis["RFM Score"] = (
    rfm_analysis["R_Score"]
    + rfm_analysis["F_Score"]
    + rfm_analysis["M_Score"]
)

# Assign customer segments
def assign_rfm_segment(row):
    if (
        row["R_Score"] >= 4
        and row["F_Score"] >= 4
        and row["M_Score"] >= 4
    ):
        return "Best Customers"

    elif row["F_Score"] >= 4 and row["R_Score"] >= 3:
        return "Loyal Customers"

    elif row["R_Score"] >= 4 and row["F_Score"] <= 2:
        return "Potential Loyalists"

    elif row["R_Score"] <= 2 and row["F_Score"] >= 3:
        return "At-Risk Customers"

    else:
        return "Other Customers"


rfm_analysis["RFM Segment"] = rfm_analysis.apply(
    assign_rfm_segment,
    axis=1
)

# Display segment distribution
st.subheader("Customer Segment Distribution")

segment_counts = rfm_analysis["RFM Segment"].value_counts()

st.bar_chart(segment_counts)

# Display RFM customer details
st.subheader("RFM Customer Details")

st.dataframe(
    rfm_analysis[
        [
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score",
            "RFM Score",
            "RFM Segment"
        ]
    ],
    use_container_width=True
)
# Customer Retention Analysis
st.header("🔄 Customer Retention Analysis")

# Count orders for each customer
customer_orders = df.groupby("Customer ID").agg(
    Total_Orders=("Invoice", "nunique"),
    First_Purchase=("InvoiceDate", "min"),
    Last_Purchase=("InvoiceDate", "max")
).reset_index()

# Identify repeat customers
customer_orders["Customer Type"] = customer_orders[
    "Total_Orders"
].apply(
    lambda orders: "Repeat Customer"
    if orders > 1
    else "One-Time Customer"
)

# Calculate customer counts
repeat_customers = (
    customer_orders["Customer Type"] == "Repeat Customer"
).sum()

one_time_customers = (
    customer_orders["Customer Type"] == "One-Time Customer"
).sum()

total_customers = len(customer_orders)

retention_percentage = (
    repeat_customers / total_customers * 100
    if total_customers > 0
    else 0
)

# Display KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Repeat Customers", f"{repeat_customers:,}")
col3.metric("Repeat Customer %", f"{retention_percentage:.2f}%")

# Display chart
st.subheader("Customer Purchase Type Distribution")

customer_type_counts = customer_orders[
    "Customer Type"
].value_counts()

st.bar_chart(customer_type_counts)

# Display customer details
st.subheader("Customer Retention Details")

st.dataframe(
    customer_orders,
    use_container_width=True,
    hide_index=True
)
# Customer Purchase Behavior Analysis
st.header("🛒 Customer Purchase Behavior Analysis")

# Create order-level data
order_data = df.groupby("Invoice").agg(
    Customer_ID=("Customer ID", "first"),
    Order_Date=("InvoiceDate", "first"),
    Order_Value=("Sales", "sum"),
    Total_Quantity=("Quantity", "sum")
).reset_index()

# Calculate purchase behavior metrics
total_orders = order_data["Invoice"].nunique()

average_order_value = (
    order_data["Order_Value"].mean()
    if total_orders > 0 else 0
)

average_quantity = (
    order_data["Total_Quantity"].mean()
    if total_orders > 0 else 0
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Average Order Value", f"{average_order_value:,.2f}")
col3.metric("Average Quantity per Order", f"{average_quantity:.2f}")

# Monthly purchasing behavior
order_data["Month"] = (
    order_data["Order_Date"].dt.to_period("M").astype(str)
)

monthly_orders = (
    order_data.groupby("Month")["Invoice"]
    .nunique()
    .sort_index()
)

st.subheader("📅 Monthly Order Trend")
st.line_chart(monthly_orders)

# Monthly sales behavior
monthly_order_sales = (
    order_data.groupby("Month")["Order_Value"]
    .sum()
    .sort_index()
)

st.subheader("💰 Monthly Customer Order Sales")
st.line_chart(monthly_order_sales)

# Show order details
st.subheader("Order-Level Details")

st.dataframe(
    order_data,
    use_container_width=True,
    hide_index=True
)
# Customer Lifetime Value Analysis
st.header("💰 Customer Lifetime Value Analysis")

# Calculate historical customer value
clv_data = df.groupby("Customer ID").agg(
    Total_Spending=("Sales", "sum"),
    Total_Orders=("Invoice", "nunique"),
    First_Purchase=("InvoiceDate", "min"),
    Last_Purchase=("InvoiceDate", "max")
).reset_index()

# Calculate average order value per customer
clv_data["Average_Order_Value"] = (
    clv_data["Total_Spending"] /
    clv_data["Total_Orders"].replace(0, pd.NA)
)

# Calculate customer activity duration
clv_data["Customer_Duration_Days"] = (
    clv_data["Last_Purchase"] -
    clv_data["First_Purchase"]
).dt.days

# Display KPIs
col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Historical Customer Value",
    f"{clv_data['Total_Spending'].sum():,.2f}"
)

col2.metric(
    "Average Customer Spending",
    f"{clv_data['Total_Spending'].mean():,.2f}"
)

col3.metric(
    "Average Orders per Customer",
    f"{clv_data['Total_Orders'].mean():.2f}"
)

# Top customers by historical value
st.subheader("🏆 Top 10 Customers by Historical Value")

top_clv = clv_data.nlargest(10, "Total_Spending")

st.bar_chart(
    top_clv.set_index("Customer ID")["Total_Spending"]
)

st.dataframe(
    top_clv,
    use_container_width=True,
    hide_index=True
)

# Download report
clv_csv = clv_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Customer Value Report",
    data=clv_csv,
    file_name="retailpulse_customer_value.csv",
    mime="text/csv",
    key="download_customer_value"
)
# Customer Insights and Business Recommendations
st.header("💡 Customer Insights & Business Recommendations")

total_customers = rfm["Customer ID"].nunique()

repeat_customers_count = (rfm["Frequency"] > 1).sum()

repeat_percentage = (
    repeat_customers_count / len(rfm) * 100
    if len(rfm) > 0 else 0
)

average_spending = rfm["Monetary"].mean()

top_customer = rfm.loc[rfm["Monetary"].idxmax()]

high_value_threshold = rfm["Monetary"].quantile(0.75)

high_value_customers = (
    rfm["Monetary"] >= high_value_threshold
).sum()

# Display business insights
col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Repeat Purchase Rate",
    f"{repeat_percentage:.2f}%"
)

col3.metric(
    "Average Customer Spending",
    f"{average_spending:,.2f}"
)

st.subheader("📊 Key Customer Insights")

st.write(
    f"• The dataset contains {total_customers:,} unique customers."
)

st.write(
    f"• {repeat_customers_count:,} customers have placed more than one order."
)

st.write(
    f"• {high_value_customers:,} customers are in the top spending quartile."
)

st.write(
    f"• The highest-spending customer has historical spending "
    f"of {top_customer['Monetary']:,.2f}."
)

st.subheader("🎯 Business Recommendations")

if repeat_percentage < 50:
    st.warning(
        "Repeat purchase rate is below 50%. "
        "Consider loyalty rewards, follow-up emails, and targeted offers."
    )
else:
    st.success(
        "More than half of customers are repeat purchasers. "
        "Consider loyalty programs to maintain engagement."
    )

st.info(
    "Recommendation: Offer personalised promotions to high-value customers "
    "and analyse at-risk customers to identify opportunities for retention."
)

st.info(
    "Recommendation: Monitor monthly sales and customer purchase frequency "
    "to identify changes in customer behaviour."
)
st.success("Customer Dashboard loaded successfully!")