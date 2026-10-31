import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Retention",
    page_icon="❤️",
    layout="wide"
)

st.title("📈 RetailPulse - Improving Customer Retention")
st.write("Customer Retention and Repeat Purchase Analysis")

# ==========================================
# 2. LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILES = [
    BASE_DIR / "retail_cleaned_final.csv",
    BASE_DIR / "data" / "retail_cleaned_final.csv",
    BASE_DIR / "data" / "retail_customer_clean.csv"
]

@st.cache_data
def load_data():
    for file_path in DATA_FILES:
        if file_path.exists():
            df = pd.read_csv(file_path, low_memory=False)

            required_columns = [
                "Customer ID", "Invoice", "InvoiceDate", "Quantity", "Price"
            ]

            missing = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing columns: {', '.join(missing)}"
                )

            df["InvoiceDate"] = pd.to_datetime(
                df["InvoiceDate"],
                errors="coerce",
                dayfirst=True
            )

            df["Customer ID"] = df["Customer ID"].astype("string")
            df["Quantity"] = pd.to_numeric(
                df["Quantity"], errors="coerce"
            )
            df["Price"] = pd.to_numeric(
                df["Price"], errors="coerce"
            )

            return df

    raise FileNotFoundError(
        "Retail dataset not found in the expected locations."
    )

try:
    df = load_data()

    # Remove invalid customer and transaction records
    sales_df = df.dropna(
        subset=["Customer ID", "InvoiceDate"]
    ).copy()

    # Consider completed positive sales only
    sales_df = sales_df[
        (sales_df["Quantity"] > 0) &
        (sales_df["Price"] > 0) &
        (~sales_df["Invoice"].astype(str).str.startswith("C"))
    ].copy()

    sales_df["TotalSales"] = (
        sales_df["Quantity"] * sales_df["Price"]
    )

    st.success("Customer dataset loaded successfully!")

except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.stop()

if sales_df.empty:
    st.warning("No valid customer sales records found.")
    st.stop()

# ==========================================
# 3. CUSTOMER PURCHASE ANALYSIS
# ==========================================

st.divider()
st.subheader("👥 Customer Purchase Analysis")

customer_orders = (
    sales_df.groupby("Customer ID")["Invoice"]
    .nunique()
)

total_customers = len(customer_orders)

repeat_customers = (
    customer_orders[customer_orders > 1].count()
)

one_time_customers = (
    customer_orders[customer_orders == 1].count()
)

repeat_rate = (
    repeat_customers / total_customers * 100
    if total_customers > 0 else 0
)

# ==========================================
# 4. KEY PERFORMANCE INDICATORS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Repeat Customers",
    f"{repeat_customers:,}"
)

col3.metric(
    "One-Time Customers",
    f"{one_time_customers:,}"
)

col4.metric(
    "Repeat Purchase Rate",
    f"{repeat_rate:.2f}%"
)

# ==========================================
# 5. REPEAT VS ONE-TIME CUSTOMERS
# ==========================================

st.divider()
st.subheader("📊 Repeat vs One-Time Customers")

customer_type = pd.Series({
    "Repeat Customers": repeat_customers,
    "One-Time Customers": one_time_customers
})

fig, ax = plt.subplots(figsize=(7, 4))

customer_type.plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax,
    startangle=90
)

ax.set_ylabel("")
ax.set_title("Customer Purchase Type Distribution")

st.pyplot(fig)
plt.close(fig)

# ==========================================
# 6. MONTHLY ACTIVE CUSTOMERS
# ==========================================

st.divider()
st.subheader("📅 Monthly Active Customers")

sales_df["Month"] = (
    sales_df["InvoiceDate"].dt.to_period("M")
)

monthly_customers = (
    sales_df.groupby("Month")["Customer ID"]
    .nunique()
)

monthly_customers.index = (
    monthly_customers.index.astype(str)
)

st.line_chart(
    monthly_customers,
    x_label="Month",
    y_label="Unique Customers",
    width="stretch"
)

# ==========================================
# 7. MONTH-OVER-MONTH RETENTION RATE
# ==========================================

st.divider()
st.subheader("🔄 Monthly Customer Retention Rate")

monthly_customer_sets = (
    sales_df.groupby("Month")["Customer ID"]
    .apply(lambda x: set(x))
    .sort_index()
)

retention_results = []

for i in range(1, len(monthly_customer_sets)):
    previous_month = monthly_customer_sets.index[i - 1]
    current_month = monthly_customer_sets.index[i]

    previous_customers = monthly_customer_sets.iloc[i - 1]
    current_customers = monthly_customer_sets.iloc[i]

    retained = len(previous_customers & current_customers)

    retention = (
        retained / len(previous_customers) * 100
        if len(previous_customers) > 0 else 0
    )

    retention_results.append({
        "Month": str(current_month),
        "Previous Month": str(previous_month),
        "Retained Customers": retained,
        "Previous Month Customers": len(previous_customers),
        "Retention Rate (%)": round(retention, 2)
    })

if retention_results:
    retention_df = pd.DataFrame(retention_results)

    st.line_chart(
        retention_df.set_index("Month")["Retention Rate (%)"],
        y_label="Retention Rate (%)",
        width="stretch"
    )

    st.dataframe(
        retention_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("At least two months of sales data are required.")
    # ==========================================
# INACTIVE CUSTOMER ANALYSIS
# ==========================================

st.divider()
st.subheader("💤 Inactive Customer Analysis")

# Find each customer's last purchase date
customer_last_purchase = (
    sales_df.groupby("Customer ID")["InvoiceDate"]
    .max()
    .reset_index()
)

customer_last_purchase.columns = [
    "Customer ID",
    "Last Purchase Date"
]

# Latest date available in the dataset
reference_date = sales_df["InvoiceDate"].max()

# Calculate days since last purchase
customer_last_purchase["Days Since Last Purchase"] = (
    reference_date -
    customer_last_purchase["Last Purchase Date"]
).dt.days

# Identify inactive customers (90 days or more)
inactive_customers = customer_last_purchase[
    customer_last_purchase["Days Since Last Purchase"] >= 90
].copy()

active_customers = customer_last_purchase[
    customer_last_purchase["Days Since Last Purchase"] < 90
].copy()

# Display KPIs
col1, col2, col3 = st.columns(3)

col1.metric(
    "💤 Inactive Customers",
    f"{len(inactive_customers):,}"
)

col2.metric(
    "✅ Active Customers",
    f"{len(active_customers):,}"
)

col3.metric(
    "📊 Inactive Customer Rate",
    f"{len(inactive_customers) / len(customer_last_purchase) * 100:.2f}%"
    if len(customer_last_purchase) > 0 else "0%"
)

# Display inactive customer table
st.write("### Inactive Customer Details")

st.dataframe(
    inactive_customers.sort_values(
        "Days Since Last Purchase",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

# Download inactive customer list
st.download_button(
    label="📥 Download Inactive Customers",
    data=inactive_customers.to_csv(index=False),
    file_name="inactive_customers.csv",
    mime="text/csv"
)
# ==========================================
# CUSTOMER LIFETIME VALUE (CLV)
# ==========================================

st.divider()
st.subheader("💰 Customer Lifetime Value Analysis")

# Calculate customer lifetime value
customer_clv = (
    sales_df.groupby("Customer ID")
    .agg(
        Total_Spending=("TotalSales", "sum"),
        Total_Orders=("Invoice", "nunique"),
        Total_Quantity=("Quantity", "sum")
    )
    .reset_index()
)

# Average spending per customer
average_clv = customer_clv["Total_Spending"].mean()

# Display KPIs
col1, col2, col3 = st.columns(3)

col1.metric(
    "💰 Average Customer Value",
    f"{average_clv:,.2f}"
)

col2.metric(
    "🏆 Highest Customer Value",
    f"{customer_clv['Total_Spending'].max():,.2f}"
)

col3.metric(
    "👥 Total Customers",
    f"{len(customer_clv):,}"
)

# Top 10 valuable customers
st.write("### 🏆 Top 10 Valuable Customers")

top_clv_customers = (
    customer_clv.sort_values(
        "Total_Spending",
        ascending=False
    ).head(10)
)

st.dataframe(
    top_clv_customers,
    use_container_width=True,
    hide_index=True
)

# Customer value chart
st.write("### 📊 Top 10 Customers by Spending")

chart_data = top_clv_customers.set_index(
    "Customer ID"
)["Total_Spending"]

st.bar_chart(
    chart_data,
    horizontal=True,
    x_label="Total Spending ",
    width="stretch"
)

# Download CLV data
st.download_button(
    label="📥 Download Customer Lifetime Value",
    data=customer_clv.to_csv(index=False),
    file_name="customer_lifetime_value.csv",
    mime="text/csv"
)
# ==========================================
# CUSTOMER RETENTION COHORT ANALYSIS
# ==========================================

st.divider()
st.subheader("📊 Customer Retention Cohort Analysis")

# Create a copy of valid customer transactions
cohort_df = sales_df.copy()

# Extract transaction month
cohort_df["InvoiceMonth"] = (
    cohort_df["InvoiceDate"].dt.to_period("M")
)

# Find each customer's first purchase month
cohort_df["CohortMonth"] = (
    cohort_df.groupby("Customer ID")["InvoiceMonth"]
    .transform("min")
)

# Convert periods into month numbers
cohort_df["InvoiceMonthNumber"] = (
    cohort_df["InvoiceMonth"].apply(lambda x: x.year * 12 + x.month)
)

cohort_df["CohortMonthNumber"] = (
    cohort_df["CohortMonth"].apply(lambda x: x.year * 12 + x.month)
)

# Calculate months since first purchase
cohort_df["CohortIndex"] = (
    cohort_df["InvoiceMonthNumber"] -
    cohort_df["CohortMonthNumber"] + 1
)

# Count unique customers in each cohort
cohort_data = (
    cohort_df.groupby(
        ["CohortMonth", "CohortIndex"]
    )["Customer ID"]
    .nunique()
    .reset_index()
)

cohort_pivot = cohort_data.pivot(
    index="CohortMonth",
    columns="CohortIndex",
    values="Customer ID"
)

# Calculate retention percentage
cohort_size = cohort_pivot.iloc[:, 0]

retention_matrix = (
    cohort_pivot.divide(cohort_size, axis=0) * 100
)

retention_matrix.index = retention_matrix.index.astype(str)

# Display cohort table
st.write("### 📅 Monthly Customer Retention (%)")

st.dataframe(
    retention_matrix.round(2).style.format("{:.2f}%"),
    use_container_width=True
)

# Display cohort heatmap
st.write("### 🔥 Customer Retention Heatmap")

fig, ax = plt.subplots(
    figsize=(14, 6)
)

image = ax.imshow(
    retention_matrix.fillna(0),
    aspect="auto",
    cmap="YlGnBu",
    vmin=0,
    vmax=100
)

ax.set_title("Customer Cohort Retention (%)")
ax.set_xlabel("Months Since First Purchase")
ax.set_ylabel("First Purchase Month")

ax.set_xticks(range(len(retention_matrix.columns)))
ax.set_xticklabels(retention_matrix.columns)

ax.set_yticks(range(len(retention_matrix.index)))
ax.set_yticklabels(retention_matrix.index)

fig.colorbar(
    image,
    ax=ax,
    label="Retention Percentage"
)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
# ==========================================
# CUSTOMER WIN-BACK RECOMMENDATIONS
# ==========================================

st.divider()
st.subheader("📩 Customer Win-Back Recommendations")

# Calculate customer purchase information
winback_df = (
    sales_df.groupby("Customer ID")
    .agg(
        Total_Orders=("Invoice", "nunique"),
        Total_Spending=("TotalSales", "sum"),
        Last_Purchase=("InvoiceDate", "max")
    )
    .reset_index()
)

# Calculate days since last purchase
reference_date = sales_df["InvoiceDate"].max()

winback_df["Days_Inactive"] = (
    reference_date - winback_df["Last_Purchase"]
).dt.days

# Create recommendation based on inactivity
def get_winback_recommendation(days):

    if days >= 180:
        return "Send special comeback offer"

    elif days >= 120:
        return "Offer loyalty discount"

    elif days >= 90:
        return "Send personalized product offers"

    else:
        return "Customer is active - maintain engagement"


winback_df["Recommendation"] = (
    winback_df["Days_Inactive"]
    .apply(get_winback_recommendation)
)

# Filter inactive customers
winback_customers = winback_df[
    winback_df["Days_Inactive"] >= 90
].copy()

# Display KPIs
col1, col2, col3 = st.columns(3)

col1.metric(
    "💤 Inactive Customers",
    f"{len(winback_customers):,}"
)

col2.metric(
    "🎁 Comeback Offers",
    f"{(winback_customers['Days_Inactive'] >= 180).sum():,}"
)

col3.metric(
    "📩 Customers to Contact",
    f"{len(winback_customers):,}"
)

# Display recommendations
st.write("### 🎯 Customer Win-Back Action Plan")

st.dataframe(
    winback_customers.sort_values(
        "Days_Inactive",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

# Recommendation summary
st.write("### 📊 Recommendation Distribution")

recommendation_counts = (
    winback_customers["Recommendation"]
    .value_counts()
)

st.bar_chart(
    recommendation_counts,
    horizontal=True,
    x_label="Number of Customers",
    width="stretch"
)

# Download recommendation list
st.download_button(
    label="📥 Download Win-Back Recommendations",
    data=winback_customers.to_csv(index=False),
    file_name="customer_winback_recommendations.csv",
    mime="text/csv"
)
# ==========================================
# CUSTOMER RETENTION RISK ANALYSIS
# ==========================================

st.divider()
st.subheader("📉 Customer Retention Risk Analysis")

# Calculate customer purchase details
risk_df = (
    sales_df.groupby("Customer ID")
    .agg(
        Total_Orders=("Invoice", "nunique"),
        Total_Spending=("TotalSales", "sum"),
        Last_Purchase=("InvoiceDate", "max")
    )
    .reset_index()
)

# Calculate inactive days
reference_date = sales_df["InvoiceDate"].max()

risk_df["Days_Inactive"] = (
    reference_date - risk_df["Last_Purchase"]
).dt.days

# Assign risk category using business rules
def calculate_risk(row):

    if row["Days_Inactive"] >= 180:
        return "High Risk"

    elif row["Days_Inactive"] >= 90:
        return "Medium Risk"

    else:
        return "Low Risk"


risk_df["Risk_Level"] = risk_df.apply(
    calculate_risk,
    axis=1
)

# Count risk categories
risk_counts = (
    risk_df["Risk_Level"]
    .value_counts()
    .reindex(
        ["High Risk", "Medium Risk", "Low Risk"],
        fill_value=0
    )
)

# Display KPIs
col1, col2, col3 = st.columns(3)

col1.metric(
    "🔴 High Risk Customers",
    f"{risk_counts['High Risk']:,}"
)

col2.metric(
    "🟠 Medium Risk Customers",
    f"{risk_counts['Medium Risk']:,}"
)

col3.metric(
    "🟢 Low Risk Customers",
    f"{risk_counts['Low Risk']:,}"
)

# Display risk distribution
st.write("### 📊 Customer Risk Distribution")

st.bar_chart(
    risk_counts,
    x_label="Risk Level",
    y_label="Number of Customers",
    width="stretch"
)

# Filter customer risk
selected_risk = st.selectbox(
    "Select Risk Category",
    ["All", "High Risk", "Medium Risk", "Low Risk"],
    key="retention_risk_filter"
)

if selected_risk == "All":
    filtered_risk = risk_df
else:
    filtered_risk = risk_df[
        risk_df["Risk_Level"] == selected_risk
    ]

st.write("### 👥 Customer Risk Details")

st.dataframe(
    filtered_risk.sort_values(
        "Days_Inactive",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

# Download report
st.download_button(
    label="📥 Download Customer Risk Report",
    data=risk_df.to_csv(index=False),
    file_name="customer_retention_risk.csv",
    mime="text/csv"
)
# ==========================================
# CUSTOMER PURCHASE FREQUENCY ANALYSIS
# ==========================================

st.divider()
st.subheader("📊 Customer Purchase Frequency Analysis")

# Calculate orders per customer
frequency_df = (
    sales_df.groupby("Customer ID")
    .agg(
        Total_Orders=("Invoice", "nunique"),
        Total_Spending=("TotalSales", "sum")
    )
    .reset_index()
)

# Calculate average orders
average_orders = frequency_df["Total_Orders"].mean()

# Classify purchase frequency
def classify_frequency(orders):

    if orders >= 10:
        return "Frequent Buyers"

    elif orders >= 3:
        return "Occasional Buyers"

    else:
        return "Rare Buyers"


frequency_df["Purchase_Category"] = (
    frequency_df["Total_Orders"].apply(classify_frequency)
)

# Display KPIs
col1, col2, col3 = st.columns(3)

col1.metric(
    "📦 Average Orders per Customer",
    f"{average_orders:.2f}"
)

col2.metric(
    "👥 Total Customers",
    f"{len(frequency_df):,}"
)

col3.metric(
    "🔁 Frequent Buyers",
    f"{(frequency_df['Purchase_Category'] == 'Frequent Buyers').sum():,}"
)

# Purchase frequency distribution
st.write("### 📈 Purchase Frequency Distribution")

frequency_counts = (
    frequency_df["Purchase_Category"]
    .value_counts()
    .reindex(
        ["Frequent Buyers", "Occasional Buyers", "Rare Buyers"],
        fill_value=0
    )
)

st.bar_chart(
    frequency_counts,
    horizontal=True,
    x_label="Number of Customers",
    width="stretch"
)

# Customer frequency details
st.write("### 👥 Customer Purchase Details")

selected_category = st.selectbox(
    "Select Purchase Category",
    ["All", "Frequent Buyers", "Occasional Buyers", "Rare Buyers"],
    key="purchase_frequency_filter"
)

if selected_category == "All":
    filtered_frequency = frequency_df
else:
    filtered_frequency = frequency_df[
        frequency_df["Purchase_Category"] == selected_category
    ]

st.dataframe(
    filtered_frequency.sort_values(
        "Total_Orders",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

# Download report
st.download_button(
    label="📥 Download Purchase Frequency Report",
    data=frequency_df.to_csv(index=False),
    file_name="customer_purchase_frequency.csv",
    mime="text/csv"
)
# ==========================================
# CUSTOMER SPENDING BEHAVIOUR ANALYSIS
# ==========================================

st.divider()
st.subheader("🛍️ Customer Spending Behaviour Analysis")

# Calculate customer spending
spending_df = (
    sales_df.groupby("Customer ID")
    .agg(
        Total_Spending=("TotalSales", "sum"),
        Total_Orders=("Invoice", "nunique"),
        Total_Quantity=("Quantity", "sum")
    )
    .reset_index()
)

# Classify customers based on spending
def spending_category(amount):

    if amount >= 1000:
        return "High Spenders"

    elif amount >= 300:
        return "Medium Spenders"

    else:
        return "Low Spenders"


spending_df["Spending_Category"] = (
    spending_df["Total_Spending"].apply(spending_category)
)

# Display KPIs
col1, col2, col3 = st.columns(3)

col1.metric(
    "💰 Average Customer Spending",
    f"£{spending_df['Total_Spending'].mean():,.2f}"
)

col2.metric(
    "🏆 High Spenders",
    f"{(spending_df['Spending_Category'] == 'High Spenders').sum():,}"
)

col3.metric(
    "🛒 Total Customers",
    f"{len(spending_df):,}"
)

# Spending distribution
st.write("### 📊 Customer Spending Distribution")

spending_counts = (
    spending_df["Spending_Category"]
    .value_counts()
    .reindex(
        ["High Spenders", "Medium Spenders", "Low Spenders"],
        fill_value=0
    )
)

st.bar_chart(
    spending_counts,
    horizontal=True,
    x_label="Number of Customers",
    width="stretch"
)

# Top 10 spending customers
st.write("### 🏆 Top 10 Highest Spending Customers")

top_spenders = spending_df.sort_values(
    "Total_Spending",
    ascending=False
).head(10)

st.dataframe(
    top_spenders,
    use_container_width=True,
    hide_index=True
)

# Customer spending filter
selected_spending = st.selectbox(
    "Select Spending Category",
    ["All", "High Spenders", "Medium Spenders", "Low Spenders"],
    key="spending_category_filter"
)

if selected_spending == "All":
    filtered_spending = spending_df
else:
    filtered_spending = spending_df[
        spending_df["Spending_Category"] == selected_spending
    ]

st.write("### 👥 Customer Spending Details")

st.dataframe(
    filtered_spending.sort_values(
        "Total_Spending",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

# Download report
st.download_button(
    label="📥 Download Spending Behaviour Report",
    data=spending_df.to_csv(index=False),
    file_name="customer_spending_behaviour.csv",
    mime="text/csv"
)
# ==========================================
# CUSTOMER PURCHASE INTERVAL ANALYSIS
# ==========================================

st.header("🗓️ Customer Purchase Interval Analysis")

# Get unique customer purchases
purchase_dates = sales_df[
    ["Customer ID", "Invoice", "InvoiceDate"]
].drop_duplicates()

purchase_dates = purchase_dates.sort_values(
    ["Customer ID", "InvoiceDate"]
)

# Calculate days between purchases
purchase_dates["Days_Between_Purchases"] = (
    purchase_dates.groupby("Customer ID")["InvoiceDate"]
    .diff()
    .dt.days
)

# Calculate average purchase interval
interval_df = (
    purchase_dates.groupby("Customer ID")["Days_Between_Purchases"]
    .mean()
    .reset_index()
)

interval_df = interval_df.dropna()

interval_df["Days_Between_Purchases"] = (
    interval_df["Days_Between_Purchases"].round(1)
)

# KPIs
if not interval_df.empty:

    avg_interval = interval_df["Days_Between_Purchases"].mean()

    fast_customers = (
        interval_df["Days_Between_Purchases"] <= 30
    ).sum()

    slow_customers = (
        interval_df["Days_Between_Purchases"] > 90
    ).sum()

    c1, c2, c3 = st.columns(3)

    c1.metric("Average Purchase Interval", f"{avg_interval:.1f} days")
    c2.metric("Frequent Purchase Customers", f"{fast_customers:,}")
    c3.metric("Slow Purchase Customers", f"{slow_customers:,}")

    # Purchase interval chart
    st.subheader("Customer Purchase Interval Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(
        interval_df["Days_Between_Purchases"],
        bins=20,
        edgecolor="black"
    )

    ax.set_xlabel("Days Between Purchases")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Purchase Interval Distribution")

    st.pyplot(fig)
    plt.close(fig)

    # Customer details
    st.subheader("Customer-wise Purchase Interval")

    st.dataframe(
        interval_df.sort_values(
            "Days_Between_Purchases"
        ),
        use_container_width=True
    )

    # Download
    csv = interval_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Purchase Interval Report",
        data=csv,
        file_name="customer_purchase_interval.csv",
        mime="text/csv"
    )

else:
    st.info("Not enough repeat purchase data to calculate intervals.")
    # ==========================================
# CUSTOMER REVENUE CONTRIBUTION ANALYSIS
# ==========================================

st.header("💰 Customer Revenue Contribution Analysis")

# Calculate customer revenue
revenue_df = (
    sales_df.groupby("Customer ID")
    .agg(
        Customer_Revenue=("TotalSales", "sum"),
        Total_Orders=("Invoice", "nunique")
    )
    .reset_index()
)

# Calculate revenue contribution percentage
total_revenue = revenue_df["Customer_Revenue"].sum()

revenue_df["Revenue_Contribution_%"] = (
    revenue_df["Customer_Revenue"] / total_revenue * 100
).round(2)

# Sort customers by revenue
revenue_df = revenue_df.sort_values(
    "Customer_Revenue",
    ascending=False
)

# KPIs
c1, c2, c3 = st.columns(3)

c1.metric("Total Customer Revenue", f"{total_revenue:,.2f}")
c2.metric("Total Customers", f"{len(revenue_df):,}")

top_10_contribution = revenue_df.head(10)["Revenue_Contribution_%"].sum()

c3.metric("Top 10 Customer Contribution", f"{top_10_contribution:.2f}%")

# Top 10 customers chart
st.subheader("Top 10 Customers by Revenue")

top_10 = revenue_df.head(10)

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    top_10["Customer ID"].astype(str),
    top_10["Customer_Revenue"]
)

ax.set_xlabel("Customer Revenue")
ax.set_ylabel("Customer ID")
ax.set_title("Top 10 Revenue-Contributing Customers")

ax.invert_yaxis()

st.pyplot(fig)
plt.close(fig)

# Customer revenue table
st.subheader("Customer Revenue Details")

st.dataframe(
    revenue_df,
    use_container_width=True
)

# Download report
csv = revenue_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Revenue Contribution Report",
    data=csv,
    file_name="customer_revenue_contribution.csv",
    mime="text/csv"
)
# ==========================================
# CUSTOMER ORDER VALUE ANALYSIS
# ==========================================

st.header("🛒 Customer Order Value Analysis")

# Calculate invoice-wise order value
order_value_df = (
    sales_df.groupby(["Customer ID", "Invoice"])
    .agg(
        Order_Value=("TotalSales", "sum"),
        Order_Date=("InvoiceDate", "min")
    )
    .reset_index()
)

# Calculate KPIs
average_order_value = order_value_df["Order_Value"].mean()
highest_order_value = order_value_df["Order_Value"].max()
total_orders = order_value_df["Invoice"].nunique()

c1, c2, c3 = st.columns(3)

c1.metric("Average Order Value", f"{average_order_value:,.2f}")
c2.metric("Highest Order Value", f"{highest_order_value:,.2f}")
c3.metric("Total Orders", f"{total_orders:,}")

# Order value distribution
st.subheader("Order Value Distribution")

fig, ax = plt.subplots(figsize=(10, 4))

ax.hist(
    order_value_df["Order_Value"],
    bins=30,
    edgecolor="black"
)

ax.set_xlabel("Order Value")
ax.set_ylabel("Number of Orders")
ax.set_title("Customer Order Value Distribution")

st.pyplot(fig)
plt.close(fig)

# Top 10 highest-value orders
st.subheader("Top 10 Highest-Value Orders")

top_orders = order_value_df.nlargest(10, "Order_Value")

st.dataframe(top_orders, use_container_width=True)

# Customer-wise average order value
st.subheader("Customer-wise Average Order Value")

customer_order_value = (
    order_value_df.groupby("Customer ID")["Order_Value"]
    .mean()
    .reset_index(name="Average_Order_Value")
    .sort_values("Average_Order_Value", ascending=False)
)

st.dataframe(
    customer_order_value,
    use_container_width=True
)

# Download report
csv = order_value_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Order Value Report",
    data=csv,
    file_name="customer_order_value.csv",
    mime="text/csv"
)
# ==========================================
# 8. TOP REPEAT CUSTOMERS
# ==========================================

st.divider()
st.subheader("🏆 Top 10 Repeat Customers")

repeat_customer_details = (
    sales_df.groupby("Customer ID")
    .agg(
        Total_Orders=("Invoice", "nunique"),
        Total_Spending=("TotalSales", "sum"),
        Last_Purchase=("InvoiceDate", "max")
    )
    .reset_index()
)

top_repeat_customers = (
    repeat_customer_details[
        repeat_customer_details["Total_Orders"] > 1
    ]
    .sort_values("Total_Orders", ascending=False)
    .head(10)
)

st.dataframe(
    top_repeat_customers,
    use_container_width=True,
    hide_index=True
)

# ==========================================
# 9. CUSTOMER RETENTION RECOMMENDATIONS
# ==========================================

st.divider()
st.subheader("💡 Customer Retention Recommendations")

if repeat_rate < 30:
    st.warning(
        "Repeat purchase rate is low. "
        "Consider loyalty rewards and follow-up campaigns."
    )
elif repeat_rate < 60:
    st.info(
        "There is room to improve repeat purchases. "
        "Try personalized offers and customer engagement."
    )
else:
    st.success(
        "Repeat purchase rate is strong. "
        "Continue loyalty programs and customer engagement."
    )

st.markdown("""
**Suggested Business Actions:**

1. Offer discounts for repeat purchases.
2. Introduce customer loyalty reward points.
3. Send personalized product recommendations.
4. Contact customers who have not purchased recently.
5. Provide special offers to loyal customers.
""")

# ==========================================
# 10. DOWNLOAD CUSTOMER ANALYSIS
# ==========================================

st.divider()
st.subheader("📥 Download Customer Analysis")

csv_data = repeat_customer_details.to_csv(index=False)

st.download_button(
    label="Download Customer Retention Data",
    data=csv_data,
    file_name="customer_retention_analysis.csv",
    mime="text/csv"
)

st.success("Customer Retention Analysis Completed!")