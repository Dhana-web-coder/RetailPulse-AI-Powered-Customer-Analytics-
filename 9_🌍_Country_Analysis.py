import streamlit as st
import pandas as pd
from pathlib import Path

st.title("🌍 Country Analysis")
st.subheader("Retail Sales by Country")

BASE_DIR = Path(r"C:\Users\acer\Desktop\RERAILPULSE")
DATA_PATH = BASE_DIR / "retail_cleaned_final.csv"

if not DATA_PATH.exists():
    st.error("Dataset file not found!")
    st.stop()

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["TotalSales"] = df["Quantity"] * df["Price"]

df = df.dropna(subset=["Country", "Quantity", "Price"])
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

country_summary = (
    df.groupby("Country")
    .agg(
        TotalSales=("TotalSales", "sum"),
        TotalOrders=("Invoice", "nunique"),
        TotalQuantity=("Quantity", "sum"),
        TotalCustomers=("Customer ID", "nunique")
    )
    .reset_index()
    .sort_values("TotalSales", ascending=False)
)
# ==========================================
# COUNTRY-WISE SALES MAP
# ==========================================

import plotly.express as px

st.divider()
st.subheader("🌍 Country-wise Sales Map")

map_df = country_summary.copy()

map_df["Sales Label"] = map_df["TotalSales"].apply(
    lambda x: f"{x:,.2f}"
)

fig = px.choropleth(
    map_df,
    locations="Country",
    locationmode="country names",
    color="TotalSales",
    hover_name="Country",
    hover_data={
        "TotalSales": ":,.2f",
        "TotalOrders": True,
        "TotalCustomers": True
    },
    color_continuous_scale="Viridis",
    title="Country-wise Total Sales"
)

fig.update_traces(
    hovertemplate=
    "<b>Country: %{hovertext}</b><br>" +
    "Total Sales: %{z:,.2f}<br>" +
    "<extra></extra>"
)

fig.update_layout(
    height=550,
    coloraxis_colorbar_title="Sales Amount",
    margin=dict(l=0, r=0, t=50, b=0)
)

st.plotly_chart(fig, width="stretch")

# Country-wise Sales Amount
st.subheader("💰 Country-wise Sales Amount")

sales_display = map_df[
    ["Country", "TotalSales"]
].sort_values("TotalSales", ascending=False)

sales_display["TotalSales"] = sales_display["TotalSales"].map(
    lambda x: f"{x:,.2f}"
)

st.dataframe(sales_display, width="stretch")
c1, c2, c3 = st.columns(3)

c1.metric("Total Countries", country_summary["Country"].nunique())
c2.metric("Total Sales", f"{country_summary['TotalSales'].sum():,.2f}")
c3.metric("Total Orders", f"{country_summary['TotalOrders'].sum():,}")

st.divider()

st.subheader("🏆 Top 10 Countries by Sales")

top_countries = country_summary.head(10)

st.bar_chart(
    top_countries.set_index("Country")["TotalSales"],
    width="stretch"
)

st.subheader("📊 Country-wise Sales Details")

# ==========================================
# TOP 10 COUNTRIES BY ORDERS
# ==========================================

st.divider()

st.subheader("🏆 Top 10 Countries by Orders")

top_order_countries = (
    country_summary
    .sort_values("TotalOrders", ascending=False)
    .head(10)
)

st.bar_chart(
    top_order_countries.set_index("Country")["TotalOrders"],
    horizontal=True,
    width="stretch"
)

st.dataframe(
    top_order_countries[
        ["Country", "TotalOrders", "TotalSales"]
    ],
    width="stretch"
)

st.dataframe(country_summary, width="stretch")

# ==========================================
# MONTHLY SALES BY COUNTRY
# ==========================================

st.divider()

st.subheader("📅 Monthly Sales by Country")

monthly_df = df.copy()

monthly_df["InvoiceDate"] = pd.to_datetime(
    monthly_df["InvoiceDate"], errors="coerce"
)

monthly_df = monthly_df.dropna(subset=["InvoiceDate"])

monthly_df["Month"] = (
    monthly_df["InvoiceDate"].dt.to_period("M").astype(str)
)

monthly_sales = (
    monthly_df.groupby(["Month", "Country"])["TotalSales"]
    .sum()
    .reset_index()
)

available_countries = sorted(monthly_sales["Country"].unique())

selected_countries = st.multiselect(
    "Select Countries",
    available_countries,
    default=available_countries[:5]
)

filtered_monthly = monthly_sales[
    monthly_sales["Country"].isin(selected_countries)
]

if not filtered_monthly.empty:
    monthly_pivot = filtered_monthly.pivot(
        index="Month",
        columns="Country",
        values="TotalSales"
    ).fillna(0)

    st.line_chart(monthly_pivot, width="stretch")
else:
    st.warning("Please select at least one country.")

    # ==========================================
# AVERAGE ORDER VALUE BY COUNTRY
# ==========================================

st.divider()

st.subheader("💰 Average Order Value by Country")

# Calculate Average Order Value
aov_df = country_summary.copy()

aov_df["AverageOrderValue"] = (
    aov_df["TotalSales"] / aov_df["TotalOrders"]
)

# Sort highest average order value first
aov_df = aov_df.sort_values(
    "AverageOrderValue",
    ascending=False
)

# Show Top 10 Countries
top_aov = aov_df.head(10)

st.bar_chart(
    top_aov.set_index("Country")["AverageOrderValue"],
    horizontal=True,
    width="stretch"
)

# Display details
st.dataframe(
    top_aov[
        ["Country", "TotalSales", "TotalOrders", "AverageOrderValue"]
    ],
    width="stretch"
)

# ==========================================
# COUNTRY SALES CONTRIBUTION
# ==========================================

st.divider()

st.subheader("📊 Country Sales Contribution (%)")

contribution_df = country_summary.copy()

total_country_sales = contribution_df["TotalSales"].sum()

if total_country_sales > 0:

    contribution_df["SalesContribution"] = (
        contribution_df["TotalSales"]
        / total_country_sales
        * 100
    )

    contribution_df = contribution_df.sort_values(
        "SalesContribution",
        ascending=False
    )

    top_contribution = contribution_df.head(10)

    st.bar_chart(
        top_contribution.set_index("Country")[
            "SalesContribution"
        ],
        horizontal=True,
        width="stretch"
    )

    st.dataframe(
        top_contribution[
            ["Country", "TotalSales", "SalesContribution"]
        ],
        width="stretch"
    )

else:
    st.warning("Sales data is not available.")

    # ==========================================
# COUNTRY-WISE TOP 5 PRODUCTS
# ==========================================

st.divider()

st.subheader("🛍️ Country-wise Top 5 Products")

product_country_df = df.dropna(
    subset=["Country", "StockCode", "Description"]
).copy()

# Calculate product quantity and sales
product_country_summary = (
    product_country_df.groupby(
        ["Country", "StockCode", "Description"]
    )
    .agg(
        TotalQuantity=("Quantity", "sum"),
        TotalSales=("TotalSales", "sum")
    )
    .reset_index()
)

# Select a country
product_country_list = sorted(
    product_country_summary["Country"].unique()
)

selected_product_country = st.selectbox(
    "Select Country to View Top 5 Products",
    product_country_list,
    key="top5_product_country"
)

# Filter selected country
country_products = product_country_summary[
    product_country_summary["Country"]
    == selected_product_country
]

# Display Top 5 products
top_5_products = country_products.sort_values(
    "TotalQuantity",
    ascending=False
).head(5)

st.write(f"Top 5 Products in {selected_product_country}")

st.bar_chart(
    top_5_products.set_index("Description")["TotalQuantity"],
    horizontal=True,
    width="stretch"
)

st.dataframe(
    top_5_products[
        ["StockCode", "Description", "TotalQuantity", "TotalSales"]
    ],
    width="stretch"
)

# ==========================================
# COUNTRY-WISE AVERAGE QUANTITY
# ==========================================

st.divider()
st.subheader("📦 Country-wise Average Quantity")

quantity_summary = country_summary.copy()

quantity_summary["AverageQuantity"] = (
    quantity_summary["TotalQuantity"] /
    quantity_summary["TotalOrders"]
)

quantity_summary = quantity_summary.sort_values(
    "AverageQuantity",
    ascending=False
)

st.bar_chart(
    quantity_summary.head(10).set_index("Country")["AverageQuantity"],
    horizontal=True,
    width="stretch"
)

st.dataframe(
    quantity_summary[
        ["Country", "TotalQuantity", "TotalOrders", "AverageQuantity"]
    ],
    width="stretch"
)

# ==========================================
# COUNTRY-WISE SALES RANKING
# ==========================================

st.divider()
st.subheader("🏆 Country-wise Sales Ranking")

ranking_df = country_summary.copy()

ranking_df["SalesRank"] = ranking_df["TotalSales"].rank(
    method="dense",
    ascending=False
).astype(int)

ranking_df = ranking_df.sort_values("SalesRank")

st.dataframe(
    ranking_df[
        ["SalesRank", "Country", "TotalSales", "TotalOrders"]
    ],
    width="stretch"
)

top_country = ranking_df.iloc[0]

st.success(
    f"🏆 Top Country: {top_country['Country']} "
    f"| Sales: {top_country['TotalSales']:,.2f}"
)
# ==========================================
# COUNTRY-WISE SALES PERFORMANCE SUMMARY
# ==========================================

st.divider()
st.subheader("📊 Country-wise Sales Performance Summary")

performance_df = country_summary.copy()

performance_df["AverageOrderValue"] = (
    performance_df["TotalSales"] /
    performance_df["TotalOrders"]
)

performance_df["SalesContribution"] = (
    performance_df["TotalSales"] /
    performance_df["TotalSales"].sum()
) * 100

st.dataframe(
    performance_df[
        [
            "Country",
            "TotalSales",
            "TotalOrders",
            "TotalCustomers",
            "AverageOrderValue",
            "SalesContribution"
        ]
    ].sort_values("TotalSales", ascending=False),
    width="stretch"
)
# ==========================================
# COUNTRY-WISE CUSTOMER ANALYSIS
# ==========================================

st.divider()
st.subheader("👥 Country-wise Customer Analysis")

customer_country_df = country_summary.copy()

customer_country_df = customer_country_df.sort_values(
    "TotalCustomers",
    ascending=False
)

st.bar_chart(
    customer_country_df.head(10).set_index("Country")["TotalCustomers"],
    horizontal=True,
    width="stretch"
)

st.dataframe(
    customer_country_df[
        ["Country", "TotalCustomers", "TotalOrders", "TotalSales"]
    ],
    width="stretch"
)
st.subheader("🔎 Search Country")

selected_country = st.selectbox(
    "Select Country",
    country_summary["Country"].tolist()
)

country_data = country_summary[
    country_summary["Country"] == selected_country
]

st.dataframe(country_data, width="stretch")
# ==========================================
# TOP 10 COUNTRIES BY CUSTOMERS
# ==========================================

st.divider()

st.subheader("👥 Top 10 Countries by Customers")

top_customer_countries = (
    country_summary
    .sort_values("TotalCustomers", ascending=False)
    .head(10)
)

st.bar_chart(
    top_customer_countries.set_index("Country")["TotalCustomers"],
    horizontal=True,
    width="stretch"
)

st.dataframe(
    top_customer_countries[
        ["Country", "TotalCustomers", "TotalOrders", "TotalSales"]
    ],
    width="stretch"
)