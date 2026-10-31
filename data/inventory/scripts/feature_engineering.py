import pandas as pd

# Step 1: Load dataset
df = pd.read_csv(
    "data/retailpulse_combined.csv",
    low_memory=False
)

# Step 2: Display dataset information
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())
# Step 2: Create Sales column
df["Sales"] = df["Quantity"] * df["Price"]

# Display first 5 rows
print("\nSales Column:")
print(df[["Quantity", "Price", "Sales"]].head())

# Calculate total sales
print("\nTotal Sales:")
print(round(df["Sales"].sum(), 2))
import matplotlib.pyplot as plt

# Monthly Sales Graph
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Sales"] = df["Quantity"] * df["Price"]

df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

monthly_sales = df.groupby("Month")["Sales"].sum()

monthly_sales.plot(kind="line", marker="o", figsize=(12, 5))

plt.title("RetailPulse - Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()