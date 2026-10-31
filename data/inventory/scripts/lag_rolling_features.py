import pandas as pd

# Load dataset
df = pd.read_csv("data/retail_2009_2010.csv", low_memory=False)

# Convert invoice date
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

# Calculate sales
df["Sales"] = df["Quantity"] * df["Price"]

# Remove invalid rows
df = df.dropna(subset=["InvoiceDate", "Sales"])
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

# Calculate daily sales
daily_sales = (
    df.set_index("InvoiceDate")["Sales"]
    .resample("D")
    .sum()
    .reset_index()
)

daily_sales.columns = ["Date", "Sales"]

# Create lag and rolling features
daily_sales["Sales_Lag_1"] = daily_sales["Sales"].shift(1)
daily_sales["Sales_Lag_7"] = daily_sales["Sales"].shift(7)

daily_sales["Rolling_Avg_7"] = (
    daily_sales["Sales"].shift(1).rolling(7).mean()
)

# Display results
print("Lag and Rolling Features Created!")
print(daily_sales.tail(10))

# Save results
daily_sales.to_csv("lag_rolling_features.csv", index=False)

print("Saved: lag_rolling_features.csv")
