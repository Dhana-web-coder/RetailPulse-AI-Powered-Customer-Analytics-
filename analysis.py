import pandas as pd
from pathlib import Path

#Project folder location

BASE_DIR = Path(__file__).resolve().parent

#Load datasets from the main project folder

file_2009 = BASE_DIR / "data" / "retail_2009_2010.csv"
file_2010 = BASE_DIR / "retail_2010_2011.csv"

df_2009 = pd.read_csv(file_2009, low_memory=False)
df_2010 = pd.read_csv(file_2010, low_memory=False)

#Clean column names

df_2009.columns = df_2009.columns.str.strip()
df_2010.columns = df_2010.columns.str.strip()

#Check dataset dates

df_2009["InvoiceDate"] = pd.to_datetime(
df_2009["InvoiceDate"], errors="coerce"
)

df_2010["InvoiceDate"] = pd.to_datetime(
df_2010["InvoiceDate"], errors="coerce"
)

print("2009–2010 Dataset Shape:", df_2009.shape)
print("2009–2010 Start Date:", df_2009["InvoiceDate"].min())
print("2009–2010 End Date:", df_2009["InvoiceDate"].max())

print("\n2010–2011 Dataset Shape:", df_2010.shape)
print("2010–2011 Start Date:", df_2010["InvoiceDate"].min())
print("2010–2011 End Date:", df_2010["InvoiceDate"].max())

#Combine datasets

df = pd.concat([df_2009, df_2010], ignore_index=True)

print("\nCombined Dataset Shape:", df.shape)

#Convert numeric columns

df["Quantity"] = pd.to_numeric(
df["Quantity"], errors="coerce"
)

df["Price"] = pd.to_numeric(
df["Price"], errors="coerce"
)

#Remove invalid records

df = df.dropna(
subset=["InvoiceDate", "Quantity", "Price"]
)

#Calculate sales

df["Sales"] = df["Quantity"] * df["Price"]

#Create monthly sales data

df["Month"] = df["InvoiceDate"].dt.to_period("M")

monthly_sales = (
df.groupby("Month")["Sales"]
.sum()
.sort_index()
)

monthly_sales.index = monthly_sales.index.to_timestamp()
monthly_sales = monthly_sales.asfreq("MS", fill_value=0)

#Feature Engineering: Lag and Rolling Average

features = pd.DataFrame(index=monthly_sales.index)

features["Monthly_Sales"] = monthly_sales
features["Lag_1"] = monthly_sales.shift(1)
features["Lag_2"] = monthly_sales.shift(2)
features["Lag_3"] = monthly_sales.shift(3)

features["Rolling_Avg_3"] = (
monthly_sales.shift(1).rolling(window=3).mean()
)

features["Rolling_Avg_6"] = (
monthly_sales.shift(1).rolling(window=6).mean()
)

#Save output inside data folder

output_folder = BASE_DIR / "data"
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "retailpulse_features.csv"
features.to_csv(output_file, index_label="Month")

print("\nFeature Engineering Completed Successfully!")
print("Output saved to:", output_file)
print("\nFeature Preview:")
print(features.tail(10))