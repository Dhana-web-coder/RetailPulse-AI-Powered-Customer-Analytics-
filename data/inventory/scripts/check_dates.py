import pandas as pd

df = pd.read_csv(
    "retail_cleaned_final.csv",
    low_memory=False
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

print("First date:")
print(df["InvoiceDate"].min())

print("\nLast date:")
print(df["InvoiceDate"].max())

print("\nTotal rows:")
print(len(df))