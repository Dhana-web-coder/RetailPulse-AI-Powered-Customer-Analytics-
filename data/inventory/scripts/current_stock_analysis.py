import pandas as pd
# Step 1: Load reorder point data
df = pd.read_csv("inventory_reorder_points.csv")
# Step 2: Create sample stock values
# These are examples, not actual warehouse stock
df["Current_Stock"] = 100
# Step 3: Check stock status
df["Stock_Status"] = df.apply(
    lambda row: (
        "Low Stock"
        if row["Current_Stock"] <= row["Reorder_Point"]
        else "Sufficient Stock"
    ),
    axis=1
)
# Step 4: Display results
print("\nCurrent Stock Level Analysis")
print(
    df[
        [
            "StockCode",
            "Description",
            "Reorder_Point",
            "Current_Stock",
            "Stock_Status"
        ]
    ].head(10).to_string(index=False)
)
# Step 5: Save results
df.to_csv("current_stock_analysis.csv", index=False)
print("\nCurrent stock analysis saved successfully!")
