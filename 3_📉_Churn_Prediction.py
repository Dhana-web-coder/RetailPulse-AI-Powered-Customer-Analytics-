import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction")
st.write("RetailPulse - Machine Learning Dashboard")

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

DATA_PATH = Path(r"C:\Users\acer\Desktop\RERAILPULSE\customer_churn_data.csv")

# Try the data folder if the file is not in project root
if not DATA_PATH.exists():
    DATA_PATH = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "customer_churn_data.csv"
    )

if not DATA_PATH.exists():
    st.error("customer_churn_data.csv file not found.")
    st.info(
        "Place customer_churn_data.csv in the project root "
        "or inside the data folder."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

required_columns = ["Recency", "Frequency", "Monetary", "Churn"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

df = df[required_columns].copy()

for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna()
df = df[
    (df["Recency"] >= 0)
    & (df["Frequency"] >= 0)
    & (df["Monetary"] >= 0)
    & (df["Churn"].isin([0, 1]))
]

df["Churn"] = df["Churn"].astype(int)

if df.empty or df["Churn"].nunique() != 2:
    st.error(
        "The dataset must contain valid customer records "
        "for both Churn = 0 and Churn = 1."
    )
    st.stop()

# --------------------------------------------------
# 3. DATASET OVERVIEW
# --------------------------------------------------

st.header("1. Customer Churn Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Churned Customers", f"{(df['Churn'] == 1).sum():,}")
col3.metric("Active Customers", f"{(df['Churn'] == 0).sum():,}")
col4.metric(
    "Churn Rate",
    f"{(df['Churn'].mean() * 100):.2f}%"
)

st.subheader("Churn Class Distribution")
st.write("0 = Not Churn | 1 = Churn")

class_counts = (
    df["Churn"]
    .value_counts()
    .sort_index()
    .rename_axis("Churn Status")
    .reset_index(name="Customer Count")
)

st.dataframe(class_counts, use_container_width=True)

fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x="Churn", ax=ax)
ax.set_title("Customer Churn Class Distribution")
ax.set_xlabel("Churn Status (0 = No, 1 = Yes)")
ax.set_ylabel("Number of Customers")
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# 4. TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

X = df[["Recency", "Frequency", "Monetary"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

churn_class_index = list(model.classes_).index(1)

y_probability = model.predict_proba(X_test)[
    :, churn_class_index
]

# --------------------------------------------------
# 5. MODEL PERFORMANCE
# --------------------------------------------------

st.header("2. Model Performance")

accuracy = accuracy_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)

col1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Training Customers", f"{len(X_train):,}")
col3.metric("Testing Customers", f"{len(X_test):,}")

st.info(
    "Accuracy shows the percentage of correct predictions. "
    "Precision and Recall are also important for churn prediction."
)

# --------------------------------------------------
# 6. CLASSIFICATION REPORT
# --------------------------------------------------

st.header("3. Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.subheader("Churn Class Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Churn Precision",
    f"{report.get('1', {}).get('precision', 0):.2f}"
)

col2.metric(
    "Churn Recall",
    f"{report.get('1', {}).get('recall', 0):.2f}"
)

col3.metric(
    "Churn F1-Score",
    f"{report.get('1', {}).get('f1-score', 0):.2f}"
)

st.dataframe(report_df.round(3), use_container_width=True)

# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------

st.header("4. Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cm_df = pd.DataFrame(
    cm,
    index=["Actual Not Churn", "Actual Churn"],
    columns=["Predicted Not Churn", "Predicted Churn"]
)

st.dataframe(cm_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Churn", "Churn"],
    yticklabels=["Not Churn", "Churn"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Customer Churn Confusion Matrix")

st.pyplot(fig)
plt.close(fig)

st.caption("Rows show actual classes; columns show predicted classes.")

# --------------------------------------------------
# 8. FEATURE IMPORTANCE
# --------------------------------------------------

st.header("5. Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(7, 4))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    ax=ax
)

ax.set_title("Factors Influencing Churn Prediction")
ax.set_xlabel("Feature Importance")
ax.set_ylabel("Feature")

st.pyplot(fig)
plt.close(fig)

st.dataframe(importance_df.round(4), use_container_width=True)

st.caption(
    "Feature importance shows how much each feature contributes "
    "to the Random Forest model. It does not prove causation."
)

# --------------------------------------------------
# 9. CUSTOMER CHURN PREDICTIONS
# --------------------------------------------------

st.header("6. Customer Churn Predictions")

results = X_test.copy()

results["Actual Churn"] = y_test
results["Predicted Churn"] = y_pred
results["Churn Probability (%)"] = (
    y_probability * 100
).round(2)

results["Churn Status"] = results["Predicted Churn"].map({
    0: "Not Churn",
    1: "Churn Risk"
})

def risk_level(probability):
    if probability >= 70:
        return "High Risk"
    elif probability >= 40:
        return "Medium Risk"
    return "Low Risk"

results["Risk Level"] = results["Churn Probability (%)"].apply(
    risk_level
)

st.dataframe(
    results.reset_index(drop=True),
    use_container_width=True
)

st.subheader("Churn Risk Level Distribution")

risk_counts = results["Risk Level"].value_counts()

st.bar_chart(risk_counts)

csv_data = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Churn Predictions CSV",
    data=csv_data,
    file_name="retailpulse_churn_predictions.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 10. PREDICT INDIVIDUAL CUSTOMER CHURN
# --------------------------------------------------

st.header("7. Predict Customer Churn")

st.write("Enter customer details to estimate churn risk.")

with st.form("churn_prediction_form"):

    recency = st.number_input(
        "Recency (Days Since Last Purchase)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency (Number of Purchases)",
        min_value=0,
        value=5
    )

    monetary = st.number_input(
        "Monetary (Total Purchase Amount)",
        min_value=0.0,
        value=1000.0
    )

    submit = st.form_submit_button("Predict Churn")

if submit:

    customer_data = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    prediction = model.predict(customer_data)[0]

    customer_probabilities = model.predict_proba(customer_data)[0]
    probability = customer_probabilities[churn_class_index] * 100

    st.metric("Churn Risk Probability", f"{probability:.2f}%")

    if probability >= 70:
        st.error("🔴 High Churn Risk")
    elif probability >= 40:
        st.warning("🟠 Medium Churn Risk")
    else:
        st.success("🟢 Low Churn Risk")

    if prediction == 1:
        st.write("Model prediction: Customer is at risk of churning.")
    else:
        st.write("Model prediction: Customer is predicted not to churn.")

# --------------------------------------------------
# 11. ROC CURVE & AUC SCORE
# --------------------------------------------------

st.header("8. ROC Curve & AUC Score")

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

auc_score = roc_auc_score(
    y_test,
    y_probability
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {auc_score:.3f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Prediction"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Customer Churn Prediction")
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

st.metric("AUC Score", f"{auc_score:.3f}")

st.caption(
    "AUC measures how well the model distinguishes churned "
    "customers from non-churned customers. Higher is generally better."
)

# --------------------------------------------------
# 12. DASHBOARD FOOTER
# --------------------------------------------------

st.success("Customer Churn Prediction Dashboard loaded successfully!")
# --------------------------------------------------
# 13. BUSINESS RECOMMENDATIONS
# --------------------------------------------------

st.header("9. Business Recommendations")

st.write(
    "Recommended actions to help reduce customer churn "
    "and improve customer retention."
)

if not results.empty:

    high_risk = (results["Risk Level"] == "High Risk").sum()
    medium_risk = (results["Risk Level"] == "Medium Risk").sum()
    low_risk = (results["Risk Level"] == "Low Risk").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("High-Risk Customers", int(high_risk))
    col2.metric("Medium-Risk Customers", int(medium_risk))
    col3.metric("Low-Risk Customers", int(low_risk))

    st.subheader("Recommended Actions")

    if high_risk > 0:
        st.error(
            "🔴 High Risk: Contact these customers personally, "
            "understand their concerns, and consider suitable "
            "retention offers."
        )

    if medium_risk > 0:
        st.warning(
            "🟠 Medium Risk: Send personalized offers, product "
            "recommendations, and follow-up messages."
        )

    if low_risk > 0:
        st.success(
            "🟢 Low Risk: Maintain engagement through loyalty "
            "rewards, regular updates, and quality service."
        )

    st.info(
        "Note: These recommendations are based on predicted churn "
        "risk. Review customer preferences and business costs before "
        "offering discounts."
    )

else:
    st.info("No customer prediction results are available.")
    # --------------------------------------------------
# 14. CUSTOMER RETENTION ANALYSIS
# --------------------------------------------------

st.header("10. Customer Retention Analysis")

st.write(
    "Analyze customer churn risk and identify opportunities "
    "to improve customer retention."
)

if not results.empty and "Risk Level" in results.columns:

    total_customers = len(results)

    high_risk = (results["Risk Level"] == "High Risk").sum()
    medium_risk = (results["Risk Level"] == "Medium Risk").sum()
    low_risk = (results["Risk Level"] == "Low Risk").sum()

    churn_risk_count = high_risk + medium_risk

    churn_risk_percentage = (
        churn_risk_count / total_customers * 100
        if total_customers > 0 else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers Analyzed",
        f"{total_customers:,}"
    )

    col2.metric(
        "Customers at Risk",
        f"{churn_risk_count:,}"
    )

    col3.metric(
        "Customers at Risk (%)",
        f"{churn_risk_percentage:.2f}%"
    )

    st.subheader("Customer Risk Distribution")

    retention_data = pd.DataFrame({
        "Risk Level": [
            "High Risk",
            "Medium Risk",
            "Low Risk"
        ],
        "Customers": [
            int(high_risk),
            int(medium_risk),
            int(low_risk)
        ]
    })

    st.bar_chart(
        retention_data.set_index("Risk Level")
    )

    st.subheader("Customer Retention Strategies")

    st.write(
        f"🔴 High Risk: Give priority to {high_risk:,} "
        "customers and investigate their reasons for inactivity."
    )

    st.write(
        f"🟠 Medium Risk: Engage {medium_risk:,} customers "
        "with personalized communication and relevant offers."
    )

    st.write(
        f"🟢 Low Risk: Continue building loyalty among "
        f"{low_risk:,} customers through good service."
    )

    st.info(
        "Note: These figures represent the customers in the "
        "model's prediction results, not necessarily all customers. "
        "Customers at risk are predicted to be high or medium risk; "
        "this is not a guarantee that they will leave."
    )

else:
    st.warning(
        "Customer prediction results are not available "
        "for retention analysis."
    )
    # --------------------------------------------------
# 15. REVENUE AT RISK ANALYSIS
# --------------------------------------------------

st.header("11. Revenue at Risk Analysis")

st.write(
    "Estimate the historical spending associated with "
    "customers who are predicted to have churn risk."
)

if (
    not results.empty
    and "Risk Level" in results.columns
    and "Monetary" in results.columns
):

    high_risk_data = results[
        results["Risk Level"] == "High Risk"
    ]

    medium_risk_data = results[
        results["Risk Level"] == "Medium Risk"
    ]

    high_risk_revenue = high_risk_data["Monetary"].sum()
    medium_risk_revenue = medium_risk_data["Monetary"].sum()

    total_revenue_at_risk = (
        high_risk_revenue + medium_risk_revenue
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "High-Risk Customer Spending",
        f"{high_risk_revenue:,.2f}"
    )

    col2.metric(
        "Medium-Risk Customer Spending",
        f"{medium_risk_revenue:,.2f}"
    )

    col3.metric(
        "Total Spending at Risk",
        f"{total_revenue_at_risk:,.2f}"
    )

    st.subheader("Spending by Risk Level")

    revenue_chart = pd.DataFrame({
        "Risk Level": [
            "High Risk",
            "Medium Risk"
        ],
        "Historical Spending": [
            high_risk_revenue,
            medium_risk_revenue
        ]
    })

    st.bar_chart(
        revenue_chart.set_index("Risk Level")
    )

    st.info(
        "Important: These amounts represent historical Monetary "
        "values for customers in the prediction results. They are "
        "not guaranteed future revenue losses. Confirm the meaning "
        "of the Monetary column and currency before business use."
    )

else:
    st.warning(
        "Revenue analysis requires prediction results containing "
        "both Risk Level and Monetary columns."
    )
    # --------------------------------------------------
# 16. CUSTOMER SEGMENTATION
# --------------------------------------------------

st.header("12. Customer Segmentation (RFM Analysis)")

rfm_file = Path(__file__).resolve().parent.parent / "rfm_results.csv"

if rfm_file.exists():

    rfm_df = pd.read_csv(rfm_file)

    st.write("Customer segments based on Recency, Frequency, and Monetary value.")

    if "Segment" in rfm_df.columns:

        segment_counts = (
            rfm_df["Segment"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = ["Customer Segment", "Customers"]

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Segmented Customers",
            f"{len(rfm_df):,}"
        )

        col2.metric(
            "Number of Segments",
            f"{rfm_df['Segment'].nunique()}"
        )

        st.subheader("Customer Segment Distribution")

        st.bar_chart(
            segment_counts.set_index("Customer Segment")
        )

        st.subheader("Customer Segment Details")

        st.dataframe(rfm_df, use_container_width=True)

        csv_data = rfm_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Customer Segmentation CSV",
            data=csv_data,
            file_name="retailpulse_customer_segments.csv",
            mime="text/csv"
        )

    else:
        st.warning("The RFM file does not contain a 'Segment' column.")

else:
    st.warning(f"RFM results file not found: {rfm_file}")