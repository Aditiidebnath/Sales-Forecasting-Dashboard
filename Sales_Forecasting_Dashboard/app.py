
import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title(" Sales Forecasting & Demand Intelligence System")

st.write("Interactive Dashboard")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("train.csv")

# Convert Order Date
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Create Date Features
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month_name()

# -----------------------------
# Sidebar
# -----------------------------
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# ==========================================
# PAGE 1 : SALES OVERVIEW
# ==========================================

if page == "Sales Overview":

    st.header("Sales Overview Dashboard")

    yearly = pd.read_csv("yearly_sales.csv")
    monthly = pd.read_csv("monthly_sales.csv")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            yearly,
            x="Year",
            y="Sales",
            title="Total Sales by Year",
            color="Sales"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        monthly["Date"] = pd.to_datetime(monthly["Date"])

        fig2 = px.line(
            monthly,
            x="Date",
            y="Sales",
            markers=True,
            title="Monthly Sales Trend"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sales by Region and Category")

    region = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    category = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    filtered = df[
        (df["Region"] == region) &
        (df["Category"] == category)
    ]

    sales = (
        filtered.groupby("Sub-Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig3 = px.bar(
        sales,
        x="Sub-Category",
        y="Sales",
        color="Sales",
        title="Sales by Sub-Category"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # ==========================================
# PAGE 2 : FORECAST EXPLORER
# ==========================================

elif page == "Forecast Explorer":

    st.header(" Forecast Explorer")

    forecast = pd.read_csv("forecast_results.csv")
    forecast["Date"] = pd.to_datetime(forecast["Date"])

    option = st.selectbox(
        "Forecast Type",
        ["Category", "Region"]
    )

    if option == "Category":
        value = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique())
        )
    else:
        value = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique())
        )

    months = st.slider(
        "Forecast Horizon (Months Ahead)",
        min_value=1,
        max_value=3,
        value=3
    )

    forecast_display = forecast.head(months)

    fig = px.line(
        forecast_display,
        x="Date",
        y="Forecast",
        markers=True,
        title=f"{months} Month Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("MAE", "13915.32")

    with col2:
        st.metric("RMSE", "18893.85")

    st.success("Best Model Used: XGBoost")

    # ==========================================
# PAGE 3 : ANOMALY REPORT
# ==========================================

elif page == "Anomaly Report":

    st.header(" Sales Anomaly Report")

    anomalies = pd.read_csv("anomalies.csv")
    anomalies["Order Date"] = pd.to_datetime(anomalies["Order Date"])

    # Line chart showing anomaly points
    fig = px.line(
        anomalies,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Detected Sales Anomalies"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomalies")

    st.dataframe(anomalies)

    st.write(f"**Total Anomalies Detected:** {len(anomalies)}")

    # ==========================================
# PAGE 4 : PRODUCT DEMAND SEGMENTS
# ==========================================

elif page == "Product Demand Segments":

    st.header(" Product Demand Segments")

    cluster = pd.read_csv("cluster_results.csv")

    st.subheader("Demand Cluster Visualization")

    fig = px.scatter(
        cluster,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_name="Sub-Category",
        size="Total_Sales",
        title="Product Demand Clusters"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Details")

    st.dataframe(cluster)

    st.write("### Cluster Summary")

    summary = cluster.groupby("Cluster")["Sub-Category"].apply(list)

    for c in summary.index:
        st.write(f"**Cluster {c}:**")
        st.write(", ".join(summary[c]))
