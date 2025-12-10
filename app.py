"""
NovaMart Marketing Analytics Dashboard
======================================

Masters of AI in Business - Data Visualization Assignment

This Streamlit app implements the full NovaMart marketing analytics dashboard:
- Campaign performance
- Customer insights
- Product performance
- Geographic analysis
- Attribution & funnel
- ML model evaluation

Place all CSVs inside a `data/` folder at the repo root and run:
    streamlit run app.py
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc


# --------------------------------------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="NovaMart Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------------

DATA_FILES: Dict[str, str] = {
    "campaigns": "campaign_performance.csv",
    "customers": "customer_data.csv",
    "products": "product_sales.csv",
    "leads": "lead_scoring_results.csv",
    "feature_importance": "feature_importance.csv",
    "learning_curve": "learning_curve.csv",
    "geographic": "geographic_data.csv",
    "attribution": "channel_attribution.csv",
    "funnel": "funnel_data.csv",
    "journey": "customer_journey.csv",
    "correlation": "correlation_matrix.csv",
}


@st.cache_data(show_spinner=True)
def load_data(data_dir: str = ".") -> Dict[str, pd.DataFrame]:
    """
    Load all CSVs into a dictionary of DataFrames.

    Parameters
    ----------
    data_dir : str
        Directory containing the data files (relative to this app).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of named DataFrames.
    """
    base_path = Path(data_dir)
    data: Dict[str, pd.DataFrame] = {}

    for key, filename in DATA_FILES.items():
        path = base_path / filename
        if not path.exists():
            st.error(f"Missing data file: {path}")
            st.stop()
        df = pd.read_csv(path)
        data[key] = df

    # Post-processing: campaign data
    campaigns = data["campaigns"].copy()
    campaigns["date"] = pd.to_datetime(campaigns["date"])
    # Ensure year is int
    if campaigns["year"].dtype != "int64":
        campaigns["year"] = campaigns["year"].astype(int)
    data["campaigns"] = campaigns

    # Post-processing: correlation matrix (use metric names as index)
    corr = data["correlation"].copy()
    if "Unnamed: 0" in corr.columns:
        corr = corr.rename(columns={"Unnamed: 0": "metric"})
        corr = corr.set_index("metric")
    data["correlation"] = corr

    return data


# --------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------


def format_currency(value: float) -> str:
    """Format as Indian Rupees in crores for dashboard KPIs."""
    if pd.isna(value):
        return "₹0.00 Cr"
    crores = value / 1e7  # 1 Cr = 10,000,000
    return f"₹{crores:,.2f} Cr"


def format_int(value: float) -> str:
    """Format large integers nicely with commas."""
    if pd.isna(value):
        return "0"
    return f"{int(round(value)):,}"


def safe_divide(numerator: float, denominator: float) -> float:
    """Avoid division by zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


# --------------------------------------------------------------------------------
# Page: Executive Overview
# --------------------------------------------------------------------------------


def page_executive_overview(data: Dict[str, pd.DataFrame]) -> None:
    st.title("NovaMart Marketing Analytics Dashboard")
    st.subheader("🏠 Executive Overview")

    campaigns = data["campaigns"]
    customers = data["customers"]

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_revenue = campaigns["revenue"].sum()
        st.metric("Total Revenue", format_currency(total_revenue))

    with col2:
        total_conversions = campaigns["conversions"].sum()
        st.metric("Total Conversions", format_int(total_conversions))

    with col3:
        total_customers = customers["customer_id"].nunique()
        st.metric("Total Customers", format_int(total_customers))

    with col4:
        total_spend = campaigns["spend"].sum()
        overall_roas = safe_divide(total_revenue, total_spend)
        st.metric("Overall ROAS", f"{overall_roas:,.2f}x")

    st.markdown("---")

    # Revenue trend (monthly) & Channel performance bar
    col_left, col_right = st.columns((2.5, 2))

    with col_left:
        st.markdown("### Revenue Trend Over Time (Monthly)")

        df = campaigns.copy()
        df = df.set_index("date").sort_index()
        monthly = df["revenue"].resample("M").sum().reset_index()
        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)

        fig = px.line(
            monthly,
            x="date",
            y="revenue",
            markers=True,
            labels={"date": "Month", "revenue": "Revenue"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Channel Performance (Revenue / Conversions / ROAS)")

        metric_label = st.selectbox(
            "Select metric", ["Revenue", "Conversions", "ROAS"], key="exec_channel_metric"
        )
        metric_map = {
            "Revenue": "revenue",
            "Conversions": "conversions",
            "ROAS": "roas",
        }
        metric_col = metric_map[metric_label]

        channel_agg = (
            campaigns.groupby("channel")
            .agg(
                spend=("spend", "sum"),
                revenue=("revenue", "sum"),
                conversions=("conversions", "sum"),
            )
            .reset_index()
        )
        # Compute channel-level ROAS if needed
        channel_agg["roas"] = channel_agg.apply(
            lambda row: safe_divide(row["revenue"], row["spend"]), axis=1
        )

        fig_bar = px.bar(
            channel_agg.sort_values(metric_col, ascending=True),
            x=metric_col,
            y="channel",
            orientation="h",
            labels={metric_col: metric_label, "channel": "Channel"},
        )
        fig_bar.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)


# --------------------------------------------------------------------------------
# Page: Campaign Analytics
# --------------------------------------------------------------------------------


def page_campaign_analytics(data: Dict[str, pd.DataFrame]) -> None:
    st.header("📈 Campaign Analytics")

    campaigns = data["campaigns"]

    # -----------------------------------------------------------------------------
    # 2.1 Line Chart - Revenue Trend Over Time (daily/weekly/monthly + channel)
    # -----------------------------------------------------------------------------
    st.subheader("Revenue Trend Over Time")

    c1, c2, c3 = st.columns(3)

    with c1:
        agg_level = st.selectbox(
            "Aggregation level", ["Daily", "Weekly", "Monthly"], key="rev_agg"
        )
    with c2:
        min_date = campaigns["date"].min().date()
        max_date = campaigns["date"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="rev_date",
        )
        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
    with c3:
        channels = sorted(campaigns["channel"].unique())
        selected_channels = st.multiselect(
            "Channels", channels, default=channels, key="rev_channels"
        )

    df = campaigns.copy()
    df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]
    if selected_channels:
        df = df[df["channel"].isin(selected_channels)]

    if agg_level == "Daily":
        df_grouped = (
            df.groupby(["date", "channel"], as_index=False)["revenue"].sum()
        )
        x_col = "date"
    elif agg_level == "Weekly":
        df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
        df_grouped = (
            df.groupby(["week", "channel"], as_index=False)["revenue"].sum()
        )
        x_col = "week"
    else:  # Monthly
        df["month_period"] = df["date"].dt.to_period("M").apply(lambda r: r.start_time)
        df_grouped = (
            df.groupby(["month_period", "channel"], as_index=False)["revenue"].sum()
        )
        x_col = "month_period"

    fig_rev = px.line(
        df_grouped,
        x=x_col,
        y="revenue",
        color="channel",
        labels={x_col: "Date", "revenue": "Revenue", "channel": "Channel"},
    )
    fig_rev.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 1.2 Grouped Bar - Regional Performance by Quarter (Year selector)
    # 1.3 Stacked Bar - Campaign Type Contribution (absolute vs 100%)
    # -----------------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Regional Performance by Quarter")

        years = sorted(campaigns["year"].unique())
        selected_year = st.selectbox("Year", years, index=len(years) - 1)

        df_year = campaigns[campaigns["year"] == selected_year]
        regional = (
            df_year.groupby(["region", "quarter"], as_index=False)["revenue"].sum()
        )

        fig_grouped = px.bar(
            regional,
            x="quarter",
            y="revenue",
            color="region",
            barmode="group",
            labels={"quarter": "Quarter", "revenue": "Revenue", "region": "Region"},
        )
        fig_grouped.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_grouped, use_container_width=True)

    with col_right:
        st.markdown("### Campaign Type Contribution (Monthly Spend)")

        mode = st.radio(
            "View",
            ["Absolute (₹)", "100% Stacked"],
            index=0,
            horizontal=True,
            key="campaign_type_mode",
        )

        df_ct = campaigns.copy()
        df_ct["year_month"] = df_ct["date"].dt.to_period("M").astype(str)

        monthly_spend = (
            df_ct.groupby(["year_month", "campaign_type"], as_index=False)["spend"].sum()
        )

        if mode == "Absolute (₹)":
            y_col = "spend"
            fig_stack = px.bar(
                monthly_spend,
                x="year_month",
                y=y_col,
                color="campaign_type",
                labels={
                    "year_month": "Year-Month",
                    "spend": "Spend",
                    "campaign_type": "Campaign Type",
                },
            )
        else:
            # Convert to percentages per month
            total_by_month = (
                monthly_spend.groupby("year_month")["spend"].transform("sum")
            )
            monthly_spend["spend_pct"] = (
                monthly_spend["spend"] / total_by_month * 100.0
            )
            y_col = "spend_pct"
            fig_stack = px.bar(
                monthly_spend,
                x="year_month",
                y=y_col,
                color="campaign_type",
                labels={
                    "year_month": "Year-Month",
                    "spend_pct": "Spend Share (%)",
                    "campaign_type": "Campaign Type",
                },
            )

        fig_stack.update_layout(barmode="stack", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 2.2 Stacked Area Chart - Cumulative Conversions by Channel (Region filter)
    # -----------------------------------------------------------------------------
    st.markdown("### Cumulative Conversions by Channel")

    regions = sorted(campaigns["region"].unique())
    selected_regions = st.multiselect(
        "Filter by region", regions, default=regions, key="cum_regions"
    )

    df_conv = campaigns.copy()
    if selected_regions:
        df_conv = df_conv[df_conv["region"].isin(selected_regions)]

    df_conv = (
        df_conv.groupby(["date", "channel"], as_index=False)["conversions"].sum()
        .sort_values(["channel", "date"])
    )

    # Cumulative per channel
    df_conv["cumulative_conversions"] = df_conv.groupby("channel")[
        "conversions"
    ].cumsum()

    fig_area = px.area(
        df_conv,
        x="date",
        y="cumulative_conversions",
        color="channel",
        labels={
            "date": "Date",
            "cumulative_conversions": "Cumulative Conversions",
            "channel": "Channel",
        },
    )
    fig_area.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 4.2 Bubble Chart - Channel Performance Matrix (CTR vs Conversion Rate)
    # -----------------------------------------------------------------------------
    st.markdown("### Channel Performance Matrix (CTR vs Conversion Rate)")

    channel_agg = (
        campaigns.groupby("channel")
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            spend=("spend", "sum"),
        )
        .reset_index()
    )
    channel_agg["ctr"] = channel_agg["clicks"] / channel_agg["impressions"] * 100.0
    channel_agg["conversion_rate"] = (
        channel_agg["conversions"] / channel_agg["clicks"]
    ) * 100.0

    fig_bubble = px.scatter(
        channel_agg,
        x="ctr",
        y="conversion_rate",
        size="spend",
        color="channel",
        hover_data=["impressions", "clicks", "conversions", "spend"],
        labels={
            "ctr": "CTR (%)",
            "conversion_rate": "Conversion Rate (%)",
            "spend": "Total Spend",
            "channel": "Channel",
        },
    )
    fig_bubble.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 4.4 Calendar Heatmap - Daily Performance (Revenue)
    # -----------------------------------------------------------------------------
    st.markdown("### Calendar Heatmap - Daily Revenue")

    metric_option = st.selectbox(
        "Metric", ["Revenue", "Spend", "Conversions"], key="cal_metric"
    )
    metric_col_map = {
        "Revenue": "revenue",
        "Spend": "spend",
        "Conversions": "conversions",
    }
    metric_col = metric_col_map[metric_option]

    years = sorted(campaigns["year"].unique())
    selected_year = st.selectbox(
        "Calendar year", years, index=len(years) - 1, key="cal_year"
    )

    df_cal = campaigns[campaigns["year"] == selected_year].copy()
    df_cal["date"] = pd.to_datetime(df_cal["date"])
    daily = df_cal.groupby("date", as_index=False)[metric_col].sum()
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["weekday"] = daily["date"].dt.weekday  # 0=Mon, 6=Sun

    pivot = daily.pivot_table(
        index="weekday", columns="week", values=metric_col, aggfunc="sum"
    )

    # Ensure all weekdays present
    pivot = pivot.reindex(index=range(7))

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig_cal = px.imshow(
        pivot,
        aspect="auto",
        origin="lower",
        labels=dict(x="Week of Year", y="Day of Week", color=metric_option),
        x=pivot.columns,
        y=day_labels,
    )
    fig_cal.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_cal, use_container_width=True)


# --------------------------------------------------------------------------------
# Page: Customer Insights
# --------------------------------------------------------------------------------


def page_customer_insights(data: Dict[str, pd.DataFrame]) -> None:
    st.header("👥 Customer Insights")

    customers = data["customers"]

    # -----------------------------------------------------------------------------
    # 3.1 Histogram - Customer Age Distribution
    # -----------------------------------------------------------------------------
    st.subheader("Age Distribution")

    col1, col2 = st.columns([2, 1])

    with col1:
        segments = sorted(customers["customer_segment"].unique())
        selected_segments = st.multiselect(
            "Filter by customer segment",
            segments,
            default=segments,
            key="age_segments",
        )

        df_age = customers.copy()
        if selected_segments:
            df_age = df_age[df_age["customer_segment"].isin(selected_segments)]

        min_age = int(df_age["age"].min())
        max_age = int(df_age["age"].max())
        default_bins = 10
        bins = st.slider(
            "Number of bins", min_value=5, max_value=30, value=default_bins, key="age_bins"
        )

        fig_age = px.histogram(
            df_age,
            x="age",
            nbins=bins,
            color="customer_segment",
            barmode="overlay",
            opacity=0.7,
            labels={"age": "Age", "customer_segment": "Segment"},
        )
        fig_age.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        st.markdown(
            """
            **Insights to look for:**
            - Core age band (e.g., 25–40)
            - How different segments skew younger/older
            """
        )

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 3.2 Box Plot - Lifetime Value by Segment
    # 3.3 Violin Plot - Satisfaction by NPS Category & Acquisition Channel
    # -----------------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Lifetime Value by Customer Segment")

        show_points = st.checkbox(
            "Overlay individual customers", value=False, key="ltv_points"
        )

        fig_ltv = px.box(
            customers,
            x="customer_segment",
            y="lifetime_value",
            points="all" if show_points else False,
            labels={"customer_segment": "Customer Segment", "lifetime_value": "LTV"},
        )
        fig_ltv.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_ltv, use_container_width=True)

    with col_right:
        st.markdown("### Satisfaction Score by NPS Category & Acquisition Channel")

        acq_channels = sorted(customers["acquisition_channel"].unique())
        selected_channels = st.multiselect(
            "Acquisition channels",
            acq_channels,
            default=acq_channels,
            key="nps_channels",
        )

        df_sat = customers.copy()
        if selected_channels:
            df_sat = df_sat[df_sat["acquisition_channel"].isin(selected_channels)]

        fig_violin = px.violin(
            df_sat,
            x="nps_category",
            y="satisfaction_score",
            color="acquisition_channel",
            box=True,
            points=False,
            labels={
                "nps_category": "NPS Category",
                "satisfaction_score": "Satisfaction Score",
                "acquisition_channel": "Acquisition Channel",
            },
        )
        fig_violin.update_layout(legend_title_text="Acquisition Channel", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 4.1 Scatter Plot - Income vs Lifetime Value (trendline toggle)
    # -----------------------------------------------------------------------------
    st.markdown("### Income vs Lifetime Value")

    show_trend = st.checkbox("Show overall trend line", value=True, key="trendline")

    fig_scatter = px.scatter(
        customers,
        x="income",
        y="lifetime_value",
        color="customer_segment",
        hover_data=["age_group", "income_bracket"],
        labels={
            "income": "Income",
            "lifetime_value": "Lifetime Value",
            "customer_segment": "Segment",
        },
    )

    if show_trend and len(customers) > 1:
        x_vals = customers["income"].values
        y_vals = customers["lifetime_value"].values
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = m * x_line + b
        fig_scatter.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Trend line",
            )
        )

    fig_scatter.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 5.3 Sunburst Chart - Customer Segmentation (Region -> City Tier -> Segment)
    # -----------------------------------------------------------------------------
    st.markdown("### Customer Segmentation Hierarchy")

    seg = (
        customers.groupby(["region", "city_tier", "customer_segment"])
        .size()
        .reset_index(name="count")
    )

    fig_sunburst = px.sunburst(
        seg,
        path=["region", "city_tier", "customer_segment"],
        values="count",
        labels={
            "region": "Region",
            "city_tier": "City Tier",
            "customer_segment": "Segment",
            "count": "Customer Count",
        },
    )
    fig_sunburst.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_sunburst, use_container_width=True)


# --------------------------------------------------------------------------------
# Page: Product Performance
# --------------------------------------------------------------------------------


def page_product_performance(data: Dict[str, pd.DataFrame]) -> None:
    st.header("📦 Product Performance")

    products = data["products"]

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        regions = sorted(products["region"].unique())
        selected_regions = st.multiselect(
            "Regions", regions, default=regions, key="prod_regions"
        )
    with col2:
        years = sorted(products["year"].unique())
        selected_years = st.multiselect(
            "Years", years, default=years, key="prod_years"
        )

    df_prod = products.copy()
    if selected_regions:
        df_prod = df_prod[df_prod["region"].isin(selected_regions)]
    if selected_years:
        df_prod = df_prod[df_prod["year"].isin(selected_years)]

    # -----------------------------------------------------------------------------
    # 5.2 Treemap - Product Sales Hierarchy (Category > Subcategory > Product)
    # -----------------------------------------------------------------------------
    st.subheader("Product Sales Hierarchy")

    fig_tree = px.treemap(
        df_prod,
        path=["category", "subcategory", "product_name"],
        values="sales",
        color="profit_margin",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=df_prod["profit_margin"].median(),
        labels={
            "category": "Category",
            "subcategory": "Subcategory",
            "product_name": "Product",
            "sales": "Sales",
            "profit_margin": "Profit Margin",
        },
    )
    fig_tree.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # Category comparison & regional product performance
    # -----------------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Sales by Category")

        cat_agg = (
            df_prod.groupby("category", as_index=False)[["sales", "profit"]].sum()
        )

        fig_cat = px.bar(
            cat_agg,
            x="category",
            y="sales",
            text="sales",
            labels={"category": "Category", "sales": "Sales"},
        )
        fig_cat.update_traces(texttemplate="%{text:.2s}", textposition="outside")
        fig_cat.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_right:
        st.markdown("### Regional Product Performance (Sales by Region & Category)")

        reg_cat = (
            df_prod.groupby(["region", "category"], as_index=False)["sales"].sum()
        )

        fig_reg_cat = px.bar(
            reg_cat,
            x="region",
            y="sales",
            color="category",
            barmode="group",
            labels={
                "region": "Region",
                "sales": "Sales",
                "category": "Category",
            },
        )
        fig_reg_cat.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_reg_cat, use_container_width=True)


# --------------------------------------------------------------------------------
# Page: Geographic Analysis
# --------------------------------------------------------------------------------


def page_geographic_analysis(data: Dict[str, pd.DataFrame]) -> None:
    st.header("🗺️ Geographic Analysis")

    geo = data["geographic"]

    # -----------------------------------------------------------------------------
    # 6.1 Map - State-wise Performance (metric toggle)
    # -----------------------------------------------------------------------------
    st.subheader("State-wise Performance Map")

    metric_option = st.selectbox(
        "Select metric",
        ["total_revenue", "total_customers", "market_penetration", "yoy_growth"],
        format_func=lambda x: {
            "total_revenue": "Total Revenue",
            "total_customers": "Total Customers",
            "market_penetration": "Market Penetration (%)",
            "yoy_growth": "YoY Growth (%)",
        }[x],
        key="geo_metric",
    )

    fig_geo1 = px.scatter_geo(
        geo,
        lat="latitude",
        lon="longitude",
        color=metric_option,
        hover_name="state",
        size=None,
        projection="natural earth",
        labels={metric_option: metric_option.replace("_", " ").title()},
    )
    fig_geo1.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_geo1, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 6.2 Bubble Map - Store Performance (bubble size & satisfaction color)
    # -----------------------------------------------------------------------------
    st.subheader("Store Performance Bubble Map")

    fig_geo2 = px.scatter_geo(
        geo,
        lat="latitude",
        lon="longitude",
        size="store_count",
        color="customer_satisfaction",
        hover_name="state",
        hover_data=["region", "total_customers", "total_revenue"],
        projection="natural earth",
        labels={
            "store_count": "Store Count",
            "customer_satisfaction": "Customer Satisfaction",
        },
    )
    fig_geo2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_geo2, use_container_width=True)


# --------------------------------------------------------------------------------
# Page: Attribution & Funnel
# --------------------------------------------------------------------------------


def page_attribution_funnel(data: Dict[str, pd.DataFrame]) -> None:
    st.header("🎯 Attribution & Funnel")

    attribution = data["attribution"]
    funnel = data["funnel"]
    corr = data["correlation"]
    journey = data["journey"]

    col1, col2 = st.columns(2)

    # -----------------------------------------------------------------------------
    # 5.1 Donut Chart - Attribution Model Comparison
    # -----------------------------------------------------------------------------
    with col1:
        st.subheader("Channel Attribution Models")

        model_options = [
            "first_touch",
            "last_touch",
            "linear",
            "time_decay",
            "position_based",
        ]
        selected_model = st.selectbox(
            "Attribution model", model_options, key="attrib_model"
        )

        fig_donut = px.pie(
            attribution,
            names="channel",
            values=selected_model,
            hole=0.4,
            labels={"channel": "Channel"},
        )
        fig_donut.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    # -----------------------------------------------------------------------------
    # 5.4 Funnel Chart - Marketing Funnel
    # -----------------------------------------------------------------------------
    with col2:
        st.subheader("Marketing Funnel")

        fig_funnel = px.funnel(
            funnel,
            y="stage",
            x="visitors",
            labels={"stage": "Funnel Stage", "visitors": "Visitors"},
        )
        fig_funnel.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 4.3 Correlation Heatmap - Marketing Metrics
    # -----------------------------------------------------------------------------
    st.subheader("Correlation Matrix of Marketing Metrics")

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation"),
    )
    fig_corr.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # Bonus: Sankey Diagram - Customer Journey Paths
    # -----------------------------------------------------------------------------
    st.subheader("Customer Journey Paths (Sankey Diagram)")

    # Build Sankey from touchpoint_1 -> touchpoint_2 -> touchpoint_3 -> touchpoint_4
    df_j = journey.copy()

    # Get unique labels
    labels = pd.unique(
        pd.concat(
            [
                df_j["touchpoint_1"],
                df_j["touchpoint_2"],
                df_j["touchpoint_3"],
                df_j["touchpoint_4"],
            ]
        )
    ).tolist()
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Build source, target, value lists for each step
    edges = []
    for step_from, step_to in [
        ("touchpoint_1", "touchpoint_2"),
        ("touchpoint_2", "touchpoint_3"),
        ("touchpoint_3", "touchpoint_4"),
    ]:
        df_step = (
            df_j.groupby([step_from, step_to], as_index=False)["customer_count"].sum()
        )
        for _, row in df_step.iterrows():
            edges.append(
                (
                    label_to_idx[row[step_from]],
                    label_to_idx[row[step_to]],
                    int(row["customer_count"]),
                )
            )

    if edges:
        source, target, value = zip(*edges)
        sankey_fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=15,
                        line=dict(color="black", width=0.5),
                        label=labels,
                    ),
                    link=dict(
                        source=list(source),
                        target=list(target),
                        value=list(value),
                    ),
                )
            ]
        )
        sankey_fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(sankey_fig, use_container_width=True)
    else:
        st.info("Insufficient journey data to build Sankey diagram.")


# --------------------------------------------------------------------------------
# Page: ML Model Evaluation
# --------------------------------------------------------------------------------


def page_ml_evaluation(data: Dict[str, pd.DataFrame]) -> None:
    st.header("🤖 ML Model Evaluation - Lead Scoring")

    leads = data["leads"]
    feat_imp = data["feature_importance"]
    lc = data["learning_curve"]

    # -----------------------------------------------------------------------------
    # 7.1 Confusion Matrix - with threshold slider
    # -----------------------------------------------------------------------------
    st.subheader("Confusion Matrix (Conversion vs. Predicted Class)")

    col1, col2 = st.columns([1, 3])

    with col1:
        threshold = st.slider(
            "Probability threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key="threshold",
        )

        y_true = leads["actual_converted"].values
        y_scores = leads["predicted_probability"].values
        y_pred = (y_scores >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        st.markdown("**Counts**")
        st.write(f"TP: {tp}  |  FP: {fp}  |  TN: {tn}  |  FN: {fn}")

        st.markdown("**Rates**")
        tpr = safe_divide(tp, tp + fn)  # recall
        fpr = safe_divide(fp, fp + tn)
        st.write(f"TPR (Recall): {tpr:.2%}")
        st.write(f"FPR: {fpr:.2%}")

    with col2:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 7.2 ROC Curve
    # -----------------------------------------------------------------------------
    st.subheader("ROC Curve")

    y_true = leads["actual_converted"].values
    y_scores = leads["predicted_probability"].values
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {roc_auc:.2f})",
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(dash="dash"),
        )
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 7.3 Learning Curve
    # -----------------------------------------------------------------------------
    st.subheader("Learning Curve")

    show_bands = st.checkbox(
        "Show confidence bands (± std)", value=True, key="lc_bands"
    )

    fig_lc = go.Figure()
    fig_lc.add_trace(
        go.Scatter(
            x=lc["training_size"],
            y=lc["train_score"],
            mode="lines+markers",
            name="Train Score",
        )
    )
    fig_lc.add_trace(
        go.Scatter(
            x=lc["training_size"],
            y=lc["validation_score"],
            mode="lines+markers",
            name="Validation Score",
        )
    )

    if show_bands:
        # Train band
        fig_lc.add_trace(
            go.Scatter(
                x=list(lc["training_size"]) + list(lc["training_size"][::-1]),
                y=list(
                    lc["train_score"] + lc["train_score_std"]
                )
                + list(
                    (lc["train_score"] - lc["train_score_std"])[::-1]
                ),
                fill="toself",
                opacity=0.2,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            )
        )
        # Validation band
        fig_lc.add_trace(
            go.Scatter(
                x=list(lc["training_size"]) + list(lc["training_size"][::-1]),
                y=list(
                    lc["validation_score"] + lc["validation_score_std"]
                )
                + list(
                    (lc["validation_score"] - lc["validation_score_std"])[::-1]
                ),
                fill="toself",
                opacity=0.2,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            )
        )

    fig_lc.update_layout(
        xaxis_title="Training Size",
        yaxis_title="Score",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_lc, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 7.4 Feature Importance
    # -----------------------------------------------------------------------------
    st.subheader("Feature Importance")

    sort_order = st.radio(
        "Sort by",
        ["Descending", "Ascending"],
        index=0,
        horizontal=True,
        key="fi_sort",
    )
    show_error = st.checkbox(
        "Show error bars (± std)", value=True, key="fi_error"
    )

    fi = feat_imp.copy()
    fi = fi.sort_values("importance", ascending=(sort_order == "Ascending"))

    fig_fi = go.Figure()

    fig_fi.add_trace(
        go.Bar(
            x=fi["importance"],
            y=fi["feature"],
            orientation="h",
            name="Importance",
            error_x=dict(
                type="data",
                array=fi["importance_std"] if show_error else [0] * len(fi),
                visible=show_error,
            ),
        )
    )
    fig_fi.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# --------------------------------------------------------------------------------
# Sidebar Navigation & Main
# --------------------------------------------------------------------------------


def sidebar() -> str:
    st.sidebar.title("NovaMart Dashboard")
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Executive Overview",
            "📈 Campaign Analytics",
            "👥 Customer Insights",
            "📦 Product Performance",
            "🗺️ Geographic Analysis",
            "🎯 Attribution & Funnel",
            "🤖 ML Model Evaluation",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Masters of AI in Business · Data Visualization Assignment")
    return page


def main() -> None:
    data = load_data(".")
    page = sidebar()

    if page == "🏠 Executive Overview":
        page_executive_overview(data)
    elif page == "📈 Campaign Analytics":
        page_campaign_analytics(data)
    elif page == "👥 Customer Insights":
        page_customer_insights(data)
    elif page == "📦 Product Performance":
        page_product_performance(data)
    elif page == "🗺️ Geographic Analysis":
        page_geographic_analysis(data)
    elif page == "🎯 Attribution & Funnel":
        page_attribution_funnel(data)
    elif page == "🤖 ML Model Evaluation":
        page_ml_evaluation(data)


if __name__ == "__main__":
    main()
