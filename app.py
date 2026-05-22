import os

# Must be set before keras is imported (triggered on first joblib.load of a model)
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_utils import (
    DATASET_META,
    SENSOR_COLS,
    available_datasets,
    load_all_datasets,
    dataset_overview,
    lifetime_summary,
    sensor_distribution_long,
    sensor_variance_ranking,
    lifecycle_sensor_profile,
    create_latest_test_fleet,
    baseline_predict_rul,
    load_cached_predictions,
    load_model_dict,
    predict_rul_with_model,
    cost_analysis_summary,
)

st.set_page_config(
    page_title="NASA C-MAPSS Dashboard",
    page_icon="✈️",
    layout="wide",
)

DATA_DIR = "data"
PREDICTION_DIR = "predictions"
MODEL_DIR = "models"
MODEL_NAMES = ["Conv1D+LSTM", "AE+LSTM", "Pure LSTM"]

st.markdown(
    """
    <style>
    .main {background-color: #f3f6f8;}
    .block-container {padding-top: 1.4rem;}
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e6e9ef;
        padding: 14px 16px;
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .status-card {
        background-color: white;
        border: 2px solid #ffa600;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    .status-title {font-size: 16px; font-weight: 700; color: #182c3a;}
    .status-value {font-size: 22px; font-weight: 800; color: #182c3a; margin-top: 4px;}
    .status-label {font-size: 13px; font-weight: 800; color: #ffa600; margin-top: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)

found_datasets = available_datasets(DATA_DIR)
if not found_datasets:
    st.error("Dataset files were not found in the data folder.")
    st.stop()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Filters")
page = st.sidebar.radio(
    "Dashboard",
    ["Overview", "Maintenance", "Cost Analysis"],
    index=0,
)
selected_dataset = st.sidebar.selectbox("Dataset", found_datasets, index=0)

if page in ("Maintenance", "Cost Analysis"):
    selected_model_name = st.sidebar.selectbox("Prediction Model", MODEL_NAMES, index=0)
else:
    selected_model_name = MODEL_NAMES[0]

train_data, test_data, rul_data = load_all_datasets(DATA_DIR, found_datasets)
train = train_data[selected_dataset]
test = test_data[selected_dataset]
rul = rul_data[selected_dataset]
overview_df = dataset_overview(train_data, test_data)

# -----------------------------
# Helper charts
# -----------------------------
def clean_layout(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="#eef3f8",
        margin=dict(l=30, r=30, t=55, b=30),
        font=dict(size=12),
    )
    return fig


def bar_chart(df, x, y, title):
    fig = px.bar(df, x=x, y=y, title=title, text_auto=True)
    return clean_layout(fig)


def line_chart(df, x, y, title):
    fig = px.line(df, x=x, y=y, title=title)
    return clean_layout(fig)


def box_chart(df, x, y, title):
    fig = px.box(df, x=x, y=y, title=title)
    return clean_layout(fig, height=520)


def hist_chart(df, x, title, nbins=40):
    fig = px.histogram(df, x=x, nbins=nbins, title=title)
    return clean_layout(fig)


def lifetime_distribution_chart(lifetime_df):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Engine Lifetimes", "Lifetime Distribution"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Histogram(x=lifetime_df["Lifetime"], nbinsx=25, name="Lifetime", showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Box(y=lifetime_df["Lifetime"], name="", boxpoints="outliers", showlegend=False),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Lifetime (cycles)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Cycles", row=1, col=2)
    fig.update_layout(title="Engine Lifetime Distribution")
    return clean_layout(fig, height=470)


def multi_sensor_lifecycle_chart(train, sensors):
    rows = int(np.ceil(len(sensors) / 2))
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=sensors,
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
    )
    for idx, sensor in enumerate(sensors):
        profile_df = lifecycle_sensor_profile(train, sensor)
        r = idx // 2 + 1
        c = idx % 2 + 1
        fig.add_trace(
            go.Scatter(
                x=profile_df["Lifecycle Percentage"],
                y=profile_df[sensor],
                mode="lines",
                name=sensor,
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="Lifecycle (%)", row=r, col=c)
        fig.update_yaxes(title_text="Value", row=r, col=c)
    fig.update_layout(title="Sensor Behavior Across Engine Lifecycles")
    return clean_layout(fig, height=max(360, rows * 280))


@st.cache_resource
def _load_model_dict_cached(dataset_id: str):
    return load_model_dict(MODEL_DIR, dataset_id)


def get_fleet_predictions(dataset_id: str, model_name: str):
    model_dict = _load_model_dict_cached(dataset_id)
    if model_dict is not None and model_name in model_dict:
        try:
            return predict_rul_with_model(
                test_data[dataset_id],
                rul_data[dataset_id],
                train_data[dataset_id],
                model_dict[model_name],
                dataset_id,
            )
        except Exception as e:
            st.warning(f"Model prediction failed ({e}). Falling back to baseline.")
    cached = load_cached_predictions(PREDICTION_DIR, dataset_id)
    if cached is not None:
        return cached
    fleet = create_latest_test_fleet(test_data[dataset_id], rul_data[dataset_id], dataset_id)
    return baseline_predict_rul(fleet, train_data[dataset_id])


def status_from_rul(rul_value, critical_threshold=30, warning_threshold=80):
    if rul_value <= critical_threshold:
        return "CRITICAL"
    if rul_value <= warning_threshold:
        return "WARNING"
    return "HEALTHY"


def status_color(status):
    return {"CRITICAL": "#ff1f1f", "WARNING": "#ffa600", "HEALTHY": "#128a2e"}.get(status, "#ffa600")

# -----------------------------
# Page: Overview
# -----------------------------
if page == "Overview":
    st.title("NASA C-MAPSS Turbofan Engine Overview")
    current_row = overview_df[overview_df["Dataset"] == selected_dataset].iloc[0]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Dataset", current_row["Dataset"])
    c2.metric("Engines", int(current_row["Train Engines"]))
    c3.metric("Total Cycles", int(current_row["Total Cycles"]))
    c4.metric("Average Cycle", current_row["Average Cycle"])
    c5.metric("Operating Condition", DATASET_META[selected_dataset]["operating_condition"].split(" ")[0])
    c6.metric("Fault Mode", DATASET_META[selected_dataset]["fault_mode"].split(" ")[0])

    st.subheader("Dataset Description")
    st.dataframe(overview_df, use_container_width=True)

    st.subheader("Engine Lifetime Summary Statistics")
    st.dataframe(lifetime_summary(train), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(bar_chart(overview_df, "Dataset", "Total Cycles", "Total Cycles by Dataset"), use_container_width=True)
    with col2:
        st.plotly_chart(bar_chart(overview_df, "Dataset", "Average Lifetime", "Average Lifetime by Dataset"), use_container_width=True)

    st.plotly_chart(bar_chart(overview_df, "Dataset", "Train Engines", "Number of Engines by Dataset"), use_container_width=True)

    distribution_df = sensor_distribution_long(train)
    st.plotly_chart(box_chart(distribution_df, "Sensor", "Normalized Value", "Normalized Sensor Distribution"), use_container_width=True)

    variance_df = sensor_variance_ranking(train)
    st.plotly_chart(bar_chart(variance_df, "Sensor", "Variance", "Sensor Variance Ranking"), use_container_width=True)

    lifetime_df = train.groupby("unit_id", as_index=False)["time_cycles"].max()
    lifetime_df.rename(columns={"time_cycles": "Lifetime"}, inplace=True)
    st.plotly_chart(lifetime_distribution_chart(lifetime_df), use_container_width=True)

    selected_sensors = st.multiselect(
        "Sensors for Lifecycle Behavior",
        SENSOR_COLS,
        default=SENSOR_COLS[:6],
        max_selections=6,
    )
    if selected_sensors:
        st.plotly_chart(multi_sensor_lifecycle_chart(train, selected_sensors), use_container_width=True)

# -----------------------------
# Page: Predictive Maintenance
# -----------------------------
elif page == "Maintenance":
    st.title("Aircraft Engine Predictive Maintenance Dashboard")
    st.caption(f"Model: **{selected_model_name}** | Dataset: **{selected_dataset}**")

    fleet = get_fleet_predictions(selected_dataset, selected_model_name).copy()
    fleet = fleet.sort_values("unit_id")

    warning_threshold = 80
    critical_threshold = 30

    fleet["status"] = fleet["predicted_RUL"].apply(lambda x: status_from_rul(x, critical_threshold, warning_threshold))
    fleet["engine_label"] = "Engine " + fleet["unit_id"].astype(str)

    selected_engine_label = st.sidebar.selectbox("Engine", fleet["engine_label"].tolist(), index=0)
    selected_engine_id = int(selected_engine_label.replace("Engine ", ""))
    selected = fleet[fleet["unit_id"] == selected_engine_id].iloc[0]

    predicted_rul = float(selected["predicted_RUL"])
    actual_rul = float(selected["true_RUL"])
    selected_status = status_from_rul(predicted_rul, critical_threshold, warning_threshold)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fleet Size", len(fleet))
    k2.metric("Average Predicted RUL", round(fleet["predicted_RUL"].mean(), 2))
    k3.metric("Warning Engines", int((fleet["predicted_RUL"] <= warning_threshold).sum()))
    k4.metric("Critical Engines", int((fleet["predicted_RUL"] <= critical_threshold).sum()))

    st.markdown(
        f"""
        <div class="status-card" style="border-color: {status_color(selected_status)};">
            <div class="status-title">Engine #{selected_engine_id}</div>
            <div class="status-value">{predicted_rul:.0f} cycles remaining</div>
            <div class="status-label" style="color: {status_color(selected_status)};">Status: {selected_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_rul,
                number={"font": {"size": 44}},
                title={"text": "Remaining Useful Life (cycles)"},
                gauge={
                    "axis": {"range": [0, 300]},
                    "bar": {"color": "#ffa600"},
                    "steps": [
                        {"range": [0, critical_threshold], "color": "#ff4d4d"},
                        {"range": [critical_threshold, warning_threshold], "color": "#ffca3a"},
                        {"range": [warning_threshold, 300], "color": "#83e283"},
                    ],
                    "threshold": {"line": {"color": "#182c3a", "width": 3}, "thickness": 0.75, "value": predicted_rul},
                },
            )
        )
        st.plotly_chart(clean_layout(gauge, height=360), use_container_width=True)

    with col2:
        compare_df = pd.DataFrame({"RUL Type": ["Actual", "Predicted"], "Cycles": [actual_rul, predicted_rul]})
        fig_compare = px.bar(compare_df, x="RUL Type", y="Cycles", text="Cycles", title="Actual vs Predicted RUL")
        fig_compare.update_traces(texttemplate="%{text:.0f}", marker_color="#ffa600")
        st.plotly_chart(clean_layout(fig_compare, height=360), use_container_width=True)

    fig_fleet = px.scatter(
        fleet,
        x="unit_id",
        y="predicted_RUL",
        color="predicted_RUL",
        color_continuous_scale=["red", "orange", "green"],
        hover_data=["unit_id", "true_RUL", "predicted_RUL", "status"],
        title="Fleet-Wide RUL Overview",
        labels={"unit_id": "Engine Index", "predicted_RUL": "Predicted RUL (cycles)"},
    )
    fig_fleet.add_hline(y=warning_threshold, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    fig_fleet.add_hline(y=critical_threshold, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    st.plotly_chart(clean_layout(fig_fleet, height=520), use_container_width=True)

# -----------------------------
# Page: Cost Analysis
# -----------------------------
else:
    st.title("Predictive Maintenance Cost Analysis")
    st.caption(f"Model: **{selected_model_name}** | Dataset: **{selected_dataset}**")

    fleet = get_fleet_predictions(selected_dataset, selected_model_name).copy()
    planned_cost = st.sidebar.number_input("Planned Cost", min_value=1.0, value=10000.0, step=1000.0)
    unplanned_cost = st.sidebar.number_input("Unplanned Cost", min_value=1.0, value=80000.0, step=5000.0)
    target_availability = st.sidebar.slider("Target Availability (%)", min_value=50, max_value=100, value=90, step=5)
    warning_threshold = st.sidebar.number_input("Warning Threshold", min_value=1, value=80, step=5)
    critical_threshold = st.sidebar.number_input("Critical Threshold", min_value=1, value=30, step=5)

    cost = cost_analysis_summary(
        fleet=fleet,
        planned_cost=planned_cost,
        unplanned_cost=unplanned_cost,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )

    a, b, c, d = st.columns(4)
    a.metric("Traditional Cost", f"${cost['traditional_total_cost']:,.0f}")
    b.metric("Predictive Cost", f"${cost['predictive_total_cost']:,.0f}")
    c.metric("Savings", f"${cost['savings']:,.0f}")
    d.metric("Savings Rate", f"{cost['savings_rate']:.1f}%")

    left, right = st.columns(2)
    with left:
        cost_df = pd.DataFrame({
            "Strategy": ["Traditional", "Predictive"],
            "Total Cost": [cost["traditional_total_cost"], cost["predictive_total_cost"]],
        })
        fig_cost = px.bar(cost_df, x="Strategy", y="Total Cost", text="Total Cost", title="Maintenance Cost Comparison")
        fig_cost.update_traces(texttemplate="$%{text:,.0f}", textposition="inside", marker_color=["red", "green"])
        st.plotly_chart(clean_layout(fig_cost, height=430), use_container_width=True)

    with right:
        events_df = pd.DataFrame({
            "Maintenance Type": ["Scheduled", "Scheduled", "Unscheduled", "Unscheduled"],
            "Strategy": ["Traditional", "Predictive", "Traditional", "Predictive"],
            "Events": [cost["traditional_scheduled"], cost["predictive_scheduled"], cost["traditional_unscheduled"], cost["predictive_unscheduled"]],
        })
        fig_events = px.bar(events_df, x="Maintenance Type", y="Events", color="Strategy", barmode="group", title="Maintenance Events Breakdown")
        st.plotly_chart(clean_layout(fig_events, height=430), use_container_width=True)

    left, right = st.columns(2)
    with left:
        pie_df = pd.DataFrame({
            "Category": ["Savings", "Remaining Cost"],
            "Amount": [cost["savings"], cost["predictive_total_cost"]],
        })
        fig_pie = px.pie(pie_df, values="Amount", names="Category", title="Cost Savings Distribution")
        st.plotly_chart(clean_layout(fig_pie, height=430), use_container_width=True)

    with right:
        availability_df = pd.DataFrame({
            "Strategy": ["Traditional", "Predictive"],
            "Operational Readiness (%)": [cost["traditional_availability"], cost["predictive_availability"]],
        })
        fig_avail = px.bar(availability_df, x="Strategy", y="Operational Readiness (%)", title="Fleet Availability")
        fig_avail.update_traces(marker_color=["orange", "green"])
        fig_avail.add_hline(y=target_availability, line_dash="dash", line_color="red", annotation_text=f"Target: {target_availability}%")
        fig_avail.update_yaxes(range=[0, 105])
        st.plotly_chart(clean_layout(fig_avail, height=430), use_container_width=True)
