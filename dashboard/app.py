"""
Streamlit Dashboard for Predictive Maintenance
Interactive visualization and monitoring dashboard.
"""

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="wrench",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .risk-critical { color: #ff4b4b; font-weight: bold; }
    .risk-high { color: #ffa500; font-weight: bold; }
    .risk-medium { color: #ffff00; font-weight: bold; }
    .risk-low { color: #00ff00; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)


# Configuration
CONFIG = {
    "api_url": "http://localhost:8000",
    "models_dir": "artifacts/models",
    "reports_dir": "artifacts/reports",
    "data_dir": "data/transformed",
}


def load_evaluation_report():
    """Load model evaluation report."""
    report_path = Path(CONFIG["reports_dir"]) / "evaluation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return None


def load_training_metrics():
    """Load training metrics."""
    metrics_path = Path(CONFIG["models_dir"]) / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def load_test_data():
    """Load test dataset for visualization."""
    X_test_path = Path(CONFIG["data_dir"]) / "X_test.csv"
    y_test_path = Path(CONFIG["data_dir"]) / "y_test.csv"

    if X_test_path.exists() and y_test_path.exists():
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)
        return X_test, y_test
    return None, None


def get_risk_color(level: str) -> str:
    """Get color for risk level."""
    colors = {
        "CRITICAL": "#ff4b4b",
        "HIGH": "#ffa500",
        "MEDIUM": "#ffeb3b",
        "LOW": "#4caf50",
    }
    return colors.get(level, "#808080")


def main():
    """Main dashboard application."""

    # Sidebar
    st.sidebar.title("Predictive Maintenance")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Navigation",
        ["Overview", "Model Performance", "Predictions", "Data Explorer", "System Health"],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Version:** 1.0.0\n\n**Last Updated:** " + datetime.now().strftime("%Y-%m-%d")
    )

    # Main content based on page selection
    if page == "Overview":
        render_overview()
    elif page == "Model Performance":
        render_model_performance()
    elif page == "Predictions":
        render_predictions()
    elif page == "Data Explorer":
        render_data_explorer()
    elif page == "System Health":
        render_system_health()


def render_overview():
    """Render overview page."""
    st.title("Predictive Maintenance Overview")

    # Load data
    report = load_evaluation_report()

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    if report and report.get("summary"):
        summary = report["summary"]
        with col1:
            st.metric("Models Evaluated", summary.get("total_models_evaluated", 0))
        with col2:
            st.metric("Best RMSE", f"{summary.get('best_rmse', 0):.2f}")
        with col3:
            st.metric("Best R2", f"{summary.get('best_r2', 0):.3f}")
        with col4:
            best_model = report.get("best_model", {}).get("name", "N/A")
            st.metric("Best Model", best_model.replace("_", " ").title())
    else:
        with col1:
            st.metric("Models Evaluated", "N/A")
        with col2:
            st.metric("Best RMSE", "N/A")
        with col3:
            st.metric("Best R2", "N/A")
        with col4:
            st.metric("Best Model", "N/A")

    st.markdown("---")

    # Model Comparison Chart
    if report and report.get("detailed_metrics"):
        st.subheader("Model Performance Comparison")

        metrics_data = []
        for model_name, metrics in report["detailed_metrics"].items():
            metrics_data.append(
                {
                    "Model": model_name.replace("_", " ").title(),
                    "RMSE": metrics.get("rmse", 0),
                    "MAE": metrics.get("mae", 0),
                    "R2": metrics.get("r2", 0),
                }
            )

        df_metrics = pd.DataFrame(metrics_data)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                df_metrics,
                x="Model",
                y="RMSE",
                title="RMSE by Model",
                color="RMSE",
                color_continuous_scale="Blues_r",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                df_metrics,
                x="Model",
                y="R2",
                title="R2 Score by Model",
                color="R2",
                color_continuous_scale="Greens",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    if report and report.get("recommendations"):
        st.subheader("Recommendations")
        for rec in report["recommendations"]:
            st.info(rec)


def render_model_performance():
    """Render model performance page."""
    st.title("Model Performance Analysis")

    report = load_evaluation_report()

    if not report or not report.get("detailed_metrics"):
        st.warning("No evaluation data available. Please run model evaluation first.")
        return

    # Model selector
    model_names = list(report["detailed_metrics"].keys())
    selected_model = st.selectbox(
        "Select Model", model_names, format_func=lambda x: x.replace("_", " ").title()
    )

    metrics = report["detailed_metrics"][selected_model]

    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MSE", f"{metrics.get('mse', 0):.4f}")
    with col2:
        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
    with col3:
        st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
    with col4:
        st.metric("R2 Score", f"{metrics.get('r2', 0):.4f}")

    st.markdown("---")

    # Additional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
    with col2:
        st.metric("Max Error", f"{metrics.get('max_error', 0):.2f}")
    with col3:
        st.metric("Std Error", f"{metrics.get('std_error', 0):.4f}")

    # Check for evaluation plots
    plot_path = Path(CONFIG["reports_dir"]) / f"{selected_model}_evaluation.png"
    if plot_path.exists():
        st.subheader("Evaluation Plots")
        st.image(str(plot_path))


def render_predictions():
    """Render predictions page."""
    st.title("Make Predictions")

    st.subheader("Single Prediction")

    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            unit_number = st.number_input("Unit Number", min_value=1, value=1)
            time_in_cycles = st.number_input("Time in Cycles", min_value=1, value=100)
            op_setting_1 = st.number_input("Op Setting 1", value=0.0)
            op_setting_2 = st.number_input("Op Setting 2", value=0.0)
            op_setting_3 = st.number_input("Op Setting 3", value=100.0)

        with col2:
            sensors_1_7 = {}
            for i in range(1, 8):
                sensors_1_7[f"sensor_{i}"] = st.number_input(f"Sensor {i}", value=50.0)

        with col3:
            sensors_8_14 = {}
            for i in range(8, 15):
                sensors_8_14[f"sensor_{i}"] = st.number_input(f"Sensor {i}", value=50.0)

        col4, col5 = st.columns(2)
        with col4:
            sensors_15_21 = {}
            for i in range(15, 22):
                sensors_15_21[f"sensor_{i}"] = st.number_input(f"Sensor {i}", value=50.0)

        with col5:
            model_name = st.selectbox(
                "Model",
                [
                    "random_forest",
                    "gradient_boosting",
                    "linear_regression",
                    "ridge",
                    "lasso",
                    "svr",
                ],
            )

        submitted = st.form_submit_button("Predict RUL")

        if submitted:
            # Prepare data
            reading = {
                "unit_number": unit_number,
                "time_in_cycles": time_in_cycles,
                "op_setting_1": op_setting_1,
                "op_setting_2": op_setting_2,
                "op_setting_3": op_setting_3,
                **sensors_1_7,
                **sensors_8_14,
                **sensors_15_21,
            }

            try:
                response = requests.post(
                    f"{CONFIG['api_url']}/predict",
                    json={"readings": [reading], "model_name": model_name},
                    timeout=10,
                )

                if response.status_code == 200:
                    result = response.json()
                    pred = result["predictions"][0]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Predicted RUL:** {pred['predicted_rul']:.2f} cycles")
                    with col2:
                        risk_color = get_risk_color(pred["risk_level"])
                        st.markdown(
                            f"**Risk Level:** <span style='color:{risk_color}'>"
                            f"{pred['risk_level']}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.error(f"API Error: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Make sure the API server is running.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown("---")

    # Batch Prediction
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} records")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            st.info("Batch prediction would be processed here...")


def render_data_explorer():
    """Render data explorer page."""
    st.title("Data Explorer")

    X_test, y_test = load_test_data()

    if X_test is None:
        st.warning("No test data available.")
        return

    st.subheader("Dataset Overview")
    st.write(f"**Samples:** {len(X_test)}")
    st.write(f"**Features:** {len(X_test.columns)}")

    # Feature distribution
    st.subheader("Feature Distribution")
    feature = st.selectbox("Select Feature", X_test.columns)

    fig = px.histogram(X_test, x=feature, nbins=50, title=f"Distribution of {feature}")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.subheader("Feature Correlations")
    if st.checkbox("Show correlation matrix"):
        sensor_cols = [col for col in X_test.columns if "sensor" in col][:10]
        if sensor_cols:
            corr = X_test[sensor_cols].corr()
            fig = px.imshow(corr, title="Sensor Correlation Matrix", color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)

    # RUL distribution
    if y_test is not None:
        st.subheader("RUL Distribution")
        fig = px.histogram(y_test, x=y_test.columns[0], nbins=50, title="RUL Distribution")
        st.plotly_chart(fig, use_container_width=True)


def render_system_health():
    """Render system health page."""
    st.title("System Health")

    # API Health Check
    st.subheader("API Status")
    try:
        response = requests.get(f"{CONFIG['api_url']}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            st.success("API is healthy")
            st.json(health)
        else:
            st.error("API returned error")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API")
    except Exception as e:
        st.error(f"Error: {str(e)}")

    st.markdown("---")

    # Model Status
    st.subheader("Model Status")
    models_dir = Path(CONFIG["models_dir"])
    if models_dir.exists():
        model_files = list(models_dir.glob("*_model.pkl")) + list(models_dir.glob("*.h5"))
        if model_files:
            for model_file in model_files:
                st.write(f"- {model_file.name}")
        else:
            st.warning("No models found")
    else:
        st.warning("Models directory not found")

    st.markdown("---")

    # Resource Usage (placeholder)
    st.subheader("Resource Usage")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPU Usage", "45%")
    with col2:
        st.metric("Memory Usage", "2.1 GB")
    with col3:
        st.metric("Disk Usage", "5.2 GB")


if __name__ == "__main__":
    main()
