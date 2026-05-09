from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forecasting import train_and_forecast


st.set_page_config(
    page_title="ClimateCast AI",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
            :root {
                --bg: #070d1b;
                --panel: rgba(15, 25, 48, 0.78);
                --panel-strong: rgba(20, 31, 58, 0.92);
                --line: rgba(148, 163, 184, 0.18);
                --text: #f8fafc;
                --muted: #c7d2fe;
                --purple: #8b5cf6;
                --blue: #3b82f6;
                --green: #57d879;
                --gold: #f7b733;
            }

            html, body, [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 22% 18%, rgba(59, 130, 246, 0.16), transparent 28rem),
                    radial-gradient(circle at 85% 7%, rgba(139, 92, 246, 0.14), transparent 24rem),
                    linear-gradient(135deg, #050916 0%, #091124 48%, #070d1b 100%);
                color: var(--text);
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stSidebar"] {
                background: rgba(9, 17, 36, 0.96);
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span {
                color: #e5e7eb;
            }

            .block-container {
                padding-top: 1.7rem;
                padding-bottom: 2rem;
                max-width: 1500px;
            }

            .app-title {
                display: flex;
                align-items: center;
                gap: 0.95rem;
                padding: 0.4rem 0 1.15rem;
                border-bottom: 1px solid var(--line);
                margin-bottom: 1.35rem;
            }

            .logo-mark {
                position: relative;
                display: grid;
                place-items: center;
                width: 3.5rem;
                height: 3.5rem;
                font-size: 2.25rem;
            }

            .brand-name {
                font-size: 1.72rem;
                line-height: 1;
                font-weight: 800;
                color: var(--text);
                letter-spacing: 0;
            }

            .brand-name span {
                color: #7aa2ff;
            }

            .brand-subtitle {
                margin-top: 0.42rem;
                color: #cbd5e1;
                font-size: 0.88rem;
            }

            .section-label {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: #dbeafe;
                font-size: 0.86rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.03em;
                margin: 0.45rem 0 1rem;
            }

            .upload-frame {
                display: grid;
                place-items: center;
                min-height: 10.6rem;
                border: 1px dashed rgba(203, 213, 225, 0.38);
                border-radius: 8px;
                background: rgba(8, 14, 30, 0.42);
                margin-bottom: 0.9rem;
                text-align: center;
                color: #cbd5e1;
            }

            .uploaded-file,
            .about-box,
            .metric-card,
            .chart-card,
            .table-card,
            .component-card {
                border: 1px solid var(--line);
                background: linear-gradient(145deg, rgba(15, 25, 48, 0.94), rgba(8, 14, 30, 0.78));
                border-radius: 8px;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
            }

            .uploaded-file {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0.72rem 0.78rem;
                margin: 0.8rem 0 1.2rem;
            }

            .uploaded-file small {
                color: #cbd5e1;
            }

            .about-box {
                padding: 1rem 1.05rem;
                color: #cbd5e1;
                line-height: 1.55;
                margin-top: 1.2rem;
            }

            .about-title {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: #dbeafe;
                font-weight: 800;
                margin-bottom: 0.72rem;
            }

            .hero-row {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 1rem;
                margin-bottom: 1.35rem;
            }

            .hero-title {
                font-size: clamp(2rem, 4vw, 2.65rem);
                line-height: 1.08;
                font-weight: 850;
                letter-spacing: 0;
                color: white;
                margin: 0;
            }

            .hero-copy {
                margin-top: 0.68rem;
                color: #cbd5e1;
                font-size: 1.05rem;
            }

            .mode-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.48rem;
                white-space: nowrap;
                color: #b79cff;
                border: 1px solid rgba(139, 92, 246, 0.8);
                background: rgba(25, 18, 52, 0.72);
                padding: 0.6rem 0.95rem;
                border-radius: 6px;
                font-weight: 700;
            }

            .metric-card {
                min-height: 6.45rem;
                padding: 1.15rem 1.2rem;
                display: flex;
                align-items: center;
                gap: 1.05rem;
                margin-bottom: 1.35rem;
            }

            .metric-icon {
                width: 3.9rem;
                height: 3.9rem;
                display: grid;
                place-items: center;
                border-radius: 999px;
                font-size: 1.8rem;
                flex: 0 0 auto;
            }

            .metric-icon.purple { background: rgba(139, 92, 246, 0.32); color: #f472ff; }
            .metric-icon.blue { background: rgba(59, 130, 246, 0.28); color: #60a5fa; }
            .metric-icon.green { background: rgba(34, 197, 94, 0.28); color: #7ee787; }
            .metric-icon.gold { background: rgba(247, 183, 51, 0.32); color: #ffd166; }

            .metric-label {
                color: #c7d2fe;
                font-size: 0.92rem;
                font-weight: 700;
                margin-bottom: 0.45rem;
            }

            .metric-value {
                color: white;
                font-size: 1.2rem;
                font-weight: 850;
            }

            .chart-card,
            .table-card {
                padding: 1.05rem 1.05rem 0.8rem;
                margin-bottom: 1.35rem;
            }

            .panel-heading {
                color: white;
                font-size: 1.18rem;
                font-weight: 850;
                margin-bottom: 0.2rem;
            }

            .panel-subtitle {
                color: #cbd5e1;
                margin-bottom: 1rem;
            }

            .component-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.65rem;
                margin-top: 1rem;
            }

            .component-card {
                padding: 0.72rem;
                min-height: 11.1rem;
            }

            .empty-state {
                border: 1px solid var(--line);
                background: linear-gradient(145deg, rgba(15, 25, 48, 0.94), rgba(8, 14, 30, 0.78));
                border-radius: 8px;
                padding: 3rem 2rem;
                min-height: 24rem;
                display: grid;
                place-items: center;
                text-align: center;
                color: #cbd5e1;
            }

            .empty-state strong {
                display: block;
                color: white;
                font-size: 1.35rem;
                margin-bottom: 0.55rem;
            }

            .component-title {
                text-align: center;
                color: white;
                font-size: 0.82rem;
                font-weight: 800;
                margin-bottom: 0.3rem;
            }

            div[data-testid="stFileUploader"] {
                margin-top: -11.05rem;
                min-height: 11.05rem;
                opacity: 0;
            }

            div[data-testid="stFileUploader"] section {
                min-height: 10.8rem;
                cursor: pointer;
            }

            .stButton > button,
            .stDownloadButton > button {
                width: 100%;
                min-height: 2.85rem;
                border-radius: 6px;
                border: 1px solid rgba(139, 92, 246, 0.8);
                color: white;
                font-weight: 800;
                background: linear-gradient(135deg, #7c3aed, #5b5ee8);
            }

            .stDownloadButton > button {
                color: #b79cff;
                background: rgba(15, 23, 42, 0.45);
            }

            [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div {
                color: var(--purple);
            }

            [data-testid="stDataFrame"] {
                border-radius: 8px;
                overflow: hidden;
            }

            @media (max-width: 900px) {
                .hero-row {
                    flex-direction: column;
                }

                .component-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_csv_source(uploaded_file):
    if uploaded_file is None:
        return None, "No CSV selected", None

    data = uploaded_file.getvalue()
    return BytesIO(data), uploaded_file.name, len(data)


def available_numeric_columns(source):
    df = pd.read_csv(source)
    df.columns = df.columns.str.lower().str.strip()
    columns = df.select_dtypes(include=["number"]).columns.tolist()
    if "date" in columns:
        columns.remove("date")
    return columns


def load_clean_from_source(source, target):
    df = pd.read_csv(source)
    df.columns = df.columns.str.lower().str.strip()
    if "date" not in df.columns:
        raise ValueError("The dataset must contain a 'date' column.")
    if target not in df.columns:
        raise ValueError(f"The dataset must contain the target column '{target}'.")

    df = df[["date", target]].rename(columns={"date": "ds", target: "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna().sort_values("ds").drop_duplicates("ds").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def forecast_data(csv_bytes, target, periods):
    if csv_bytes is None:
        raise ValueError("Upload a CSV file before generating a forecast.")

    source = BytesIO(csv_bytes)
    df = load_clean_from_source(source, target)
    model, forecast = train_and_forecast(df, periods)
    fitted = model.predict(df[["ds"]])
    components = model.predict(model.make_future_dataframe(periods=periods, freq="D"))
    return df, forecast, fitted, components


def format_target(target):
    labels = {
        "meantemp": "meantemp (°C)",
        "humidity": "humidity (%)",
        "wind_speed": "wind speed",
        "meanpressure": "mean pressure",
    }
    return labels.get(target, target.replace("_", " ").title())


def y_axis_label(target):
    if "temp" in target:
        return "Temperature (°C)"
    if target == "humidity":
        return "Humidity (%)"
    if "pressure" in target:
        return "Pressure"
    if "wind" in target:
        return "Wind Speed"
    return format_target(target)


def metric_card(icon, label, value, color):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon {color}">{icon}</div>
            <div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def line_chart(df, forecast, target):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["ds"],
            y=df["y"],
            name="Actual",
            mode="lines",
            line=dict(color="#4f9cff", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Forecast",
            mode="lines",
            line=dict(color="#6ee785", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            name="Upper Bound",
            mode="lines",
            line=dict(color="rgba(203,213,225,0.72)", width=1, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            name="Lower Bound",
            mode="lines",
            line=dict(color="rgba(203,213,225,0.72)", width=1, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(110, 231, 133, 0.12)",
        )
    )
    fig.add_vline(
        x=forecast["ds"].min(),
        line_width=1,
        line_dash="dash",
        line_color="#8b5cf6",
    )
    fig.add_annotation(
        x=forecast["ds"].min(),
        y=forecast["yhat_upper"].max(),
        text="Forecast Start",
        showarrow=False,
        bgcolor="rgba(89, 64, 177, 0.42)",
        font=dict(color="#c4b5fd", size=11),
        yshift=8,
    )
    fig.update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=18, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#ffffff"),
        ),
        xaxis=dict(
            title="Date",
            gridcolor="rgba(148,163,184,0.10)",
            color="#e5e7eb",
            linecolor="rgba(203,213,225,0.7)",
        ),
        yaxis=dict(
            title=y_axis_label(target),
            gridcolor="rgba(148,163,184,0.10)",
            color="#e5e7eb",
            linecolor="rgba(203,213,225,0.7)",
        ),
        hovermode="x unified",
    )
    return fig


def mini_chart(data, x, y, color):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[x],
            y=data[y],
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=color.replace("1)", "0.12)") if color.startswith("rgba") else "rgba(96,165,250,0.10)",
        )
    )
    fig.update_layout(
        height=118,
        margin=dict(l=2, r=2, t=4, b=2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def render_sidebar():
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "forecast_requested" not in st.session_state:
        st.session_state.forecast_requested = False

    st.sidebar.markdown(
        """
        <div class="app-title">
            <div class="logo-mark">⛅</div>
            <div>
                <div class="brand-name">ClimateCast <span>AI</span></div>
                <div class="brand-subtitle">AI-Powered Weather Forecasting</div>
            </div>
        </div>
        <div class="section-label">☷ Inputs</div>
        <div style="font-weight:800;margin-bottom:.7rem;">Upload Weather CSV</div>
        <div class="upload-frame">
            <div>
                <div style="font-size:2rem;margin-bottom:.4rem;">☁</div>
                <div>Drag & drop your CSV file here</div>
                <div style="margin:.45rem 0;color:#94a3b8;">or</div>
                <div style="display:inline-block;border:1px solid rgba(203,213,225,.5);border-radius:6px;padding:.58rem 1.35rem;font-weight:800;">Browse Files</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.sidebar.file_uploader(
        "Upload Weather CSV",
        type=["csv"],
        key=f"weather_csv_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    source, file_name, file_size = read_csv_source(uploaded)
    csv_error = None
    try:
        numeric_columns = available_numeric_columns(source) if source is not None else []
    except Exception as exc:
        csv_error = str(exc)
        numeric_columns = []

    size_label = f"{file_size / 1024:.1f} KB" if file_size else "Upload a weather CSV to begin"
    status_dot = "●" if uploaded else "○"
    status_color = "#6ee785" if uploaded else "#94a3b8"
    st.sidebar.markdown(
        f"""
        <div class="uploaded-file">
            <div>
                <strong>▣ {file_name}</strong><br>
                <small>{size_label}</small>
            </div>
            <div style="color:{status_color};font-size:1.2rem;">{status_dot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("✕ Remove CSV", width="stretch", disabled=uploaded is None):
        st.session_state.uploader_key += 1
        st.session_state.forecast_requested = False
        st.rerun()
    if csv_error:
        st.sidebar.error(f"Could not read CSV: {csv_error}")

    if uploaded is None:
        st.session_state.forecast_requested = False

    if numeric_columns:
        target = st.sidebar.selectbox(
            "Select Target Variable",
            options=numeric_columns,
            format_func=format_target,
            index=0,
        )
    else:
        st.sidebar.selectbox(
            "Select Target Variable",
            options=["Upload a CSV first"],
            index=0,
            disabled=True,
        )
        target = None

    periods = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30, step=1)
    generate = st.sidebar.button(
        "⚙ Generate Forecast",
        width="stretch",
        disabled=uploaded is None or not numeric_columns,
    )
    if generate:
        st.session_state.forecast_requested = True

    st.sidebar.markdown(
        """
        <div class="about-box">
            <div class="about-title">ⓘ About ClimateCast AI</div>
            ClimateCast AI uses Facebook Prophet under the hood to analyze historical weather
            patterns and predict future values with high accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return uploaded, target, periods, generate, numeric_columns


inject_styles()

try:
    uploaded_placeholder, target, periods, generate, numeric_columns = render_sidebar()
except Exception as exc:
    st.error(f"Could not read the weather CSV: {exc}")
    st.stop()

if numeric_columns and target not in numeric_columns:
    target = numeric_columns[0]

st.markdown(
    """
    <div class="hero-row">
        <div>
            <h1 class="hero-title">ClimateCast AI Dashboard</h1>
            <div class="hero-copy">Forecast future weather with confidence.</div>
        </div>
        <div class="mode-pill">☾ Dark Mode</div>
    </div>
    """,
    unsafe_allow_html=True,
)

csv_bytes = uploaded_placeholder.getvalue() if uploaded_placeholder else None
ready_to_forecast = (
    uploaded_placeholder is not None
    and bool(numeric_columns)
    and st.session_state.forecast_requested
)

if uploaded_placeholder is None:
    st.markdown(
        """
        <div class="empty-state">
            <div>
                <div style="font-size:2.4rem;margin-bottom:.65rem;">☁</div>
                <strong>Upload a weather CSV to generate a forecast</strong>
                Choose a CSV from the sidebar to enable the forecast controls.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not numeric_columns:
    st.markdown(
        """
        <div class="empty-state">
            <div>
                <div style="font-size:2.4rem;margin-bottom:.65rem;">▦</div>
                <strong>No numeric target columns found</strong>
                Upload a CSV with a date column and at least one numeric weather column.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not st.session_state.forecast_requested:
    st.markdown(
        """
        <div class="empty-state">
            <div>
                <div style="font-size:2.4rem;margin-bottom:.65rem;">↗</div>
                <strong>Ready to forecast</strong>
                Select your target variable and horizon, then generate the forecast.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif ready_to_forecast:
    try:
        with st.spinner("Training Prophet and generating the forecast..."):
            df, forecast, fitted, components = forecast_data(csv_bytes, target, periods)
    except Exception as exc:
        st.error(f"Forecast generation failed: {exc}")
    else:
        metric_cols = st.columns(4)
        with metric_cols[0]:
            metric_card("♨", "Target Variable", format_target(target), "purple")
        with metric_cols[1]:
            metric_card("▦", "Forecast Horizon", f"{periods} Days", "blue")
        with metric_cols[2]:
            metric_card("↗", "Historical Records", f"{len(df):,} Days", "green")
        with metric_cols[3]:
            metric_card("◎", "Model", "Prophet", "gold")

        st.markdown('<div class="chart-card"><div class="panel-heading">Forecast vs Actual</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(df, forecast, target), width="stretch", config={"displayModeBar": True})
        st.markdown("</div>", unsafe_allow_html=True)

        left, right = st.columns([0.95, 1.05])

        with left:
            st.markdown(
                f'<div class="table-card"><div class="panel-heading">Forecast Table <span style="font-weight:500;">(Next {periods} Days)</span></div>',
                unsafe_allow_html=True,
            )
            table = forecast.rename(
                columns={
                    "ds": "Date",
                    "yhat": "Forecast",
                    "yhat_lower": "Lower Bound",
                    "yhat_upper": "Upper Bound",
                }
            )
            table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
            for column in ["Forecast", "Lower Bound", "Upper Bound"]:
                table[column] = table[column].round(2)
            st.dataframe(table, width="stretch", height=245, hide_index=True)
            st.download_button(
                "⇩ Download Forecast CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name=f"climatecast_{target}_forecast.csv",
                mime="text/csv",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown(
                """
                <div class="table-card">
                    <div class="panel-heading">Components</div>
                    <div class="panel-subtitle">Decomposed time series components</div>
                    <div class="component-grid">
                """,
                unsafe_allow_html=True,
            )

            component_cols = st.columns(4)
            residuals = df[["ds", "y"]].copy()
            residuals["residual"] = residuals["y"] - fitted["yhat"].values
            component_specs = [
                ("Trend", components.tail(365), "trend", "#4f9cff"),
                ("Yearly Seasonality", components.tail(365), "yearly", "#6ee785"),
                ("Weekly Seasonality", components.tail(180), "weekly", "#fbbf24"),
                ("Residuals", residuals.tail(180), "residual", "#ff4646"),
            ]

            for col, (title, data, y_col, color) in zip(component_cols, component_specs):
                with col:
                    st.markdown(f'<div class="component-card"><div class="component-title">{title}</div>', unsafe_allow_html=True)
                    st.plotly_chart(mini_chart(data, "ds", y_col, color), width="stretch", config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)
