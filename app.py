import os
import glob
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import os
import urllib.request
import joblib
import streamlit as st
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# AgriPredict AI - robust Streamlit dashboard
# Auto-detects model/dataset files. Dashboard still opens if
# the model or dataset is missing; missing items are explained.
# ---------------------------------------------------------

st.set_page_config(
    page_title="AgriPredict AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "Crop", "Crop_Year", "Season", "State",
    "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"
]
TARGET = "Yield"

st.markdown("""
<style>
.block-container {max-width: 1450px; padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {padding: 24px 28px; border-radius: 18px;
       background: linear-gradient(115deg,#12372A,#436850);
       color: white; margin-bottom: 18px;}
.hero h1 {color: white; margin: 0 0 8px 0;}
.hero p {color: #e6f0e8; margin: 0;}
div[data-testid="stMetric"] {padding: 14px; border-radius: 12px;
       border: 1px solid rgba(128,128,128,.22);}
</style>
""", unsafe_allow_html=True)


def find_files():
    """Look for files in common Colab/local project folders."""
    roots = [Path.cwd(), Path("/content"), Path("/content/drive/MyDrive")]
    seen = set()
    model_files, csv_files = [], []
    for root in roots:
        if not root.exists():
            continue
        try:
            for pattern in ("*.pkl", "*.pickle", "*.joblib"):
                for p in root.rglob(pattern):
                    if p.is_file() and str(p) not in seen:
                        seen.add(str(p))
                        model_files.append(p)
            for p in root.rglob("*.csv"):
                if p.is_file() and str(p) not in seen:
                    seen.add(str(p))
                    csv_files.append(p)
        except (PermissionError, OSError):
            continue
    # Prefer familiar names, but do not require exact filenames.
    model_files.sort(key=lambda p: (
        0 if "agripredict" in p.name.lower() else 1,
        0 if "model" in p.name.lower() else 1,
        len(str(p))
    ))
    csv_files.sort(key=lambda p: (
        0 if "crop_yield" in p.name.lower() else 1,
        0 if "yield" in p.name.lower() else 1,
        len(str(p))
    ))
    return model_files, csv_files


@st.cache_data(show_spinner=False)
def read_csv(path, mtime):
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def read_model(path, mtime):
    suffix = Path(path).suffix.lower()
    if suffix == ".joblib":
        import joblib
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def make_demo_data(n=240):
    """Clearly labelled synthetic data used only to preview the UI."""
    rng = np.random.default_rng(42)
    crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Pulses"]
    seasons = ["Kharif", "Rabi", "Whole Year", "Summer", "Winter"]
    states = ["Madhya Pradesh", "Punjab", "Maharashtra", "Uttar Pradesh", "Gujarat"]
    crop = rng.choice(crops, n)
    area = rng.uniform(10, 900, n)
    rainfall = rng.uniform(350, 1600, n)
    fertilizer = rng.uniform(100, 15000, n)
    pesticide = rng.uniform(5, 900, n)
    base = pd.Series(crop).map({
        "Rice": 3.0, "Wheat": 2.8, "Maize": 2.5,
        "Cotton": 1.8, "Sugarcane": 65.0, "Pulses": 0.9
    }).to_numpy()
    yield_values = np.maximum(
        0.05, base + rainfall / 1200 + fertilizer / 12000
        + rng.normal(0, 0.35, n)
    )
    return pd.DataFrame({
        "Crop": crop,
        "Crop_Year": rng.integers(2000, 2024, n),
        "Season": rng.choice(seasons, n),
        "State": rng.choice(states, n),
        "Area": area,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide,
        "Yield": yield_values,
    })


def normalize_columns(data):
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]
    # Case-insensitive matching for common column names.
    lookup = {str(c).strip().lower(): c for c in data.columns}
    aliases = {
        "crop_year": ["crop_year", "year"],
        "annual_rainfall": ["annual_rainfall", "rainfall"],
        "fertilizer": ["fertilizer", "fertiliser"],
        "pesticide": ["pesticide", "pesticides"],
        "yield": ["yield", "crop_yield"],
        "crop": ["crop"],
        "season": ["season"],
        "state": ["state", "region"],
        "area": ["area"],
    }
    rename = {}
    for desired, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lookup:
                original = lookup[candidate]
                canonical = next((x for x in FEATURES + [TARGET]
                                  if x.lower() == desired), desired)
                if original != canonical:
                    rename[original] = canonical
                break
    return data.rename(columns=rename)


model_files, csv_files = find_files()
model = None
model_path = None
model_error = None
for candidate in model_files:
    try:
        model = read_model(str(candidate), candidate.stat().st_mtime)
        model_path = str(candidate)
        break
    except Exception as exc:
        model_error = f"{candidate.name}: {exc}"

df = None
data_path = None
data_error = None
for candidate in csv_files:
    try:
        candidate_df = normalize_columns(read_csv(str(candidate), candidate.stat().st_mtime))
        if all(c in candidate_df.columns for c in FEATURES + [TARGET]):
            df = candidate_df
            data_path = str(candidate)
            break
    except Exception as exc:
        data_error = f"{candidate.name}: {exc}"

demo_mode = False
if df is None:
    df = make_demo_data()
    demo_mode = True

# Numeric conversions, without crashing on dirty cells.
for col in ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide", TARGET]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)
if TARGET in df.columns:
    chart_df = df.dropna(subset=[TARGET]).copy()
else:
    chart_df = df.copy()

# Sidebar
st.sidebar.markdown("## 🌾 AgriPredict AI")
st.sidebar.caption("Crop Yield Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Predict Yield", "Dataset Explorer", "Model Status", "About"]
)
st.sidebar.divider()
if demo_mode:
    st.sidebar.warning("Demo preview data is being shown.")
else:
    st.sidebar.success("Dataset loaded")

# Hero header
st.markdown("""
<div class="hero">
  <h1>🌾 AgriPredict AI</h1>
  <p>Agricultural data analytics and crop-yield prediction dashboard</p>
</div>
""", unsafe_allow_html=True)

if demo_mode:
    st.warning(
        "DEMO PREVIEW: No matching dataset CSV was found. Charts use synthetic "
        "example data so the interface can load. Upload your real crop_yield.csv "
        "to see real dataset results. Predictions remain unavailable unless a model is found."
    )
if model is None:
    st.info(
        "Prediction model not detected yet. The dashboard and charts still work. "
        "Upload your .pkl/.pickle/.joblib model file to enable predictions."
    )

if page == "Dashboard":
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", f"{len(df):,}")
    c2.metric("Crop Types", f"{df['Crop'].nunique():,}" if "Crop" in df else "N/A")
    c3.metric("States / Regions", f"{df['State'].nunique():,}" if "State" in df else "N/A")
    if "Crop_Year" in df and df["Crop_Year"].notna().any():
        c4.metric("Year Range", f"{int(df['Crop_Year'].min())}–{int(df['Crop_Year'].max())}")
    else:
        c4.metric("Year Range", "N/A")

    left, right = st.columns(2)
    with left:
        crop_counts = df["Crop"].fillna("Unknown").value_counts().head(12).rename_axis("Crop").reset_index(name="Records")
        fig = px.bar(crop_counts, x="Records", y="Crop", orientation="h",
                     title="Most Represented Crops", color="Records",
                     color_continuous_scale="Greens")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        season_counts = df["Season"].fillna("Unknown").value_counts().rename_axis("Season").reset_index(name="Records")
        fig = px.pie(season_counts, names="Season", values="Records", hole=0.48,
                     title="Records by Season")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        if "Crop_Year" in df:
            yearly = df.dropna(subset=["Crop_Year"]).groupby("Crop_Year", as_index=False).size()
            yearly = yearly.rename(columns={"size": "Records"})
            fig = px.line(yearly, x="Crop_Year", y="Records", markers=True,
                          title="Records by Year")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if not chart_df.empty:
            crop_yield = chart_df.groupby("Crop", as_index=False)[TARGET].median().sort_values(TARGET, ascending=False).head(12)
            fig = px.bar(crop_yield, x=TARGET, y="Crop", orientation="h",
                         title="Median Yield by Crop (dataset units)",
                         color=TARGET, color_continuous_scale="Teal")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Charts are descriptive, not proof of cause and effect.")

elif page == "Predict Yield":
    st.subheader("🌱 Predict Crop Yield")
    if model is None:
        st.error("Prediction is disabled because no saved model file was found.")
        st.write("Upload your trained model (.pkl, .pickle or .joblib) to the Colab Files panel, then rerun the app.")
    elif demo_mode:
        st.warning("A model was found, but the real dataset was not. Confirm model input categories and units before using predictions.")
    if model is not None:
        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                crop_options = sorted(df["Crop"].dropna().astype(str).unique().tolist())
                season_options = sorted(df["Season"].dropna().astype(str).unique().tolist())
                state_options = sorted(df["State"].dropna().astype(str).unique().tolist())
                crop = st.selectbox("Crop", crop_options or ["Rice"])
                season = st.selectbox("Season", season_options or ["Kharif"])
                state = st.selectbox("State / Region", state_options or ["Unknown"])
                years = df["Crop_Year"].dropna()
                default_year = int(years.max()) if len(years) else 2023
                year = st.number_input("Crop Year", min_value=1900, max_value=2100, value=default_year)
            with col2:
                def median_or(col, default):
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    return max(0.0, float(vals.median())) if len(vals) else default
                area = st.number_input("Area (training-data units)", min_value=0.0, value=median_or("Area", 100.0))
                rainfall = st.number_input("Annual Rainfall (training-data units)", min_value=0.0, value=median_or("Annual_Rainfall", 800.0))
                fertilizer = st.number_input("Fertilizer (training-data units)", min_value=0.0, value=median_or("Fertilizer", 1000.0))
                pesticide = st.number_input("Pesticide (training-data units)", min_value=0.0, value=median_or("Pesticide", 100.0))
            submitted = st.form_submit_button("Predict Yield", type="primary", use_container_width=True)
        if submitted:
            if area <= 0:
                st.error("Area must be greater than zero.")
            else:
                row = pd.DataFrame([{
                    "Crop": crop, "Crop_Year": year, "Season": season, "State": state,
                    "Area": area, "Annual_Rainfall": rainfall,
                    "Fertilizer": fertilizer, "Pesticide": pesticide
                }])[FEATURES]
                try:
                    pred = float(np.asarray(model.predict(row)).ravel()[0])
                    if not np.isfinite(pred):
                        raise ValueError("Model returned a non-finite prediction.")
                    st.success("Prediction generated")
                    st.metric("Estimated Yield", f"{pred:,.4f}")
                    st.caption("Output unit is the same as the target used during model training. Verify it before interpretation.")
                    result = row.copy()
                    result["Predicted_Yield"] = pred
                    st.dataframe(result, use_container_width=True)
                    st.download_button("Download prediction CSV",
                        result.to_csv(index=False).encode("utf-8"),
                        "agripredict_prediction.csv", "text/csv")
                except Exception as exc:
                    st.error(f"Prediction could not run: {exc}")
                    st.info("The saved model may expect different feature names, column order, categories or preprocessing. Check the training code.")

elif page == "Dataset Explorer":
    st.subheader("Dataset Explorer")
    f1, f2 = st.columns(2)
    crop_options = sorted(df["Crop"].dropna().unique().tolist())
    state_options = sorted(df["State"].dropna().unique().tolist())
    with f1:
        chosen_crops = st.multiselect("Filter crops", crop_options, default=crop_options)
    with f2:
        chosen_states = st.multiselect("Filter states / regions", state_options, default=state_options)
    filtered = df[df["Crop"].isin(chosen_crops) & df["State"].isin(chosen_states)]
    st.metric("Rows after filtering", f"{len(filtered):,}")
    st.dataframe(filtered.head(1000), use_container_width=True)
    st.download_button("Download CSV", filtered.to_csv(index=False).encode("utf-8"),
                       "agripredict_data.csv", "text/csv")
    q1, q2, q3 = st.columns(3)
    q1.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    q2.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
    q3.metric("Valid Yield Values", f"{int(df[TARGET].notna().sum()):,}")
    st.write("Missing values by column")
    miss = df.isna().sum().rename("Missing").reset_index()
    miss.columns = ["Column", "Missing"]
    st.dataframe(miss, use_container_width=True)
    vals = df[TARGET].dropna()
    if len(vals) >= 4:
        q1v, q3v = vals.quantile([0.25, 0.75])
        iqr = q3v - q1v
        outlier_count = int(((vals < q1v - 1.5 * iqr) | (vals > q3v + 1.5 * iqr)).sum())
        st.info(f"{outlier_count:,} yield values are flagged by the IQR rule. Investigate them; do not delete automatically.")

elif page == "Model Status":
    st.subheader("Model & Dataset Status")
    if model_path:
        st.success(f"Model loaded: {model_path}")
    else:
        st.error("No model file detected.")
        st.write("Searched for .pkl, .pickle and .joblib files in the current folder and common Colab folders.")
        if model_error:
            st.caption(f"Last model loading error: {model_error}")
    if data_path:
        st.success(f"Dataset loaded: {data_path}")
    else:
        st.warning("No matching real dataset CSV detected. Synthetic demo data is being used for dashboard preview.")
        if data_error:
            st.caption(f"Last CSV loading error: {data_error}")
    st.write("Required input columns:", FEATURES)
    st.write("Required target column:", TARGET)
    with st.expander("Files detected by the app"):
        st.write("Model files:", [str(p) for p in model_files] or "None")
        st.write("CSV files:", [str(p) for p in csv_files] or "None")

elif page == "About":
    st.subheader("About AgriPredict AI")
    st.write(
        "AgriPredict AI is a machine-learning portfolio project for exploring "
        "agricultural records and estimating crop yield from a saved model."
    )
    st.markdown("""
    **Project workflow**
    1. Load and inspect agricultural records.
    2. Enter crop and environmental features.
    3. Run the saved model, when available.
    4. Review data quality and prediction outputs.

    **Responsible use**
    - Predictions are estimates, not guarantees of farm production.
    - Verify target units and feature definitions.
    - Document data source, model type, validation method and limitations.
    - Do not use a prediction as the sole basis for costly farming decisions.
    """)
    st.write("Features expected by the app:", FEATURES)
    st.write("Target column:", TARGET)

st.divider()
st.caption("AgriPredict AI • Streamlit portfolio dashboard • Confirm data source, units and model validity before publication.")
