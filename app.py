import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download 


st.set_page_config(
    page_title="AgriPredict AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "crop_yield.csv")

FEATURES = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="kShreya06/agripredict-model",
        filename="agripredict_model.pkl"
    )
    return joblib.load(model_path)

@st.cache_data
def load_dataset():
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "crop_yield.csv"
    )

    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)

    return pd.DataFrame()
try:
    model = load_model()
except Exception as error:
    st.error("Model load nahi ho pa raha.")
    st.exception(error)
    st.stop()


df = load_dataset()


model_features = list(
    getattr(model, "feature_names_in_", FEATURES)
)

if set(model_features) != set(FEATURES):
    st.error(
        "The model input features do not match this app. "
        f"Expected: {model_features}"
    )
    st.stop()


if not df.empty:
    missing_columns = [
        column for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Dataset is missing columns: {missing_columns}"
        )
        st.stop()
# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def numeric_values(column, fallback):
    if column not in df.columns:
        return fallback

    values = pd.to_numeric(
        df[column], errors="coerce"
    ).dropna()

    if values.empty:
        return fallback

    return values


def get_categories(column, fallback):
    if column not in df.columns:
        return fallback

    values = df[column].dropna().unique().tolist()
    values = sorted(values, key=lambda value: str(value))

    return values if values else fallback


def format_number(value):
    return f"{value:,.2f}"


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🌾 AgriPredict AI")
    st.caption("Crop Yield Prediction Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Predict Yield",
            "Dataset Explorer",
            "Prediction History",
            "About Project"
        ]
    )

    st.divider()
    st.caption("Machine Learning • Streamlit • Python")

# --------------------------------------------------
# 6. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("🌱 Agriculture Intelligence Dashboard")
    st.write(
        "Explore agricultural data and estimate crop yield "
        "using the trained machine-learning pipeline."
    )

    if df.empty:
        st.warning(
            "Dataset not found. Dataset insights are unavailable."
        )
    else:
        total_records = len(df)

        total_crops = (
            df["Crop"].nunique()
            if "Crop" in df.columns else 0
        )

        total_states = (
            df["State"].nunique()
            if "State" in df.columns else 0
        )

        total_seasons = (
            df["Season"].nunique()
            if "Season" in df.columns else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Dataset Records", f"{total_records:,}")
        col2.metric("Unique Crops", total_crops)
        col3.metric("States", total_states)
        col4.metric("Seasons", total_seasons)

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Crop Distribution")

            if "Crop" in df.columns:
                crop_counts = (
                    df["Crop"].value_counts().head(10)
                )
                st.bar_chart(crop_counts)

        with right:
            st.subheader("Season Distribution")

            if "Season" in df.columns:
                season_counts = df["Season"].value_counts()
                st.bar_chart(season_counts)

        if "Crop_Year" in df.columns:
            st.subheader("Records by Year")

            yearly_counts = (
                df.groupby("Crop_Year")
                .size()
                .sort_index()
            )

            st.line_chart(yearly_counts)

        st.caption(
            "Charts describe the supplied dataset. "
            "They do not by themselves establish model accuracy."
        )

# --------------------------------------------------
# 7. CROP YIELD PREDICTION
# --------------------------------------------------

elif page == "Predict Yield":

    st.title("🔮 Predict Crop Yield")
    st.write(
        "Enter agricultural conditions to generate a model prediction."
    )

    if df.empty:
        st.error(
            "The dataset is required to populate valid crop categories."
        )
        st.stop()

    crop_options = get_categories("Crop", ["Rice"])
    season_options = get_categories("Season", ["Kharif"])
    state_options = get_categories("State", ["Assam"])

    years = numeric_values("Crop_Year", [2020])
    areas = numeric_values("Area", [1000])
    rainfalls = numeric_values("Annual_Rainfall", [1000])
    fertilizers = numeric_values("Fertilizer", [1000])
    pesticides = numeric_values("Pesticide", [100])

    with st.form("prediction_form"):

        st.subheader("Crop Information")

        c1, c2 = st.columns(2)

        with c1:
            crop = st.selectbox("Crop", crop_options)

            crop_year = st.number_input(
                "Crop Year",
                min_value=int(years.min()),
                max_value=max(
                    int(years.max()), int(years.min())
                ),
                value=int(years.max()),
                step=1
            )

            season = st.selectbox("Season", season_options)

            state = st.selectbox("State", state_options)

        with c2:
            area = st.number_input(
                "Cultivated Area",
                min_value=0.000001,
                value=float(areas.median()),
                step=1.0
            )

            rainfall = st.number_input(
                "Annual Rainfall",
                min_value=0.0,
                value=float(rainfalls.median()),
                step=10.0
            )

            fertilizer = st.number_input(
                "Fertilizer",
                min_value=0.0,
                value=float(fertilizers.median()),
                step=100.0
            )

            pesticide = st.number_input(
                "Pesticide",
                min_value=0.0,
                value=float(pesticides.median()),
                step=1.0
            )

        submitted = st.form_submit_button(
            "🌾 Predict Crop Yield",
            use_container_width=True
        )

    if submitted:

        if area <= 0:
            st.error("Area must be greater than zero.")

        elif any(
            not np.isfinite(value)
            for value in [
                area, rainfall, fertilizer, pesticide
            ]
        ):
            st.error("Please enter valid numeric values.")

        else:
            input_values = {
                "Crop": crop,
                "Crop_Year": crop_year,
                "Season": season,
                "State": state,
                "Area": area,
                "Annual_Rainfall": rainfall,
                "Fertilizer": fertilizer,
                "Pesticide": pesticide
            }

            input_df = pd.DataFrame(
                [input_values],
                columns=model_features
            )

            try:
                prediction = float(
                    np.asarray(
                        model.predict(input_df)
                    ).reshape(-1)[0]
                )

                if not np.isfinite(prediction):
                    st.error(
                        "The model returned an invalid prediction."
                    )
                else:
                    st.success("Prediction generated successfully!")

                    st.metric(
                        "Predicted Crop Yield",
                        format_number(prediction)
                    )

                    st.caption(
                        "The unit must be verified from the training "
                        "dataset and target definition before practical use."
                    )

                    if "prediction_history" not in st.session_state:
                        st.session_state.prediction_history = []

                    record = {
                        **input_values,
                        "Predicted_Yield": prediction
                    }

                    st.session_state.prediction_history.append(
                        record
                    )

                    with st.expander("View input details"):
                        st.dataframe(
                            input_df,
                            use_container_width=True
                        )

            except Exception as error:
                st.error(
                    "Prediction failed. Check model compatibility "
                    "and input categories."
                )
                st.exception(error)

# --------------------------------------------------
# 8. DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.title("📊 Dataset Explorer")

    if df.empty:
        st.warning("Dataset is unavailable.")
    else:
        st.write("Dataset shape:", df.shape)

        st.subheader("Preview")
        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.subheader("Missing Values")
        missing = df.isna().sum().rename("Missing Values")
        st.dataframe(missing, use_container_width=True)

        st.subheader("Numerical Summary")
        st.dataframe(
            df.describe(include="all").transpose(),
            use_container_width=True
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Dataset CSV",
            data=csv_data,
            file_name="crop_yield_data.csv",
            mime="text/csv"
        )

# --------------------------------------------------
# 9. PREDICTION HISTORY
# --------------------------------------------------

elif page == "Prediction History":

    st.title("🧾 Prediction History")

    history = st.session_state.get(
        "prediction_history", []
    )

    if not history:
        st.info(
            "No predictions in this session yet. "
            "Open Predict Yield to generate one."
        )
    else:
        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.download_button(
            "Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Session History"):
            st.session_state.prediction_history = []
            st.rerun()

# --------------------------------------------------
# 10. ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.title("ℹ️ About AgriPredict AI")

    st.markdown("""
    AgriPredict is a machine-learning application designed
    to estimate crop yield from agricultural input features.

    **Input features**
    - Crop
    - Crop year
    - Season
    - State
    - Cultivated area
    - Annual rainfall
    - Fertilizer
    - Pesticide

    **Important limitations**
    - Prediction quality must be established using held-out test data.
    - Output units depend on the original target variable.
    - Predictions are estimates, not guaranteed farming outcomes.
    - Model performance may vary across locations and years.
    """)

    st.write("Model type:", type(model).__name__)
    st.write("Dataset available:", not df.empty)

    if hasattr(model, "n_features_in_"):
        st.write(
            "Model input feature count:",
            model.n_features_in_
        )

# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "AgriPredict AI • Educational and analytical prototype"
)