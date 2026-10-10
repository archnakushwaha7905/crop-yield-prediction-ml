import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="AgriPredict | Crop Yield AI",
    page_icon="🌱",
    layout="wide"
)

# ---------- Custom Website Design ----------
st.markdown("""
<style>
.stApp {
    background: #f3f8f1;
    color: #193b2b;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.hero {
    background: linear-gradient(120deg, #123d2b, #2e8950);
    padding: 38px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin: 0;
}
.hero p {
    color: #e5f5e8;
    font-size: 17px;
    margin-top: 12px;
}
.tag {
    display: inline-block;
    background: #dff4df;
    color: #21633b;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
    margin-bottom: 12px;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #dce9d8;
    box-shadow: 0 5px 18px rgba(25,70,40,.06);
    margin-bottom: 15px;
}
.card h3 {
    color: #2d7947;
    font-size: 16px;
}
.metric {
    color: #164b2e;
    font-size: 27px;
    font-weight: 700;
}
.section-title {
    color: #174c30;
    font-size: 27px;
    font-weight: 700;
    margin: 22px 0 6px 0;
}
.section-sub {
    color: #627467;
    margin-bottom: 20px;
}
div.stButton > button {
    background: #217342;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 22px;
    font-weight: 600;
    width: 100%;
}
div.stButton > button:hover {
    background: #15532f;
    color: white;
    border: none;
}
div[data-testid="stForm"] {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #dce9d8;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 14px;
    border: 1px solid #dce9d8;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="tag">🌱 AI-POWERED AGRICULTURE</div>
    <h1>AgriPredict</h1>
    <p>Predict your crop yield with machine learning.
    Make smarter, data-driven farming decisions.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Load Dataset ----------
DATA_PATH = Path("dataset/crop_yield_dataset.csv")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def train_model(data_path, modified_time):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    target_candidates = [
        "yield (kg/ha)",
        "yield",
        "crop yield",
        "crop_yield",
        "yield_kg_per_ha"
    ]

    target = next(
        (c for c in df.columns
         if c.strip().lower() in target_candidates),
        None
    )

    if target is None:
        raise ValueError(
            "Yield column not found. Available columns: "
            + ", ".join(df.columns)
        )

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    if len(df) < 5:
        raise ValueError("The dataset contains too few valid records.")

    if len(df) > 20000:
        df = df.sample(20000, random_state=42)

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Remove columns that contain no useful values
    X = X.dropna(axis=1, how="all")

    # Remove ID-like columns
    X = X.drop(
        columns=[
            c for c in X.columns
            if c.strip().lower() in ["id", "index", "unnamed: 0"]
        ],
        errors="ignore"
    )

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    if not transformers:
        raise ValueError("No usable input features were found.")

    preprocessor = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions)
    }

    return model, X, y, target, metrics, numeric_cols, categorical_cols


if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Please place crop_yield_dataset.csv "
        "inside the dataset folder."
    )
    st.stop()

df = load_data(str(DATA_PATH))

# ---------- Overview Cards ----------
st.markdown('<div class="section-title">Dashboard Overview</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">An overview of your agricultural dataset.</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f'<div class="card"><h3>📊 Dataset Records</h3>'
        f'<div class="metric">{len(df):,}</div></div>',
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f'<div class="card"><h3>🧾 Available Features</h3>'
        f'<div class="metric">{len(df.columns) - 1}</div></div>',
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        '<div class="card"><h3>🤖 Prediction Engine</h3>'
        '<div class="metric">Random Forest</div></div>',
        unsafe_allow_html=True
    )

# ---------- Train Model ----------
st.markdown('<div class="section-title">Model Analytics</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Train the model and evaluate its performance.</div>',
    unsafe_allow_html=True
)

if st.button("⚙️ Train Machine Learning Model"):
    with st.spinner("Training model. Please wait..."):
        try:
            model, X, y, target, metrics, nums, cats = train_model(
                str(DATA_PATH),
                DATA_PATH.stat().st_mtime
            )

            st.session_state["model"] = model
            st.session_state["features"] = X
            st.session_state["target"] = target
            st.session_state["metrics"] = metrics
            st.session_state["numeric_cols"] = nums
            st.session_state["categorical_cols"] = cats

            st.success("Model trained successfully!")

        except Exception as e:
            st.error(f"Unable to train the model: {e}")

if "metrics" in st.session_state:
    m1, m2 = st.columns(2)

    with m1:
        st.metric(
            "R² Score",
            f"{st.session_state['metrics']['r2']:.3f}"
        )

    with m2:
        st.metric(
            "Mean Absolute Error",
            f"{st.session_state['metrics']['mae']:.2f}"
        )

# ---------- Prediction Section ----------
st.markdown('<div class="section-title">🌾 Predict Crop Yield</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Enter the agricultural conditions below '
    'to estimate crop yield.</div>',
    unsafe_allow_html=True
)

if "model" not in st.session_state:
    st.info("Train the machine learning model first to enable predictions.")
else:
    features = st.session_state["features"]
    numeric_cols = st.session_state["numeric_cols"]
    categorical_cols = st.session_state["categorical_cols"]

    with st.form("prediction_form"):
        st.markdown("### Agricultural Conditions")

        input_data = {}

        columns = st.columns(2)

        for i, col in enumerate(features.columns):
            with columns[i % 2]:
                if col in numeric_cols:
                    series = pd.to_numeric(
                        features[col], errors="coerce"
                    ).dropna()

                    if series.empty:
                        input_data[col] = 0.0
                    else:
                        default = float(series.median())
                        minimum = float(series.min())
                        maximum = float(series.max())

                        if minimum == maximum:
                            input_data[col] = st.number_input(
                                col.replace("_", " ").title(),
                                value=default
                            )
                        else:
                            input_data[col] = st.number_input(
                                col.replace("_", " ").title(),
                                min_value=minimum,
                                max_value=maximum,
                                value=default
                            )
                else:
                    options = (
                        features[col].dropna().astype(str)
                        .value_counts().index.tolist()
                    )

                    if not options:
                        options = ["Unknown"]

                    input_data[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options
                    )

        submitted = st.form_submit_button("🌱 Predict Yield")

    if submitted:
        try:
            input_df = pd.DataFrame(
                [input_data],
                columns=features.columns
            )

            result = st.session_state["model"].predict(input_df)[0]

            target = st.session_state["target"]
            unit = "kg/ha" if "kg/ha" in target.lower() or target.lower() == "yield" else "yield units"

            st.markdown("---")
            st.markdown("### Prediction Result")

            st.markdown(
                f"""
                <div class="hero">
                    <p>Estimated Crop Yield</p>
                    <h1>{result:,.2f} {unit}</h1>
                    <p>Estimate generated using the trained Random Forest model.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- Dataset Preview ----------
with st.expander("📁 Explore Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

# ---------- Footer ----------
st.markdown("""
<hr>
<div style="text-align:center; color:#617365; padding:12px;">
    <b>AgriPredict</b> · Crop Yield Prediction Using Machine Learning
    <br>Built with Python, Streamlit and Scikit-learn
</div>
""", unsafe_allow_html=True)