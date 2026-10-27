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

# ---------- Website Design ----------
st.markdown("""
<style>
.stApp {
    background-color: #f3f8f1;
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
    padding: 30px;
    border-radius: 20px;
    color: white !important;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 42px;
}

.hero p {
    color: white !important;
    font-size: 17px;
}

.tag {
    display: inline-block;
    background: #dff4df;
    color: #21633b !important;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
    margin-bottom: 12px;
}

.section-title {
    color: #174c30 !important;
    font-size: 27px;
    font-weight: bold;
    margin-top: 22px;
}

.section-sub {
    color: #435b49 !important;
    margin-bottom: 20px;
}

.card {
    background: #ffffff;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #dce9d8;
    margin-bottom: 15px;
}

.card h3 {
    color: #21633b !important;
    font-size: 16px;
}

.metric {
    color: #123d2b !important;
    -webkit-text-fill-color: #123d2b !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

.stMarkdown,
.stMarkdown p,
.stMarkdown li {
    color: #193b2b;
}

label,
.stSelectbox label,
.stNumberInput label {
    color: #193b2b !important;
}

input,
textarea {
    color: #193b2b !important;
    background-color: #ffffff !important;
}

/* FIX: Model Analytics numbers */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    padding: 18px !important;
    border-radius: 14px;
    border: 1px solid #dce9d8;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #193b2b !important;
    -webkit-text-fill-color: #193b2b !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div,
div[data-testid="stMetricValue"] span {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-size: 32px !important;
    font-weight: 900 !important;
    opacity: 1 !important;
    visibility: visible !important;
    text-shadow: none !important;
}

div[data-testid="stMetricDelta"] {
    color: #193b2b !important;
    -webkit-text-fill-color: #193b2b !important;
}

div[data-testid="stForm"] {
    background: #ffffff;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #dce9d8;
}

div.stButton > button {
    background: #217342;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 12px 22px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background: #15532f;
    color: white !important;
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

# ---------- Dataset Path ----------
DATA_PATH = (
    Path(__file__).resolve().parent
    / "dataset"
    / "crop_yield_dataset.csv"
)


@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    return data


@st.cache_resource
def train_model(data_path, modified_time):
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()

    target_names = {
        "yield (kg/ha)",
        "yield",
        "crop yield",
        "crop_yield",
        "yield_kg_per_ha"
    }

    target = next(
        (
            col for col in data.columns
            if col.strip().lower() in target_names
        ),
        None
    )

    if target is None:
        raise ValueError(
            "Yield column not found. Available columns: "
            + ", ".join(data.columns)
        )

    data[target] = pd.to_numeric(
        data[target], errors="coerce"
    )
    data = data.dropna(subset=[target])

    if len(data) < 5:
        raise ValueError(
            "Dataset has too few valid records."
        )

    if len(data) > 20000:
        data = data.sample(20000, random_state=42)

    X = data.drop(columns=[target]).copy()
    y = data[target].copy()

    X = X.dropna(axis=1, how="all")

    X = X.drop(
        columns=[
            col for col in X.columns
            if col.strip().lower()
            in ["id", "index", "unnamed: 0"]
        ],
        errors="ignore"
    )

    if X.shape[1] == 0:
        raise ValueError("No usable input features found.")

    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = [
        col for col in X.columns
        if col not in numeric_cols
    ]

    transformers = []

    if numeric_cols:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_cols
        ))

    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]),
            categorical_cols
        ))

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

    return (
        model, X, target, metrics,
        numeric_cols, categorical_cols
    )


# ---------- Load Dataset ----------
if not DATA_PATH.exists():
    st.error(
        "Dataset not found! Keep your CSV file here: "
        "dataset/crop_yield_dataset.csv"
    )
    st.stop()

try:
    df = load_data(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime
    )
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ---------- Dashboard Overview ----------
st.markdown(
    '<div class="section-title">Dashboard Overview</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-sub">'
    'Overview of your agricultural dataset.</div>',
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
        f'<div class="card"><h3>🧾 Dataset Columns</h3>'
        f'<div class="metric">{len(df.columns)}</div></div>',
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        '<div class="card"><h3>🤖 Prediction Engine</h3>'
        '<div class="metric">Random Forest</div></div>',
        unsafe_allow_html=True
    )


# ---------- Model Analytics ----------
st.markdown(
    '<div class="section-title">Model Analytics</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-sub">'
    'Train and evaluate the machine learning model.</div>',
    unsafe_allow_html=True
)

if st.button("⚙️ Train Machine Learning Model"):
    with st.spinner("Training model. Please wait..."):
        try:
            (
                model, X, target, metrics,
                nums, cats
            ) = train_model(
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


# ---------- Display Model Numbers ----------
if "metrics" in st.session_state:
    m1, m2 = st.columns(2)

    with m1:
        st.metric(
            label="R² Score",
            value=f"{st.session_state['metrics']['r2']:.3f}"
        )

    with m2:
        st.metric(
            label="Mean Absolute Error",
            value=f"{st.session_state['metrics']['mae']:.2f}"
        )


# ---------- Crop Yield Prediction ----------
st.markdown(
    '<div class="section-title">🌾 Predict Crop Yield</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-sub">'
    'Enter the agricultural conditions below '
    'to estimate crop yield.</div>',
    unsafe_allow_html=True
)

if "model" not in st.session_state:
    st.info(
        "Click 'Train Machine Learning Model' first "
        "to enable crop yield predictions."
    )
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
                label = col.replace("_", " ").title()

                if col in numeric_cols:
                    series = pd.to_numeric(
                        features[col], errors="coerce"
                    ).dropna()

                    if series.empty:
                        input_data[col] = st.number_input(
                            label, value=0.0, key=f"num_{col}"
                        )
                    else:
                        default = float(series.median())
                        minimum = float(series.min())
                        maximum = float(series.max())

                        if minimum == maximum:
                            input_data[col] = st.number_input(
                                label,
                                value=default,
                                key=f"num_{col}"
                            )
                        else:
                            input_data[col] = st.number_input(
                                label,
                                min_value=minimum,
                                max_value=maximum,
                                value=default,
                                key=f"num_{col}"
                            )

                elif col in categorical_cols:
                    options = (
                        features[col].dropna().astype(str)
                        .value_counts().index.tolist()
                    )

                    if not options:
                        options = ["Unknown"]

                    input_data[col] = st.selectbox(
                        label, options, key=f"cat_{col}"
                    )

        submitted = st.form_submit_button("🌱 Predict Yield")

    if submitted:
        try:
            input_df = pd.DataFrame(
                [input_data],
                columns=features.columns
            )

            result = st.session_state["model"].predict(
                input_df
            )[0]

            target = st.session_state["target"]

            if "kg/ha" in target.lower():
                unit = "kg/ha"
            elif target.lower() == "yield":
                unit = "yield units"
            else:
                unit = target

            st.markdown("---")
            st.markdown("### Prediction Result")

            st.markdown(
                f"""
                <div class="hero">
                    <p>Estimated Crop Yield</p>
                    <h1>{result:,.2f} {unit}</h1>
                    <p>Estimate generated using the trained
                    Random Forest model.</p>
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
<div style="text-align:center; color:#435b49; padding:12px;">
    <b>AgriPredict</b> · Crop Yield Prediction Using Machine Learning
    <br>Built with Python, Streamlit and Scikit-learn
</div>
""", unsafe_allow_html=True)