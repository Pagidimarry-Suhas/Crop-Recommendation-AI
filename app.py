import os, joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; color: #1a3a1a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #74c69d !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
    border-left: 5px solid #40916c;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
.metric-card h3 { color: #1b4332; margin: 0; font-size: 0.95rem; font-family: 'DM Sans'; font-weight: 500; }
.metric-card p  { color: #1b4332; margin: 4px 0 0; font-size: 1.6rem; font-weight: 700; }

/* Prediction box */
.prediction-box {
    background: linear-gradient(135deg, #1b4332, #40916c);
    color: white;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(27,67,50,0.35);
    margin: 16px 0;
}
.prediction-box .crop-name {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    text-transform: capitalize;
    letter-spacing: 1px;
}
.prediction-box .confidence {
    font-size: 1.1rem;
    opacity: 0.85;
    margin-top: 8px;
}

/* Top-N cards */
.alt-card {
    background: #f0faf4;
    border: 1px solid #b7e4c7;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stButton > button {
    background: linear-gradient(135deg, #2d6a4f, #52b788);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 36px;
    font-size: 1.05rem;
    font-family: 'DM Sans';
    font-weight: 500;
    width: 100%;
    cursor: pointer;
    transition: transform 0.15s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(64,145,108,0.4); }
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ───────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model   = joblib.load(os.path.join(BASE, "models", "best_model.pkl"))
    scaler  = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
    le      = joblib.load(os.path.join(BASE, "models", "label_encoder.pkl"))
    meta    = joblib.load(os.path.join(BASE, "models", "metadata.pkl"))
    df      = pd.read_csv(os.path.join(BASE, "data", "crop_data.csv"))
    return model, scaler, le, meta, df

model, scaler, le, meta, df = load_artifacts()
FEATURES = meta["feature_names"]

# ── Sidebar Inputs ───────────────────────────────────────────────────────────
st.sidebar.markdown("## 🌿 Soil & Weather Inputs")
st.sidebar.markdown("Adjust the sliders to match your field conditions.")
st.sidebar.markdown("---")

N         = st.sidebar.slider("Nitrogen (N) — kg/ha",          0,   140, 60)
P         = st.sidebar.slider("Phosphorus (P) — kg/ha",        5,   145, 45)
K         = st.sidebar.slider("Potassium (K) — kg/ha",         5,   205, 40)
temp      = st.sidebar.slider("Temperature — °C",              8.0, 44.0, 25.0, step=0.5)
humidity  = st.sidebar.slider("Humidity — %",                  14.0,100.0, 65.0, step=0.5)
ph        = st.sidebar.slider("Soil pH",                        3.5,  9.5,  6.5, step=0.1)
rainfall  = st.sidebar.slider("Rainfall — mm",                  20.0,300.0,100.0, step=5.0)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Best Model:** {meta['best_model_name']}")
res = meta["results_summary"]
for name, vals in res.items():
    st.sidebar.markdown(f"• **{name}** — Test Acc: `{vals['test_acc']:.3f}`")

# ── Header ───────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("<div style='font-size:4rem; padding-top:8px'>🌾</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin-bottom:0'>Crop Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#555; margin-top:2px'>AI-powered recommendations based on soil nutrients & weather</p>", unsafe_allow_html=True)

st.markdown("---")

# ── Predict ──────────────────────────────────────────────────────────────────
input_arr = np.array([[N, P, K, temp, humidity, ph, rainfall]])

tab_pred, tab_eda, tab_models, tab_data = st.tabs([
    "🔮 Prediction", "📊 EDA & Insights", "🤖 Model Performance", "📋 Dataset"
])

# ──────────────────────────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("### Predict the Best Crop for Your Field")
    st.markdown("Adjust the sliders in the sidebar, then click **Recommend**.")

    if st.button("🌱 Get Crop Recommendation"):
        # Best model uses raw features (Random Forest)
        proba = model.predict_proba(input_arr)[0]
        top_indices = np.argsort(proba)[::-1][:5]
        top_crops   = [(le.classes_[i], proba[i]) for i in top_indices]

        best_crop, best_conf = top_crops[0]

        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size:1rem; opacity:0.7; letter-spacing:2px; text-transform:uppercase">Recommended Crop</div>
            <div class="crop-name">🌿 {best_crop.title()}</div>
            <div class="confidence">Confidence: {best_conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Show top-5
        st.markdown("#### Alternative Recommendations")
        for i, (crop, conf) in enumerate(top_crops[1:], 2):
            bar = "█" * int(conf * 30)
            st.markdown(f"""
            <div class="alt-card">
                <span><b>#{i}</b> &nbsp; {crop.title()}</span>
                <span style="color:#2d6a4f; font-weight:600">{conf*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bar chart
        st.markdown("#### Confidence Distribution (Top 5)")
        fig, ax = plt.subplots(figsize=(8, 3))
        crops_list = [c.title() for c, _ in top_crops]
        confs_list = [c * 100 for _, c in top_crops]
        colors_bar = ["#1b4332"] + ["#74c69d"] * 4
        bars = ax.barh(crops_list[::-1], confs_list[::-1], color=colors_bar[::-1], height=0.55)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 105)
        for bar, val in zip(bars, confs_list[::-1]):
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=10, color="#1b4332", fontweight="bold")
        ax.set_facecolor("#f8fdf9"); fig.patch.set_facecolor("#f8fdf9")
        ax.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Input summary
        st.markdown("#### Your Input Summary")
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("🧪 Nitrogen", f"{N} kg/ha"),
            ("🧪 Phosphorus", f"{P} kg/ha"),
            ("🧪 Potassium", f"{K} kg/ha"),
            ("🌡️ Temperature", f"{temp}°C"),
            ("💧 Humidity", f"{humidity}%"),
            ("⚗️ Soil pH", str(ph)),
            ("🌧️ Rainfall", f"{rainfall} mm"),
        ]
        for i, (label, val) in enumerate(metrics):
            col = [c1, c2, c3, c4][i % 4]
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{label}</h3><p>{val}</p>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("👈  Adjust the sliders in the sidebar, then click **Get Crop Recommendation**.")

# ──────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown("### Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Crop Distribution")
        crop_counts = df["label"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_bar = plt.cm.Greens(np.linspace(0.4, 0.9, len(crop_counts)))[::-1]
        crop_counts.plot(kind="barh", ax=ax, color=colors_bar)
        ax.set_xlabel("Count"); ax.set_ylabel("")
        ax.set_facecolor("#f8fdf9"); fig.patch.set_facecolor("#f8fdf9")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 5))
        corr = df[FEATURES].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="Greens",
                    ax=ax, linewidths=0.5, annot_kws={"size": 9}, square=True)
        ax.set_title("Feature Correlations", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("#### Feature Distributions by Crop")
    selected_feat = st.selectbox("Select Feature:", FEATURES, index=0)
    top_crops_eda = df["label"].value_counts().head(10).index.tolist()
    sub = df[df["label"].isin(top_crops_eda)]

    fig, ax = plt.subplots(figsize=(12, 4))
    for crop in top_crops_eda:
        vals = sub[sub["label"] == crop][selected_feat]
        ax.plot([], [])  # cycle color
    palette = sns.color_palette("Greens_d", len(top_crops_eda))
    for i, crop in enumerate(top_crops_eda):
        vals = sub[sub["label"] == crop][selected_feat]
        ax.hist(vals, bins=18, alpha=0.6, label=crop.title(), color=palette[i])
    ax.set_xlabel(selected_feat); ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of '{selected_feat}' across top 10 crops")
    ax.legend(fontsize=8, ncol=2)
    ax.set_facecolor("#f8fdf9"); fig.patch.set_facecolor("#f8fdf9")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("#### Box Plots — Feature vs Crop")
    feat2 = st.selectbox("Select Feature for Box Plot:", FEATURES, index=3, key="box_feat")
    fig, ax = plt.subplots(figsize=(14, 5))
    grouped = [df[df["label"] == c][feat2].values for c in sorted(df["label"].unique())]
    bp = ax.boxplot(grouped, patch_artist=True, notch=False,
                    boxprops=dict(facecolor="#74c69d", color="#1b4332"),
                    medianprops=dict(color="#1b4332", linewidth=2),
                    whiskerprops=dict(color="#40916c"),
                    capprops=dict(color="#40916c"),
                    flierprops=dict(marker="o", color="#95d5b2", alpha=0.5))
    ax.set_xticks(range(1, len(df["label"].unique()) + 1))
    ax.set_xticklabels(sorted(df["label"].unique()), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(feat2); ax.set_title(f"{feat2} Distribution by Crop")
    ax.set_facecolor("#f8fdf9"); fig.patch.set_facecolor("#f8fdf9")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
with tab_models:
    st.markdown("### Model Performance")

    # Summary table
    st.markdown("#### Accuracy Comparison")
    rows = []
    for name, vals in meta["results_summary"].items():
        rows.append({
            "Model": name,
            "CV Accuracy": f"{vals['cv_acc']*100:.2f}%",
            "Test Accuracy": f"{vals['test_acc']*100:.2f}%",
            "Best Params": str(vals['params'])
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        img_path = os.path.join(BASE, "reports", "model_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Model Accuracy Comparison", use_container_width=True)

    with col2:
        img_path = os.path.join(BASE, "reports", "feature_importance.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Feature Importances — Random Forest", use_container_width=True)

    img_path = os.path.join(BASE, "reports", "confusion_matrix.png")
    if os.path.exists(img_path):
        st.markdown(f"#### Confusion Matrix — {meta['best_model_name']}")
        st.image(img_path, use_container_width=True)

    st.markdown("#### About the Models")
    with st.expander("🌲 Random Forest"):
        st.markdown("""
        **Random Forest** is an ensemble of decision trees that trains on random subsets of features.
        - Robust to outliers & non-linear patterns
        - Provides feature importances natively
        - Tuned via `n_estimators`, `max_depth`, `min_samples_split`
        """)
    with st.expander("🤖 Support Vector Machine (SVM)"):
        st.markdown("""
        **SVM** finds the optimal hyperplane that maximises the margin between classes.
        - Effective in high-dimensional spaces
        - Kernel trick enables non-linear boundaries
        - Tuned via `C`, `kernel`, `gamma`
        """)
    with st.expander("📍 K-Nearest Neighbours (KNN)"):
        st.markdown("""
        **KNN** classifies a sample by majority vote among its K nearest neighbours.
        - Instance-based, no training phase
        - Sensitive to feature scale (uses StandardScaler)
        - Tuned via `n_neighbors`, `weights`, `metric`
        """)

# ──────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples",  len(df))
    c2.metric("Features",       len(FEATURES))
    c3.metric("Crop Classes",   df["label"].nunique())

    st.markdown("#### Sample Records")
    st.dataframe(df.sample(50, random_state=1).reset_index(drop=True),
                 use_container_width=True, height=400)

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df[FEATURES].describe().T.style.format("{:.2f}"),
                 use_container_width=True)
