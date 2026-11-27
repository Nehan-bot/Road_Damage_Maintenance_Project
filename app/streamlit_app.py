
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RoadSight AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# DATABASE
# ============================================================

try:
    from database.db_manager import (
        initialize_database,
        save_prediction,
        get_prediction_history,
        get_prediction_details,
    )
except ImportError as error:
    st.error("Could not import database/db_manager.py")
    st.exception(error)
    st.stop()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

YOLO_MODEL_PATH = (
    ROOT_DIR
    / "experiments"
    / "yolo_final"
    / "yolo26s_640_final"
    / "weights"
    / "best.pt"
)

RF_MODEL_PATH = (
    ROOT_DIR
    / "experiments"
    / "random_forest"
    / "random_forest_priority.joblib"
)

CLASS_NAMES = {
    0: "longitudinal crack",
    1: "transverse crack",
    2: "alligator crack",
    3: "pothole",
}

FEATURE_COLUMNS = [
    "total_detections",
    "longitudinal crack_count",
    "longitudinal crack_mean_confidence",
    "transverse crack_count",
    "transverse crack_mean_confidence",
    "alligator crack_count",
    "alligator crack_mean_confidence",
    "pothole_count",
    "pothole_mean_confidence",
]


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "System"

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.25

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


NAV_PAGES = ["Home", "Dashboard", "Image Analysis", "Prediction History", "Model & Connection"]
if "nav_choice" not in st.session_state:
    st.session_state.nav_choice = st.session_state.page

def _navigate_to(destination):
    # Streamlit invokes callbacks before rendering widgets on the next rerun.
    st.session_state.page = destination
    st.session_state.nav_choice = destination

def _sync_navigation():
    st.session_state.page = st.session_state.nav_choice


# ============================================================
# THEME STYLING
# ============================================================

def apply_theme(theme):
    light = """
      --page-bg:#f3f6fb; --surface:#ffffff; --surface-2:#eaf0f7;
      --text:#15263d; --muted:#52657c; --border:#d9e2ee;
      --accent:#2563eb; --accent-strong:#174ea6;
    """
    dark = """
      --page-bg:#0b1220; --surface:#111d2e; --surface-2:#1a2a40;
      --text:#f3f7fc; --muted:#b8c7d9; --border:#2b3c52;
      --accent:#60a5fa; --accent-strong:#2563eb;
    """
    if theme == "Light":
        palette = f".stApp {{{light}}}"
        scheme = "light"
    elif theme == "Dark":
        palette = f".stApp {{{dark}}}"
        scheme = "dark"
    else:
        palette = f"""
        @media (prefers-color-scheme:light) {{ .stApp {{{light}}} }}
        @media (prefers-color-scheme:dark) {{ .stApp {{{dark}}} }}
        """
        scheme = "light dark"

    css = f"""
    <style>
    :root {{ color-scheme:{scheme}; }}
    {palette}
    .stApp {{background:var(--page-bg)!important;color:var(--text)!important;}}
    [data-testid="stSidebar"] {{background:#0d1b2d!important;border-right:1px solid #23364c;}}
    [data-testid="stSidebar"] * {{color:#eef4fb;}}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{color:#c4d1e0;}}
    [data-testid="stSidebar"] hr {{border-color:#2b3d53;}}
    .block-container {{max-width:1540px;padding-top:1.25rem;padding-bottom:2.3rem;}}
    h1,h2,h3,h4,h5,p,li,label,[data-testid="stMarkdownContainer"] {{color:var(--text);}}
    [data-testid="stCaptionContainer"],.rg-muted {{color:var(--muted)!important;}}
    [data-testid="stMetric"] {{background:var(--surface)!important;border:1px solid var(--border)!important;
      border-radius:16px!important;padding:16px 18px!important;box-shadow:0 2px 8px rgba(20,40,70,.04);}}
    [data-testid="stMetricLabel"],[data-testid="stMetricValue"] {{color:var(--text)!important;}}
    .rg-kicker {{color:var(--accent);font-weight:800;letter-spacing:.11em;font-size:.76rem;text-transform:uppercase;}}
    .rg-hero-title {{font-size:clamp(2.5rem,4.6vw,4.4rem);line-height:1.02;font-weight:850;
      letter-spacing:-.045em;color:var(--text);margin:.65rem 0 .5rem;}}
    .rg-subtitle {{font-size:clamp(1.05rem,1.5vw,1.3rem);color:var(--muted);font-weight:550;line-height:1.45;}}
    .rg-lead {{font-size:1.04rem;line-height:1.75;color:var(--muted);}}
    .rg-section {{background:var(--surface);border:1px solid var(--border);border-radius:17px;
      padding:1.15rem 1.25rem;height:100%;}}
    .rg-section h3 {{font-size:1.12rem;margin:.35rem 0 .55rem;}}
    .rg-section p {{color:var(--muted);line-height:1.65;margin-bottom:.15rem;}}
    .rg-stat {{background:var(--surface-2);border:1px solid var(--border);border-radius:14px;
      padding:1rem;text-align:center;height:100%;}}
    .rg-stat-number {{font-size:1.55rem;font-weight:800;color:var(--text);}}
    .rg-stat-label {{font-size:.88rem;color:var(--muted);line-height:1.45;}}
    .rg-theme-label {{text-align:right;color:var(--muted);font-size:.72rem;font-weight:800;letter-spacing:.12em;margin-bottom:.3rem;}}
    div.stButton>button {{border-radius:11px!important;min-height:2.75rem;font-weight:650!important;
      border:1px solid var(--border)!important;transition:transform .12s ease,box-shadow .12s ease;}}
    div.stButton>button:hover {{transform:translateY(-1px);box-shadow:0 5px 16px rgba(30,65,110,.12);}}
    div.stButton>button[kind="primary"] {{background:var(--accent-strong)!important;color:#fff!important;border-color:var(--accent-strong)!important;}}
    [data-testid="stRadio"] label {{color:var(--text)!important;background:var(--surface)!important;
      border:1px solid var(--border)!important;border-radius:999px!important;padding:.35rem .7rem!important;}}
    [data-testid="stRadio"] label p {{font-size:.86rem;font-weight:650;}}
    [data-testid="stFileUploader"],[data-testid="stExpander"],[data-testid="stAlert"] {{
      background:var(--surface)!important;color:var(--text)!important;border-color:var(--border)!important;border-radius:14px!important;}}
    [data-testid="stDataFrame"],[data-testid="stTable"] {{border:1px solid var(--border);border-radius:12px;overflow:hidden;}}
    [data-testid="stTextInput"] input,[data-testid="stSelectbox"] div,[data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input {{background:var(--surface)!important;color:var(--text)!important;border-color:var(--border)!important;}}
    [data-testid="stHeader"] {{background:transparent!important;}}
    .rg-footer {{color:var(--muted);font-size:.82rem;text-align:center;}}
    @media(max-width:900px) {{.block-container{{padding-left:1rem;padding-right:1rem;}}.rg-hero-title{{font-size:2.5rem;}}}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

database_connected = False
database_error = None

try:
    initialize_database()
    database_connected = True
except Exception as error:
    database_error = str(error)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    rf_model = joblib.load(str(RF_MODEL_PATH))
    return yolo_model, rf_model


models_available = YOLO_MODEL_PATH.exists() and RF_MODEL_PATH.exists()
models_loaded = False
model_error = None
yolo_model = None
rf_model = None

# Keep the Home page quick: model files are loaded on first image analysis.
if st.session_state.page == "Image Analysis" and models_available:
    try:
        yolo_model, rf_model = load_models()
        models_loaded = True
    except Exception as error:
        model_error = str(error)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(result):
    counts = {
        name: 0 for name in CLASS_NAMES.values()
    }

    confidence_sums = {
        name: 0.0 for name in CLASS_NAMES.values()
    }

    detections = result.boxes

    if detections is not None and len(detections) > 0:
        class_ids = detections.cls.cpu().numpy().astype(int)
        confidences = detections.conf.cpu().numpy()

        for class_id, confidence in zip(
            class_ids, confidences
        ):
            class_name = CLASS_NAMES.get(class_id)

            if class_name is None:
                continue

            counts[class_name] += 1
            confidence_sums[class_name] += float(confidence)

    features = {
        "total_detections": sum(counts.values())
    }

    for class_name in CLASS_NAMES.values():
        count = counts[class_name]

        mean_confidence = (
            confidence_sums[class_name] / count
            if count > 0
            else 0.0
        )

        features[f"{class_name}_count"] = count
        features[
            f"{class_name}_mean_confidence"
        ] = mean_confidence

    feature_df = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    return feature_df, counts


# ============================================================
# SIDEBAR — compact branded navigation
with st.sidebar:
    logo_path = ROOT_DIR / "app" / "assets" / "roadsight_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=86)
    st.markdown(
        "<div style='font-size:1.55rem;font-weight:850;letter-spacing:-.04em;"
        "color:#f3f7fc;margin-top:-.4rem'>RoadSight AI</div>",
        unsafe_allow_html=True,
    )
    st.caption("Road Damage Intelligence")
    st.divider()

    st.radio(
        "Workspace",
        NAV_PAGES,
        key="nav_choice",
        on_change=_sync_navigation,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### System Status")
    status_yolo = "Model files found" if models_available else "Model files missing"
    status_rf = "Model files found" if models_available else "Model files missing"
    status_db = "Connected" if database_connected else "Connection error"
    for label, status in [
        ("YOLO26s Detector", status_yolo),
        ("Random Forest", status_rf),
        ("SQLite Database", status_db),
    ]:
        st.markdown(
            "<div style='background:#14263a;border:1px solid #2b4058;"
            "border-radius:12px;padding:.65rem .8rem;margin:.35rem 0'>"
            "<span style='color:#65d69a'>●</span> "
            f"<span style='color:#edf4fb;font-size:.9rem'>{label}</span>"
            f"<br><span style='color:#a9bad0;font-size:.78rem'>{status}</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("M.Tech Research Prototype")
    st.caption("Image-based road damage decision support")


# ============================================================
# COMMON HEADER — appearance selector sits below Streamlit's toolbar
header_left, header_right = st.columns([5.1, 2.0], vertical_alignment="center")
with header_left:
    st.markdown(
        "<div style='font-size:1.55rem;font-weight:850;letter-spacing:-.04em;"
        "margin:.15rem 0;color:var(--text)'>RoadSight AI</div>"
        "<div style='font-size:.92rem;color:var(--muted);margin-top:-.25rem'>"
        "Road Damage Intelligence &amp; Maintenance Prioritization</div>",
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown("<div class='rg-theme-label'>APPEARANCE</div>", unsafe_allow_html=True)
    st.radio(
        "Appearance",
        ["System", "Light", "Dark"],
        key="theme_choice",
        horizontal=True,
        label_visibility="collapsed",
    )

apply_theme(st.session_state.theme_choice)
page = st.session_state.page

if page != "Home":
    st.caption(
        "Research prototype: visual priority estimates are not verified "
        "real-world maintenance urgency."
    )


# ============================================================
# LOAD HISTORY FOR DASHBOARD AND HISTORY PAGES
# ============================================================

history = pd.DataFrame()

if database_connected:
    try:
        history = get_prediction_history(limit=100)
    except Exception:
        history = pd.DataFrame()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Home":
    hero_left, hero_right = st.columns([1.02, 1.15], gap="large")
    with hero_left:
        with st.container(border=True):
            st.markdown('<span class="rg-kicker">AI for safer roads</span>', unsafe_allow_html=True)
            st.markdown('<div class="rg-hero-title">RoadSight AI</div>', unsafe_allow_html=True)
            st.markdown('<div class="rg-subtitle">Road Damage Intelligence &amp; Maintenance Prioritization</div>', unsafe_allow_html=True)
            st.write(
                "An AI-assisted research system that identifies common road damage "
                "from images and estimates maintenance priority to support safer, "
                "more durable and cost-aware road infrastructure."
            )
            st.button(
                "Get Started — Analyse Image  →",
                type="primary",
                use_container_width=True,
                on_click=_navigate_to,
                args=("Image Analysis",),
            )
            st.button(
                "Go to Dashboard",
                use_container_width=True,
                on_click=_navigate_to,
                args=("Dashboard",),
            )
    with hero_right:
        hero_image = ROOT_DIR / "app" / "assets" / "road_safety_hero.jpg"
        if hero_image.exists():
            st.image(str(hero_image), use_container_width=True)
        else:
            st.markdown(
                '<div class="rg-hero"><span class="rg-kicker">Computer vision</span>'
                '<h2>Detect damage.<br>Prioritize maintenance.</h2>'
                '<p class="rg-muted">Upload a road image to inspect model detections.</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("## Why road safety matters")
    safety1, safety2, safety3 = st.columns(3, gap="medium")
    with safety1:
        st.markdown('<div class="rg-section"><h3>Safer journeys</h3><p class="rg-muted">Road defects can contribute to loss of vehicle control, crashes and injuries. Timely inspection and maintenance help reduce avoidable hazards.</p></div>', unsafe_allow_html=True)
    with safety2:
        st.markdown('<div class="rg-section"><h3>Global challenge</h3><p class="rg-muted">Road traffic injuries remain a major global public-health concern. The World Health Organization estimates about 1.19 million road deaths annually.</p></div>', unsafe_allow_html=True)
    with safety3:
        st.markdown('<div class="rg-section"><h3>Our approach</h3><p class="rg-muted">RoadSight AI combines object detection and a priority classifier to help organize image-based observations for further review.</p></div>', unsafe_allow_html=True)

    st.markdown("## How RoadSight AI works")
    step1, step2, step3, step4 = st.columns(4, gap="medium")
    for col, number, title, desc in [
        (step1, "01", "Input image", "Upload a road photograph."),
        (step2, "02", "Detect damage", "YOLO26s detects cracks and potholes."),
        (step3, "03", "Estimate priority", "Random Forest predicts a priority category from detection features."),
        (step4, "04", "Review & act", "Use the output as a decision-support cue for inspection."),
    ]:
        with col:
            st.markdown(f'<div class="rg-section"><span class="rg-kicker">{number}</span><h3>{title}</h3><p class="rg-muted">{desc}</p></div>', unsafe_allow_html=True)

    st.info("Research limitation: image-based predictions may miss defects or misclassify severity. They should be checked by qualified personnel and are not a substitute for field inspection.")
    st.markdown(
        "Global statistic: the World Health Organization estimates about "
        "1.19 million road traffic deaths each year. "
        "[Source: WHO road traffic injuries fact sheet](https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries).",
        help="The statistic is a global estimate and is not a result of this project."
    )

elif page == "Dashboard":

    st.subheader("Project Dashboard")
    st.write(
        "Monitor model readiness and review recent road "
        "damage analysis activity."
    )

    total_records = len(history) if not history.empty else 0

    high_count = 0
    medium_count = 0
    low_count = 0

    if (
        not history.empty
        and "predicted_priority" in history.columns
    ):
        priority = history["predicted_priority"]

        high_count = int((priority == "High").sum())
        medium_count = int((priority == "Medium").sum())
        low_count = int((priority == "Low").sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Recent Analyses", total_records)
    col2.metric("High Priority", high_count)
    col3.metric("Medium Priority", medium_count)
    col4.metric("Low Priority", low_count)

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("System Status")

        status_data = pd.DataFrame(
            {
                "Component": [
                    "YOLO26s Detector",
                    "Random Forest",
                    "SQLite Database",
                ],
                "Status": [
                    "Files found" if models_available else "Unavailable",
                    "Files found" if models_available else "Unavailable",
                    "Connected" if database_connected else "Error",
                ],
            }
        )

        st.dataframe(
            status_data,
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.subheader("Priority Distribution")

        if total_records > 0:
            priority_data = pd.DataFrame(
                {
                    "Priority": [
                        "High",
                        "Medium",
                        "Low",
                    ],
                    "Records": [
                        high_count,
                        medium_count,
                        low_count,
                    ],
                }
            )

            st.bar_chart(
                priority_data.set_index("Priority")
            )
        else:
            st.info(
                "No saved analyses yet. "
                "Analyze an image to populate the dashboard."
            )

    st.divider()

    st.subheader("Recent Activity")

    if not history.empty:
        st.dataframe(
            history.head(5),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No recent prediction records.")

    st.button(
        "Go to Image Analysis",
        type="primary",
        on_click=_navigate_to,
        args=("Image Analysis",),
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif page == "Image Analysis":

    st.subheader("Image Analysis")
    st.write(
        "Upload a road image to detect damage and estimate "
        "its visual maintenance priority."
    )

    if not models_available:
        st.error(
            "The trained model files are missing. Check the model paths "
            "on the Model & Connection page."
        )
        st.stop()

    if not models_loaded:
        with st.spinner("Loading trained models for your first analysis..."):
            try:
                yolo_model, rf_model = load_models()
                models_loaded = True
            except Exception as error:
                st.error("Could not load the trained models.")
                st.exception(error)
                st.stop()

    uploaded_file = st.file_uploader(
        "Upload road image",
        type=["jpg", "jpeg", "png"],
    )

    st.slider(
        "Detection confidence threshold",
        min_value=0.05,
        max_value=0.90,
        step=0.05,
        key="confidence",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Unable to read this image.")
            st.stop()

        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

        if st.button(
            "Analyze Road Damage",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "Detecting damage and estimating priority..."
            ):
                try:
                    results = yolo_model.predict(
                        source=image,
                        conf=st.session_state.confidence,
                        verbose=False,
                    )

                    result = results[0]

                    feature_df, counts = extract_features(result)

                    total_detections = int(
                        feature_df.iloc[0]["total_detections"]
                    )

                    annotated_image = result.plot()[:, :, ::-1]

                    prediction = "Insufficient detections"
                    probability_dict = {}
                    classification_success = False

                    if total_detections > 0:
                        prediction = rf_model.predict(
                            feature_df
                        )[0]

                        if hasattr(rf_model, "predict_proba"):
                            probabilities = rf_model.predict_proba(
                                feature_df
                            )[0]

                            probability_dict = {
                                str(class_name): round(
                                    float(probability) * 100,
                                    2,
                                )
                                for class_name, probability in zip(
                                    rf_model.classes_,
                                    probabilities,
                                )
                            }

                        classification_success = True

                    if (
                        classification_success
                        or total_detections == 0
                    ):
                        save_prediction(
                            image_filename=uploaded_file.name,
                            counts=counts,
                            feature_df=feature_df,
                            prediction=prediction,
                            probabilities=probability_dict,
                        )

                    st.session_state.analysis_result = {
                        "annotated_image": annotated_image,
                        "counts": counts,
                        "features": feature_df,
                        "total": total_detections,
                        "prediction": prediction,
                        "probabilities": probability_dict,
                        "filename": uploaded_file.name,
                    }

                    st.success("Analysis completed.")

                except Exception as error:
                    st.error("Image analysis failed.")
                    st.exception(error)

    # Display the most recent analysis after reruns.
    analysis = st.session_state.analysis_result

    if analysis is not None:
        st.divider()
        st.subheader("Detection Results")

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(
                analysis["annotated_image"],
                caption="YOLO26s annotated image",
                use_container_width=True,
            )

        with col2:
            st.metric(
                "Total Detections",
                analysis["total"],
            )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Damage Class": list(
                            analysis["counts"].keys()
                        ),
                        "Detections": list(
                            analysis["counts"].values()
                        ),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("Visual Maintenance Priority")

            prediction = analysis["prediction"]

            if prediction == "High":
                st.error("🔴 HIGH PRIORITY")
            elif prediction == "Medium":
                st.warning("🟡 MEDIUM PRIORITY")
            elif prediction == "Low":
                st.success("🟢 LOW PRIORITY")
            else:
                st.info(str(prediction))

        if analysis["probabilities"]:
            st.subheader("Classifier Probability Estimates")

            st.dataframe(
                pd.DataFrame(
                    {
                        "Priority": list(
                            analysis["probabilities"].keys()
                        ),
                        "Probability (%)": list(
                            analysis["probabilities"].values()
                        ),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("View Extracted Model Features"):
            st.dataframe(
                analysis["features"],
                hide_index=True,
                use_container_width=True,
            )

        st.caption(
            "Priority estimates are based on image-derived "
            "features and require further independent validation."
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    st.subheader("Prediction History")
    st.write("Review recent road image analyses.")

    if not database_connected:
        st.error("SQLite database is not available.")
        st.stop()

    if history.empty:
        st.info("No prediction records have been saved yet.")
    else:
        st.metric("Recent Records", len(history))

        st.dataframe(
            history,
            hide_index=True,
            use_container_width=True,
        )
        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Prediction History as CSV",
            data=csv_data,
            file_name="roadsight_prediction_history.csv",
            mime="text/csv",
        )

        st.subheader("View Prediction Details")

        selected_id = st.selectbox(
            "Prediction ID",
            history["id"].tolist(),
        )

        if st.button("View Selected Prediction"):
            record = get_prediction_details(selected_id)

            if record is not None:
                st.write(f"**Prediction ID:** {record[0]}")
                st.write(f"**Analysis time (UTC):** {record[1]}")
                st.write(f"**Image filename:** {record[2]}")
                st.write(f"**Total detections:** {record[3]}")
                st.write(f"**Predicted priority:** {record[5]}")

                st.write("**Damage Counts**")
                st.json(json.loads(record[4]))

                st.write("**Classifier Probabilities**")
                st.json(json.loads(record[6]))

                st.write("**Extracted Features**")
                st.json(json.loads(record[7]))
            else:
                st.warning("Prediction record was not found.")


# ============================================================
# MODEL AND CONNECTION SETTINGS
# ============================================================

elif page == "Model & Connection":

    st.subheader("Model & Connection Settings")

    st.markdown("### Model Status")

    if models_available:
        st.success("YOLO26s and Random Forest model files found.")
        st.info(
            "Weights load when you open Image Analysis to keep Home fast. "
            "After loading, Streamlit caches the models for subsequent use."
        )
    else:
        st.error("One or more model files are missing.")

    st.markdown("### Model Paths")

    st.code(str(YOLO_MODEL_PATH))
    st.code(str(RF_MODEL_PATH))

    if model_error:
        st.error(model_error)

    st.markdown("### Detection Configuration")

    st.write(
        f"Current confidence threshold: "
        f"{st.session_state.confidence:.2f}"
    )

    st.markdown("### SQLite Connection")

    if database_connected:
        st.success("Database initialization succeeded.")
        st.caption(
            "The app uses the SQLite connection managed by "
            "database/db_manager.py."
        )
    else:
        st.error("Database initialization failed.")
        st.exception(database_error)

    st.markdown("### Damage Classes")

    st.dataframe(
        pd.DataFrame(
            {
                "Class ID": list(CLASS_NAMES.keys()),
                "Damage Class": list(CLASS_NAMES.values()),
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    "<div class='rg-footer'>RoadSight AI · M.Tech Research Prototype · "
    "Road Damage Intelligence &amp; Maintenance Prioritization</div>",
    unsafe_allow_html=True,
)
