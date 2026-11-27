
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


# ============================================================
# THEME STYLING
# ============================================================

def apply_theme(theme):
    # Streamlit's built-in theme is app-wide; this CSS palette keeps all
    # custom surfaces and text readable in each selected mode.
    if theme == "Dark":
        css = """
        <style>
        :root { color-scheme: dark; }
        .stApp { --page-bg:#0b1220; --surface:#111d2e; --surface-2:#17263a;
          --text:#f1f5f9; --muted:#b8c7d9; --border:#2b3b50; --accent:#60a5fa;
          --accent-strong:#2563eb; --sidebar-bg:#0d1b2d; }
        </style>"""
    elif theme == "Light":
        css = """
        <style>
        :root { color-scheme: light; }
        .stApp { --page-bg:#f4f7fb; --surface:#ffffff; --surface-2:#eaf0f7;
          --text:#172b45; --muted:#526174; --border:#d8e1ec; --accent:#2563eb;
          --accent-strong:#174ea6; --sidebar-bg:#10243a; }
        </style>"""
    else:
        css = """
        <style>
        @media (prefers-color-scheme: dark) {
          .stApp { --page-bg:#0b1220; --surface:#111d2e; --surface-2:#17263a;
            --text:#f1f5f9; --muted:#b8c7d9; --border:#2b3b50; --accent:#60a5fa;
            --accent-strong:#2563eb; --sidebar-bg:#0d1b2d; }
        }
        @media (prefers-color-scheme: light) {
          .stApp { --page-bg:#f4f7fb; --surface:#ffffff; --surface-2:#eaf0f7;
            --text:#172b45; --muted:#526174; --border:#d8e1ec; --accent:#2563eb;
            --accent-strong:#174ea6; --sidebar-bg:#10243a; }
        }
        </style>"""
    global_css = """
    <style>
    .stApp { background:var(--page-bg); color:var(--text); }
    [data-testid="stSidebar"] { background:var(--sidebar-bg); }
    [data-testid="stSidebar"] * { color:#f1f5f9; }
    [data-testid="stMetric"], .rg-card {
      background:var(--surface); border:1px solid var(--border);
      border-radius:14px; padding:16px;
    }
    [data-testid="stMetric"] label, .rg-muted { color:var(--muted)!important; }
    .block-container { padding-top:1.15rem; padding-bottom:2rem; max-width:1500px; }
    h1,h2,h3,h4,p,li,label { color:var(--text); }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background:var(--surface); border:1px solid var(--border);
      border-radius:20px; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { padding:1.15rem; }
    .rg-kicker { display:inline-block; color:var(--accent); font-weight:700;
      letter-spacing:.08em; font-size:.78rem; text-transform:uppercase; }
    .rg-hero-title { font-size:clamp(2.3rem,4.5vw,4rem); line-height:1.05;
      font-weight:800; color:var(--text); margin:.5rem 0; }
    .rg-subtitle { font-size:1.15rem; color:var(--muted); }
    .rg-section { background:var(--surface); border:1px solid var(--border);
      border-radius:16px; padding:1.15rem; height:100%; }
    .rg-stat { background:var(--surface-2); border-radius:12px; padding:1rem;
      text-align:center; height:100%; }
    div.stButton > button[kind="primary"] { background:var(--accent-strong);
      border-color:var(--accent-strong); color:#fff; border-radius:10px; font-weight:600; }
    div.stButton > button { border-radius:10px; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border-radius:12px; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] div,
    [data-testid="stMultiSelect"] div, [data-testid="stDateInput"] input {
      background:var(--surface)!important; color:var(--text)!important;
      border-color:var(--border)!important;
    }
    [data-testid="stFileUploader"], [data-testid="stExpander"],
    [data-testid="stAlert"], [data-testid="stPopover"] {
      background:var(--surface)!important; color:var(--text)!important;
      border-color:var(--border)!important;
    }
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
      color:var(--text)!important;
    }
    [data-testid="stHeader"] { background:transparent; }
    </style>
    """
    st.markdown(css + global_css, unsafe_allow_html=True)


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


models_loaded = False
model_error = None
yolo_model = None
rf_model = None

if YOLO_MODEL_PATH.exists() and RF_MODEL_PATH.exists():
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
# SIDEBAR — CHATGPT STYLE NAVIGATION
# ============================================================

with st.sidebar:
    logo_path = ROOT_DIR / "assets" / "roadsight_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=92)
    st.markdown("## RoadSight AI")
    st.caption("Road Damage Intelligence")
    st.divider()

    page = st.radio(
        "Workspace",
        ["Home", "Dashboard", "Image Analysis", "Prediction History", "Model & Connection"],
        key="page",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### System Status")
    if models_loaded:
        st.success("YOLO26s · Ready")
        st.success("Random Forest · Ready")
    else:
        st.error("Models · Not ready")
    if database_connected:
        st.success("SQLite · Connected")
    else:
        st.error("SQLite · Error")

    st.divider()
    st.caption("M.Tech Research Prototype\n\nImage-based road damage analysis")


# ============================================================
# COMMON HEADER
# ============================================================

header_left, header_right = st.columns([5, 2])
with header_left:
    st.markdown("### RoadSight AI")
    st.caption("Road Damage Intelligence & Maintenance Prioritization")
with header_right:
    st.radio(
        "Appearance",
        ["System", "Light", "Dark"],
        key="theme_choice",
        horizontal=True,
        label_visibility="collapsed",
    )

apply_theme(st.session_state.theme_choice)

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
    hero_left, hero_right = st.columns([1.05, 1], gap="large")
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
            if st.button("Get Started — Analyse Image  →", type="primary", use_container_width=True):
                st.session_state.page = "Image Analysis"
                st.rerun()
            if st.button("Go to Dashboard", use_container_width=True):
                st.session_state.page = "Dashboard"
                st.rerun()
    with hero_right:
        hero_image = ROOT_DIR / "assets" / "road_safety_hero.jpg"
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
                    "Ready" if models_loaded else "Unavailable",
                    "Ready" if models_loaded else "Unavailable",
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
        on_click=lambda: st.session_state.update(
            page="Image Analysis"
        ),
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

    if not models_loaded:
        st.error(
            "The trained models are not available. "
            "Check the Model & Connection page."
        )
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

    if models_loaded:
        st.success("YOLO26s model loaded")
        st.success("Random Forest model loaded")
    else:
        st.error("One or more models could not be loaded.")

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

st.caption(
    "RoadSight AI | M.Tech Research Prototype | "
    "Road Damage Intelligence & Maintenance Prioritization"
)
