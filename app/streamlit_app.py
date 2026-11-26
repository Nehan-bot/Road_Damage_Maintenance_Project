
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import joblib


# ============================================================
# 1. PROJECT ROOT AND PYTHON IMPORT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

# Fix database module import when running through Streamlit
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# 2. DATABASE IMPORTS
# ============================================================

try:
    from database.db_manager import (
        initialize_database,
        save_prediction,
        get_prediction_history,
        get_prediction_details,
    )

except ImportError as error:
    st.error(
        "Database module could not be imported. "
        "Check that database/db_manager.py exists."
    )
    st.exception(error)
    st.stop()


# ============================================================
# 3. MODEL PATHS AND CONFIGURATION
# ============================================================

YOLO_MODEL_PATH = (
    ROOT_DIR
    / "runs"
    / "detect"
    / "experiments"
    / "yolo_final"
    / "yolo26n_rtx4070_corrected_balanced600"
    / "weights"
    / "best.pt"
)

RF_MODEL_PATH = (
    ROOT_DIR
    / "experiments"
    / "random_forest"
    / "random_forest_priority.joblib"
)

CONFIDENCE_THRESHOLD = 0.25

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
# 4. INITIALIZE DATABASE
# ============================================================

try:
    initialize_database()

except Exception as error:
    st.error(f"Database initialization failed: {error}")
    st.stop()


# ============================================================
# 5. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Damage Detection System",
    page_icon="🛣️",
    layout="wide",
)

st.title(
    "🛣️ Road Damage Detection and "
    "Intelligent Maintenance Prioritization"
)

st.markdown(
    """
    Upload a road image to identify visible road damage
    and estimate its visual maintenance priority.

    **Integrated technologies:**
    - YOLO for road damage detection
    - Random Forest for priority classification
    - SQLite for prediction history
    """
)

st.info(
    "Research prototype: priority predictions represent "
    "image-based visual severity, not verified real-world "
    "maintenance urgency."
)


# ============================================================
# 6. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    rf_model = joblib.load(str(RF_MODEL_PATH))

    return yolo_model, rf_model


if not YOLO_MODEL_PATH.exists():

    st.error(
        f"YOLO model file not found:\n{YOLO_MODEL_PATH}"
    )
    st.stop()


if not RF_MODEL_PATH.exists():

    st.error(
        f"Random Forest model file not found:\n{RF_MODEL_PATH}"
    )
    st.stop()


try:

    yolo_model, rf_model = load_models()

    st.success(
        "YOLO and Random Forest models loaded successfully."
    )

except Exception as error:

    st.error("Model loading failed.")
    st.exception(error)
    st.stop()


# ============================================================
# 7. FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(result):

    counts = {
        class_name: 0
        for class_name in CLASS_NAMES.values()
    }

    confidence_sums = {
        class_name: 0.0
        for class_name in CLASS_NAMES.values()
    }

    detections = result.boxes

    if detections is not None and len(detections) > 0:

        class_ids = (
            detections.cls.cpu().numpy().astype(int)
        )

        confidences = (
            detections.conf.cpu().numpy()
        )

        for class_id, confidence in zip(
            class_ids,
            confidences
        ):

            class_name = CLASS_NAMES.get(class_id)

            if class_name is None:
                continue

            counts[class_name] += 1

            confidence_sums[class_name] += float(
                confidence
            )

    features = {
        "total_detections": sum(counts.values())
    }

    for class_name in CLASS_NAMES.values():

        count = counts[class_name]

        if count > 0:

            mean_confidence = (
                confidence_sums[class_name] / count
            )

        else:

            mean_confidence = 0.0

        features[
            f"{class_name}_count"
        ] = count

        features[
            f"{class_name}_mean_confidence"
        ] = mean_confidence

    feature_df = pd.DataFrame(
        [
            [
                features[column]
                for column in FEATURE_COLUMNS
            ]
        ],
        columns=FEATURE_COLUMNS,
    )

    return feature_df, counts


# ============================================================
# 8. IMAGE UPLOAD SECTION
# ============================================================

st.divider()

st.header("📷 Road Image Analysis")

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
)


if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error("Unable to read the uploaded image.")
        st.stop()

    st.subheader("Uploaded Road Image")

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True,
    )

    analyze_button = st.button(
        "Analyze Road Damage",
        type="primary",
        use_container_width=True,
    )

    # ========================================================
    # 9. ANALYZE IMAGE
    # ========================================================

    if analyze_button:

        with st.spinner(
            "Detecting road damage and calculating priority..."
        ):

            try:

                # YOLO inference
                results = yolo_model.predict(
                    source=image,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )

                result = results[0]

                # Extract features
                feature_df, counts = extract_features(
                    result
                )

                total_detections = int(
                    feature_df.iloc[0][
                        "total_detections"
                    ]
                )

                # Generate annotated image
                annotated_image = result.plot()

                # Convert BGR to RGB
                annotated_image = (
                    annotated_image[:, :, ::-1]
                )

            except Exception as error:

                st.error("Image analysis failed.")
                st.exception(error)
                st.stop()

        # ====================================================
        # 10. DISPLAY YOLO RESULTS
        # ====================================================

        st.divider()

        st.header("🔍 Road Damage Detection Results")

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            st.image(
                annotated_image,
                caption="YOLO annotated detection image",
                use_container_width=True,
            )

        with col2:

            st.subheader("Detected Damage Counts")

            detection_table = pd.DataFrame({
                "Damage Class": list(counts.keys()),
                "Detections": list(counts.values()),
            })

            st.dataframe(
                detection_table,
                hide_index=True,
                use_container_width=True,
            )

            st.metric(
                "Total Detections",
                total_detections,
            )

            st.caption(
                f"Confidence threshold: "
                f"{CONFIDENCE_THRESHOLD}"
            )

        # ====================================================
        # 11. RANDOM FOREST CLASSIFICATION
        # ====================================================

        st.divider()

        st.header(
            "🚧 Maintenance Priority Assessment"
        )

        prediction = "Insufficient detections"

        probability_dict = {}

        classification_success = False

        if total_detections == 0:

            st.warning(
                "No road damage was detected at the selected "
                "confidence threshold. This does not prove "
                "that the road is damage-free. Manual inspection "
                "is recommended."
            )

        else:

            try:

                # Random Forest prediction
                prediction = rf_model.predict(
                    feature_df
                )[0]

                # Class probabilities
                if hasattr(rf_model, "predict_proba"):

                    probabilities = (
                        rf_model.predict_proba(
                            feature_df
                        )[0]
                    )

                    probability_dict = {
                        str(class_name): round(
                            float(probability) * 100,
                            2
                        )
                        for class_name, probability in zip(
                            rf_model.classes_,
                            probabilities
                        )
                    }

                classification_success = True

                st.subheader(
                    "Predicted Visual Priority"
                )

                if prediction == "High":

                    st.error("🔴 HIGH PRIORITY")

                elif prediction == "Medium":

                    st.warning("🟡 MEDIUM PRIORITY")

                elif prediction == "Low":

                    st.success("🟢 LOW PRIORITY")

                else:

                    st.info(str(prediction))

                # Display probability estimates
                if probability_dict:

                    st.subheader(
                        "Classifier Probability Estimates"
                    )

                    probability_table = pd.DataFrame({
                        "Priority": list(
                            probability_dict.keys()
                        ),
                        "Model Probability (%)": list(
                            probability_dict.values()
                        ),
                    })

                    st.dataframe(
                        probability_table,
                        hide_index=True,
                        use_container_width=True,
                    )

            except Exception as error:

                prediction = "Classification error"

                st.error(
                    "Random Forest classification failed."
                )

                st.exception(error)

        # ====================================================
        # 12. SAVE PREDICTION IN SQLITE
        # ====================================================

        if (
            classification_success
            or total_detections == 0
        ):

            try:

                save_prediction(
                    image_filename=uploaded_file.name,
                    counts=counts,
                    feature_df=feature_df,
                    prediction=prediction,
                    probabilities=probability_dict,
                )

                st.success(
                    "✅ Prediction saved successfully "
                    "in the SQLite database."
                )

            except Exception as error:

                st.error(
                    "Failed to save prediction in database."
                )

                st.exception(error)

        # ====================================================
        # 13. DISPLAY EXTRACTED FEATURES
        # ====================================================

        with st.expander(
            "View Extracted Model Features"
        ):

            st.dataframe(
                feature_df,
                hide_index=True,
                use_container_width=True,
            )

        st.caption(
            "The classifier was trained using a small "
            "manually reviewed dataset. Its predictions "
            "require further independent validation."
        )


# ============================================================
# 14. PREDICTION HISTORY
# ============================================================

st.divider()

st.header("📊 Prediction History")

try:

    history = get_prediction_history(
        limit=100
    )

    if history.empty:

        st.info(
            "No prediction records found. "
            "Analyze a road image to create the first record."
        )

    else:

        # ----------------------------------------------------
        # SUMMARY METRICS
        # ----------------------------------------------------

        total_records = len(history)

        high_count = (
            history["predicted_priority"] == "High"
        ).sum()

        medium_count = (
            history["predicted_priority"] == "Medium"
        ).sum()

        low_count = (
            history["predicted_priority"] == "Low"
        ).sum()

        other_count = (
            ~history["predicted_priority"].isin(
                ["High", "Medium", "Low"]
            )
        ).sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Recent Records",
            total_records
        )

        col2.metric(
            "High Priority",
            int(high_count)
        )

        col3.metric(
            "Medium Priority",
            int(medium_count)
        )

        col4.metric(
            "Low Priority",
            int(low_count)
        )

        if other_count > 0:

            st.caption(
                f"Other or unclassified records: "
                f"{int(other_count)}"
            )

        # ----------------------------------------------------
        # HISTORY TABLE
        # ----------------------------------------------------

        st.subheader("Recent Road Analyses")

        st.dataframe(
            history,
            hide_index=True,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # SELECT AND VIEW RECORD
        # ----------------------------------------------------

        st.subheader(
            "View Individual Prediction Details"
        )

        selected_id = st.selectbox(
            "Select Prediction ID",
            history["id"].tolist(),
        )

        if st.button(
            "View Selected Prediction"
        ):

            record = get_prediction_details(
                selected_id
            )

            if record is not None:

                st.write(
                    f"**Prediction ID:** {record[0]}"
                )

                st.write(
                    f"**Analysis time (UTC):** {record[1]}"
                )

                st.write(
                    f"**Image filename:** {record[2]}"
                )

                st.write(
                    f"**Total detections:** {record[3]}"
                )

                st.write(
                    f"**Predicted priority:** {record[5]}"
                )

                st.write("**Damage Counts**")

                st.json(
                    json.loads(record[4])
                )

                st.write(
                    "**Classifier Probability Estimates**"
                )

                st.json(
                    json.loads(record[6])
                )

                st.write(
                    "**Extracted Model Features**"
                )

                st.json(
                    json.loads(record[7])
                )

            else:

                st.warning(
                    "Selected prediction record was not found."
                )

except Exception as error:

    st.error(
        "Unable to retrieve prediction history."
    )

    st.exception(error)


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "Road Damage Detection and Intelligent Maintenance "
    "Prioritization System | M.Tech Research Project"
)
