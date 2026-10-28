"""
Diabetic Retinopathy - Dataset Browser + Predictor
==================================================
Streamlit app for:
1. Loading the APTOS 2019 dataset
2. Browsing retinal images by severity
3. Predicting diabetic retinopathy using a trained model

NOTE:
A trained model is required for prediction.
"""

import os
import random
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 224

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

CLASS_DESCRIPTIONS = {
    "No DR": "No visible signs of diabetic retinopathy.",
    "Mild": "Early stage — microaneurysms present.",
    "Moderate": "More extensive retinal changes and blood vessel damage.",
    "Severe": "Significant blockage of blood vessels; high risk of progression.",
    "Proliferative DR": "Most advanced stage — abnormal new vessel growth; urgent care needed.",
}

# Model paths
H5_PATH = os.path.join("model", "dr_mobilenetv2.h5")
TFLITE_PATH = os.path.join("model", "dr_mobilenetv2_int8.tflite")


st.set_page_config(
    page_title="DR Dataset + Predictor",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data(show_spinner="Loading APTOS 2019 dataset...")


def load_dataset():

    try:

        import kagglehub

        # Download / locate dataset
        dataset_path = kagglehub.dataset_download(
            "mariaherrerot/aptos2019"
        )

        csv_path = None
        img_dir = None

        # ----------------------------------------------------
        # Find train CSV
        # ----------------------------------------------------

        for root, dirs, files in os.walk(dataset_path):

            for f in files:

                if f.lower() in ["train.csv", "train_1.csv"]:

                    csv_path = os.path.join(root, f)

                    # Prefer train_1.csv if available
                    if f.lower() == "train_1.csv":
                        break

            if csv_path is not None and os.path.basename(csv_path).lower() == "train_1.csv":
                break

        # ----------------------------------------------------
        # Find train_images directory
        # ----------------------------------------------------

        for root, dirs, files in os.walk(dataset_path):

            for d in dirs:

                if d.lower() == "train_images":

                    possible_path = os.path.join(root, d)

                    if os.path.isdir(possible_path):

                        img_dir = possible_path

                        # Continue searching in case there is
                        # another nested train_images directory
                        nested = os.path.join(
                            possible_path,
                            "train_images"
                        )

                        if os.path.isdir(nested):

                            img_dir = nested

                        break

            if img_dir is not None:
                break

        # ----------------------------------------------------
        # Check dataset
        # ----------------------------------------------------

        if csv_path is None:

            st.error(
                "Could not find train.csv or train_1.csv "
                "in the downloaded dataset."
            )

            return None

        if img_dir is None:

            st.error(
                "Could not find train_images folder "
                "in the downloaded dataset."
            )

            return None

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        df = pd.read_csv(csv_path)

        # Find ID column
        if "id_code" in df.columns:

            id_col = "id_code"

        else:

            id_col = df.columns[0]

        # Find diagnosis column
        if "diagnosis" in df.columns:

            label_col = "diagnosis"

        else:

            label_col = df.columns[1]

        # ----------------------------------------------------
        # Create image paths
        # ----------------------------------------------------

        def get_image_path(image_id):

            image_id = str(image_id)

            # Normal expected path
            path = os.path.join(
                img_dir,
                image_id + ".png"
            )

            if os.path.exists(path):
                return path

            # Try JPG
            path = os.path.join(
                img_dir,
                image_id + ".jpg"
            )

            if os.path.exists(path):
                return path

            # Try JPEG
            path = os.path.join(
                img_dir,
                image_id + ".jpeg"
            )

            if os.path.exists(path):
                return path

            return None

        df["filepath"] = df[id_col].apply(
            get_image_path
        )

        # Remove missing images
        df = df[
            df["filepath"].notna()
        ].reset_index(drop=True)

        # Convert labels
        df["label"] = df[label_col].astype(int)

        # Convert label to class name
        df["class_name"] = df["label"].apply(
            lambda x: CLASS_NAMES[x]
            if 0 <= x < len(CLASS_NAMES)
            else "Unknown"
        )

        return df

    except Exception as e:

        st.error(
            f"Error while loading dataset: {e}"
        )

        return None


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_keras_model():

    if os.path.exists(H5_PATH):

        try:

            return tf.keras.models.load_model(
                H5_PATH
            )

        except Exception as e:

            st.error(
                f"Could not load Keras model: {e}"
            )

            return None

    return None


@st.cache_resource
def load_tflite_interpreter():

    if os.path.exists(TFLITE_PATH):

        try:

            interpreter = tf.lite.Interpreter(
                model_path=TFLITE_PATH
            )

            interpreter.allocate_tensors()

            return interpreter

        except Exception as e:

            st.error(
                f"Could not load TFLite model: {e}"
            )

            return None

    return None


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(image: Image.Image):

    image = image.convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    arr = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    return np.expand_dims(
        arr,
        axis=0
    )


# ============================================================
# KERAS PREDICTION
# ============================================================

def predict_keras(model, arr):

    predictions = model.predict(
        arr,
        verbose=0
    )

    return predictions[0]


# ============================================================
# TFLITE PREDICTION
# ============================================================

def predict_tflite(interpreter, arr):

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    input_info = input_details[0]

    input_dtype = input_info["dtype"]

    # --------------------------------------------------------
    # Quantized model
    # --------------------------------------------------------

    if input_dtype == np.uint8:

        scale, zero_point = input_info["quantization"]

        if scale > 0:

            arr = (
                arr / scale + zero_point
            ).astype(
                np.uint8
            )

        else:

            arr = arr.astype(
                np.uint8
            )

    else:

        arr = arr.astype(
            input_dtype
        )

    interpreter.set_tensor(
        input_info["index"],
        arr
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    # Dequantize output if required
    out_scale, out_zero_point = (
        output_details[0]["quantization"]
    )

    if out_scale > 0:

        output = (
            output.astype(np.float32)
            - out_zero_point
        ) * out_scale

    return output


# ============================================================
# RUN PREDICTION
# ============================================================

def run_prediction(
    image,
    model,
    interpreter,
    mode
):

    arr = preprocess(image)

    if (
        mode.startswith("Full")
        and model is not None
    ):

        return predict_keras(
            model,
            arr
        )

    if interpreter is not None:

        return predict_tflite(
            interpreter,
            arr
        )

    if model is not None:

        return predict_keras(
            model,
            arr
        )

    return None


# ============================================================
# SHOW PREDICTION
# ============================================================

def show_prediction(preds):

    if preds is None:

        st.error(
            "No prediction could be generated."
        )

        return

    # Convert output to numpy
    preds = np.asarray(
        preds,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Handle binary model output
    # --------------------------------------------------------

    if len(preds) == 1:

        value = float(preds[0])

        # If sigmoid output
        if value >= 0.5:

            top_idx = 1
            confidence = value

        else:

            top_idx = 0
            confidence = 1 - value

        final_preds = np.array([
            1 - value,
            value
        ])

        # Pad to 5 classes for display
        padded = np.zeros(5)

        padded[0] = final_preds[0]
        padded[1] = final_preds[1]

        preds = padded

    else:

        # Softmax output
        total = np.sum(preds)

        if total > 0:

            preds = preds / total

        top_idx = int(
            np.argmax(preds)
        )

        confidence = float(
            preds[top_idx]
        )

    top_class = CLASS_NAMES[top_idx]

    st.markdown(
        f"### Prediction: **{top_class}** "
        f"({confidence * 100:.1f}% confidence)"
    )

    st.info(
        CLASS_DESCRIPTIONS[top_class]
    )

    # Display probabilities
    for i, name in enumerate(CLASS_NAMES):

        probability = float(
            preds[i]
        )

        st.markdown(
            f"**{name}** — "
            f"{probability * 100:.1f}%"
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "👁️ Diabetic Retinopathy — "
    "Dataset Browser & Predictor"
)


# Load models
model = load_keras_model()

interpreter = load_tflite_interpreter()

model_ready = (
    model is not None
    or interpreter is not None
)


# ------------------------------------------------------------
# Model warning
# ------------------------------------------------------------

if not model_ready:

    st.warning(
        "No trained model found in `model/`. "
        "Dataset browsing is available, but prediction "
        "requires a trained model."
    )


# ============================================================
# TABS
# ============================================================

tab_browse, tab_predict = st.tabs(
    [
        "📂 Browse Dataset",
        "🔍 Predict"
    ]
)


# ============================================================
# TAB 1 — BROWSE DATASET
# ============================================================

with tab_browse:

    df = load_dataset()

    if df is None:

        st.error(
            "Could not load the APTOS 2019 dataset."
        )

    elif len(df) == 0:

        st.error(
            "Dataset loaded, but no matching training images were found."
        )

    else:

        st.success(
            f"Loaded {len(df)} labeled fundus images."
        )

        # ----------------------------------------------------
        # Dataset statistics
        # ----------------------------------------------------

        st.subheader(
            "Class distribution"
        )

        counts = (
            df["class_name"]
            .value_counts()
            .reindex(
                CLASS_NAMES
            )
            .fillna(0)
        )

        st.bar_chart(
            counts
        )

        # ----------------------------------------------------
        # Browse images
        # ----------------------------------------------------

        st.subheader(
            "Browse images"
        )

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            selected_class = st.selectbox(
                "Filter by class",
                ["All"] + CLASS_NAMES
            )

        with col2:

            n_images = st.slider(
                "Number of images to show",
                3,
                24,
                9,
                step=3
            )

        # Filter
        if selected_class == "All":

            filtered = df

        else:

            filtered = df[
                df["class_name"]
                == selected_class
            ]

        if len(filtered) == 0:

            st.warning(
                "No images found for this class."
            )

        else:

            sample = filtered.sample(
                min(
                    n_images,
                    len(filtered)
                ),
                random_state=None
            )

            cols = st.columns(3)

            for i, (_, row) in enumerate(
                sample.iterrows()
            ):

                with cols[i % 3]:

                    st.image(
                        row["filepath"],
                        caption=row["class_name"],
                        use_container_width=True
                    )


# ============================================================
# TAB 2 — PREDICTION
# ============================================================

with tab_predict:

    if not model_ready:

        st.info(
            "A trained model is required to enable predictions."
        )

        st.markdown(
            """
            ### Required model files

            Place either of these files inside the `model` folder:

            - `dr_mobilenetv2.h5`
            - `dr_mobilenetv2_int8.tflite`

            Example:

            `MajorProjectPY/model/dr_mobilenetv2.h5`
            """
        )

    else:

        # ----------------------------------------------------
        # Inference mode
        # ----------------------------------------------------

        mode = st.radio(
            "Inference mode",
            [
                "Full Keras model (higher precision)",
                "Quantized TFLite (edge/on-device)"
            ],
            index=0 if model is not None else 1,
            horizontal=True
        )

        # ----------------------------------------------------
        # Image source
        # ----------------------------------------------------

        source = st.radio(
            "Image source",
            [
                "Upload my own",
                "Pick a random image from dataset"
            ],
            horizontal=True
        )

        image = None

        # ----------------------------------------------------
        # Upload image
        # ----------------------------------------------------

        if source == "Upload my own":

            uploaded_file = st.file_uploader(
                "Upload a retinal fundus image",
                type=[
                    "png",
                    "jpg",
                    "jpeg"
                ]
            )

            if uploaded_file is not None:

                image = Image.open(
                    uploaded_file
                )

        # ----------------------------------------------------
        # Random dataset image
        # ----------------------------------------------------

        else:

            df = load_dataset()

            if df is not None:

                if st.button(
                    "Pick random image"
                ):

                    row = df.sample(
                        1
                    ).iloc[0]

                    image = Image.open(
                        row["filepath"]
                    )

                    st.caption(
                        "Ground-truth label from dataset: "
                        f"**{row['class_name']}**"
                    )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if image is not None:

            colA, colB = st.columns(
                [1, 1]
            )

            with colA:

                st.image(
                    image,
                    caption="Selected image",
                    use_container_width=True
                )

            with colB:

                with st.spinner(
                    "Analyzing..."
                ):

                    preds = run_prediction(
                        image,
                        model,
                        interpreter,
                        mode
                    )

                show_prediction(
                    preds
                )

        st.caption(
            "⚠️ This is a research/demo tool, "
            "not a medical device. Always confirm "
            "results with a qualified ophthalmologist."
        )