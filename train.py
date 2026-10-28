import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 5

DATASET_ROOT = os.path.expanduser(
    r"~\.cache\kagglehub\datasets\mariaherrerot\aptos2019\versions\3"
)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "dr_mobilenetv2.h5"
)


# ============================================================
# FIND DATASET
# ============================================================

print("\nSearching for APTOS dataset...")

csv_path = None
image_dir = None


# Find train_1.csv
for root, dirs, files in os.walk(DATASET_ROOT):

    for file in files:

        if file.lower() in ["train_1.csv", "train.csv"]:

            csv_path = os.path.join(root, file)

            if file.lower() == "train_1.csv":
                break

    if csv_path is not None:
        break


# Find train_images
for root, dirs, files in os.walk(DATASET_ROOT):

    for directory in dirs:

        if directory.lower() == "train_images":

            possible = os.path.join(
                root,
                directory
            )

            nested = os.path.join(
                possible,
                "train_images"
            )

            if os.path.isdir(nested):

                image_dir = nested

            elif os.path.isdir(possible):

                image_dir = possible

            break

    if image_dir is not None:
        break


print("CSV:", csv_path)
print("Images:", image_dir)


if csv_path is None:
    raise FileNotFoundError(
        "Could not find train_1.csv"
    )

if image_dir is None:
    raise FileNotFoundError(
        "Could not find train_images"
    )


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(csv_path)

print("\nCSV columns:")
print(df.columns.tolist())

if "id_code" in df.columns:
    id_col = "id_code"
else:
    id_col = df.columns[0]

if "diagnosis" in df.columns:
    label_col = "diagnosis"
else:
    label_col = df.columns[1]


# ============================================================
# CREATE IMAGE PATHS
# ============================================================

def find_image(image_id):

    image_id = str(image_id)

    extensions = [
        ".png",
        ".jpg",
        ".jpeg"
    ]

    for ext in extensions:

        path = os.path.join(
            image_dir,
            image_id + ext
        )

        if os.path.exists(path):
            return path

    return None


df["filepath"] = df[id_col].apply(
    find_image
)

df["label"] = df[label_col].astype(int)

# Remove images that cannot be found
df = df[
    df["filepath"].notna()
].reset_index(drop=True)


print("\nTotal usable images:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts().sort_index())


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(
    train_df["label"]
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"]
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:")
print(class_weights)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


# ============================================================
# DATASETS
# ============================================================

train_paths = train_df["filepath"].values
train_labels = train_df["label"].values

val_paths = val_df["filepath"].values
val_labels = val_df["label"].values


train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)


# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.05
    ),

    tf.keras.layers.RandomZoom(
        0.1
    ),
])


def train_preprocess(path, label):

    image, label = load_image(
        path,
        label
    )

    image = augmentation(
        image,
        training=True
    )

    return image, label


train_dataset = (
    train_dataset
    .shuffle(1000)
    .map(
        train_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


val_dataset = (
    val_dataset
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# MOBILE NET V2
# ============================================================

print("\nCreating MobileNetV2 model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    ),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False


inputs = tf.keras.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)

x = base_model(
    inputs,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.3
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = tf.keras.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n======================================")
print("STARTING TRAINING")
print("======================================\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving final model...")

model.save(
    MODEL_PATH
)

print("\n======================================")
print("TRAINING COMPLETE!")
print("======================================")

print(
    "\nModel saved at:"
)

print(
    os.path.abspath(MODEL_PATH)
)

print("\nYou can now run the Streamlit app.")