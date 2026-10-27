
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset path
DATA_DIR = Path(
    r"C:\Users\tejas\OneDrive\Desktop\vyshu\archive\Forest Fire Dataset\Training"
)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Find the two image folders
fire_dir = DATA_DIR / "fire"

no_fire_dir = next(
    (
        folder for folder in DATA_DIR.iterdir()
        if folder.is_dir()
        and folder.name.lower().replace(" ", "_") in ["no_fire", "nofire"]
    ),
    None
)

if not fire_dir.exists() or no_fire_dir is None:
    raise FileNotFoundError(
        "Could not find fire and no-fire folders. Check the dataset path and folder names."
    )

print("Loading images...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["fire", no_fire_dir.name]
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["fire", no_fire_dir.name]
)

# Create the CNN model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
print("Class names:", train_ds.class_names)
print("Fire folder:", fire_dir)
print("No-fire folder:", no_fire_dir)

print("Starting CNN training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save the trained model
Path("models").mkdir(exist_ok=True)
model.save("models/forest_fire_cnn.keras")

print("Training complete! Model saved in the models folder.")
