import json
from pathlib import Path

import tensorflow as tf

DATASET_DIR = Path("dataset/audio_mels")
MODELS_DIR = Path("dataset/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "voice_cnn.keras"
BEST_MODEL_PATH = MODELS_DIR / "voice_cnn_best.keras"
CLASS_INDICES_PATH = MODELS_DIR / "voice_class_indices.json"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 60
SEED = 21


def build_datasets():
    for user_dir in DATASET_DIR.iterdir():
        if user_dir.is_dir() and any(user_dir.glob("*.png")):
            break
    else:
        raise SystemExit("dataset/audio_mels no contiene espectrogramas.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
    )

    return train_ds, val_ds


def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1))
    x = tf.keras.layers.RandomTranslation(0.05, 0.05)(inputs)
    x = tf.keras.layers.RandomZoom(0.05)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    train_ds, val_ds = build_datasets()
    class_names = train_ds.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: ((tf.cast(x, tf.float32) / 255.0), y), num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda x, y: ((tf.cast(x, tf.float32) / 255.0), y), num_parallel_calls=AUTOTUNE
    )
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    model = build_model(num_classes)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(str(MODEL_PATH))
    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as fp:
        json.dump({i: name for i, name in enumerate(class_names)}, fp, indent=4, ensure_ascii=False)

    print(f"Modelo final guardado en {MODEL_PATH}")
    print(f"Mejor modelo guardado en {BEST_MODEL_PATH}")
    print(f"Clases guardadas en {CLASS_INDICES_PATH}")
    print("Entrenamiento de voz completado.")


if __name__ == "__main__":
    main()
