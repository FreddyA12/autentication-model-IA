import os
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuración
EMBEDDINGS_DIR = "dataset/face/embeddings"
MODELS_DIR = "dataset/face/models"
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings_dataset.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# Hiperparámetros
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

def load_data():
    if not os.path.exists(EMBEDDINGS_PATH):
        print("No se encontró el archivo de embeddings")
        return None, None, None
    
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data['idx_to_label']

def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(512,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    # Cargar datos
    X, y, idx_to_label = load_data()
    if X is None: return

    num_classes = len(idx_to_label)
    print(f"Cargadas {len(X)} muestras, {num_classes} clases")

    # Dividir
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Construir modelo
    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entrenar
    print("Entrenando...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, "face_classifier.keras"),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ],
        verbose=1
    )

    # Guardar mapeo de clases
    with open(os.path.join(MODELS_DIR, "class_indices.json"), 'w') as f:
        json.dump(idx_to_label, f, indent=4)
    
    print("Entrenamiento completado. Modelo guardado.")

if __name__ == "__main__":
    main()
