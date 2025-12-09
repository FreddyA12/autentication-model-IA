"""
Script para entrenar el modelo MLP de voz.
Uso: python dataset/voice/scripts/3_train_voice_mlp.py
"""

import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Configuración
EMBEDDINGS_DIR = Path("dataset/voice/embeddings")
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "voice_embeddings.npy"
LABELS_FILE = EMBEDDINGS_DIR / "voice_labels.npy"
LABEL_MAP_FILE = EMBEDDINGS_DIR / "voice_label_map.json"

MODELS_DIR = Path("dataset/voice/models")
MODEL_PATH = MODELS_DIR / "voice_mlp_best.keras"
CLASS_INDICES_PATH = MODELS_DIR / "voice_class_indices.json"
HISTORY_PATH = MODELS_DIR / "voice_training_history.png"

EPOCHS = 200
BATCH_SIZE = 4
TEST_SIZE = 0.2

def build_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Capa 1: Potente pero con menos restricciones
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Capa 2: Refinamiento
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4), 
        
        # Salida
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Learning rate un poco más alto para aprender más rápido
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    if not EMBEDDINGS_FILE.exists():
        print(f"No existe {EMBEDDINGS_FILE}")
        return

    # Cargar datos
    X = np.load(EMBEDDINGS_FILE)
    y = np.load(LABELS_FILE)
    with open(LABEL_MAP_FILE, 'r') as f:
        label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}
    
    num_classes = len(label_map)
    print(f"Datos: {X.shape}, Clases: {num_classes}")

    # Data Augmentation simple
    if len(X) < 100:
        print("Aplicando augmentation...")
        X_aug, y_aug = [], []
        for emb, label in zip(X, y):
            X_aug.append(emb)
            y_aug.append(label)
            X_aug.append(emb + np.random.normal(0, 0.01, emb.shape))
            y_aug.append(label)
        X, y = np.array(X_aug, dtype=np.float32), np.array(y_aug)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y
    )

    classes = np.unique(y_train)
    class_weights_values = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_values)}
    print("Class weights:", class_weights)

    # Entrenar
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(X.shape[1], num_classes)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True,
                                        monitor='val_accuracy', mode='max'),
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=5, min_lr=1e-5)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1,
                        class_weight=class_weights)

    # Evaluar
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nAccuracy Final: {acc*100:.2f}%")
    
    # Guardar mapa de clases
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(label_map, f, indent=4)

    # Gráfica
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.savefig(HISTORY_PATH)
    print(f"Gráfica guardada en {HISTORY_PATH}")
    print("Listo. Siguiente: python dataset/voice/scripts/4_predict_voice.py")

if __name__ == "__main__":
    main()
