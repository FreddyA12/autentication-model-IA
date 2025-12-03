"""
=============================================================================
PASO 3: ENTRENAR TU PROPIA RED NEURONAL (CNN) SOBRE LOS EMBEDDINGS
=============================================================================

Aquí es donde TÚ entrenas tu propio modelo.

FaceNet ya hizo el trabajo pesado:
- Extracción de rasgos faciales
- Invariancia a luz, ángulos, expresiones
- Separación de identidades

Tu red neuronal SOLO necesita aprender a clasificar los embeddings.

Arquitectura (la que usan bancos, universidades, papers de seguridad):

    Input (512)
        ↓
    Dense(256, relu)
        ↓
    Dropout(0.3)
        ↓
    Dense(128, relu)
        ↓
    Dropout(0.2)
        ↓
    Dense(num_clases, softmax)

Uso:
    python dataset/scripts/3_train_classifier.py
"""

import os
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
EMBEDDINGS_DIR = "dataset/embeddings"
MODELS_DIR = "dataset/models"
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings_dataset.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# Hiperparámetros
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2


def load_data():
    """Carga los embeddings extraídos en el paso 2."""
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"❌ No se encontraron embeddings en {EMBEDDINGS_PATH}\n"
            "   Ejecuta primero: python dataset/scripts/2_preprocess_and_extract_embeddings.py"
        )
    
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
    
    return data['X'], data['y'], data['idx_to_label']


def build_model(num_classes):
    """
    Construye la red neuronal para clasificar embeddings.
    
    Esta es la arquitectura que usan:
    - Bancos para autenticación
    - Universidades en investigación
    - Papers de fusión biométrica
    - Compañías de seguridad
    """
    model = models.Sequential([
        # Entrada: embedding de 512 dimensiones
        layers.Input(shape=(512,)),
        
        # Capa oculta 1
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),  # Previene overfitting
        
        # Capa oculta 2
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        
        # Capa de salida: una neurona por clase
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def plot_training_history(history, save_path):
    """Guarda gráficas del entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Accuracy durante el entrenamiento', fontsize=14)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Loss durante el entrenamiento', fontsize=14)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   📊 Gráfica guardada: {save_path}")


def main():
    print("="*70)
    print("PASO 3: ENTRENAR TU RED NEURONAL")
    print("="*70)
    print("""
    Aquí entrenas TU PROPIO MODELO sobre los embeddings de FaceNet.
    
    FaceNet ya extrajo las características faciales.
    Tu red solo aprende a clasificar esos embeddings.
    
    Arquitectura:
    ┌────────────────────────────────────────┐
    │  Input (512)                           │
    │     ↓                                  │
    │  Dense(256, relu) + Dropout(0.3)       │
    │     ↓                                  │
    │  Dense(128, relu) + Dropout(0.2)       │
    │     ↓                                  │
    │  Dense(num_clases, softmax)            │
    └────────────────────────────────────────┘
    """)
    
    # Cargar datos
    print("📂 Cargando embeddings...")
    X, y, idx_to_label = load_data()
    num_classes = len(idx_to_label)
    
    print(f"   X shape: {X.shape} (muestras, embedding_dim)")
    print(f"   y shape: {y.shape} (etiquetas)")
    print(f"   Clases: {num_classes}")
    for idx, name in idx_to_label.items():
        count = (y == idx).sum()
        print(f"      {idx}: {name} ({count} muestras)")
    
    # Split train/validation
    print(f"\n📊 Dividiendo datos ({int((1-VALIDATION_SPLIT)*100)}% train, {int(VALIDATION_SPLIT*100)}% val)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=VALIDATION_SPLIT, 
        random_state=42, 
        stratify=y
    )
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Val: {len(X_val)} muestras")
    
    # Construir modelo
    print("\n🏗️  Construyendo modelo...")
    model = build_model(num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📋 Arquitectura del modelo:")
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "face_classifier_best.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Entrenar
    print("\n" + "="*60)
    print("🚀 ENTRENANDO...")
    print("="*60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluar
    print("\n" + "="*60)
    print("📊 EVALUACIÓN FINAL")
    print("="*60)
    
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    
    print("\n🎯 Reporte de clasificación:")
    class_names = [idx_to_label[i] for i in range(num_classes)]
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    print("📉 Matriz de confusión:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    accuracy = (y_pred == y_val).mean()
    print(f"\n✅ Accuracy final: {accuracy*100:.2f}%")
    
    # Guardar modelo
    model_path = os.path.join(MODELS_DIR, "face_classifier.keras")
    model.save(model_path)
    print(f"\n💾 Modelo guardado: {model_path}")
    
    # Guardar mapeo de clases
    class_indices_path = os.path.join(MODELS_DIR, "class_indices.json")
    with open(class_indices_path, 'w') as f:
        json.dump(idx_to_label, f, indent=4)
    print(f"💾 Mapeo de clases: {class_indices_path}")
    
    # Guardar gráfica
    plot_path = os.path.join(MODELS_DIR, "training_history.png")
    plot_training_history(history, plot_path)
    
    print("\n" + "="*60)
    print("✅ PASO 3 COMPLETADO")
    print("="*60)
    print(f"""
    Tu modelo entrenado está en: {model_path}
    
    Siguiente paso:
        python dataset/scripts/4_predict.py
    
    El modelo predice así:
    
        imagen → MTCNN → FaceNet → embedding → TU MODELO → probabilidades
                                                              ↓
                                                    Freddy: 92%
                                                    Melanie: 7%
                                                    Jose: 1%
    
        Si max_prob >= 50% → Es esa persona
        Si max_prob <  50% → DESCONOCIDO
    """)


if __name__ == "__main__":
    main()
