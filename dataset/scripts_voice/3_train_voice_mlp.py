"""
=============================================================================
PASO 3: ENTRENAR TU MLP (MULTI-LAYER PERCEPTRON)
=============================================================================

Aquí entrenas TU PROPIA red neuronal sobre los embeddings de ECAPA-TDNN.
Es exactamente igual a entrenar el clasificador de rostros sobre FaceNet.

ECAPA-TDNN ya extrajo las características.
Tu MLP solo aprende a clasificar esas características.

Arquitectura:
    Input(192) → Dense(256) → Dropout → Dense(128) → Dropout → Dense(num_clases)

Uso:
    python dataset/scripts_voice/3_train_voice_mlp.py
"""

import os
import numpy as np
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
EMBEDDINGS_DIR = Path("dataset/embeddings")
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "voice_embeddings.npy"
LABELS_FILE = EMBEDDINGS_DIR / "voice_labels.npy"
LABEL_MAP_FILE = EMBEDDINGS_DIR / "voice_label_map.json"

MODELS_DIR = Path("dataset/models")
MODEL_PATH = MODELS_DIR / "voice_mlp.keras"
BEST_MODEL_PATH = MODELS_DIR / "voice_mlp_best.keras"
CLASS_INDICES_PATH = MODELS_DIR / "voice_class_indices.json"
HISTORY_PATH = MODELS_DIR / "voice_training_history.png"

EPOCHS = 50
BATCH_SIZE = 16
TEST_SIZE = 0.2
RANDOM_SEED = 42


def build_mlp(input_dim, num_classes):
    """
    Construye la arquitectura MLP
    
    Args:
        input_dim: Dimensión de entrada (192 para ECAPA-TDNN)
        num_classes: Número de personas a clasificar
    
    Returns:
        Modelo compilado
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(256, activation='relu', name='dense1'),
        layers.Dropout(0.3, name='dropout1'),
        
        layers.Dense(128, activation='relu', name='dense2'),
        layers.Dropout(0.3, name='dropout2'),
        
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='VoiceMLP')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    print("="*70)
    print("PASO 3: ENTRENAR TU MLP DE VOZ")
    print("="*70)
    print("""
    Aquí entrenas TU PROPIO MODELO sobre los embeddings de ECAPA-TDNN.
    
    ECAPA-TDNN ya extrajo las características de voz (192 números).
    Tu MLP aprende a clasificar esos embeddings.
    
    Arquitectura de TU red neuronal:
    ┌────────────────────────────────────────┐
    │  Input (192)                           │
    │     ↓                                  │
    │  Dense(256, relu) + Dropout(0.3)       │
    │     ↓                                  │
    │  Dense(128, relu) + Dropout(0.3)       │
    │     ↓                                  │
    │  Dense(num_clases, softmax)            │
    └────────────────────────────────────────┘
    
    Esto ES entrenamiento supervisado con epochs, backprop, etc.
    """)
    
    # Verificar que existen los embeddings
    if not EMBEDDINGS_FILE.exists():
        print(f"❌ No se encontró {EMBEDDINGS_FILE}")
        print("   Ejecuta primero: python dataset/scripts_voice/2_generate_voice_embeddings.py")
        return
    
    # Cargar datos
    print("📂 Cargando embeddings...")
    X = np.load(EMBEDDINGS_FILE)
    y = np.load(LABELS_FILE)
    
    with open(LABEL_MAP_FILE, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
        # Convertir keys a int
        label_map = {int(k): v for k, v in label_map.items()}
    
    num_classes = len(label_map)
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Clases: {num_classes}")
    
    print("\n📊 Distribución de datos:")
    for label_idx, person_name in label_map.items():
        count = (y == label_idx).sum()
        print(f"   {label_idx}: {person_name:12s} → {count} muestras")
    
    # Dividir en train/test
    train_pct = int((1-TEST_SIZE)*100)
    val_pct = int(TEST_SIZE*100)
    print(f"\n🔀 Dividiendo datos ({train_pct}% train, {val_pct}% val)...")
    
    # Si hay muy pocas muestras, no usar stratify
    if len(X) < 20:
        print("   ⚠️  Dataset pequeño: usando split sin stratify")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_SEED
        )
    
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Val:   {len(X_val)} muestras")
    
    # Construir modelo
    print("\n🏗️  Construyendo MLP...")
    model = build_mlp(input_dim=X.shape[1], num_classes=num_classes)
    
    print("\n📋 Arquitectura del modelo:")
    model.summary()
    
    print(f"\n{'='*70}")
    print(f"Total de parámetros: {model.count_params():,}")
    print(f"Estos son los pesos que TÚ vas a entrenar con tus datos.")
    print(f"{'='*70}")
    
    # Callbacks
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(BEST_MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Entrenar
    print("\n" + "="*70)
    print("🚀 ENTRENAMIENTO EN PROGRESO...")
    print("="*70 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluación final
    print("\n" + "="*70)
    print("📊 EVALUACIÓN FINAL")
    print("="*70 + "\n")
    
    # Predicciones
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Reporte de clasificación
    print("🎯 Reporte de clasificación:\n")
    target_names = [label_map[i] for i in range(num_classes)]
    
    # Obtener labels presentes en validación
    labels_in_val = np.unique(y_val)
    target_names_subset = [target_names[i] for i in labels_in_val]
    
    print(classification_report(
        y_val, y_pred, 
        labels=labels_in_val,
        target_names=target_names_subset, 
        digits=4,
        zero_division=0
    ))
    
    # Matriz de confusión
    print("\n📉 Matriz de confusión:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    print("\n   (Filas = real, Columnas = predicho)")
    
    # Accuracy final
    accuracy = (y_pred == y_val).mean()
    print(f"\n✅ Accuracy final: {accuracy*100:.2f}%")
    
    # Guardar modelo final
    model.save(str(MODEL_PATH))
    print(f"\n💾 Modelo guardado: {MODEL_PATH}")
    
    # Guardar mapeo de clases
    with open(CLASS_INDICES_PATH, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)
    print(f"💾 Mapeo de clases: {CLASS_INDICES_PATH}")
    
    # Graficar historial de entrenamiento
    print("\n📊 Generando gráficas...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(HISTORY_PATH, dpi=150)
    print(f"   ✅ Gráfica guardada: {HISTORY_PATH}")
    
    # Análisis por clase
    print("\n" + "="*70)
    print("📈 ANÁLISIS POR CLASE")
    print("="*70)
    
    for label_idx, person_name in label_map.items():
        mask = (y_val == label_idx)
        if mask.sum() == 0:
            continue
        
        correct = (y_pred[mask] == label_idx).sum()
        total = mask.sum()
        acc = correct / total * 100
        
        print(f"\n{person_name.upper()}")
        print(f"   Muestras en validación: {total}")
        print(f"   Correctas: {correct}")
        print(f"   Accuracy: {acc:.2f}%")
        
        # Confianza promedio
        confidences = y_pred_probs[mask, label_idx]
        avg_conf = confidences.mean()
        print(f"   Confianza promedio: {avg_conf*100:.2f}%")
    
    print("\n" + "="*70)
    print("✅ PASO 3 COMPLETADO")
    print("="*70)
    print(f"""
    Tu modelo MLP está entrenado y guardado en:
        {BEST_MODEL_PATH}
    
    Parámetros del modelo: {model.count_params():,}
    Accuracy alcanzado: {accuracy*100:.2f}%
    
    El modelo predice así:
    
        audio.wav → ECAPA-TDNN → embedding(192) → TU MLP → probabilidades
                                                      ↓
                                                 Freddy: 95%
                                                 Melanie: 4%
                                                 Rafael: 1%
    
    Siguiente paso:
        python dataset/scripts_voice/4_predict_voice.py <audio.wav>
    """)


if __name__ == "__main__":
    main()
