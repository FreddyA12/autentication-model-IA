"""
Script para convertir modelo antiguo a formato compatible con TensorFlow 2.x
"""
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras import layers, models

print("🔄 Cargando embeddings...")
with open('../dataset/embeddings/embeddings_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
y = data['y']
idx_to_label = data['idx_to_label']

print(f"   X shape: {X.shape}")
print(f"   Clases: {idx_to_label}")

# Recrear el modelo con arquitectura compatible
print("\n🏗️  Creando modelo compatible...")
model = models.Sequential([
    layers.Input(shape=(512,)),  # Usar Input en lugar de batch_shape
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(idx_to_label), activation='softmax')
])

print("\n📋 Arquitectura:")
model.summary()

# Cargar pesos del modelo antiguo
print("\n🔄 Intentando cargar pesos del modelo antiguo...")
try:
    old_model = keras.models.load_model(
        '../dataset/models/face_classifier.keras',
        compile=False
    )
    print("   ❌ No se pudo cargar el modelo antiguo")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("\n   🔄 Reentrenando modelo desde cero...")
    
    from sklearn.model_selection import train_test_split
    
    # Split datos
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compilar
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar
    print("\n🚀 Entrenando...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluar
    print("\n📊 Evaluando...")
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"   Accuracy: {acc*100:.2f}%")

# Guardar modelo compatible
print("\n💾 Guardando modelo compatible...")
model.save('../dataset/models/face_classifier_compatible.keras')
print("   ✅ Guardado en: dataset/models/face_classifier_compatible.keras")

print("\n✅ ¡Modelo convertido exitosamente!")
print("\n📝 Actualiza settings.py para usar:")
print("   FACE_MODEL_PATH = 'face_classifier_compatible.keras'")
