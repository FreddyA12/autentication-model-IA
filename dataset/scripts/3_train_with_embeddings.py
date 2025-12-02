

import os
import pickle
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# También una red neuronal simple como alternativa
import tensorflow as tf
from tensorflow.keras import layers, models

EMBEDDINGS_DIR = "dataset/embeddings"
MODELS_DIR = "dataset/models"

os.makedirs(MODELS_DIR, exist_ok=True)


def load_embeddings():
    """Carga los embeddings pre-extraídos."""
    embeddings_path = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"No se encontraron embeddings en {embeddings_path}.\n"
            "Ejecuta primero: python dataset/scripts/2_preprocess_aligned.py"
        )
    
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['embeddings'], data['labels']


def prepare_data(embeddings_dict, labels_dict):
    """Prepara los datos para entrenamiento."""
    X = []
    y = []
    
    for person in embeddings_dict:
        for emb in embeddings_dict[person]:
            X.append(emb)
            y.append(person)
    
    X = np.array(X)
    y = np.array(y)
    
    # Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le


def train_svm_classifier(X, y, le):
    """
    Entrena un clasificador SVM sobre los embeddings.
    SVM funciona muy bien con embeddings de alta dimensión.
    """
    print("\n" + "="*50)
    print("ENTRENANDO CLASIFICADOR SVM")
    print("="*50)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDatos de entrenamiento: {len(X_train)}")
    print(f"Datos de prueba: {len(X_test)}")
    
    # Entrenar SVM con kernel RBF
    print("\nEntrenando SVM...")
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,  # Para obtener probabilidades
        class_weight='balanced'
    )
    svm.fit(X_train, y_train)
    
    # Evaluar
    print("\n" + "-"*40)
    print("EVALUACIÓN EN TEST SET")
    print("-"*40)
    
    y_pred = svm.predict(X_test)
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    # Accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    
    # Cross-validation
    print("\nValidación cruzada (5-fold):")
    cv_scores = cross_val_score(svm, X, y, cv=5)
    print(f"  Scores: {cv_scores}")
    print(f"  Media: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    
    # Guardar modelo
    model_path = os.path.join(MODELS_DIR, "face_svm.pkl")
    joblib.dump(svm, model_path)
    print(f"\nModelo SVM guardado en: {model_path}")
    
    return svm, accuracy


def train_neural_classifier(X, y, le, num_classes):
    """
    Entrena una red neuronal pequeña sobre los embeddings.
    """
    print("\n" + "="*50)
    print("ENTRENANDO CLASIFICADOR NEURAL")
    print("="*50)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Modelo pequeño (embeddings ya son features de alta calidad)
    model = models.Sequential([
        layers.Input(shape=(512,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Entrenar
    print("\nEntrenando...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluar
    print("\n" + "-"*40)
    print("EVALUACIÓN EN TEST SET")
    print("-"*40)
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    accuracy = (y_pred == y_test).mean()
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    
    # Guardar modelo
    model_path = os.path.join(MODELS_DIR, "face_embedding_classifier.keras")
    model.save(model_path)
    print(f"\nModelo neural guardado en: {model_path}")
    
    return model, accuracy


def save_label_encoder(le):
    """Guarda el mapeo de clases."""
    class_indices = {i: name for i, name in enumerate(le.classes_)}
    
    path = os.path.join(MODELS_DIR, "class_indices.json")
    with open(path, 'w') as f:
        json.dump(class_indices, f, indent=4)
    
    print(f"Mapeo de clases guardado en: {path}")


def main():
    print("="*60)
    print("ENTRENAMIENTO CON FACE EMBEDDINGS")
    print("="*60)
    
    # Cargar embeddings
    print("\nCargando embeddings...")
    embeddings_dict, labels_dict = load_embeddings()
    
    for person, embs in embeddings_dict.items():
        print(f"  {person}: {len(embs)} embeddings")
    
    # Preparar datos
    X, y, le = prepare_data(embeddings_dict, labels_dict)
    num_classes = len(le.classes_)
    
    print(f"\nTotal de muestras: {len(X)}")
    print(f"Dimensión de embeddings: {X.shape[1]}")
    print(f"Número de clases: {num_classes}")
    
    # Entrenar ambos clasificadores
    svm_model, svm_acc = train_svm_classifier(X, y, le)
    nn_model, nn_acc = train_neural_classifier(X, y, le, num_classes)
    
    # Guardar label encoder
    save_label_encoder(le)
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"\n  SVM Accuracy:    {svm_acc*100:.2f}%")
    print(f"  Neural Accuracy: {nn_acc*100:.2f}%")
    print("\nModelos guardados en:", MODELS_DIR)
    print("\nPara predecir, usa:")
    print("  python dataset/scripts/4_predict_embeddings.py")


if __name__ == "__main__":
    main()
