"""
Predicción usando Face Embeddings (FaceNet + SVM/Neural).

Este script:
1. Detecta y alinea la cara con MTCNN
2. Extrae el embedding con FaceNet
3. Clasifica con SVM o red neural
"""

import os
import json
import numpy as np
import torch
import cv2
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import tensorflow as tf

# Configuración
TEST_DIR = "dataset/test_data/test_data"
CLEAN_DIR = "dataset/dataset_clean"
MODELS_DIR = "dataset/models"
EMBEDDINGS_DIR = "dataset/embeddings"

IMG_SIZE = 160
CONFIDENCE_THRESHOLD = 0.60  # Umbral para marcar como desconocido
DISTANCE_THRESHOLD = 1.0     # Umbral de distancia para embeddings

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Cargar modelos
print("\nCargando modelos...")

# MTCNN para detección y alineamiento
mtcnn = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    post_process=True,
    device=device,
    keep_all=False
)

# FaceNet para embeddings
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Clasificadores
svm_path = os.path.join(MODELS_DIR, "face_svm.pkl")
nn_path = os.path.join(MODELS_DIR, "face_embedding_classifier.keras")

svm_model = None
nn_model = None

if os.path.exists(svm_path):
    svm_model = joblib.load(svm_path)
    print("  ✓ SVM cargado")

if os.path.exists(nn_path):
    nn_model = tf.keras.models.load_model(nn_path)
    print("  ✓ Red neural cargada")

# Mapeo de clases
class_indices_path = os.path.join(MODELS_DIR, "class_indices.json")
with open(class_indices_path, 'r') as f:
    idx_to_class = json.load(f)

# Cargar embeddings de referencia para verificación por distancia
embeddings_path = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")
reference_embeddings = None
if os.path.exists(embeddings_path):
    import pickle
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    reference_embeddings = data['embeddings']
    print("  ✓ Embeddings de referencia cargados")


def extract_embedding(img_path):
    """
    Extrae el embedding de una imagen.
    Retorna el embedding o None si no se detecta cara.
    """
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return None
    
    # Detectar y alinear
    face_tensor = mtcnn(img)
    
    if face_tensor is None:
        return None
    
    # Extraer embedding
    with torch.no_grad():
        face_tensor = face_tensor.unsqueeze(0).to(device)
        embedding = facenet(face_tensor)
    
    return embedding.cpu().numpy().flatten()


def find_closest_person(embedding):
    """
    Encuentra la persona más cercana por distancia euclidiana.
    Retorna (nombre, distancia_mínima).
    """
    if reference_embeddings is None:
        return None, float('inf')
    
    min_dist = float('inf')
    closest_person = None
    
    for person, embs in reference_embeddings.items():
        for ref_emb in embs:
            dist = np.linalg.norm(embedding - ref_emb)
            if dist < min_dist:
                min_dist = dist
                closest_person = person
    
    return closest_person, min_dist


def predict_svm(embedding):
    """Predicción con SVM."""
    if svm_model is None:
        return None, 0
    
    proba = svm_model.predict_proba([embedding])[0]
    idx = np.argmax(proba)
    conf = proba[idx]
    name = idx_to_class[str(idx)]
    
    return name, conf


def predict_nn(embedding):
    """Predicción con red neural."""
    if nn_model is None:
        return None, 0
    
    proba = nn_model.predict(np.array([embedding]), verbose=0)[0]
    idx = np.argmax(proba)
    conf = proba[idx]
    name = idx_to_class[str(idx)]
    
    return name, conf


def predict(img_path, method='svm'):
    """
    Predicción completa con verificación de umbral.
    
    Args:
        img_path: Ruta a la imagen
        method: 'svm', 'neural', o 'distance'
    
    Returns:
        (nombre, confianza, info_extra)
    """
    embedding = extract_embedding(img_path)
    
    if embedding is None:
        return "No detectado", 0, {}
    
    # Verificación por distancia (siempre útil)
    closest, min_dist = find_closest_person(embedding)
    
    if method == 'distance':
        # Clasificación puramente por distancia
        if min_dist > DISTANCE_THRESHOLD:
            return "DESCONOCIDO", 1 - min_dist/2, {'distancia': min_dist}
        return closest, 1 - min_dist/2, {'distancia': min_dist}
    
    elif method == 'svm':
        name, conf = predict_svm(embedding)
    else:
        name, conf = predict_nn(embedding)
    
    # Verificar umbral de confianza
    if conf < CONFIDENCE_THRESHOLD:
        return "DESCONOCIDO", conf, {'distancia': min_dist, 'closest': closest}
    
    # Verificar consistencia con distancia
    if min_dist > DISTANCE_THRESHOLD * 1.2:
        return "DESCONOCIDO", conf, {'distancia': min_dist, 'pred': name}
    
    return name, conf, {'distancia': min_dist}


def test_images():
    """Prueba con imágenes de test."""
    print("\n" + "="*60)
    print("PROBANDO IMÁGENES EXTERNAS")
    print("="*60 + "\n")
    
    if not os.path.exists(TEST_DIR):
        print(f"Directorio no encontrado: {TEST_DIR}")
        return
    
    for file in sorted(os.listdir(TEST_DIR)):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        path = os.path.join(TEST_DIR, file)
        
        # Probar con SVM
        name, conf, info = predict(path, method='svm')
        
        if name == "No detectado":
            print(f"❌ {file:<20} → No se detectó rostro")
        elif name == "DESCONOCIDO":
            dist = info.get('distancia', 0)
            print(f"⚠️  {file:<20} → DESCONOCIDO (conf: {conf*100:.1f}%, dist: {dist:.2f})")
        else:
            dist = info.get('distancia', 0)
            print(f"✅ {file:<20} → {name} (conf: {conf*100:.1f}%, dist: {dist:.2f})")


def test_dataset_samples():
    """Prueba con muestras del dataset."""
    import random
    
    print("\n" + "="*60)
    print("PROBANDO MUESTRAS DEL DATASET")
    print("="*60 + "\n")
    
    correct = 0
    total = 0
    
    for person in sorted(os.listdir(CLEAN_DIR)):
        folder = os.path.join(CLEAN_DIR, person)
        if not os.path.isdir(folder):
            continue
        
        imgs = [f for f in os.listdir(folder) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not imgs:
            continue
        
        # Probar 3 muestras aleatorias
        samples = random.sample(imgs, min(3, len(imgs)))
        
        for sample in samples:
            path = os.path.join(folder, sample)
            name, conf, info = predict(path, method='svm')
            
            total += 1
            dist = info.get('distancia', 0)
            
            if name == person:
                correct += 1
                emoji = "✅"
            elif name == "DESCONOCIDO":
                emoji = "⚠️"
            else:
                emoji = "❌"
            
            print(f"{emoji} {person:<10} | {sample:<25} → {name:<12} (conf: {conf*100:.1f}%, dist: {dist:.2f})")
    
    if total > 0:
        print(f"\nAccuracy en muestras: {correct}/{total} ({correct/total*100:.1f}%)")


if __name__ == "__main__":
    print("="*60)
    print("PREDICCIÓN CON FACE EMBEDDINGS")
    print("="*60)
    
    if svm_model is None and nn_model is None:
        print("\n❌ No se encontraron modelos entrenados.")
        print("   Ejecuta primero:")
        print("   1. python dataset/scripts/2_preprocess_aligned.py")
        print("   2. python dataset/scripts/3_train_with_embeddings.py")
    else:
        test_images()
        test_dataset_samples()
