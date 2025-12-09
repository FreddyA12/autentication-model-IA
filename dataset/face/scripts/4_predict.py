"""
=============================================================================
PASO 4: PREDICCIÓN
=============================================================================

Cómo funciona la predicción:

1. Detectas cara con MTCNN
2. Alinear
3. Pasas esa cara por FaceNet
4. Obtienes embedding (512)
5. El embedding va a tu modelo entrenado
6. Tu modelo dice:
       Freddy: 92%
       Melanie: 7%
       Jose: 1%

7. Aplicas umbral:
   - Si la clase más probable está >= 50% → ES esa persona
   - Si está < 50% → NO ES (DESCONOCIDO)

Uso:
    python dataset/scripts/4_predict.py
"""

import os
import json
import random
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import tensorflow as tf

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
TEST_DIR = "dataset/face/test_data/test_data"
CLEAN_DIR = "dataset/face/aligned"
MODELS_DIR = "dataset/face/models"

IMG_SIZE = 160
CONFIDENCE_THRESHOLD = 0.80  # 80% - Si está debajo, es DESCONOCIDO

# ============================================================================
# CARGAR MODELOS
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Dispositivo: {device}")

print("\n📦 Cargando modelos...")

# MTCNN para detectar y alinear caras
mtcnn = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    post_process=True,
    device=device,
    keep_all=False
)

# FaceNet para extraer embeddings
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# TU modelo entrenado (clasificador)
model_path = os.path.join(MODELS_DIR, "face_classifier.keras")
classifier = None
if os.path.exists(model_path):
    classifier = tf.keras.models.load_model(model_path)
    print(f"   ✅ Clasificador cargado: {model_path}")
else:
    # Intentar cargar el modelo best
    model_path = os.path.join(MODELS_DIR, "face_classifier_best.keras")
    if os.path.exists(model_path):
        classifier = tf.keras.models.load_model(model_path)
        print(f"   ✅ Clasificador cargado: {model_path}")
    else:
        print(f"   ❌ No se encontró clasificador")

# Mapeo de clases
class_indices_path = os.path.join(MODELS_DIR, "class_indices.json")
idx_to_label = None
if os.path.exists(class_indices_path):
    with open(class_indices_path, 'r') as f:
        idx_to_label = json.load(f)
    # Convertir keys a int
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    print(f"   ✅ Clases: {list(idx_to_label.values())}")
else:
    print(f"   ❌ No se encontró mapeo de clases")


def extract_embedding(img_path):
    """
    Pipeline completo:
    imagen → MTCNN (detectar/alinear) → FaceNet → embedding
    """
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return None
    
    # Detectar y alinear cara
    face_tensor = mtcnn(img)
    
    if face_tensor is None:
        return None
    
    # Extraer embedding con FaceNet
    with torch.no_grad():
        face_tensor = face_tensor.unsqueeze(0).to(device)
        embedding = facenet(face_tensor)
    
    return embedding.cpu().numpy().flatten()


def predict(img_path):
    """
    Predicción completa:
    
    1. Extrae embedding de la imagen
    2. Pasa por tu clasificador
    3. Obtiene probabilidades para cada clase
    4. Aplica umbral
    
    Returns:
        (nombre_predicho, confianza, todas_las_probabilidades)
    """
    if classifier is None or idx_to_label is None:
        return "ERROR", 0, {}
    
    # Extraer embedding
    embedding = extract_embedding(img_path)
    
    if embedding is None:
        return "NO_DETECTADO", 0, {}
    
    # Pasar por TU clasificador
    probs = classifier.predict(np.array([embedding]), verbose=0)[0]
    
    # Obtener predicción
    pred_idx = np.argmax(probs)
    pred_conf = probs[pred_idx]
    pred_name = idx_to_label[pred_idx]
    
    # Crear diccionario de probabilidades
    all_probs = {idx_to_label[i]: float(probs[i]) for i in range(len(probs))}
    
    # Aplicar umbral
    if pred_conf < CONFIDENCE_THRESHOLD:
        return "DESCONOCIDO", pred_conf, all_probs
    
    return pred_name, pred_conf, all_probs


# ============================================================================
# PRUEBAS
# ============================================================================

if classifier is None:
    print("\n❌ No se encontró el modelo clasificador.")
    print("   Ejecuta primero:")
    print("   1. python dataset/scripts/2_preprocess_and_extract_embeddings.py")
    print("   2. python dataset/scripts/3_train_classifier.py")
else:
    print("\n" + "="*70)
    print("🧪 PROBANDO IMÁGENES EXTERNAS")
    print("="*70)
    print(f"   Directorio: {TEST_DIR}")
    print(f"   Umbral de confianza: {CONFIDENCE_THRESHOLD*100:.0f}%\n")

    if os.path.exists(TEST_DIR):
        for file in sorted(os.listdir(TEST_DIR)):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = os.path.join(TEST_DIR, file)
            name, conf, probs = predict(path)

            if name == "NO_DETECTADO":
                print(f"   ❌ {file:<20} → No se detectó rostro")
            elif name == "DESCONOCIDO":
                print(f"   ⚠️  {file:<20} → DESCONOCIDO (max: {conf*100:.1f}%)")
                for person, prob in sorted(probs.items(), key=lambda x: -x[1]):
                    print(f"       └─ {person}: {prob*100:.1f}%")
            else:
                print(f"   ✅ {file:<20} → {name} ({conf*100:.1f}%)")

    print("\n" + "="*70)
    print("🧪 PROBANDO MUESTRAS DEL DATASET")
    print("="*70 + "\n")

    correct = 0
    total = 0

    for person in sorted(os.listdir(CLEAN_DIR)):
        folder = os.path.join(CLEAN_DIR, person)

        if not os.path.isdir(folder):
            continue

        imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".png",".jpeg"))]

        if len(imgs) == 0:
            continue

        # Probar 3 muestras aleatorias
        samples = random.sample(imgs, min(3, len(imgs)))

        for sample in samples:
            path = os.path.join(folder, sample)
            pred_name, conf, probs = predict(path)

            total += 1

            if pred_name == person:
                correct += 1
                emoji = "✅"
            elif pred_name == "DESCONOCIDO":
                emoji = "⚠️"
            else:
                emoji = "❌"

            print(f"   {emoji} {person:<10} | {sample:<25} → {pred_name:<12} ({conf*100:.1f}%)")

    if total > 0:
        print(f"\n   📊 Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
