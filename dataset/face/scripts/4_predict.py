import os
import json
import random
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Configuración
TEST_DIR = "dataset/face/test_data"
CLEAN_DIR = "dataset/face/aligned"
MODELS_DIR = "dataset/face/models"

IMG_SIZE = 160
CONFIDENCE_THRESHOLD = 0.80

# Cargar Modelos
print("Cargando modelos...")
mtcnn = MTCNN()
facenet = FaceNet()

model_path = os.path.join(MODELS_DIR, "face_classifier.keras")
classifier = None

if os.path.exists(model_path):
    classifier = tf.keras.models.load_model(model_path)
    print(f"Clasificador cargado: {model_path}")
else:
    print("No se encontró el clasificador")

# Mapeo de clases
class_indices_path = os.path.join(MODELS_DIR, "class_indices.json")
idx_to_label = None
if os.path.exists(class_indices_path):
    with open(class_indices_path, 'r') as f:
        idx_to_label = json.load(f)
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}

def extract_embedding(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None: return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        return None
    
    # Detectar caras
    results = mtcnn.detect_faces(img_rgb)
    if not results:
        return None
    
    # Mejor cara
    best_face = max(results, key=lambda x: x['confidence'])
    x, y, w, h = best_face['box']
    x, y = max(0, x), max(0, y)
    
    # Recortar
    face = img_rgb[y:y+h, x:x+w]
    try:
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    except:
        return None
        
    # Extraer embedding
    samples = np.expand_dims(face_resized, axis=0)
    embedding = facenet.embeddings(samples)[0]
    
    return embedding

def predict(img_path):
    if classifier is None or idx_to_label is None:
        return "ERROR", 0, {}
    
    embedding = extract_embedding(img_path)
    if embedding is None:
        return "NO_DETECTADO", 0, {}
    
    probs = classifier.predict(np.array([embedding]), verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_conf = probs[pred_idx]
    pred_name = idx_to_label[pred_idx]
    
    all_probs = {idx_to_label[i]: float(probs[i]) for i in range(len(probs))}
    
    if pred_conf < CONFIDENCE_THRESHOLD:
        return "DESCONOCIDO", pred_conf, all_probs
    
    return pred_name, pred_conf, all_probs

if __name__ == "__main__":
    if classifier is None:
        print("Modelo no cargado.")
    else:
        print("\n--- PRUEBA CON DATOS EXTERNOS ---")
        if os.path.exists(TEST_DIR):
            for file in sorted(os.listdir(TEST_DIR)):
                if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                path = os.path.join(TEST_DIR, file)
                name, conf, probs = predict(path)

                if name == "NO_DETECTADO":
                    print(f"{file:<20} -> No detectado")
                elif name == "DESCONOCIDO":
                    print(f"{file:<20} -> DESCONOCIDO ({conf*100:.1f}%)")
                else:
                    print(f"{file:<20} -> {name} ({conf*100:.1f}%)")

        print("\n--- PRUEBA CON DATASET ---")
        correct = 0
        total = 0

        if os.path.exists(CLEAN_DIR):
            for person in sorted(os.listdir(CLEAN_DIR)):
                folder = os.path.join(CLEAN_DIR, person)
                if not os.path.isdir(folder): continue

                imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".png",".jpeg"))]
                if not imgs: continue

                samples = random.sample(imgs, min(3, len(imgs)))

                for sample in samples:
                    path = os.path.join(folder, sample)
                    pred_name, conf, probs = predict(path)
                    total += 1
                    
                    if pred_name == person: correct += 1
                    
                    print(f"{person:<10} | {sample:<25} -> {pred_name:<12} ({conf*100:.1f}%)")

            if total > 0:
                print(f"\nPrecisión: {correct}/{total} ({correct/total*100:.1f}%)")
