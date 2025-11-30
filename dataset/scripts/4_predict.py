import os
import json
import cv2
import random
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

TEST_DIR = "dataset/test_data/test_data"
CLEAN_DIR = "dataset/dataset_clean"
MODELS_DIR = "dataset/models"
MODEL_PATH = os.path.join(MODELS_DIR, "faces_cnn_best.keras")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")

IMG_SIZE = (160, 160)

print("\n Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    idx_to_class = json.load(f)

detector = MTCNN()

def detectar_rostro(path):
    img = cv2.imread(path)
    if img is None:
        return None, None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        return None, img

    box = faces[0]["box"]
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    rostro = rgb[y:y+h, x:x+w]

    try:
        rostro = cv2.resize(rostro, IMG_SIZE)
    except:
        return None, img

    rostro = rostro.astype("float32") / 255.0
    return rostro, img

def predecir(path):
    rostro, _ = detectar_rostro(path)

    if rostro is None:
        return "No detectado", 0

    rostro = np.expand_dims(rostro, axis=0)
    preds = model.predict(rostro, verbose=0)[0]

    idx = int(np.argmax(preds))
    name = idx_to_class[str(idx)]
    conf = preds[idx] * 100
    return name, conf


print("\n Probando imágenes externas...\n")

for file in os.listdir(TEST_DIR):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(TEST_DIR, file)
    pred, conf = predecir(path)

    if pred == "No detectado":
        print(f"❌ {file} → No se detectó rostro")
    else:
        print(f"✅ {file} → {pred} ({conf:.2f}%)")


print("\n Probando muestras internas del dataset_clean...\n")

for person in os.listdir(CLEAN_DIR):
    folder = os.path.join(CLEAN_DIR, person)

    if not os.path.isdir(folder):
        continue

    imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".png",".jpeg"))]

    if len(imgs) == 0:
        continue

    sample = random.choice(imgs)
    sample_path = os.path.join(folder, sample)

    pred, conf = predecir(sample_path)

    print(f" {person:<10} | archivo: {sample:<20} → {pred} ({conf:.2f}%)")
