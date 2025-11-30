import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

RAW_DIR = "dataset/dataset_raw"
CLEAN_DIR = "dataset/dataset_clean"
IMG_SIZE = 160

os.makedirs(CLEAN_DIR, exist_ok=True)
detector = MTCNN()

def procesar(path):
    img = cv2.imread(path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    if len(faces) == 0:
        return None

    f = max(faces, key=lambda x: x["box"][2] * x["box"][3])
    x, y, w, h = f["box"]

    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)

    crop = rgb[y:y+h, x:x+w]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

for user in os.listdir(RAW_DIR):
    src = os.path.join(RAW_DIR, user)
    dst = os.path.join(CLEAN_DIR, user)

    if not os.path.isdir(src):
        continue

    os.makedirs(dst, exist_ok=True)
    print(f"Procesando {user}...")

    for img_name in tqdm(os.listdir(src)):
        p = os.path.join(src, img_name)
        rostro = procesar(p)
        if rostro is not None:
            cv2.imwrite(os.path.join(dst, img_name), rostro)

print("Preprocesamiento completado — dataset_clean listo")
