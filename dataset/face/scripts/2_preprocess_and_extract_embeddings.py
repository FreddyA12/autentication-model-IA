import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Configuración
RAW_DIR = "dataset/face/processed"
CLEAN_DIR = "dataset/face/aligned"
EMBEDDINGS_DIR = "dataset/face/embeddings"
IMG_SIZE = 160

# Cargar Modelos
print("Cargando modelos...")
detector = MTCNN()
embedder = FaceNet()

def process_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None: return None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        return None, None
    
    # Detectar caras
    results = detector.detect_faces(img_rgb)
    if not results:
        return None, None
    
    # Obtener la mejor cara
    best_face = max(results, key=lambda x: x['confidence'])
    x, y, w, h = best_face['box']
    x, y = max(0, x), max(0, y)
    
    # Recortar y redimensionar
    face = img_rgb[y:y+h, x:x+w]
    try:
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    except:
        return None, None
        
    # Extraer embedding
    #(batch_size, height, width, channels)
    samples = np.expand_dims(face_resized, axis=0)
    
    #(convoluciones, normalizaciones, dense layers) 512 D
    embedding = embedder.embeddings(samples)[0]
    
    return face_resized, embedding

def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    all_embeddings = []
    all_labels = []
    label_to_idx = {}
    idx_to_label = {}
    
    if not os.path.exists(RAW_DIR):
        print(f"No se encontró el directorio {RAW_DIR}")
        return

    persons = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    
    for idx, person in enumerate(persons):
        label_to_idx[person] = idx
        idx_to_label[idx] = person
        
        src_dir = os.path.join(RAW_DIR, person)
        dst_dir = os.path.join(CLEAN_DIR, person)
        os.makedirs(dst_dir, exist_ok=True)
        
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Procesando {person} ({len(images)} imágenes)...")
        
        for img_name in tqdm(images):
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(dst_dir, img_name)
            
            face_np, embedding = process_image(src_path)
            
            if embedding is not None:
                # Guardar imagen alineada
                face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(dst_path, face_bgr)
                
                # Guardar embedding
                all_embeddings.append(embedding)
                all_labels.append(idx)
    
    # Guardar datos
    X = np.array(all_embeddings)
    y = np.array(all_labels)
    
    data = {
        'X': X,
        'y': y,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label
    }
    
    embeddings_path = os.path.join(EMBEDDINGS_DIR, "embeddings_dataset.pkl")
    with open(embeddings_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Guardados {len(X)} embeddings en {embeddings_path}")

if __name__ == "__main__":
    main()
