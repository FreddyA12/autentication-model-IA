"""
Preprocesamiento mejorado para reconocimiento facial:
1. Detecta rostros con MTCNN
2. Alinea las caras usando los landmarks de los ojos
3. Extrae embeddings con FaceNet (InceptionResnetV1)
4. Guarda imágenes alineadas + embeddings para entrenamiento

Esto produce datos de MUCHA mejor calidad para entrenar.
"""

import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from tqdm import tqdm
import pickle

# Configuración
RAW_DIR = "dataset/dataset_raw"
CLEAN_DIR = "dataset/dataset_clean"
EMBEDDINGS_DIR = "dataset/embeddings"
IMG_SIZE = 160

# Usar GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Inicializar MTCNN (detector de caras con landmarks)
mtcnn = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,  # Normaliza la imagen para FaceNet
    device=device,
    keep_all=False  # Solo la cara más grande/confiable
)

# FaceNet para extraer embeddings (pre-entrenado en VGGFace2)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def align_face(img_path):
    """
    Detecta, alinea y recorta la cara de una imagen.
    Retorna la imagen alineada (tensor) y la imagen como array numpy.
    """
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return None, None
    
    # MTCNN detecta y alinea automáticamente usando los landmarks
    # Retorna tensor normalizado listo para FaceNet
    face_tensor = mtcnn(img)
    
    if face_tensor is None:
        return None, None
    
    # Convertir tensor a imagen numpy para guardar
    # El tensor viene normalizado (-1, 1), lo convertimos a (0, 255)
    face_np = face_tensor.permute(1, 2, 0).numpy()
    face_np = ((face_np + 1) / 2 * 255).astype(np.uint8)
    
    return face_tensor, face_np


def extract_embedding(face_tensor):
    """
    Extrae el embedding de 512 dimensiones de una cara.
    """
    if face_tensor is None:
        return None
    
    with torch.no_grad():
        face_tensor = face_tensor.unsqueeze(0).to(device)
        embedding = facenet(face_tensor)
    
    return embedding.cpu().numpy().flatten()


def process_dataset():
    """
    Procesa todo el dataset:
    1. Alinea y guarda las imágenes
    2. Extrae y guarda los embeddings
    """
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    all_embeddings = {}
    all_labels = {}
    
    for person in os.listdir(RAW_DIR):
        src_dir = os.path.join(RAW_DIR, person)
        dst_dir = os.path.join(CLEAN_DIR, person)
        
        if not os.path.isdir(src_dir):
            continue
        
        os.makedirs(dst_dir, exist_ok=True)
        
        images = [f for f in os.listdir(src_dir) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not images:
            continue
        
        print(f"\n{'='*50}")
        print(f"Procesando: {person} ({len(images)} imágenes)")
        print(f"{'='*50}")
        
        person_embeddings = []
        processed = 0
        failed = 0
        
        for img_name in tqdm(images, desc=f"  {person}"):
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(dst_dir, img_name)
            
            # Alinear cara
            face_tensor, face_np = align_face(src_path)
            
            if face_tensor is None:
                failed += 1
                continue
            
            # Extraer embedding
            embedding = extract_embedding(face_tensor)
            
            if embedding is None:
                failed += 1
                continue
            
            # Guardar imagen alineada
            face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, face_bgr)
            
            # Guardar embedding
            person_embeddings.append(embedding)
            processed += 1
        
        print(f"  ✓ Procesadas: {processed}")
        print(f"  ✗ Fallidas: {failed}")
        
        if person_embeddings:
            all_embeddings[person] = np.array(person_embeddings)
            all_labels[person] = [person] * len(person_embeddings)
    
    # Guardar embeddings
    embeddings_path = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")
    with open(embeddings_path, 'wb') as f:
        pickle.dump({
            'embeddings': all_embeddings,
            'labels': all_labels
        }, f)
    
    print(f"\n{'='*50}")
    print("RESUMEN")
    print(f"{'='*50}")
    for person, embs in all_embeddings.items():
        print(f"  {person}: {len(embs)} embeddings")
    print(f"\nEmbeddings guardados en: {embeddings_path}")
    print(f"Imágenes alineadas en: {CLEAN_DIR}")


def verify_alignment():
    """
    Verifica la calidad del alineamiento mostrando estadísticas.
    """
    print("\n" + "="*50)
    print("VERIFICACIÓN DE CALIDAD")
    print("="*50)
    
    embeddings_path = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")
    
    if not os.path.exists(embeddings_path):
        print("No se encontraron embeddings. Ejecuta process_dataset() primero.")
        return
    
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    
    # Calcular distancias intra-clase e inter-clase
    print("\nDistancias promedio entre embeddings:")
    print("-" * 40)
    
    for person, embs in embeddings.items():
        if len(embs) < 2:
            continue
        
        # Distancia intra-clase (entre imágenes de la misma persona)
        distances = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                dist = np.linalg.norm(embs[i] - embs[j])
                distances.append(dist)
        
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        print(f"  {person}: {avg_dist:.3f} ± {std_dist:.3f}")
    
    # Distancia inter-clase (entre diferentes personas)
    persons = list(embeddings.keys())
    if len(persons) >= 2:
        print("\nDistancias entre personas diferentes:")
        print("-" * 40)
        
        for i in range(len(persons)):
            for j in range(i+1, len(persons)):
                embs1 = embeddings[persons[i]]
                embs2 = embeddings[persons[j]]
                
                # Calcular distancia promedio entre las dos personas
                distances = []
                for e1 in embs1[:50]:  # Limitar para eficiencia
                    for e2 in embs2[:50]:
                        dist = np.linalg.norm(e1 - e2)
                        distances.append(dist)
                
                avg_dist = np.mean(distances)
                print(f"  {persons[i]} vs {persons[j]}: {avg_dist:.3f}")


if __name__ == "__main__":
    print("="*60)
    print("PREPROCESAMIENTO MEJORADO CON FACENET")
    print("="*60)
    print("\nEste script:")
    print("  1. Detecta rostros con MTCNN")
    print("  2. Alinea las caras usando landmarks de los ojos")
    print("  3. Extrae embeddings de 512 dimensiones con FaceNet")
    print("  4. Guarda imágenes alineadas + embeddings")
    print()
    
    process_dataset()
    verify_alignment()
    
    print("\n" + "="*60)
    print("¡PREPROCESAMIENTO COMPLETADO!")
    print("="*60)
    print("\nAhora puedes entrenar con:")
    print("  python dataset/scripts/3_train_with_embeddings.py")
