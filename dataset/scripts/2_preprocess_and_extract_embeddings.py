"""
=============================================================================
PASO 2: PREPROCESAR IMÁGENES Y EXTRAER EMBEDDINGS CON FACENET
=============================================================================

Este script hace TODO el trabajo pesado:
1. Lee imágenes de dataset_raw/
2. Detecta rostros con MTCNN
3. Alinea y recorta las caras (160x160)
4. Extrae embeddings de 512 dimensiones con FaceNet
5. Guarda:
   - Imágenes procesadas en dataset_clean/
   - Embeddings en dataset/embeddings/

FaceNet NO se entrena, solo se usa como extractor de características.
Los embeddings son vectores de 512 números que representan la identidad facial.

Uso:
    python dataset/scripts/2_preprocess_and_extract_embeddings.py
"""

import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from tqdm import tqdm
import pickle

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
RAW_DIR = "dataset/dataset_raw"
CLEAN_DIR = "dataset/dataset_clean"
EMBEDDINGS_DIR = "dataset/embeddings"
IMG_SIZE = 160

# ============================================================================
# INICIALIZAR MODELOS (FaceNet NO se entrena, solo se usa)
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Dispositivo: {device}")

# MTCNN: Detecta caras y extrae landmarks (ojos, nariz, boca)
# Esto permite ALINEAR las caras para que todas estén en la misma posición
print("📦 Cargando MTCNN (detector de rostros)...")
mtcnn = MTCNN(
    image_size=IMG_SIZE,
    margin=20,              # Margen alrededor de la cara
    min_face_size=40,       # Tamaño mínimo de cara a detectar
    thresholds=[0.6, 0.7, 0.7],  # Umbrales de detección
    post_process=True,      # Normaliza para FaceNet
    device=device,
    keep_all=False          # Solo la cara más grande/confiable
)

# FaceNet: Convierte una cara en un vector de 512 números (embedding)
# Este modelo está PRE-ENTRENADO en millones de caras (VGGFace2)
# NO lo entrenamos, solo lo usamos para extraer características
print("📦 Cargando FaceNet (extractor de embeddings)...")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("✅ Modelos cargados\n")


def process_image(img_path):
    """
    Procesa una imagen:
    1. Detecta la cara con MTCNN
    2. Alinea usando landmarks de los ojos
    3. Recorta a 160x160
    4. Extrae embedding con FaceNet
    
    Returns:
        face_tensor: Tensor de la cara (para FaceNet)
        face_numpy: Imagen numpy (para guardar)
        embedding: Vector de 512 dimensiones
    """
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return None, None, None
    
    # MTCNN detecta y ALINEA automáticamente usando los landmarks
    face_tensor = mtcnn(img)
    
    if face_tensor is None:
        return None, None, None
    
    # Convertir tensor a imagen numpy para guardar
    # El tensor viene normalizado (-1, 1), lo convertimos a (0, 255)
    face_np = face_tensor.permute(1, 2, 0).numpy()
    face_np = ((face_np + 1) / 2 * 255).astype(np.uint8)
    
    # Extraer embedding con FaceNet
    with torch.no_grad():
        face_batch = face_tensor.unsqueeze(0).to(device)
        embedding = facenet(face_batch)
    
    embedding = embedding.cpu().numpy().flatten()
    
    return face_tensor, face_np, embedding


def main():
    print("="*70)
    print("PASO 2: PREPROCESAR Y EXTRAER EMBEDDINGS")
    print("="*70)
    print("""
    Este script:
    ┌─────────────────────────────────────────────────────────────────┐
    │  imagen  →  MTCNN  →  FaceNet  →  embedding (512 dimensiones)   │
    └─────────────────────────────────────────────────────────────────┘
    
    FaceNet convierte cada cara en un vector de 512 números.
    Estos números representan la "identidad" de la cara.
    
    - Si dos caras son de la MISMA persona → embeddings CERCANOS
    - Si son de personas DIFERENTES → embeddings LEJANOS
    """)
    
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Estructura para guardar embeddings
    all_embeddings = []  # Lista de embeddings (X)
    all_labels = []      # Lista de etiquetas (y)
    label_to_idx = {}    # Mapeo nombre → índice
    idx_to_label = {}    # Mapeo índice → nombre
    
    # Procesar cada persona
    persons = sorted([d for d in os.listdir(RAW_DIR) 
                      if os.path.isdir(os.path.join(RAW_DIR, d))])
    
    if not persons:
        print(f"❌ No se encontraron carpetas en {RAW_DIR}")
        print("   Estructura esperada:")
        print("   dataset/dataset_raw/")
        print("   ├── persona1/")
        print("   │   ├── img1.jpg")
        print("   │   └── img2.jpg")
        print("   └── persona2/")
        return
    
    print(f"📁 Personas encontradas: {persons}\n")
    
    for idx, person in enumerate(persons):
        label_to_idx[person] = idx
        idx_to_label[idx] = person
        
        src_dir = os.path.join(RAW_DIR, person)
        dst_dir = os.path.join(CLEAN_DIR, person)
        os.makedirs(dst_dir, exist_ok=True)
        
        images = [f for f in os.listdir(src_dir) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"{'='*60}")
        print(f"👤 {person.upper()} (clase {idx})")
        print(f"   Imágenes encontradas: {len(images)}")
        print(f"{'='*60}")
        
        processed = 0
        failed = 0
        
        for img_name in tqdm(images, desc=f"   Procesando"):
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(dst_dir, img_name)
            
            face_tensor, face_np, embedding = process_image(src_path)
            
            if embedding is None:
                failed += 1
                continue
            
            # Guardar imagen alineada
            face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, face_bgr)
            
            # Agregar embedding y etiqueta
            all_embeddings.append(embedding)
            all_labels.append(idx)
            processed += 1
        
        print(f"   ✅ Procesadas: {processed}")
        print(f"   ❌ Fallidas: {failed}\n")
    
    # Convertir a arrays numpy
    X = np.array(all_embeddings)
    y = np.array(all_labels)
    
    print("="*60)
    print("📊 RESUMEN DEL DATASET")
    print("="*60)
    print(f"""
    X (embeddings):
        Shape: {X.shape}
        Cada fila es un embedding de 512 dimensiones
    
    y (etiquetas):
        Shape: {y.shape}
        Cada valor es el índice de la persona
    
    Mapeo de clases:
    """)
    for idx, name in idx_to_label.items():
        count = (y == idx).sum()
        print(f"        {idx} → {name}: {count} muestras")
    
    # Guardar embeddings
    data = {
        'X': X,
        'y': y,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label
    }
    
    embeddings_path = os.path.join(EMBEDDINGS_DIR, "embeddings_dataset.pkl")
    with open(embeddings_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n💾 Datos guardados:")
    print(f"   - Embeddings: {embeddings_path}")
    print(f"   - Imágenes alineadas: {CLEAN_DIR}/")
    
    # Verificar calidad: calcular distancias
    print("\n" + "="*60)
    print("🔍 VERIFICACIÓN DE CALIDAD")
    print("="*60)
    print("""
    Para que el reconocimiento funcione bien:
    - Distancia INTRA-clase (misma persona): debe ser BAJA (< 1.0)
    - Distancia INTER-clase (diferentes personas): debe ser ALTA (> 1.0)
    """)
    
    # Distancias intra-clase
    print("Distancias INTRA-clase (misma persona):")
    for idx, name in idx_to_label.items():
        embs = X[y == idx]
        if len(embs) < 2:
            continue
        
        distances = []
        for i in range(min(50, len(embs))):
            for j in range(i+1, min(50, len(embs))):
                dist = np.linalg.norm(embs[i] - embs[j])
                distances.append(dist)
        
        if distances:
            avg = np.mean(distances)
            std = np.std(distances)
            print(f"   {name}: {avg:.3f} ± {std:.3f}")
    
    # Distancias inter-clase
    print("\nDistancias INTER-clase (diferentes personas):")
    for i in range(len(persons)):
        for j in range(i+1, len(persons)):
            embs1 = X[y == i][:50]
            embs2 = X[y == j][:50]
            
            distances = []
            for e1 in embs1:
                for e2 in embs2:
                    dist = np.linalg.norm(e1 - e2)
                    distances.append(dist)
            
            if distances:
                avg = np.mean(distances)
                print(f"   {persons[i]} vs {persons[j]}: {avg:.3f}")
    
    print("\n" + "="*60)
    print("✅ PASO 2 COMPLETADO")
    print("="*60)
    print("\nSiguiente paso:")
    print("  python dataset/scripts/3_train_classifier.py")


if __name__ == "__main__":
    main()
