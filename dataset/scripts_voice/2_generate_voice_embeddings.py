"""
=============================================================================
PASO 2: GENERAR EMBEDDINGS DE VOZ CON ECAPA-TDNN
=============================================================================

Usa ECAPA-TDNN preentrenado para extraer embeddings de audio.
Es el equivalente a usar FaceNet para rostros.

ECAPA-TDNN NO se entrena, solo extrae características (embeddings de 192 dim).

Uso:
    python dataset/scripts_voice/2_generate_voice_embeddings.py

Entrada:
    dataset/dataset_voice/
        freddy/
            freddy_001.wav
            freddy_002.wav
        melanie/
            melanie_001.wav

Salida:
    dataset/embeddings/voice_embeddings.npy       (N, 192)
    dataset/embeddings/voice_labels.npy           (N,)
    dataset/embeddings/voice_label_map.json       {0: 'freddy', 1: 'melanie', ...}
"""

import os
import numpy as np
import json
from pathlib import Path
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
DATASET_DIR = Path("dataset/dataset_voice")
EMBEDDINGS_DIR = Path("dataset/embeddings")
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "voice_embeddings.npy"
LABELS_FILE = EMBEDDINGS_DIR / "voice_labels.npy"
LABEL_MAP_FILE = EMBEDDINGS_DIR / "voice_label_map.json"

SAMPLE_RATE = 16000


def main():
    print("="*70)
    print("PASO 2: GENERAR EMBEDDINGS DE VOZ CON ECAPA-TDNN")
    print("="*70)
    print("""
    Este script usa ECAPA-TDNN preentrenado (NO lo entrenas):
    
    ┌─────────────────────────────────────────────────────────┐
    │  audio.wav  →  ECAPA-TDNN  →  embedding (192 números)   │
    └─────────────────────────────────────────────────────────┘
    
    ECAPA-TDNN es como FaceNet pero para voz:
        - Ya sabe reconocer características de voz
        - Convierte 3s de audio → vector de 192 dimensiones
        - Tú NO lo entrenas, solo lo usas
    """)
    
    # Verificar dataset
    if not DATASET_DIR.exists():
        print(f"❌ No existe {DATASET_DIR}")
        print("   Ejecuta primero: python dataset/scripts_voice/1_extract_audio.py")
        return
    
    # Crear directorio de embeddings
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo ECAPA-TDNN preentrenado
    print("\n📦 Cargando ECAPA-TDNN preentrenado...")
    print("   (Esto puede tardar la primera vez, descarga ~50MB)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Dispositivo: {device}")
    
    # Usar estrategia COPY en lugar de SYMLINK para Windows
    from speechbrain.utils.fetching import LocalStrategy
    
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY
    )
    print("   ✅ Modelo cargado\n")
    
    # Recopilar datos
    embeddings = []
    labels = []
    label_map = {}
    
    persons = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    
    if not persons:
        print(f"❌ No se encontraron carpetas de personas en {DATASET_DIR}")
        return
    
    print(f"👥 Personas encontradas: {len(persons)}")
    for idx, person_dir in enumerate(persons):
        person_name = person_dir.name
        label_map[idx] = person_name
        print(f"   {idx}: {person_name}")
    
    print(f"\n{'='*60}")
    print("EXTRAYENDO EMBEDDINGS")
    print(f"{'='*60}\n")
    
    for label_idx, person_dir in enumerate(persons):
        person_name = person_dir.name
        audio_files = list(person_dir.glob("*.wav"))
        
        if not audio_files:
            print(f"⚠️  {person_name}: No hay archivos .wav")
            continue
        
        print(f"👤 {person_name.upper()} (clase {label_idx})")
        print(f"   Archivos: {len(audio_files)}")
        
        for audio_path in tqdm(audio_files, desc=f"   Procesando", leave=False):
            try:
                # Cargar audio
                signal, fs = torchaudio.load(str(audio_path))
                
                # Asegurar que sea mono y 16kHz
                if signal.shape[0] > 1:
                    signal = torch.mean(signal, dim=0, keepdim=True)
                
                if fs != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(fs, SAMPLE_RATE)
                    signal = resampler(signal)
                
                # Extraer embedding con ECAPA-TDNN
                with torch.no_grad():
                    embedding = ecapa.encode_batch(signal)
                    embedding = embedding.squeeze().cpu().numpy()  # (192,)
                
                embeddings.append(embedding)
                labels.append(label_idx)
                
            except Exception as e:
                print(f"   ❌ Error con {audio_path.name}: {e}")
                continue
        
        print(f"   ✅ Procesados {len([l for l in labels if l == label_idx])} audios\n")
    
    # Convertir a arrays numpy
    X = np.array(embeddings)
    y = np.array(labels)
    
    print("="*60)
    print("📊 RESUMEN DEL DATASET")
    print("="*60)
    print(f"""
    X (embeddings):
        Shape: {X.shape}
        Cada fila es un embedding de 192 dimensiones
        Tipo: {X.dtype}
    
    y (etiquetas):
        Shape: {y.shape}
        Valores únicos: {np.unique(y)}
    
    Distribución por clase:
    """)
    
    for label_idx, person_name in label_map.items():
        count = (y == label_idx).sum()
        print(f"        {label_idx} → {person_name:12s}: {count} muestras")
    
    # Guardar datos
    print(f"\n💾 Guardando embeddings...")
    np.save(EMBEDDINGS_FILE, X)
    np.save(LABELS_FILE, y)
    
    with open(LABEL_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)
    
    print(f"   ✅ Embeddings: {EMBEDDINGS_FILE}")
    print(f"   ✅ Etiquetas: {LABELS_FILE}")
    print(f"   ✅ Mapeo: {LABEL_MAP_FILE}")
    
    # Análisis de calidad
    print("\n" + "="*60)
    print("🔍 ANÁLISIS DE CALIDAD")
    print("="*60)
    
    print("\nDistancias INTRA-clase (misma persona):")
    for label_idx, person_name in label_map.items():
        person_embeddings = X[y == label_idx]
        
        if len(person_embeddings) < 2:
            print(f"   {person_name:12s}: Solo 1 muestra (no se puede calcular)")
            continue
        
        # Calcular distancias entre embeddings de la misma persona
        distances = []
        for i in range(len(person_embeddings)):
            for j in range(i+1, len(person_embeddings)):
                dist = np.linalg.norm(person_embeddings[i] - person_embeddings[j])
                distances.append(dist)
        
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        print(f"   {person_name:12s}: {avg_dist:.3f} ± {std_dist:.3f}")
    
    print("\nDistancias INTER-clase (personas diferentes):")
    persons_list = list(label_map.values())
    for i in range(len(persons_list)):
        for j in range(i+1, len(persons_list)):
            emb1 = X[y == i]
            emb2 = X[y == j]
            
            if len(emb1) == 0 or len(emb2) == 0:
                continue
            
            # Tomar hasta 10 muestras de cada clase
            emb1 = emb1[:10]
            emb2 = emb2[:10]
            
            distances = []
            for e1 in emb1:
                for e2 in emb2:
                    dist = np.linalg.norm(e1 - e2)
                    distances.append(dist)
            
            avg_dist = np.mean(distances)
            print(f"   {persons_list[i]:12s} vs {persons_list[j]:12s}: {avg_dist:.3f}")
    
    print("\n💡 Interpretación:")
    print("   • Distancias INTRA-clase PEQUEÑAS = buenos embeddings")
    print("   • Distancias INTER-clase GRANDES = personas bien separadas")
    
    print("\n" + "="*60)
    print("✅ PASO 2 COMPLETADO")
    print("="*60)
    print(f"""
    Los embeddings están listos para entrenar tu MLP.
    
    Siguiente paso:
        python dataset/scripts_voice/3_train_voice_mlp.py
    """)


if __name__ == "__main__":
    main()
