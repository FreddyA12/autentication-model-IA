"""
Script para generar embeddings de voz usando YAMNet.
Uso: python dataset/voice/scripts/2_generate_voice_embeddings.py
"""

import os
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tqdm import tqdm

# Directorios y archivos de salida
DATASET_DIR = Path("dataset/voice/processed")
EMBEDDINGS_DIR = Path("dataset/voice/embeddings")
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "voice_embeddings.npy"
LABELS_FILE = EMBEDDINGS_DIR / "voice_labels.npy"
LABEL_MAP_FILE = EMBEDDINGS_DIR / "voice_label_map.json"

# Cargar modelo YAMNet
print("Cargando YAMNet...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def augment_audio(wav, sr):
    """Pequeñas variaciones para robustez (ruido + ganancia)."""
    variations = [wav]

    # Ruido leve
    noise = np.random.normal(0, 0.002, len(wav))
    variations.append(wav + noise)

    # Variación ligera de volumen
    gain = np.random.uniform(0.9, 1.1)
    variations.append(wav * gain)

    return variations

def get_embeddings(audio_path):
    """Genera embeddings promediados desde YAMNet con augmentación."""
    wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    variations = augment_audio(wav, sr)
    embeddings_list = []

    for v in variations:
        # Filtrar audios extremadamente cortos
        if len(v) < 1600:
            continue

        # Normalizar amplitud a [-1, 1]
        max_abs = np.max(np.abs(v))
        if max_abs > 0:
            v = v / max_abs

        waveform = tf.convert_to_tensor(v, dtype=tf.float32)

        # Modelo YAMNet devuelve (scores, embeddings, spectrogram)
        _, embeddings, _ = yamnet_model(waveform)

        # Saltar casos donde no se generaron embeddings
        if len(embeddings) == 0:
            continue

        # Promedio de embeddings → vector único por variación
        emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()

        # Filtrar embeddings corruptos
        if np.isnan(emb_mean).any():
            continue
        if np.max(np.abs(emb_mean)) > 5:
            continue

        embeddings_list.append(emb_mean)

    return embeddings_list

def main():
    if not DATASET_DIR.exists():
        print(f"No existe {DATASET_DIR}")
        return

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = []
    labels = []
    label_map = {}

    persons = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])

    for idx, person_dir in enumerate(persons):
        person_name = person_dir.name
        label_map[idx] = person_name
        print(f"Procesando: {person_name} ({idx})")

        # Iterar sobre cada .wav procesado
        for audio_path in tqdm(list(person_dir.glob("*.wav"))):
            try:
                embs = get_embeddings(audio_path)
                for emb in embs:
                    embeddings.append(emb)
                    labels.append(idx)
            except Exception as e:
                print(f"Error en {audio_path.name}: {e}")

    # Guardar embeddings y labels
    X = np.array(embeddings, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    np.save(EMBEDDINGS_FILE, X)
    np.save(LABELS_FILE, y)
    with open(LABEL_MAP_FILE, 'w') as f:
        json.dump(label_map, f, indent=4)

    print(f"\nGuardado: {len(X)} embeddings. Shape: {X.shape}")
    print("Listo. Siguiente: python dataset/voice/scripts/3_train_voice_mlp.py")

if __name__ == "__main__":
    main()
