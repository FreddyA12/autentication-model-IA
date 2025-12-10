"""
Script para generar embeddings de voz usando ECAPA-TDNN (SpeechBrain).
Uso: python dataset/voice/scripts/2_generate_voice_embeddings.py
"""

import os
import json
import numpy as np
from pathlib import Path
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

# Directorios y archivos de salida
DATASET_DIR = Path("dataset/voice/processed")
EMBEDDINGS_DIR = Path("dataset/voice/embeddings")
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "voice_embeddings.npy"
LABELS_FILE = EMBEDDINGS_DIR / "voice_labels.npy"
LABEL_MAP_FILE = EMBEDDINGS_DIR / "voice_label_map.json"

# Cargar modelo ECAPA-TDNN
print("Cargando ECAPA-TDNN...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

from speechbrain.utils.fetching import LocalStrategy

encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY
)

def get_embedding(audio_path):
    """Genera embedding de 192 dimensiones desde ECAPA-TDNN."""
    try:
        # Cargar audio
        signal, fs = torchaudio.load(str(audio_path))
        
        # Convertir a mono si es estéreo
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        # Resamplear a 16kHz si es necesario
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        
        # Extraer embedding
        with torch.no_grad():
            embedding = encoder.encode_batch(signal)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    except Exception as e:
        print(f"Error en {audio_path.name}: {e}")
        return None

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
        print(f"\nProcesando: {person_name} ({idx})")

        # Iterar sobre cada .wav procesado
        for audio_path in tqdm(list(person_dir.glob("*.wav"))):
            emb = get_embedding(audio_path)
            if emb is not None:
                embeddings.append(emb)
                labels.append(idx)

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
