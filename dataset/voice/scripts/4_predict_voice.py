"""
Script para predecir identidad por voz usando ECAPA-TDNN.
Uso: python dataset/voice/scripts/4_predict_voice.py
"""

import json
import numpy as np
import torch
import torchaudio
import tensorflow as tf
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier

# Configuración
MODELS_DIR = Path("dataset/voice/models")
MODEL_PATH = MODELS_DIR / "voice_mlp_best.keras"
CLASS_INDICES_PATH = MODELS_DIR / "voice_class_indices.json"
DATASET_VOICE_DIR = Path("dataset/voice/processed")
TEST_AUDIOS_DIR = Path("dataset/voice/test_audios")
CONFIDENCE_THRESHOLD = 0.70

class VoicePredictor:
    def __init__(self):
        print("Cargando modelos...")
        
        # ECAPA-TDNN
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from speechbrain.utils.fetching import LocalStrategy
        
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY
        )
        
        # MLP Classifier
        self.mlp = tf.keras.models.load_model(MODEL_PATH)
        
        # Label map
        with open(CLASS_INDICES_PATH, 'r') as f:
            self.label_map = {int(k): v for k, v in json.load(f).items()}

    def predict(self, audio_path):
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
            embedding = self.encoder.encode_batch(signal)
            embedding = embedding.squeeze().cpu().numpy()
        
        # Predicción
        probs = self.mlp.predict(np.expand_dims(embedding, axis=0), verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        pred_label = self.label_map[idx]

        if pred_label == "unknown":
            identity = "unknown"
        elif conf < CONFIDENCE_THRESHOLD:
            identity = "unknown"
        else:
            identity = pred_label

        return identity, conf, {self.label_map[i]: float(p) for i, p in enumerate(probs)}

def main():
    if not MODEL_PATH.exists():
        print(f"No existe modelo en {MODEL_PATH}")
        return

    predictor = VoicePredictor()

    # Probar dataset (sanity check)
    print("\n--- Sanity Check (Dataset) ---")
    for person_dir in DATASET_VOICE_DIR.iterdir():
        if not person_dir.is_dir(): continue
        audios = list(person_dir.glob("*.wav"))
        if audios:
            identity, conf, _ = predictor.predict(audios[0])
            check = "" if identity == person_dir.name else "x"
            print(f"{check} {person_dir.name}: Predicho {identity} ({conf:.1%})")

    # Probar nuevos audios
    print("\n--- Test Audios ---")
    if not TEST_AUDIOS_DIR.exists():
        print(f"Crear carpeta {TEST_AUDIOS_DIR} y poner audios.")
        return

    for ext in ['*.wav', '*.opus', '*.mp3', '*.m4a', '*.ogg']:
        for audio in TEST_AUDIOS_DIR.glob(ext):
            identity, conf, probs = predictor.predict(audio)
            print(f"\n {audio.name}")
            if identity == "unknown":
                print(f" DESCONOCIDO (Max conf: {conf:.1%})")
            else:
                print(f" {identity.upper()} ({conf:.1%})")
            
            # Top 3
            top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for name, p in top3:
                print(f"   {name}: {p:.1%}")

if __name__ == "__main__":
    main()
