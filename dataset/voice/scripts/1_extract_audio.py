"""
Script para extraer, limpiar y segmentar audios.
"""

import os
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np

# Configuración
INPUT_DIR = Path("dataset/voice/audio_raw")
OUTPUT_DIR = Path("dataset/voice/processed")
SAMPLE_RATE = 16000
SEGMENT_MS = 5000
HOP_MS = 5000

def remove_silence(y, sr, top_db=30):
    # Quita silencio detectando energía baja
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y
    return np.concatenate([y[start:end] for start, end in intervals])

def process_audio(audio_path, output_dir, person_name):
    try:
        print(f"Procesando: {audio_path.name}")

        # Cargar audio
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

        # Quitar silencio
        audio = remove_silence(audio, sr, top_db=30)

        # Medir duración después del VAD
        duration = len(audio) / sr
        if duration < 0.7:
            return

        seg_len = SEGMENT_MS / 1000
        hop_len = HOP_MS / 1000

        # Si es corto, guardar segmento único
        if duration < seg_len:
            count = len(list(output_dir.glob("*.wav"))) + 1
            out_path = output_dir / f"{person_name}_{count:03d}.wav"
            sf.write(str(out_path), audio, sr)
            return

        # Calcular número de segmentos
        num_segments = int((duration - seg_len) / hop_len) + 1

        for i in range(num_segments):
            start = int(i * hop_len * sr)
            end = int(start + seg_len * sr)
            segment = audio[start:end]

            # Saltar segmentos muy cortos
            if len(segment) < sr * 0.7:
                continue

            count = len(list(output_dir.glob("*.wav"))) + 1
            out_path = output_dir / f"{person_name}_{count:03d}.wav"
            sf.write(str(out_path), segment, sr)

    except Exception as e:
        print(f"Error en {audio_path.name}: {e}")

def main():
    if not INPUT_DIR.exists():
        print(f"Crear directorio: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for person_dir in INPUT_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        person_out = OUTPUT_DIR / person_name
        person_out.mkdir(exist_ok=True)

        print(f"\n--- {person_name} ---")

        for ext in ['*.opus', '*.wav', '*.mp3', '*.m4a', '*.ogg']:
            for audio_file in person_dir.glob(ext):
                process_audio(audio_file, person_out, person_name)

    print("\nListo. Siguiente: python dataset/voice/scripts/2_generate_voice_embeddings.py")

if __name__ == "__main__":
    main()
