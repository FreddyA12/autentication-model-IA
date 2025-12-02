from pathlib import Path

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.dual_auth.voice_recognition import VoiceRecognizer  # noqa: E402

MODELS_DIR = Path("dataset/models")
MODEL_PATH = MODELS_DIR / "voice_cnn_best.keras"
CLASS_INDICES_PATH = MODELS_DIR / "voice_class_indices.json"
TEST_AUDIO_DIR = Path("dataset/test_data/audio_samples")


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No existe el modelo entrenado de voz. Ejecuta 7_train_voice_model.py primero.")

    recognizer = VoiceRecognizer(MODEL_PATH, CLASS_INDICES_PATH)

    if not TEST_AUDIO_DIR.exists():
        print("No existe dataset/test_data/audio_samples. Probando con dataset/audio_raw.")
        for wav_path in Path("dataset/audio_raw").iterdir():
            if wav_path.suffix.lower() != ".wav":
                continue
            pred = recognizer.predict_from_file(wav_path)
            if pred is None:
                print(f"{wav_path.name:<20} -> Confianza insuficiente")
            else:
                name, conf = pred
                print(f"{wav_path.name:<20} -> {name} ({conf:.2f})")
        return

    audio_files = [
        p for p in TEST_AUDIO_DIR.iterdir() if p.suffix.lower() in {".wav", ".mp3", ".flac"}
    ]
    if not audio_files:
        raise SystemExit("No hay archivos de audio en dataset/test_data/audio_samples")

    for audio_path in audio_files:
        result = recognizer.predict_from_file(audio_path)
        if result is None:
            print(f"{audio_path.name:<20} -> Confianza insuficiente")
        else:
            name, conf = result
            print(f"{audio_path.name:<20} -> {name} ({conf:.2f})")


if __name__ == "__main__":
    main()
