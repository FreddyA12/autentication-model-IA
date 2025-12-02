from __future__ import annotations

from pathlib import Path

import cv2
import librosa
import numpy as np
from tqdm import tqdm

AUDIO_DIR = Path("dataset/audio_raw")
MELS_DIR = Path("dataset/audio_mels")
MELS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16_000
SEGMENT_DURATION = 2.0  # seconds (reduce para más ejemplos)
HOP_DURATION = 1.0
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
IMG_SIZE = (128, 128)


def build_segments(audio: np.ndarray) -> list[np.ndarray]:
    segment_samples = int(SEGMENT_DURATION * SAMPLE_RATE)
    hop_samples = int(HOP_DURATION * SAMPLE_RATE)
    total = len(audio)

    if total < segment_samples:
        pad = segment_samples - total
        audio = np.pad(audio, (0, pad), mode="reflect")
        total = len(audio)

    segments = []
    for start in range(0, total - segment_samples + 1, hop_samples):
        end = start + segment_samples
        segment = audio[start:end]
        if len(segment) < segment_samples:
            break
        segments.append(segment)

    return segments


def mel_to_image(mel: np.ndarray) -> np.ndarray:
    mel -= mel.min()
    if mel.max() > 0:
        mel /= mel.max()
    mel_img = (mel * 255).astype("uint8")
    mel_img = cv2.resize(mel_img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return mel_img


def process_audio_file(wav_path: Path) -> None:
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    segments = build_segments(audio)
    user_dir = MELS_DIR / wav_path.stem
    user_dir.mkdir(parents=True, exist_ok=True)

    for idx, segment in enumerate(segments):
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_img = mel_to_image(mel_db)
        out_path = user_dir / f"{wav_path.stem}_{idx:04d}.png"
        cv2.imwrite(str(out_path), mel_img)


def main() -> None:
    if not AUDIO_DIR.exists():
        raise SystemExit("No se encontraron audios en dataset/audio_raw.")

    wavs = [p for p in sorted(AUDIO_DIR.iterdir()) if p.suffix.lower() in {".wav", ".mp3"}]
    if not wavs:
        raise SystemExit("dataset/audio_raw está vacío.")

    for wav in tqdm(wavs, desc="Generando espectrogramas"):
        process_audio_file(wav)

    print("Espectrogramas listos en dataset/audio_mels.")


if __name__ == "__main__":
    main()
