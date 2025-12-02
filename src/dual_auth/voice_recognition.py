from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import librosa
import numpy as np
import tensorflow as tf


class VoiceRecognizer:
    """Small helper around the CNN trained on log-mel spectrograms."""

    SAMPLE_RATE = 16_000
    SEGMENT_DURATION = 2.5  # seconds
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 256
    IMG_SIZE = (128, 128)

    def __init__(
        self,
        model_path: Path,
        class_mapping_path: Path,
        recognition_threshold: float = 0.5,
        allow_unknown: bool = True,
        unknown_label: str = "Desconocido",
    ) -> None:
        self.model = tf.keras.models.load_model(str(model_path))
        self.recognition_threshold = recognition_threshold
        self.allow_unknown = allow_unknown
        self.unknown_label = unknown_label
        self.idx_to_class = self._load_class_mapping(class_mapping_path)
        self.class_to_idx = {name: int(idx) for idx, name in self.idx_to_class.items()}

    @staticmethod
    def _load_class_mapping(mapping_path: Path) -> Dict[str, str]:
        with open(mapping_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return {str(idx): name for idx, name in data.items()}

    def _audio_to_mel(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = librosa.to_mono(audio.T)
        if sr != self.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)

        target_length = int(self.SEGMENT_DURATION * self.SAMPLE_RATE)
        if len(audio) < target_length:
            pad = target_length - len(audio)
            audio = np.pad(audio, (0, pad), mode="reflect")
        else:
            audio = audio[:target_length]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db -= mel_db.min()
        if mel_db.max() > 0:
            mel_db /= mel_db.max()
        mel_img = (mel_db * 255).astype("uint8")
        mel_img = cv2.resize(mel_img, self.IMG_SIZE, interpolation=cv2.INTER_AREA)
        mel_img = mel_img.astype("float32") / 255.0
        mel_img = np.expand_dims(mel_img, axis=-1)  # grayscale channel
        return mel_img

    def predict_from_audio(
        self, audio: np.ndarray, sr: int
    ) -> Optional[Tuple[str, float]]:
        mel = self._audio_to_mel(audio, sr)
        sample = np.expand_dims(mel, axis=0)
        preds = self.model.predict(sample, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        if self.allow_unknown and confidence < self.recognition_threshold:
            return self.unknown_label, confidence
        label = self.idx_to_class[str(idx)]
        return label, confidence

    def predict_from_file(self, audio_path: Path) -> Optional[Tuple[str, float]]:
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        return self.predict_from_audio(audio, sr)
