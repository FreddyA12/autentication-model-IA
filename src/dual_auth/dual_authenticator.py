from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .face_recognition import FaceRecognizer
from .voice_recognition import VoiceRecognizer


@dataclass
class AuthResult:
    authenticated: bool
    face_label: Optional[str]
    face_confidence: float
    voice_label: Optional[str]
    voice_confidence: float
    reason: str


class DualAuthenticator:
    """Combina los modelos de rostro y voz para un veredicto único."""

    def __init__(
        self,
        face_recognizer: FaceRecognizer,
        voice_recognizer: VoiceRecognizer,
        expected_identity: str,
        face_threshold: float = 0.7,
        voice_threshold: float = 0.6,
    ) -> None:
        self.face_recognizer = face_recognizer
        self.voice_recognizer = voice_recognizer
        self.expected_identity = expected_identity
        self.face_threshold = face_threshold
        self.voice_threshold = voice_threshold

    def evaluate(self, frame_bgr: np.ndarray, audio_buffer: np.ndarray, audio_sr: int) -> AuthResult:
        face_pred = self.face_recognizer.predict_from_frame(frame_bgr)
        if face_pred is None:
            return AuthResult(False, None, 0.0, None, 0.0, "No se detectó rostro.")

        face_label, face_conf = face_pred
        if face_label != self.expected_identity or face_conf < self.face_threshold:
            return AuthResult(False, face_label, face_conf, None, 0.0, "Rostro no coincide.")

        voice_pred = self.voice_recognizer.predict_from_audio(audio_buffer, audio_sr)
        if voice_pred is None:
            return AuthResult(False, face_label, face_conf, None, 0.0, "Voz con baja confianza.")

        voice_label, voice_conf = voice_pred
        if voice_label != self.expected_identity or voice_conf < self.voice_threshold:
            return AuthResult(
                False,
                face_label,
                face_conf,
                voice_label,
                voice_conf,
                "Voz no coincide.",
            )

        return AuthResult(True, face_label, face_conf, voice_label, voice_conf, "Autenticado.")
