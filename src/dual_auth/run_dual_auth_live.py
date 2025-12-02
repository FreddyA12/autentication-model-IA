"""
Script de demostración en tiempo real para reconocimiento facial + de voz.
Requiere una cámara web y micrófono disponibles.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd

from .dual_authenticator import DualAuthenticator
from .face_recognition import FaceRecognizer
from .voice_recognition import VoiceRecognizer

FACE_MODEL = Path("dataset/models/faces_cnn_best.keras")
FACE_CLASSES = Path("dataset/models/class_indices.json")
VOICE_MODEL = Path("dataset/models/voice_cnn_best.keras")
VOICE_CLASSES = Path("dataset/models/voice_class_indices.json")

AUDIO_SAMPLE_RATE = 16_000
AUDIO_DURATION = 2.5  # seconds
EVAL_INTERVAL = 5.0  # seconds


def record_audio_async(out_queue: queue.Queue) -> threading.Thread:
    def _worker():
        audio = sd.rec(
            int(AUDIO_DURATION * AUDIO_SAMPLE_RATE),
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        out_queue.put(audio[:, 0])

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def draw_status(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo de autenticación dual en tiempo real.")
    parser.add_argument("--identity", required=True, help="Nombre esperado de la persona (exacto).")
    parser.add_argument("--face-threshold", type=float, default=0.75)
    parser.add_argument("--voice-threshold", type=float, default=0.55)
    args = parser.parse_args()

    if not FACE_MODEL.exists() or not VOICE_MODEL.exists():
        raise SystemExit("Entrena primero ambos modelos (rostro y voz).")

    face_recognizer = FaceRecognizer(FACE_MODEL, FACE_CLASSES)
    voice_recognizer = VoiceRecognizer(
        VOICE_MODEL,
        VOICE_CLASSES,
        recognition_threshold=args.voice_threshold,
    )
    authenticator = DualAuthenticator(
        face_recognizer,
        voice_recognizer,
        expected_identity=args.identity,
        face_threshold=args.face_threshold,
        voice_threshold=args.voice_threshold,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    audio_queue: queue.Queue = queue.Queue(maxsize=1)
    audio_thread: threading.Thread | None = None
    pending_frame: np.ndarray | None = None
    last_launch = 0.0
    last_result = None

    print("Presiona Q para salir.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            recording = audio_thread is not None and audio_thread.is_alive()

            if (not recording) and (current_time - last_launch >= EVAL_INTERVAL):
                pending_frame = frame.copy()
                audio_queue = queue.Queue(maxsize=1)
                audio_thread = record_audio_async(audio_queue)
                last_launch = current_time

            if recording and not audio_queue.empty() and pending_frame is not None:
                audio_chunk = audio_queue.get()
                last_result = authenticator.evaluate(pending_frame, audio_chunk, AUDIO_SAMPLE_RATE)
                pending_frame = None
                audio_thread = None

            overlay_text = "Listo"
            overlay_color = (0, 255, 0)
            if audio_thread is not None:
                overlay_text = "Grabando audio..."
                overlay_color = (0, 215, 255)
            if last_result is not None:
                if last_result.authenticated:
                    overlay_text = (
                        f"OK {last_result.face_label} | voz {last_result.voice_confidence:.2f}"
                    )
                    overlay_color = (0, 200, 0)
                else:
                    overlay_text = f"Fallo: {last_result.reason}"
                    overlay_color = (0, 0, 255)

            draw_status(frame, overlay_text, overlay_color)
            cv2.imshow("Dual Auth", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
