"""
Face Recognition Service

Carga MTCNN + FaceNet para extraer embeddings y el clasificador entrenado
con los scripts 1-3. Devuelve identidad o "DESCONOCIDO" según el umbral.
"""

import os
import json
from io import BytesIO
from typing import Optional, Dict

import numpy as np
import torch
from PIL import Image
from django.conf import settings
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceRecognitionService:
    """Servicio de reconocimiento facial usando FaceNet + clasificador propio."""

    def __init__(self) -> None:
        try:
            print("[FaceAuth] Iniciando FaceRecognitionService...")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"   Dispositivo: {self.device}")

            # Detector/alineador
            print("   Cargando MTCNN...")
            self.mtcnn = MTCNN(
                image_size=160,
                margin=20,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                post_process=True,
                device=self.device,
                keep_all=False,
            )

            # Extractor de embeddings
            print("   Cargando FaceNet...")
            self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

            # Clasificador entrenado (scripts 2 y 3)
            print(f"   Cargando clasificador desde: {settings.FACE_MODEL_PATH}")
            model_path = self._resolve_model_path(settings.FACE_MODEL_PATH)

            import tensorflow as tf

            self.classifier = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False,
            )

            # Mapeo de clases
            print(f"   Cargando clases desde: {settings.CLASS_INDICES_PATH}")
            if not os.path.exists(settings.CLASS_INDICES_PATH):
                raise FileNotFoundError(
                    f"No se encontró el archivo en {settings.CLASS_INDICES_PATH}"
                )
            with open(settings.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                class_data = json.load(f)
                if "idx_to_label" in class_data:
                    idx_to_label = {int(k): v for k, v in class_data["idx_to_label"].items()}
                else:
                    idx_to_label = {int(k): v for k, v in class_data.items()}
            self.idx_to_label = idx_to_label

            self.confidence_threshold = settings.CONFIDENCE_THRESHOLD

            print("[FaceAuth] Modelos cargados correctamente")
            print(f"   Clases disponibles: {list(self.idx_to_label.values())}")
            print(f"   Umbral de confianza: {self.confidence_threshold * 100}%")

        except Exception as e:
            print(f"[FaceAuth] Error al inicializar FaceRecognitionService: {e}")
            import traceback

            traceback.print_exc()
            raise

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        """Devuelve un path existente; cae al *_best.keras si el principal no existe."""
        if os.path.exists(path):
            return path
        candidate = path.replace(".keras", "_best.keras")
        if os.path.exists(candidate):
            print(f"   Modelo principal no existe, usando: {candidate}")
            return candidate
        raise FileNotFoundError(f"No se encontró el modelo en {path}")

    def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Devuelve embedding de 512 dim o None si no detecta rostro."""
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            face_tensor = self.mtcnn(img)
            if face_tensor is None:
                return None
            with torch.no_grad():
                face_batch = face_tensor.unsqueeze(0).to(self.device)
                embedding = self.facenet(face_batch)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"[FaceAuth] Error al extraer embedding: {e}")
            return None

    def predict(self, image_bytes: bytes) -> Dict:
        """Predice identidad con el clasificador sobre el embedding."""
        embedding = self.extract_embedding(image_bytes)
        if embedding is None:
            return {
                "success": False,
                "identity": None,
                "confidence": 0,
                "probabilities": {},
                "message": "No se detectó ningún rostro en la imagen",
            }

        embedding_batch = np.expand_dims(embedding, axis=0)
        predictions = self.classifier.predict(embedding_batch, verbose=0)[0]

        max_idx = int(np.argmax(predictions))
        max_confidence = float(predictions[max_idx])

        probabilities = {
            self.idx_to_label[i]: float(predictions[i]) * 100
            for i in range(len(predictions))
        }

        if max_confidence >= self.confidence_threshold:
            identity = self.idx_to_label[max_idx]
            message = f"Rostro reconocido como {identity}"
            success = True
        else:
            identity = "DESCONOCIDO"
            message = f"Confianza insuficiente (max: {max_confidence*100:.1f}%)"
            success = False

        return {
            "success": success,
            "identity": identity,
            "confidence": max_confidence * 100,
            "probabilities": probabilities,
            "message": message,
        }


_face_service = None


def get_face_service():
    """Instancia singleton del servicio."""
    global _face_service
    if _face_service is None:
        _face_service = FaceRecognitionService()
    return _face_service
