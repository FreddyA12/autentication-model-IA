"""
Face Recognition Service

Carga MTCNN + FaceNet (Keras) para extraer embeddings y el clasificador entrenado.
Devuelve identidad o "DESCONOCIDO" según el umbral.
"""

import os
import json
import numpy as np
import cv2
from typing import Optional, Dict
from io import BytesIO
from PIL import Image

import tensorflow as tf
from mtcnn import MTCNN
from keras_facenet import FaceNet
from django.conf import settings


class FaceRecognitionService:
    """Servicio de reconocimiento facial usando Keras FaceNet + clasificador propio."""

    def __init__(self) -> None:
        try:
            print("[FaceAuth] Iniciando FaceRecognitionService (TensorFlow)...")

            # Detector
            print("   Cargando MTCNN...")
            self.detector = MTCNN()

            # Extractor de embeddings
            print("   Cargando FaceNet...")
            self.embedder = FaceNet()

            # Clasificador entrenado
            print(f"   Cargando clasificador desde: {settings.FACE_MODEL_PATH}")
            model_path = self._resolve_model_path(settings.FACE_MODEL_PATH)

            # Cargar mapeo de clases
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

            # Cargar el clasificador
            try:
                self.classifier = tf.keras.models.load_model(model_path)
                print(f"   [OK] Modelo cargado exitosamente")
            except Exception as e:
                print(f"   [WARN] Error al cargar modelo: {e}")
                raise

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
        if os.path.exists(path):
            return path
        candidate = path.replace(".keras", "_best.keras")
        if os.path.exists(candidate):
            print(f"   Modelo principal no existe, usando: {candidate}")
            return candidate
        raise FileNotFoundError(f"No se encontró el modelo en {path}")

    def extract_embedding(self, image_bytes: bytes):
        """Devuelve (embedding, box, score) o (None, None, None) si no detecta."""
        try:
            # Convertir bytes a imagen numpy (BGR para cv2, pero MTCNN usa RGB)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None, None, None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detectar caras
            results = self.detector.detect_faces(img_rgb)
            if not results:
                return None, None, None

            # Tomar la cara con mayor confianza
            best_face = max(results, key=lambda x: x['confidence'])
            box = best_face['box']
            confidence = best_face['confidence']
            
            x, y, w, h = box
            # Asegurar coordenadas válidas
            x, y = max(0, x), max(0, y)
            
            # Extraer rostro
            face = img_rgb[y:y+h, x:x+w]
            
            # Redimensionar a 160x160 (requerido por FaceNet)
            face_resized = cv2.resize(face, (160, 160))
            
            # Obtener embedding
            samples = np.expand_dims(face_resized, axis=0)
            embedding = self.embedder.embeddings(samples)[0]
            
            return embedding, box, confidence

        except Exception as e:
            print(f"[FaceAuth] Error al extraer embedding: {e}")
            return None, None, None

    def predict(self, image_bytes: bytes) -> Dict:
        """Predice identidad con el clasificador sobre el embedding."""
        embedding, box, box_score = self.extract_embedding(image_bytes)
        if embedding is None:
            return {
                "success": False,
                "identity": None,
                "confidence": 0,
                "probabilities": {},
                "message": "No se detectó ningún rostro en la imagen",
                "box": None,
            }

        # Predicción
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
            "box": box,
            "box_score": box_score,
        }


_face_service = None


def get_face_service():
    """Instancia singleton del servicio."""
    global _face_service
    if _face_service is None:
        _face_service = FaceRecognitionService()
    return _face_service
