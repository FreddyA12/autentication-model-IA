import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN


class FaceRecognizer:
    """Utility wrapper around the CNN + MTCNN detection pipeline."""

    def __init__(
        self,
        model_path: Path,
        class_mapping_path: Path,
        img_size: Tuple[int, int] = (160, 160),
        detection_threshold: float = 0.9,
    ) -> None:
        self.img_size = img_size
        self.detection_threshold = detection_threshold
        self.model = tf.keras.models.load_model(str(model_path))
        self.detector = MTCNN()
        self.idx_to_class = self._load_class_mapping(class_mapping_path)
        self.class_to_idx = {name: int(idx) for idx, name in self.idx_to_class.items()}

    @staticmethod
    def _load_class_mapping(mapping_path: Path) -> Dict[str, str]:
        with open(mapping_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return {str(idx): name for idx, name in data.items()}

    def _detect_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns the aligned face crop as RGB float array."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)
        if not faces:
            return None

        best = max(
            (
                f
                for f in faces
                if f.get("confidence", 0.0) >= self.detection_threshold
            ),
            key=lambda f: f["box"][2] * f["box"][3],
            default=None,
        )
        if best is None:
            return None

        x, y, w, h = best["box"]
        x, y = max(0, x), max(0, y)
        w, h = max(1, w), max(1, h)

        crop = rgb[y : y + h, x : x + w]
        if crop.size == 0:
            return None

        return cv2.resize(crop, self.img_size)

    def predict_from_frame(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        face = self._detect_face(frame_bgr)
        if face is None:
            return None

        sample = face.astype("float32") / 255.0
        sample = np.expand_dims(sample, axis=0)
        preds = self.model.predict(sample, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = self.idx_to_class[str(idx)]
        confidence = float(preds[idx])
        return label, confidence

    def predict_from_path(self, image_path: Path) -> Optional[Tuple[str, float]]:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
        return self.predict_from_frame(frame)
