from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

RAW_DIR = "dataset/dataset_raw"
CLEAN_DIR = "dataset/dataset_clean"
IMG_SIZE = 160

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = MTCNN()
    return _detector


def procesar(src_path: str, dst_path: str) -> bool:
    detector = _get_detector()
    img = cv2.imread(src_path)
    if img is None:
        return False

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    if len(faces) == 0:
        return False

    f = max(faces, key=lambda x: x["box"][2] * x["box"][3])
    x, y, w, h = f["box"]
    x, y = max(0, x), max(0, y)
    w, h = max(1, w), max(1, h)

    crop = rgb[y : y + h, x : x + w]
    if crop.size == 0:
        return False

    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst_path, crop)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detecta y recorta rostros en paralelo.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Número de procesos en paralelo (default: núcleos-1).",
    )
    return parser


def main() -> None:
    os.makedirs(CLEAN_DIR, exist_ok=True)
    args = build_parser().parse_args()

    for user in os.listdir(RAW_DIR):
        src = os.path.join(RAW_DIR, user)
        dst = os.path.join(CLEAN_DIR, user)

        if not os.path.isdir(src):
            continue

        os.makedirs(dst, exist_ok=True)
        images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not images:
            continue

        print(f"Procesando {user} con {args.workers} procesos...")
        tasks = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for img_name in images:
                src_path = os.path.join(src, img_name)
                dst_path = os.path.join(dst, img_name)
                tasks.append(executor.submit(procesar, src_path, dst_path))

            for _ in tqdm(as_completed(tasks), total=len(tasks)):
                pass

    print("Preprocesamiento completado - dataset_clean listo")


if __name__ == "__main__":
    main()
