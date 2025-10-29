"""Calcula la homografía y proyecta detecciones al plano del campo."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd

from .field_layout import DEFAULT_FIELD_LAYOUT
from .homography_estimator import HomographyEstimator
from .keypoint_model import HomographyKeypointModel

BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = BASE_DIR / "outputs" / "Homografia.pth"
VIDEO_PATH = BASE_DIR / "VideoPruebaTesis.mp4"
DETECTIONS_WITH_TEAMS = BASE_DIR / "outputs" / "detecciones_con_equipos.csv"
H_MATRIX_OUTPUT = BASE_DIR / "outputs" / "homography_matrix.npy"
PROJECTIONS_OUTPUT = BASE_DIR / "outputs" / "detecciones_proyectadas.csv"
KEYPOINT_SNAPSHOT = BASE_DIR / "outputs" / "keypoints_frame0.json"


def _parse_bbox(raw: object) -> Tuple[float, float, float, float]:
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"Bounding box inválido: {raw!r}")
    x1, y1, x2, y2 = [float(v) for v in raw]
    return x1, y1, x2, y2


def compute_homography(frame_index: int = 0) -> Dict[str, Tuple[float, float]]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint de homografía: {CHECKPOINT_PATH}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"No se encontró el video de referencia: {VIDEO_PATH}")

    model = HomographyKeypointModel(CHECKPOINT_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {VIDEO_PATH}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"No se pudo leer el frame {frame_index} del video '{VIDEO_PATH}'.")

    keypoints, confidences = model.predict_from_frame(frame)

    estimator = HomographyEstimator(DEFAULT_FIELD_LAYOUT)
    result = estimator.estimate(keypoints)
    np.save(H_MATRIX_OUTPUT, result.matrix)

    snapshot = {name: {"x": float(pt[0]), "y": float(pt[1]), "confidence": float(confidences[name])} for name, pt in keypoints.items()}
    KEYPOINT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    KEYPOINT_SNAPSHOT.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    return keypoints


def project_detections() -> Path:
    if not H_MATRIX_OUTPUT.exists():
        raise FileNotFoundError(
            f"No se encontró la matriz de homografía en {H_MATRIX_OUTPUT}. Ejecuta compute_homography() primero."
        )
    if not DETECTIONS_WITH_TEAMS.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV de detecciones con equipos en {DETECTIONS_WITH_TEAMS}. Ejecuta el clustering previamente."
        )

    homography = np.load(H_MATRIX_OUTPUT)
    estimator = HomographyEstimator(DEFAULT_FIELD_LAYOUT)

    df = pd.read_csv(DETECTIONS_WITH_TEAMS)
    centers = []
    for bbox_raw in df["BBox"]:
        x1, y1, x2, y2 = _parse_bbox(bbox_raw)
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    projected = estimator.project_points(homography, centers)
    field_x = np.full(len(df), np.nan, dtype=float)
    field_y = np.full(len(df), np.nan, dtype=float)
    if projected.size:
        field_x = projected[:, 0]
        field_y = projected[:, 1]

    df["FieldX"] = field_x
    df["FieldY"] = field_y

    PROJECTIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROJECTIONS_OUTPUT, index=False)
    return PROJECTIONS_OUTPUT


def run() -> Path:
    compute_homography(frame_index=0)
    return project_detections()


if __name__ == "__main__":
    try:
        output = run()
        print(f"Homografía aplicada. Detecciones proyectadas guardadas en: {output}")
    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"Error al generar la homografía: {exc}")
