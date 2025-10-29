"""Homography estimation from detected keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from .field_layout import FieldLayout


@dataclass(frozen=True)
class HomographyResult:
    matrix: np.ndarray
    inliers: np.ndarray
    used_keypoints: Tuple[str, ...]


class HomographyEstimator:
    """Computes the planar homography between image coordinates and the field layout."""

    def __init__(self, layout: FieldLayout) -> None:
        if not layout.keypoints:
            raise ValueError(
                "No se definieron coordenadas de referencia en FieldLayout. "
                "Actualiza `DEFAULT_FIELD_LAYOUT` con los keypoints de la cancha."
            )
        self.layout = layout

    def estimate(self, image_keypoints: Dict[str, Tuple[float, float]]) -> HomographyResult:
        common = [name for name in self.layout.keypoints if name in image_keypoints]
        if len(common) < 4:
            raise RuntimeError(
                f"Se requieren al menos 4 keypoints para calcular la homografía; encontrados {len(common)}."
            )

        src = np.array([image_keypoints[name] for name in common], dtype=np.float32)
        dst = np.array([self.layout.keypoints[name] for name in common], dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None or mask is None:
            raise RuntimeError("cv2.findHomography no pudo estimar una matriz válida.")

        return HomographyResult(matrix=H, inliers=mask.ravel().astype(bool), used_keypoints=tuple(common))

    @staticmethod
    def project_points(matrix: np.ndarray, points: Iterable[Tuple[float, float]]) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts_h = cv2.convertPointsToHomogeneous(pts).reshape(-1, 3).T  # shape (3, N)
        projected = matrix @ pts_h  # shape (3, N)
        projected /= projected[2:3]
        return projected[:2].T.astype(np.float32)


__all__ = ["HomographyEstimator", "HomographyResult"]
