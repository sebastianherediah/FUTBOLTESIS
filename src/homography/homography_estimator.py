"""Utilities to estimate the planar homography between image and field coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from .field_layout import FieldLayout, DEFAULT_FIELD_LAYOUT


@dataclass(frozen=True)
class HomographyResult:
    matrix: np.ndarray
    inliers: np.ndarray
    used_keypoints: Tuple[str, ...]


class HomographyEstimator:
    """Estimates a homography using the set of predicted keypoints."""

    def __init__(self, layout: FieldLayout = DEFAULT_FIELD_LAYOUT) -> None:
        if not layout.keypoints:
            raise ValueError("El layout de la cancha no contiene keypoints.")
        self.layout = layout

    def estimate(
        self,
        image_keypoints: Dict[str, Tuple[float, float]],
        *,
        min_matches: int = 6,
        ransac_thresh: float = 4.0,
    ) -> HomographyResult:
        """Estimates the homography from image coordinates to field coordinates."""

        common_names = [name for name in self.layout.keypoints if name in image_keypoints]
        if len(common_names) < min_matches:
            raise RuntimeError(
                f"No hay suficientes keypoints para calcular la homografía "
                f"(obtenidos {len(common_names)}, se requieren al menos {min_matches})."
            )

        img_points = np.array([image_keypoints[name] for name in common_names], dtype=np.float32)
        field_points = np.array([self.layout.keypoints[name] for name in common_names], dtype=np.float32)

        H, mask = cv2.findHomography(img_points, field_points, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None:
            H, mask = cv2.findHomography(img_points, field_points, 0)
        if H is None or mask is None:
            raise RuntimeError("cv2.findHomography no encontró una solución válida.")

        inliers = mask.ravel().astype(bool)
        if inliers.sum() < min_matches:
            H_alt, mask_alt = cv2.findHomography(img_points, field_points, 0)
            if H_alt is not None and mask_alt is not None:
                alt_inliers = mask_alt.ravel().astype(bool)
                if alt_inliers.sum() >= 4:  # mínimo geométrico
                    H = H_alt
                    inliers = alt_inliers
                else:
                    raise RuntimeError(
                        f"Homografía inválida: solo {inliers.sum()} inliers de {len(common_names)} keypoints."
                    )
            else:
                raise RuntimeError(
                    f"Homografía inválida: solo {inliers.sum()} inliers de {len(common_names)} keypoints."
                )

        used = tuple(common_names[i] for i, is_inlier in enumerate(inliers) if is_inlier)
        return HomographyResult(matrix=H, inliers=inliers, used_keypoints=used)

    @staticmethod
    def project_points(matrix: np.ndarray, points: Iterable[Tuple[float, float]]) -> np.ndarray:
        pts = np.array(list(points), dtype=np.float32)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        homogeneous = cv2.convertPointsToHomogeneous(pts).reshape(-1, 3).T  # (3, N)
        projected = matrix @ homogeneous  # (3, N)
        projected /= projected[2:3]
        return projected[:2].T.astype(np.float32)
