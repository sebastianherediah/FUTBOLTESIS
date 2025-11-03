"""High level predictor to extract homography keypoints from video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from .config import (
    HEATMAP_STRIDE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    KEYPOINT_INDEX_TO_NAME,
    KEYPOINT_NAMES,
    MEAN_RGB,
    NUM_KEYPOINTS,
    STD_RGB,
)
from .model import HRNetHeatmapModel, load_model_from_checkpoint


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float


class HomographyKeypointPredictor:
    """Loads the trained HRNet model and produces keypoints for input frames."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: HRNetHeatmapModel = load_model_from_checkpoint(str(checkpoint_path), device=self.device)

        self.mean = torch.tensor(MEAN_RGB, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD_RGB, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _prepare_tensor(self, frame_bgr: np.ndarray) -> Tuple[Tensor, np.ndarray]:
        resized = cv2.resize(frame_bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor, resized

    @torch.inference_mode()
    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        confidence_threshold: float = 0.05,
    ) -> Tuple[List[Keypoint], np.ndarray]:
        """Returns the predicted keypoints and the resized frame used for inference."""

        input_tensor, resized_frame = self._prepare_tensor(frame_bgr)
        log_probs = self.model(input_tensor)
        probs = torch.exp(log_probs)[0, :NUM_KEYPOINTS]  # Ignore background channel

        keypoints: List[Keypoint] = []
        for idx in range(NUM_KEYPOINTS):
            heatmap = probs[idx]
            max_val = heatmap.max()
            confidence = float(max_val.item())
            if confidence < confidence_threshold:
                continue

            flat_idx = int(torch.argmax(heatmap).item())
            heatmap_h, heatmap_w = heatmap.shape
            row = flat_idx // heatmap_w
            col = flat_idx % heatmap_w

            x = float(col * HEATMAP_STRIDE)
            y = float(row * HEATMAP_STRIDE)
            name = KEYPOINT_INDEX_TO_NAME[idx]
            keypoints.append(Keypoint(name=name, x=x, y=y, confidence=confidence))

        return keypoints, resized_frame

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        confidence_threshold: float = 0.05,
        radius: int = 4,
        draw_labels: bool = True,
    ) -> np.ndarray:
        keypoints, resized_frame = self.predict(frame_bgr, confidence_threshold=confidence_threshold)
        annotated = resized_frame.copy()

        for kp in keypoints:
            center = (int(round(kp.x)), int(round(kp.y)))
            cv2.circle(annotated, center, radius, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            if draw_labels:
                cv2.putText(
                    annotated,
                    kp.name,
                    (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 0),
                    1,
                    lineType=cv2.LINE_AA,
                )
        return annotated
