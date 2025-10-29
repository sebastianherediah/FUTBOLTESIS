"""Inference utilities for the homography keypoint detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from .hrnet import HighResolutionNet


DEFAULT_KEYPOINT_NAMES: Tuple[str, ...] = tuple(f"kp_{idx:02d}" for idx in range(29))


def _letterbox(image: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resizes keeping aspect ratio and pads to a square image."""

    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_LINEAR)
    pad_w = size - resized.shape[1]
    pad_h = size - resized.shape[0]
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_h - pad_top,
        pad_left,
        pad_w - pad_left,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded, scale, (pad_left, pad_top)


@dataclass(frozen=True)
class KeypointPrediction:
    """Keypoint coordinates predicted for a frame."""

    frame_index: int
    keypoints: Dict[str, Tuple[float, float]]
    confidence: Dict[str, float]


class HomographyKeypointModel(nn.Module):
    """Wrapper around the HRNet-based keypoint regression model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        keypoint_names: Iterable[str] = DEFAULT_KEYPOINT_NAMES,
        input_size: int = 256,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = int(input_size)
        self.keypoint_names: Tuple[str, ...] = tuple(keypoint_names)
        if len(self.keypoint_names) != 29:
            raise ValueError("Se esperaban 29 nombres de keypoints para alinear con el modelo entrenado.")

        self.backbone = HighResolutionNet()
        self.prediction_head = nn.Sequential(
            nn.Conv2d(334, 334, kernel_size=1, bias=False),
            nn.BatchNorm2d(334),
            nn.ReLU(inplace=True),
            nn.Conv2d(334, 58, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint["model_state_dict"]

        backbone_state = {k.replace("backbone.", "", 1): v for k, v in model_state.items() if k.startswith("backbone.")}
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"No se pudo cargar el backbone HRNet con el checkpoint proporcionado. Missing={missing}, Unexpected={unexpected}"
            )

        head_state = {k.replace("prediction_head.", "", 1): v for k, v in model_state.items() if k.startswith("prediction_head.")}
        self.prediction_head.load_state_dict(head_state, strict=True)

        self.to(self.device)
        self.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        low_level, features = self.backbone(images)
        high_res = features[0]
        target_size = high_res.shape[-2:]
        fused: List[torch.Tensor] = [high_res]
        for branch in features[1:]:
            fused.append(F.interpolate(branch, size=target_size, mode="bilinear", align_corners=False))

        if low_level.shape[-2:] != target_size:
            low_level = F.interpolate(low_level, size=target_size, mode="bilinear", align_corners=False)
        fused.append(low_level)

        features_cat = torch.cat(fused, dim=1)
        heatmaps = self.prediction_head(features_cat)
        pooled = self.pool(heatmaps).flatten(1)  # (N, 58)
        coords = torch.sigmoid(pooled).view(-1, len(self.keypoint_names), 2)
        return coords

    @torch.inference_mode()
    def predict_from_frame(self, frame: np.ndarray) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
        """Devuelve diccionarios de coordenadas y confianza para un frame en BGR."""

        original_h, original_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        letterboxed, scale, (pad_x, pad_y) = _letterbox(rgb, self.input_size)

        tensor = self.transform(letterboxed).unsqueeze(0).to(self.device)
        coords = self.forward(tensor)[0].cpu().numpy()  # shape (29, 2)

        mapped = []
        confidences = []
        for x_norm, y_norm in coords:
            x_resized = x_norm * self.input_size - pad_x
            y_resized = y_norm * self.input_size - pad_y
            x_img = np.clip(x_resized / scale, 0, original_w - 1)
            y_img = np.clip(y_resized / scale, 0, original_h - 1)
            mapped.append((float(x_img), float(y_img)))
            # Usa la distancia al centro (heurística) como confianza provisional.
            confidences.append(float(np.clip(1.0 - abs(0.5 - float(x_norm)), 0.0, 1.0)))

        keypoints = {name: mapped[idx] for idx, name in enumerate(self.keypoint_names)}
        confidence = {name: confidences[idx] for idx, name in enumerate(self.keypoint_names)}
        return keypoints, confidence


__all__ = ["HomographyKeypointModel", "KeypointPrediction", "DEFAULT_KEYPOINT_NAMES"]
