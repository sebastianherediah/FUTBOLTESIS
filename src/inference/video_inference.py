"""Utilidades para ejecutar inferencia frame a frame con RF-DETR."""

from __future__ import annotations

from argparse import Namespace
from typing import List, Sequence

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from rfdetr import RFDETRBase


class VideoInference:
    """Realiza inferencia sobre un video completo usando un checkpoint de RF-DETR.

    El flujo carga el modelo especificado, procesa cada frame del video y
    devuelve un :class:`pandas.DataFrame` con las detecciones encontradas.
    """

    def __init__(self, model_path: str) -> None:
        # Permitir deserializar argparse.Namespace contenidos en el checkpoint.
        torch.serialization.add_safe_globals([Namespace])

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        raw_names: Sequence[str] = getattr(ckpt["args"], "class_names", [])
        num_classes = len(raw_names)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = RFDETRBase(
            pretrain_weights=model_path,
            num_classes=num_classes,
            class_names=list(raw_names),
            device=device,
        )
        self.class_names: List[str] = list(raw_names)

    def process(self, video_path: str, threshold: float = 0.5) -> pd.DataFrame:
        """Procesa un vídeo y retorna detecciones con columnas Frame/ClassName/BBox."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []
        frame_idx = 0

        with tqdm(total=total_frames, desc="Procesando frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.model.predict(frame, threshold=threshold)
                for box, cid, conf in zip(
                    detections.xyxy, detections.class_id, detections.confidence
                ):
                    x1, y1, x2, y2 = map(float, box.tolist())
                    idx = cid - 1
                    name = (
                        self.class_names[idx]
                        if 0 <= idx < len(self.class_names)
                        else f"class_{cid}"
                    )
                    results.append(
                        {
                            "Frame": frame_idx,
                            "ClassName": name,
                            "BBox": [x1, y1, x2, y2],
                        }
                    )

                frame_idx += 1
                pbar.update(1)

        cap.release()
        df = pd.DataFrame(results, columns=["Frame", "ClassName", "BBox"])
        return df


__all__ = ["VideoInference"]
