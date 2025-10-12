"""Video inference pipeline for RF-DETR football detectors."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from rfdetr import RFDETRBase


@dataclass
class InferenceResult:
    """Represents a single detection in a video frame."""

    frame: int
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]


class VideoInference:
    """Runs RF-DETR inference over a video and builds a tabular dataset."""

    def __init__(
        self,
        model_path: str | Path,
        device: Optional[str] = None,
    ) -> None:
        """Load a fine-tuned RF-DETR checkpoint ready for video inference."""

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        torch.serialization.add_safe_globals([Namespace])
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        class_names = list(getattr(checkpoint.get("args", Namespace()), "class_names", []))

        if not class_names:
            raise ValueError("Checkpoint metadata must include class names.")

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)

        self.model = RFDETRBase(
            pretrain_weights=str(model_path),
            num_classes=len(class_names),
            class_names=class_names,
            device=str(self.device),
        )
        self.model.eval()
        self.class_names = class_names

    def infer(
        self,
        video_path: str | Path,
        threshold: float = 0.5,
        stride: int = 1,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Run inference frame-by-frame and return the detections as a DataFrame."""

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if stride <= 0:
            raise ValueError("Stride must be a positive integer.")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc="Procesando frames") if show_progress else None
        results: List[InferenceResult] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                if frame_idx % stride == 0:
                    with torch.inference_mode():
                        detections = self.model.predict(frame, threshold=threshold)
                    boxes = detections.xyxy
                    class_ids = detections.class_id
                    confidences = detections.confidence

                    for box, class_id, confidence in zip(boxes, class_ids, confidences):
                        x1, y1, x2, y2 = map(float, box.tolist())
                        # RF-DETR class ids are 1-indexed in the checkpoints.
                        class_index = int(class_id) - 1
                        if 0 <= class_index < len(self.class_names):
                            class_name = self.class_names[class_index]
                        else:
                            class_name = f"class_{int(class_id)}"

                        results.append(
                            InferenceResult(
                                frame=frame_idx,
                                class_name=class_name,
                                class_id=int(class_id),
                                confidence=float(confidence),
                                bbox=[x1, y1, x2, y2],
                            )
                        )

                frame_idx += 1
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            capture.release()
            if progress_bar is not None:
                progress_bar.close()

        records = [
            {
                "frame": result.frame,
                "class_name": result.class_name,
                "class_id": result.class_id,
                "confidence": result.confidence,
                "bbox": result.bbox,
            }
            for result in results
        ]

        return pd.DataFrame.from_records(
            records,
            columns=["frame", "class_name", "class_id", "confidence", "bbox"],
        )

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path,
        format: str = "parquet",
    ) -> Path:
        """Persist the detections dataset to disk."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        format = format.lower()
        if format == "parquet":
            dataframe.to_parquet(output_path, index=False)
        elif format == "csv":
            dataframe.to_csv(output_path, index=False)
        else:
            raise ValueError("Unsupported format. Use 'parquet' or 'csv'.")

        return output_path


__all__ = ["VideoInference", "InferenceResult"]
