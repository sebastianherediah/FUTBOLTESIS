"""Utilidades para ejecutar inferencia frame a frame con RF-DETR."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from rfdetr import RFDETRBase


class VideoInference:
    """Realiza inferencia sobre un video completo usando un checkpoint de RF-DETR."""

    def __init__(self, model_path: str) -> None:
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
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

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
        return pd.DataFrame(results, columns=["Frame", "ClassName", "BBox"])


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta inferencia con RF-DETR sobre un video y exporta las detecciones."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ruta al checkpoint de RF-DETR (archivo .pth).",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Ruta al archivo de video a procesar.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confianza mínima (0-1) para conservar detecciones. Valor por defecto: 0.5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta donde guardar las detecciones en CSV. Si se omite, solo se imprime un resumen.",
    )
    return parser.parse_args(argv)


def _run_cli(args: argparse.Namespace) -> None:
    inference = VideoInference(args.model)
    detections = inference.process(args.video, threshold=args.threshold)

    if detections.empty:
        print("No se encontraron detecciones con el umbral especificado.")
    else:
        print(f"Detecciones totales: {len(detecciones)}")
        print(detections.head())

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        detections.to_csv(output_path, index=False)
        print(f"Detecciones guardadas en: {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("El parámetro --threshold debe estar en el rango [0, 1]")
    _run_cli(args)


__all__ = ["VideoInference", "main"]


if __name__ == "__main__":
    main()
