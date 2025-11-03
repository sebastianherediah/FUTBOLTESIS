"""CLI utility to render homography keypoints over a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from .config import IMAGE_HEIGHT, IMAGE_WIDTH
from .predictor import HomographyKeypointPredictor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dibuja los keypoints de homografía sobre un video.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/Homografia.pth"),
        help="Ruta al checkpoint entrenado (archivo .pth).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("VideoPruebaTesis.mp4"),
        help="Ruta al video de entrada.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/video_keypoints.mp4"),
        help="Ruta del video anotado de salida.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.05,
        help="Umbral mínimo de confianza para dibujar un keypoint (0-1).",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Si se especifica, no dibuja etiquetas de texto junto a los keypoints.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Procesa solo un número limitado de frames (útil para pruebas rápidas).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo a utilizar (cpu o cuda). Por defecto se auto-detecta.",
    )
    return parser.parse_args()


def annotate_video(
    predictor: HomographyKeypointPredictor,
    video_path: Path,
    output_path: Path,
    *,
    confidence_threshold: float,
    draw_labels: bool,
    max_frames: int | None = None,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("No se pudo inicializar el escritor de video. Prueba otro códec o ruta de salida.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    progress = tqdm(total=total_frames, desc="Aplicando keypoints", unit="frame")

    processed = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = predictor.annotate_frame(
                frame,
                confidence_threshold=confidence_threshold,
                draw_labels=draw_labels,
            )
            writer.write(annotated)
            progress.update(1)
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        progress.close()
        cap.release()
        writer.release()

    return output_path


def main() -> None:
    args = _parse_args()
    device = args.device.lower() if isinstance(args.device, str) else None
    if device not in (None, "cpu", "cuda"):
        raise ValueError("El parámetro --device debe ser 'cpu' o 'cuda'.")
    predictor = HomographyKeypointPredictor(args.checkpoint, device=device)
    output = annotate_video(
        predictor,
        args.video,
        args.output,
        confidence_threshold=args.confidence,
        draw_labels=not args.no_labels,
        max_frames=args.max_frames,
    )
    print(f"Video con keypoints guardado en: {output}")


if __name__ == "__main__":
    main()
