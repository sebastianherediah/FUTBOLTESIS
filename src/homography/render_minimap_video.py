"""Generate a video overlay with projected player/ball positions on a 2D minimap."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import IMAGE_HEIGHT, IMAGE_WIDTH
from .field_layout import DEFAULT_FIELD_LAYOUT
from .homography_estimator import HomographyEstimator
from .minimap import MinimapRenderer, PlayerPoint
from .predictor import HomographyKeypointPredictor, Keypoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render minimap overlay using homography keypoints.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/Homografia.pth"))
    parser.add_argument("--video", type=Path, default=Path("VideoPruebaTesis.mp4"))
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path("outputs/detecciones_con_equipos.csv"),
        help="CSV con las detecciones (Frame, ClassName, BBox, Team).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/minimap_preview.mp4"),
        help="Video de salida con el minimapa superpuesto.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Frame inicial (por defecto 0).")
    parser.add_argument("--max-frames", type=int, default=100, help="Número máximo de frames a procesar.")
    parser.add_argument("--device", type=str, default=None, help="cpu o cuda (auto detección por defecto).")
    parser.add_argument("--confidence", type=float, default=0.05, help="Umbral mínimo de confianza para keypoints.")
    parser.add_argument("--min-matches", type=int, default=4, help="Keypoints mínimos aceptados para estimar la homografía.")
    parser.add_argument("--flip-x", action="store_true", help="Invierte el eje X del minimapa.")
    parser.add_argument("--flip-y", action="store_true", help="Invierte el eje Y del minimapa.")
    return parser.parse_args()


def _parse_bbox(raw: object) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return tuple(float(v) for v in raw)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _keypoints_to_dict(keypoints: Iterable[Keypoint]) -> Dict[str, Tuple[float, float]]:
    return {kp.name: (kp.x, kp.y) for kp in keypoints}


def main() -> None:
    args = _parse_args()
    device = args.device.lower() if isinstance(args.device, str) else None
    if device not in (None, "cpu", "cuda"):
        raise ValueError("El parámetro --device debe ser 'cpu' o 'cuda'.")

    predictor = HomographyKeypointPredictor(args.checkpoint, device=device)
    estimator = HomographyEstimator(DEFAULT_FIELD_LAYOUT)
    minimap_renderer = MinimapRenderer(flip_x=args.flip_x, flip_y=args.flip_y)

    detections_df = pd.read_csv(args.detections)
    if "BBox" not in detections_df.columns:
        raise ValueError("El CSV de detecciones debe contener la columna 'BBox'.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {args.video}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    scale_x = IMAGE_WIDTH / original_width
    scale_y = IMAGE_HEIGHT / original_height

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("No se pudo inicializar el escritor de video.")

    progress = tqdm(total=args.max_frames, desc="Renderizando minimapa", unit="frame")
    last_h = None

    try:
        frame_index = 0
        target_start = max(0, args.start_frame)
        if target_start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_start)
        while frame_index < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            all_keypoints, resized_frame = predictor.predict(frame, confidence_threshold=0.0)
            sorted_keypoints = sorted(all_keypoints, key=lambda kp: kp.confidence, reverse=True)
            selected_keypoints = [kp for kp in sorted_keypoints if kp.confidence >= args.confidence]
            if len(selected_keypoints) < 8:
                selected_keypoints = sorted_keypoints[:8]
            keypoint_dict = _keypoints_to_dict(selected_keypoints)

            try:
                homography = estimator.estimate(keypoint_dict, min_matches=args.min_matches)
                last_h = homography.matrix
            except RuntimeError:
                if last_h is None:
                    writer.write(resized_frame)
                    frame_index += 1
                    progress.update(1)
                    continue

            current_frame_number = target_start + frame_index
            frame_detections = detections_df[detections_df["Frame"] == current_frame_number]

            players: list[PlayerPoint] = []
            ball_position: Optional[Tuple[float, float]] = None

            if not frame_detections.empty and last_h is not None:
                centres: list[Tuple[float, float]] = []
                det_rows: list[pd.Series] = []
                for _, row in frame_detections.iterrows():
                    bbox = _parse_bbox(row["BBox"])
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2.0 * scale_x
                    cy = y2 * scale_y  # usar la parte media del borde inferior de la caja
                    centres.append((cx, cy))
                    det_rows.append(row)

                if centres:
                    projected = estimator.project_points(last_h, centres)
                    centres_array = np.array(centres, dtype=np.float32)
                    if projected.shape[0] >= 2:
                        corr = np.corrcoef(centres_array[:, 0], projected[:, 0])[0, 1]
                        if corr < 0:
                            projected[:, 0] *= -1.0
                    for row, (fx, fy) in zip(det_rows, projected):
                        cls = str(row["ClassName"]).lower()
                        if cls == "jugador":
                            players.append(PlayerPoint(fx, fy, team=str(row.get("Team", "")).upper()))
                        elif cls == "balon":
                            ball_position = (float(fx), float(fy))

            minimap = minimap_renderer.render(players, ball_position=ball_position)
            composed = MinimapRenderer.overlay(resized_frame, minimap)
            writer.write(composed)

            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        writer.release()

    print(f"Video con minimapa guardado en: {args.output}")


if __name__ == "__main__":
    main()
