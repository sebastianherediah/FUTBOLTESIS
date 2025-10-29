"""Visualiza detecciones proyectándolas sobre un video y exportando un clip anotado."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from homography.field_layout import DEFAULT_FIELD_LAYOUT
from homography.minimap import MinimapRenderer, PlayerPoint


@dataclass(frozen=True)
class Detection:
    frame: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    team: Optional[str] = None
    field_point: Optional[Tuple[float, float]] = None


def _parse_bbox(raw: object) -> Optional[Tuple[int, int, int, int]]:
    """Convierte una cadena/lista en coordenadas de bbox válidas."""

    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None

    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(round(float(val))) for val in raw]
    except (TypeError, ValueError):
        return None

    return x1, y1, x2, y2


def _load_detections(csv_path: Path) -> Dict[int, List[Detection]]:
    df = pd.read_csv(csv_path)
    required = {"Frame", "ClassName", "BBox"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El CSV no contiene las columnas obligatorias: {missing}")

    has_team = "Team" in df.columns
    has_field = {"FieldX", "FieldY"}.issubset(df.columns)

    grouped: Dict[int, List[Detection]] = {}
    for row in df.itertuples(index=False):
        try:
            frame = int(getattr(row, "Frame"))
        except (TypeError, ValueError):
            continue

        class_name = str(getattr(row, "ClassName"))
        bbox = _parse_bbox(getattr(row, "BBox"))
        if bbox is None:
            continue

        team = getattr(row, "Team") if has_team else None
        if team is not None and pd.isna(team):
            team = None

        if has_field:
            field_x = getattr(row, "FieldX")
            field_y = getattr(row, "FieldY")
            if pd.isna(field_x) or pd.isna(field_y):
                field_point = None
            else:
                field_point = (float(field_x), float(field_y))
        else:
            field_point = None

        grouped.setdefault(frame, []).append(Detection(frame, class_name, bbox, team, field_point))

    return grouped


def _color_id(identifier: str) -> Tuple[int, int, int]:
    """Devuelve un color BGR determinístico."""

    palette = [
        (255, 99, 71),
        (135, 206, 235),
        (152, 251, 152),
        (255, 215, 0),
        (240, 128, 128),
        (147, 112, 219),
        (60, 179, 113),
        (70, 130, 180),
    ]
    idx = abs(hash(identifier)) % len(palette)
    r, g, b = palette[idx]
    return b, g, r  # cv2 usa BGR


def _annotate_frame(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        label_id = detection.team if detection.team else detection.class_name
        color = _color_id(label_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if detection.team:
            label = detection.team
        elif detection.class_name.lower() == "balon":
            label = "balon"
        else:
            label = detection.class_name
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    return annotated


def visualize(
    video_path: Path,
    detections_path: Path,
    output_path: Path,
    *,
    fps: Optional[float] = None,
    codec: str = "mp4v",
    draw_minimap: bool = False,
) -> Path:
    detections = _load_detections(detections_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer_fps = fps if fps and fps > 0 else input_fps

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, writer_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"No se pudo inicializar el escritor de video con el códec '{codec}'. "
            "Prueba con otro códec (por ejemplo, avc1, xvid, mjpg) o usa ffmpeg."
        )

    minimap_renderer: Optional[MinimapRenderer] = None
    if draw_minimap:
        try:
            minimap_renderer = MinimapRenderer(DEFAULT_FIELD_LAYOUT)
        except ValueError as exc:
            raise RuntimeError(
                "No se pudo inicializar el minimapa. Asegúrate de definir DEFAULT_FIELD_LAYOUT en homography/field_layout.py."
            ) from exc

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    try:
        with tqdm(total=total_frames, desc="Renderizando video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_detections = detections.get(frame_idx, [])
                annotated = _annotate_frame(frame, frame_detections)

                if minimap_renderer:
                    players: List[PlayerPoint] = []
                    ball_position: Optional[Tuple[float, float]] = None
                    for det in frame_detections:
                        if det.field_point is None:
                            continue
                        if det.class_name.lower() == "balon":
                            ball_position = det.field_point
                        elif det.class_name.lower() == "jugador":
                            players.append(PlayerPoint(det.field_point[0], det.field_point[1], det.team))
                    if players or ball_position is not None:
                        minimap = minimap_renderer.render(players, ball_position=ball_position)
                        annotated = MinimapRenderer.overlay(annotated, minimap)

                writer.write(annotated)

                frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()
        writer.release()

    return output_path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un video con detecciones superpuestas a partir de un CSV."
    )
    parser.add_argument("--video", required=True, type=Path, help="Ruta al video original.")
    parser.add_argument(
        "--detections",
        required=True,
        type=Path,
        help="CSV de detecciones con columnas Frame, ClassName y BBox.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/detecciones_visualizadas.mp4"),
        help="Ruta del video anotado de salida.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS del video de salida. Si se omite, se usa el FPS del video original.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Cuatro letras del códec (FourCC) cuando se usa OpenCV. Ejemplos: mp4v, xvid, mjpg.",
    )
    parser.add_argument(
        "--use-ffmpeg",
        action="store_true",
        help="Si se especifica, usa ffmpeg para renderizar (requiere ffmpeg instalado).",
    )
    parser.add_argument(
        "--minimap",
        action="store_true",
        help="Dibuja un minimapa 2D usando las columnas FieldX/FieldY del CSV (requiere layout definido).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    if args.use_ffmpeg:
        result = visualize_with_ffmpeg(
            args.video,
            args.detections,
            args.output,
            fps=args.fps,
            draw_minimap=args.minimap,
        )
    else:
        result = visualize(
            args.video,
            args.detections,
            args.output,
            fps=args.fps,
            codec=args.codec,
            draw_minimap=args.minimap,
        )
    print(f"Video anotado guardado en: {result}")


def visualize_with_ffmpeg(
    video_path: Path,
    detections_path: Path,
    output_path: Path,
    *,
    fps: Optional[float] = None,
    draw_minimap: bool = False,
) -> Path:
    """Alternativa usando ffmpeg via subprocess para códecs más compatibles."""

    temp_output = output_path.with_suffix(".tmp.mp4")
    visualize(video_path, detections_path, temp_output, fps=fps, codec="mp4v", draw_minimap=draw_minimap)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_output),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg falló al generar el video: {result.stderr.decode('utf-8', errors='ignore')}"
        )

    temp_output.unlink(missing_ok=True)

    return output_path


if __name__ == "__main__":
    main()
