"""Genera un video con el contador acumulado de pases por equipo."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.analytics import PassAnalyzer, build_pass_count_timeline  # type: ignore
else:
    from ..analytics import PassAnalyzer, build_pass_count_timeline


def _parse_bbox(value: object) -> Optional[List[float]]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _load_detections(csv_path: Path) -> pd.DataFrame:
    detections = pd.read_csv(csv_path)
    if "BBox" not in detections.columns:
        raise ValueError("El CSV debe contener una columna 'BBox' con las bounding boxes.")

    detections["BBox"] = detections["BBox"].apply(_parse_bbox)
    if detections["BBox"].isna().any():
        raise ValueError("Existen filas con bounding boxes inválidas en el CSV proporcionado.")

    if "TrackId" in detections.columns:
        detections["TrackId"] = pd.to_numeric(detections["TrackId"], errors="coerce").astype("Int64")

    return detections


def _ordered_team_labels(teams: Sequence[str], override: Optional[Sequence[str]]) -> List[str]:
    if override:
        normalized = [name for name in override if name in teams]
        missing = [name for name in override if name not in teams]
        return normalized + missing
    return list(dict.fromkeys(teams))


def render_pass_counter_video(
    video_path: Path,
    detections_path: Path,
    output_path: Path,
    *,
    team_order: Optional[Sequence[str]] = None,
    distance_threshold: float = 90.0,
    min_possession_frames: int = 2,
    max_gap_frames: int = 12,
    team_link_max_distance: float = 160.0,
    team_link_max_gap: int = 20,
    ball_max_interp_gap: int = 12,
    control_smoothing_window: int = 5,
    control_smoothing_min_votes: int = 2,
    font_scale: float = 0.9,
    thickness: int = 2,
    padding: int = 20,
    line_spacing: int = 8,
    box_alpha: float = 0.6,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    box_color: Tuple[int, int, int] = (15, 15, 15),
) -> Path:
    """Renderiza un video con el contador acumulado de pases.

    Parameters
    ----------
    distance_threshold:
        Distancia máxima balón-jugador para asignar posesión.
    min_possession_frames:
        Frames consecutivos mínimos para considerar un segmento de posesión.
    max_gap_frames:
        Separación máxima entre segmentos del mismo equipo para que cuenten como pase.
    team_link_max_distance:
        Distancia máxima entre posesiones consecutivas del mismo equipo al reutilizar IDs heurísticos.
    team_link_max_gap:
        Hueco máximo entre frames consecutivos del mismo equipo al reutilizar IDs heurísticos.
    ball_max_interp_gap:
        Hueco máximo que se interpolará para el balón cuando desaparece temporalmente.
    control_smoothing_window:
        Tamaño de la ventana de suavizado aplicada a la línea de control.
    control_smoothing_min_votes:
        Votos mínimos dentro de la ventana para reemplazar la asignación original.
    """

    detections = _load_detections(detections_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    analyzer = PassAnalyzer(
        distance_threshold=distance_threshold,
        min_possession_frames=min_possession_frames,
        max_gap_frames=max_gap_frames,
        team_link_max_distance=team_link_max_distance,
        team_link_max_gap=team_link_max_gap,
        ball_max_interp_gap=ball_max_interp_gap,
        control_smoothing_window=control_smoothing_window,
        control_smoothing_min_votes=control_smoothing_min_votes,
    )

    frame_range = range(total_frames) if total_frames > 0 else None
    timeline = build_pass_count_timeline(
        detections,
        analyzer=analyzer,
        team_order=team_order,
        frame_range=frame_range,
    )

    ordered_teams = _ordered_team_labels(timeline.teams, team_order)
    if not ordered_teams:
        raise ValueError(
            "No se detectaron equipos en las detecciones. "
            "Asegúrate de que la columna 'Team' esté presente y contenga valores válidos."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"No se pudo inicializar el escritor de video en {output_path}. "
            "Prueba con una ruta diferente o verifica los códecs instalados."
        )

    try:
        with tqdm(total=total_frames, desc="Generando contador de pases") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                counts = timeline.counts_for_frame(frame_idx)
                lines = [f"{team}: {counts.get(team, 0)}" for team in ordered_teams]
                _draw_counter(frame, lines, font_scale, thickness, padding, line_spacing, box_alpha, font_color, box_color)

                writer.write(frame)
                pbar.update(1)
    finally:
        cap.release()
        writer.release()

    return output_path


def _draw_counter(
    frame: np.ndarray,
    lines: Sequence[str],
    font_scale: float,
    thickness: int,
    padding: int,
    line_spacing: int,
    box_alpha: float,
    font_color: Tuple[int, int, int],
    box_color: Tuple[int, int, int],
) -> None:
    if not lines:
        return

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    line_gap = max(int(round(line_spacing * font_scale)), 2)
    box_padding = max(int(round(padding * 0.5)), 8)

    line_metrics: List[Tuple[int, int, int]] = []
    max_width = 0
    total_height = 0

    for text in lines:
        (width, height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        total_height += height + baseline
        max_width = max(max_width, width)
        line_metrics.append((width, height, baseline))

    total_height += (len(lines) - 1) * line_gap
    total_height += box_padding * 2
    box_width = max_width + box_padding * 2

    x_left = padding
    y_bottom = frame.shape[0] - padding
    y_top = max(y_bottom - total_height, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_left, y_top), (x_left + box_width, y_bottom), box_color, -1)
    cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)

    y_cursor = y_top + box_padding
    for idx, (text, (_, height, baseline)) in enumerate(zip(lines, line_metrics)):
        y_cursor += height
        cv2.putText(
            frame,
            text,
            (x_left + box_padding, y_cursor),
            font_face,
            font_scale,
            font_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        y_cursor += baseline
        if idx < len(lines) - 1:
            y_cursor += line_gap


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Renderiza un video mostrando únicamente el contador de pases por equipo."
    )
    parser.add_argument("--video", required=True, type=Path, help="Ruta al video original.")
    parser.add_argument(
        "--detections",
        required=True,
        type=Path,
        help="CSV con las detecciones enriquecidas (Frame, ClassName, BBox, Team; TrackId opcional).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Ruta del video de salida con el contador de pases.",
    )
    parser.add_argument(
        "--team-order",
        nargs="+",
        default=None,
        help="Listado con el orden deseado de los equipos (ej: COLOMBIA BRASIL).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=90.0,
        help="Distancia máxima balón-jugador para considerar posesión.",
    )
    parser.add_argument(
        "--min-possession-frames",
        type=int,
        default=2,
        help="Duración mínima (en frames) de una posesión válida.",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=12,
        help="Separación máxima entre posesiones consecutivas para formar un pase.",
    )
    parser.add_argument(
        "--team-link-max-distance",
        type=float,
        default=160.0,
        help="Distancia máxima para reutilizar IDs heurísticos cuando no hay TrackId.",
    )
    parser.add_argument(
        "--team-link-max-gap",
        type=int,
        default=20,
        help="Hueco máximo (frames) para reutilizar IDs heurísticos sin TrackId.",
    )
    parser.add_argument(
        "--ball-max-interp-gap",
        type=int,
        default=12,
        help="Máximo de frames consecutivos a interpolar cuando el balón no se detecta.",
    )
    parser.add_argument(
        "--control-smoothing-window",
        type=int,
        default=5,
        help="Tamaño de la ventana de suavizado para la posesión.",
    )
    parser.add_argument(
        "--control-smoothing-min-votes",
        type=int,
        default=2,
        help="Número mínimo de votos dentro de la ventana para reemplazar la asignación.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    render_pass_counter_video(
        args.video,
        args.detections,
        args.output,
        team_order=args.team_order,
        distance_threshold=args.distance_threshold,
        min_possession_frames=args.min_possession_frames,
        max_gap_frames=args.max_gap_frames,
        team_link_max_distance=args.team_link_max_distance,
        team_link_max_gap=args.team_link_max_gap,
        ball_max_interp_gap=args.ball_max_interp_gap,
        control_smoothing_window=args.control_smoothing_window,
        control_smoothing_min_votes=args.control_smoothing_min_votes,
    )


__all__ = ["render_pass_counter_video", "main"]


if __name__ == "__main__":
    main()
