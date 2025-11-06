"""Genera un video que muestra el contador acumulado de tiros por equipo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..analytics import ShotCountTimeline, build_shot_count_timeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Renderiza un contador acumulado de tiros sobre un video.")
    parser.add_argument("--video", required=True, type=Path, help="Ruta al video original.")
    parser.add_argument(
        "--shots",
        required=True,
        type=Path,
        help="CSV con los tiros detectados (columnas StartFrame, EndFrame, Team).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Ruta del video de salida con el contador de tiros.",
    )
    parser.add_argument(
        "--team-order",
        nargs="+",
        default=None,
        help="Orden deseado de los equipos al mostrar el contador (ej: COLOMBIA BRASIL).",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.9,
        help="Escala del texto para el contador.",
    )
    parser.add_argument("--thickness", type=int, default=2, help="Grosor de la tipografía.")
    parser.add_argument("--padding", type=int, default=20, help="Margen mínimo desde el borde de la imagen.")
    parser.add_argument("--line-spacing", type=int, default=8, help="Espacio adicional entre líneas.")
    parser.add_argument("--box-alpha", type=float, default=0.6, help="Opacidad del recuadro (0-1).")
    parser.add_argument(
        "--font-color",
        type=str,
        default="255,255,255",
        help="Color del texto en formato R,G,B.",
    )
    parser.add_argument(
        "--box-color",
        type=str,
        default="15,15,15",
        help="Color del recuadro en formato R,G,B.",
    )
    return parser.parse_args()


def _parse_color(value: str) -> Tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"Color inválido: {value}. Usa el formato R,G,B.")
    try:
        r, g, b = (int(p.strip()) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Color inválido: {value}. Usa números enteros.") from exc
    return (r, g, b)


def _load_shots(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_base = {"StartFrame", "EndFrame"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(f"El CSV de tiros debe contener las columnas: {missing_base}")

    team_column = None
    for candidate in ("Team", "AttackingTeam"):
        if candidate in df.columns:
            team_column = candidate
            break
    if team_column is None:
        raise ValueError("El CSV de tiros debe incluir la columna 'Team' o 'AttackingTeam'.")

    if team_column != "Team":
        df = df.rename(columns={team_column: "Team"})

    return df


def _draw_counter(
    frame: np.ndarray,
    lines: Sequence[str],
    *,
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

    metrics: List[Tuple[int, int, int]] = []
    max_width = 0
    total_height = 0

    for text in lines:
        (width, height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        total_height += height + baseline
        max_width = max(max_width, width)
        metrics.append((width, height, baseline))

    total_height += (len(lines) - 1) * line_gap + box_padding * 2
    box_width = max_width + box_padding * 2

    x_left = padding
    y_bottom = frame.shape[0] - padding
    y_top = max(y_bottom - total_height, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_left, y_top), (x_left + box_width, y_bottom), box_color, -1)
    cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)

    y_cursor = y_top + box_padding
    for idx, (text, (_, height, baseline)) in enumerate(zip(lines, metrics)):
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


def _render_video(
    video_path: Path,
    timeline: ShotCountTimeline,
    output_path: Path,
    *,
    team_order: Optional[Sequence[str]],
    font_scale: float,
    thickness: int,
    padding: int,
    line_spacing: int,
    box_alpha: float,
    font_color: Tuple[int, int, int],
    box_color: Tuple[int, int, int],
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo inicializar el escritor de video en {output_path}")

    ordered_teams = list(team_order) if team_order else list(timeline.teams)
    if not ordered_teams:
        ordered_teams = list(timeline.teams)

    try:
        with tqdm(total=total_frames, desc="Generando contador de tiros") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                counts = timeline.counts_for_frame(frame_idx)
                lines = [f"{team}: {counts.get(team, 0)}" for team in ordered_teams] if ordered_teams else []
                _draw_counter(
                    frame,
                    lines,
                    font_scale=font_scale,
                    thickness=thickness,
                    padding=padding,
                    line_spacing=line_spacing,
                    box_alpha=box_alpha,
                    font_color=font_color,
                    box_color=box_color,
                )

                writer.write(frame)
                pbar.update(1)
    finally:
        cap.release()
        writer.release()

    return output_path


def main() -> None:
    args = _parse_args()

    shots_df = _load_shots(args.shots)

    probe = cv2.VideoCapture(str(args.video))
    if not probe.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {args.video}")
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    timeline = build_shot_count_timeline(
        shots_df,
        total_frames=total_frames,
        team_order=args.team_order,
    )

    font_color = _parse_color(args.font_color)
    box_color = _parse_color(args.box_color)

    output = _render_video(
        args.video,
        timeline,
        args.output,
        team_order=args.team_order,
        font_scale=args.font_scale,
        thickness=args.thickness,
        padding=args.padding,
        line_spacing=args.line_spacing,
        box_alpha=args.box_alpha,
        font_color=font_color,
        box_color=box_color,
    )
    print(f"Video con contador de tiros guardado en: {output}")


if __name__ == "__main__":
    main()
