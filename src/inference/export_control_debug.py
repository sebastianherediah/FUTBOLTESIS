"""Herramientas para depurar la línea de control del balón y extraer clips."""

from __future__ import annotations

import argparse
import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.analytics import PassAnalyzer  # type: ignore
else:
    from ..analytics import PassAnalyzer


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


@dataclass
class ControlTransition:
    from_frame: int
    to_frame: int
    from_player: int
    to_player: int
    from_team: str
    to_team: str
    frame_gap: int
    missing_frames: int
    from_distance: float
    to_distance: float


def _compute_control_transitions(timeline: pd.DataFrame) -> pd.DataFrame:
    rows: List[ControlTransition] = []
    last_valid_frame: Optional[int] = None
    last_valid_player: Optional[int] = None
    last_valid_team: Optional[str] = None
    last_valid_distance: float = float("nan")

    for row in timeline.itertuples(index=False):
        player = getattr(row, "PlayerId")
        team = getattr(row, "Team")
        frame = int(getattr(row, "Frame"))
        distance = getattr(row, "Distance")

        if pd.isna(player) or pd.isna(team):
            continue

        player_id = int(player)
        team_str = str(team)
        distance_val = float(distance) if pd.notna(distance) else float("nan")

        if last_valid_player is not None and player_id != last_valid_player:
            gap = frame - last_valid_frame if last_valid_frame is not None else 0
            missing = max(gap - 1, 0)
            rows.append(
                ControlTransition(
                    from_frame=int(last_valid_frame),
                    to_frame=frame,
                    from_player=int(last_valid_player),
                    to_player=player_id,
                    from_team=str(last_valid_team),
                    to_team=team_str,
                    frame_gap=int(gap),
                    missing_frames=int(missing),
                    from_distance=float(last_valid_distance),
                    to_distance=distance_val,
                )
            )

        last_valid_frame = frame
        last_valid_player = player_id
        last_valid_team = team_str
        last_valid_distance = distance_val

    return pd.DataFrame(rows)


def _extract_clips(
    video_path: Path,
    frames: Sequence[int],
    *,
    fps: float,
    window_seconds: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        center_time = frame / fps
        start_time = max(center_time - window_seconds, 0.0)
        duration = window_seconds * 2.0
        output_path = output_dir / f"frame_{int(frame):05d}.mp4"

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"No se pudo generar el clip para el frame {frame}: {exc}") from exc


def _sorted_unique_frames(values: Iterable[int]) -> List[int]:
    return sorted({int(v) for v in values})


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta cambios de control sin filtrado y opcionalmente genera clips para depuración."
    )
    parser.add_argument("--detections", required=True, type=Path, help="CSV con detecciones enriquecidas.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "control_transitions_raw.csv",
        help="Ruta del CSV a exportar con los cambios de control detectados.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Ruta al video original. Si se proporciona se pueden generar clips por frame.",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=None,
        help="Lista de frames para extraer clips alrededor del evento.",
    )
    parser.add_argument(
        "--clip-window",
        type=float,
        default=1.0,
        help="Segundos antes y después del frame central al generar clips de depuración.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=110.0,
        help="Distancia máxima balón-jugador para asignar posesión.",
    )
    parser.add_argument(
        "--min-possession-frames",
        type=int,
        default=1,
        help="Frames consecutivos mínimos para considerar un segmento de posesión.",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=18,
        help="Separación máxima entre segmentos consecutivos para agruparlos en un pase.",
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
        help="Hueco máximo en frames para reutilizar IDs heurísticos.",
    )
    parser.add_argument(
        "--ball-max-interp-gap",
        type=int,
        default=24,
        help="Máximo de frames consecutivos a interpolar cuando el balón no se detecta.",
    )
    parser.add_argument(
        "--control-smoothing-window",
        type=int,
        default=3,
        help="Tamaño de la ventana de suavizado aplicada a la posesión.",
    )
    parser.add_argument(
        "--control-smoothing-min-votes",
        type=int,
        default=1,
        help="Número mínimo de votos dentro de la ventana para reemplazar la asignación.",
    )
    parser.add_argument(
        "--clips-output",
        type=Path,
        default=Path("outputs") / "control_debug_clips",
        help="Directorio donde se guardarán los clips generados.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    detections = _load_detections(args.detections)
    analyzer = PassAnalyzer(
        distance_threshold=args.distance_threshold,
        min_possession_frames=args.min_possession_frames,
        max_gap_frames=args.max_gap_frames,
        team_link_max_distance=args.team_link_max_distance,
        team_link_max_gap=args.team_link_max_gap,
        ball_max_interp_gap=args.ball_max_interp_gap,
        control_smoothing_window=args.control_smoothing_window,
        control_smoothing_min_votes=args.control_smoothing_min_votes,
    )

    analysis = analyzer.analyze(detections)
    transitions = _compute_control_transitions(analysis.control_timeline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(args.output, index=False)
    print(f"Transiciones exportadas en: {args.output} ({len(transitions)} filas)")

    if args.video and args.frames:
        frames = _sorted_unique_frames(args.frames)
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        _extract_clips(
            args.video,
            frames,
            fps=fps,
            window_seconds=args.clip_window,
            output_dir=args.clips_output,
        )
        print(f"Clips generados en: {args.clips_output}")


if __name__ == "__main__":
    main()
