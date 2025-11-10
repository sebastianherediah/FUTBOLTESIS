"""CLI para ejecutar únicamente la etapa de estadísticas (pases, posesión y tiros)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .match_steps import (
    CLUSTER_STAGE,
    DEFAULT_OUTPUT_ROOT,
    ensure_video_dir,
    run_stats_stage,
    stage_dir,
)


def _default_detections_path(video: Path, root: Path) -> Path:
    video_dir = ensure_video_dir(root, video)
    return stage_dir(video_dir, CLUSTER_STAGE) / "detecciones_con_equipos.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcula posesión, pases y tiros a partir de las detecciones etiquetadas.")
    parser.add_argument("--video", required=True, type=Path, help="Ruta al video original.")
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="CSV con las detecciones que incluyen la columna Team (por defecto 2_clustering/detecciones_con_equipos.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directorio donde se encuentra la carpeta del video.",
    )
    parser.add_argument(
        "--pass-distance-threshold",
        type=float,
        default=110.0,
        help="Distancia máxima balón-jugador para considerar posesión.",
    )
    parser.add_argument(
        "--pass-min-possession",
        type=int,
        default=1,
        help="Frames mínimos de posesión para consolidar un segmento.",
    )
    parser.add_argument(
        "--pass-max-gap",
        type=int,
        default=18,
        help="Hueco máximo entre posesiones para generar un pase.",
    )
    parser.add_argument(
        "--ball-max-interp",
        type=int,
        default=24,
        help="Frames máximos a interpolar cuando el balón desaparece.",
    )
    parser.add_argument(
        "--shot-min-duration",
        type=int,
        default=1,
        help="Frames mínimos consecutivos para validar un tiro.",
    )
    parser.add_argument(
        "--shot-goal-distance",
        type=float,
        default=120.0,
        help="Distancia máxima (en píxeles) entre el balón y el arco para considerar un tiro.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="resumen_equipos.csv",
        help="Nombre del CSV consolidado de estadísticas por equipo.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"No se encontró el video: {args.video}")

    detections_csv = args.detections
    if detections_csv is None:
        detections_csv = _default_detections_path(args.video, args.output_root)
    if not detections_csv.exists():
        raise FileNotFoundError(f"No se encontró el CSV con equipos: {detections_csv}")

    results = run_stats_stage(
        args.video,
        detections_csv,
        output_root=args.output_root,
        pass_distance_threshold=args.pass_distance_threshold,
        pass_min_possession=args.pass_min_possession,
        pass_max_gap=args.pass_max_gap,
        ball_max_interp=args.ball_max_interp,
        shot_min_duration=args.shot_min_duration,
        shot_goal_distance=args.shot_goal_distance,
        summary_name=args.summary_name,
    )
    print(f"Pases guardados en: {results['passes']}")
    print(f"Resumen guardado en: {results['summary']}")


if __name__ == "__main__":
    main()
