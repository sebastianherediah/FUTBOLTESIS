"""CLI para ejecutar únicamente la etapa de clustering de equipos."""

from __future__ import annotations

import argparse
from pathlib import Path
from .match_steps import DEFAULT_OUTPUT_ROOT, ensure_video_dir, run_clustering_stage, stage_dir, INFERENCE_STAGE


def _default_detections_path(video: Path, root: Path) -> Path:
    video_dir = ensure_video_dir(root, video)
    return stage_dir(video_dir, INFERENCE_STAGE) / "detecciones_raw.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aplica clustering de uniformes para asignar equipos.")
    parser.add_argument("--video", required=True, type=Path, help="Ruta al video original.")
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="CSV con las detecciones base (por defecto se busca en 1_inference/detecciones_raw.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directorio donde se encuentra la carpeta del video.",
    )
    parser.add_argument(
        "--team-labels",
        nargs=2,
        metavar=("TEAM_A", "TEAM_B"),
        default=None,
        help="Etiqueta final para cada cluster (ej: BRASIL COLOMBIA).",
    )
    parser.add_argument(
        "--cluster-random-state",
        type=int,
        default=0,
        help="Semilla utilizada por K-Means.",
    )
    parser.add_argument(
        "--labeled-name",
        type=str,
        default="detecciones_con_equipos.csv",
        help="Nombre del CSV con equipos dentro de la carpeta 2_clustering.",
    )
    parser.add_argument(
        "--assignments-name",
        type=str,
        default="assignments.csv",
        help="Nombre del CSV con los resultados del clustering.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Muestra una barra de progreso con uso básico de memoria durante el clustering.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Número de recortes procesados de forma simultánea (reduce este valor si tu RAM es limitada).",
    )
    parser.add_argument(
        "--enable-faulthandler",
        action="store_true",
        help="Activa faulthandler para imprimir la pila de Python si ocurre un segfault.",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=1,
        help="Divide el clustering en N segmentos de frames (por ejemplo, 3 para procesar el video en tercios).",
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
        raise FileNotFoundError(f"No se encontró el CSV de detecciones: {detections_csv}")

    results = run_clustering_stage(
        args.video,
        detections_csv,
        output_root=args.output_root,
        team_labels=args.team_labels,
        random_state=args.cluster_random_state,
        labeled_name=args.labeled_name,
        assignments_name=args.assignments_name,
        show_progress=args.show_progress,
        batch_size=max(1, args.batch_size),
        enable_faulthandler=args.enable_faulthandler,
        segments=max(1, args.segments),
    )
    print(f"Detecciones con equipos guardadas en: {results['detecciones']}")
    print(f"Asignaciones de clustering guardadas en: {results['assignments']}")
    for key, value in sorted(results.items()):
        if key.startswith("segment_") and value:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
