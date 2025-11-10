"""Pipeline integral para obtener estadísticas de un video completo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

from .match_steps import (
    DEFAULT_OUTPUT_ROOT,
    run_clustering_stage,
    run_inference_stage,
    run_stats_stage,
)


class ProgressTracker:
    """Simple helper to report overall pipeline progress as weighted percentages."""

    def __init__(self, steps: Sequence[Tuple[str, float]]) -> None:
        if not steps:
            raise ValueError("steps no puede estar vacío")
        self.weights = {name: float(weight) for name, weight in steps}
        self.total = sum(self.weights.values())
        if self.total <= 0.0:
            raise ValueError("La suma de los pesos debe ser positiva")
        self.completed = 0.0

    def advance(self, step_name: str, message: str) -> None:
        weight = self.weights.get(step_name, 0.0)
        self.completed += weight
        percent = min(100.0, (self.completed / self.total) * 100.0)
        print(f"[{percent:5.1f}%] {message}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta inferencia, clustering, pases y tiros sobre un video completo."
    )
    parser.add_argument("--video", required=True, type=Path, help="Ruta al video a procesar.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ModeloRF.pth"),
        help="Checkpoint del detector RF-DETR.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directorio base donde se guardarán los resultados (uno por video).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral de confianza para conservar detecciones.",
    )
    parser.add_argument(
        "--team-labels",
        nargs=2,
        metavar=("TEAM_A", "TEAM_B"),
        default=None,
        help="Etiqueta final para cada equipo (ej: BRASIL COLOMBIA).",
    )
    parser.add_argument(
        "--cluster-random-state",
        type=int,
        default=0,
        help="Semilla para el clustering de equipos.",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {args.model}")
    if not args.video.exists():
        raise FileNotFoundError(f"No se encontró el video: {args.video}")

    tracker = ProgressTracker(
        [
            ("inference", 0.45),
            ("clustering", 0.2),
            ("passes", 0.2),
            ("shots", 0.1),
            ("summary", 0.05),
        ]
    )
    print("[  0.0%] Pipeline inicializado.")

    output_root = args.output_dir

    # 1. Inferencia
    detections_csv = run_inference_stage(
        args.video,
        args.model,
        threshold=args.threshold,
        output_root=output_root,
    )
    tracker.advance("inference", f"Inferencia completada: {detections_csv}")

    # 2. Clustering
    clustering_paths = run_clustering_stage(
        args.video,
        detections_csv,
        output_root=output_root,
        team_labels=args.team_labels,
        random_state=args.cluster_random_state,
    )
    tracker.advance("clustering", f"Clustering completado: {clustering_paths['detecciones']}")

    # 3. Estadísticas (pases, posesión, tiros y resumen)
    stats_paths = run_stats_stage(
        args.video,
        clustering_paths["detecciones"],
        output_root=output_root,
        pass_distance_threshold=args.pass_distance_threshold,
        pass_min_possession=args.pass_min_possession,
        pass_max_gap=args.pass_max_gap,
        ball_max_interp=args.ball_max_interp,
        shot_min_duration=args.shot_min_duration,
    )
    tracker.advance("passes", f"Pases/posesión generados: {stats_paths['passes']}")
    tracker.advance("shots", f"Tiros detectados: {stats_paths['shots']}")
    tracker.advance("summary", f"Resumen final guardado en: {stats_paths['summary']}")

    print(f"Pipeline completado. Resultados en: {stats_paths['summary'].parent}")


if __name__ == "__main__":
    main()
