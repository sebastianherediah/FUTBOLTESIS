"""Pipeline integral para obtener estadísticas de un video completo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..analytics import PassAnalyzer
from ..clustering import Cluster
from ..inference.detect_shots_simple import detect_shots
from ..inference.video_inference import VideoInference


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
        default=Path("outputs") / "match_stats",
        help="Directorio base donde se guardarán los resultados.",
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


def _ensure_output_dir(base_dir: Path, video_path: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    video_stem = video_path.stem
    run_dir = base_dir / video_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _map_cluster_labels(assignments: pd.DataFrame, team_labels: Optional[Sequence[str]]) -> Dict[str, str]:
    unique = sorted(dict.fromkeys(assignments["Team"]))
    if team_labels:
        if len(team_labels) != len(unique):
            raise ValueError(
                f"Se esperaban {len(unique)} etiquetas para los equipos pero se recibieron {len(team_labels)}."
            )
        mapping = {label: str(team_labels[idx]).upper() for idx, label in enumerate(unique)}
    else:
        mapping = {label: label.upper() for label in unique}
    return mapping


def _format_bbox_list(box: Iterable[float]) -> str:
    return json.dumps([float(v) for v in box])


def _summarize_metrics(
    passes: pd.DataFrame,
    possession: pd.DataFrame,
    shots: pd.DataFrame,
) -> pd.DataFrame:
    pass_counts = (
        passes["Team"].astype(str).str.upper().value_counts().rename_axis("Team").reset_index(name="Passes")
        if not passes.empty
        else pd.DataFrame(columns=["Team", "Passes"])
    )
    possession_df = possession.copy()
    if "Team" in possession_df.columns:
        possession_df["Team"] = possession_df["Team"].astype(str).str.upper()
    shots_counts = (
        shots["AttackingTeam"].astype(str).str.upper().value_counts().rename_axis("Team").reset_index(name="Shots")
        if not shots.empty
        else pd.DataFrame(columns=["Team", "Shots"])
    )

    summary = pd.merge(pass_counts, possession_df, on="Team", how="outer")
    summary = pd.merge(summary, shots_counts, on="Team", how="outer")

    summary["Passes"] = summary["Passes"].fillna(0).astype(int)
    summary["Shots"] = summary["Shots"].fillna(0).astype(int)
    if "Possession" in summary.columns:
        summary["Possession"] = summary["Possession"].fillna(0.0)
    else:
        summary["Possession"] = 0.0

    return summary.sort_values("Team").reset_index(drop=True)


def main() -> None:
    args = _parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {args.model}")
    if not args.video.exists():
        raise FileNotFoundError(f"No se encontró el video: {args.video}")

    output_dir = _ensure_output_dir(args.output_dir, args.video)

    # 1. Inferencia
    inference = VideoInference(str(args.model))
    detections = inference.process(str(args.video), threshold=args.threshold)

    raw_path = output_dir / "detecciones_raw.csv"
    detections_to_save = detections.copy()
    detections_to_save["BBox"] = detections_to_save["BBox"].apply(_format_bbox_list)
    detections_to_save.to_csv(raw_path, index=False)

    # 2. Clustering de equipos
    cluster = Cluster(random_state=args.cluster_random_state)
    assignments = cluster.cluster_players(detections[["Frame", "ClassName", "BBox"]], str(args.video))
    mapping = _map_cluster_labels(assignments, args.team_labels)

    assignments["Team"] = assignments["Team"].map(mapping)
    team_lookup: Dict[Tuple[int, str], str] = {
        (int(row.Frame), json.dumps([float(v) for v in row.BBox])): str(row.Team)
        for row in assignments.itertuples(index=False)
    }

    def _assign_team(row: pd.Series) -> Optional[str]:
        key = json.dumps([float(v) for v in row["BBox"]])
        return team_lookup.get((int(row["Frame"]), key))

    detections["Team"] = detections.apply(_assign_team, axis=1)

    detections_with_team = detections.copy()
    detections_with_team["BBox"] = detections_with_team["BBox"].apply(_format_bbox_list)
    detections_with_team.to_csv(output_dir / "detecciones_con_equipos.csv", index=False)

    # 3. Análisis de pases
    pass_analyzer = PassAnalyzer(
        distance_threshold=args.pass_distance_threshold,
        min_possession_frames=args.pass_min_possession,
        max_gap_frames=args.pass_max_gap,
        ball_max_interp_gap=args.ball_max_interp,
    )
    pass_result = pass_analyzer.analyze(detections)

    pass_result.passes.to_csv(output_dir / "pases_detectados.csv", index=False)
    pass_result.possession.to_csv(output_dir / "posesion_estimada.csv", index=False)
    pass_result.control_timeline.to_csv(output_dir / "control_timeline.csv", index=False)

    # 4. Tiros (heurística simple)
    shots = detect_shots(
        detections,
        ball_class="balon",
        goal_class="arco",
        player_class="jugador",
        min_duration=args.shot_min_duration,
    )
    shots.to_csv(output_dir / "tiros_detectados.csv", index=False)

    # 5. Resumen
    summary = _summarize_metrics(pass_result.passes, pass_result.possession, shots)
    summary.to_csv(output_dir / "resumen_equipos.csv", index=False)

    report = {
        "video": str(args.video),
        "model": str(args.model),
        "frames": int(pass_result.control_timeline["Frame"].max() + 1)
        if not pass_result.control_timeline.empty
        else 0,
        "teams": summary["Team"].tolist(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Pipeline completado. Resultados en: {output_dir}")


if __name__ == "__main__":
    main()
