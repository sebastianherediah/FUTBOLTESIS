"""Shared helpers to run the FUTBOLTESIS pipeline in explicit stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..analytics import PassAnalyzer
from ..clustering import Cluster
from ..inference.detect_shots_simple import detect_shots

INFERENCE_STAGE = "1_inference"
CLUSTER_STAGE = "2_clustering"
STATS_STAGE = "3_stats"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "match_runs"


def ensure_video_dir(base_dir: Path, video_path: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    video_dir = base_dir / video_path.stem
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


def stage_dir(video_dir: Path, stage_name: str) -> Path:
    path = video_dir / stage_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_bbox_list(box: Iterable[float]) -> str:
    return json.dumps([float(v) for v in box])


def parse_bbox(raw: object) -> List[float]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"No se pudo parsear la bbox: {raw}") from None
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    raise TypeError(f"Formato de bbox no soportado: {type(raw)!r}")


def load_detections(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "BBox" not in df.columns:
        raise ValueError("El archivo de detecciones debe contener la columna 'BBox'.")
    df["BBox"] = df["BBox"].apply(parse_bbox)
    return df


def load_dataframe_with_bbox(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "BBox" in df.columns:
        df["BBox"] = df["BBox"].apply(parse_bbox)
    return df


def map_cluster_labels(assignments: pd.DataFrame, team_labels: Optional[Sequence[str]]) -> Dict[str, str]:
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


def summarize_metrics(
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
    if not possession_df.empty:
        if "Team" not in possession_df.columns:
            possession_df["Team"] = ""
        possession_df["Team"] = possession_df["Team"].astype(str).str.upper()
        keep_cols = [col for col in ("Team", "Possession") if col in possession_df.columns]
        possession_df = possession_df.loc[:, keep_cols]
    shots_counts = (
        shots["AttackingTeam"].astype(str).str.upper().value_counts().rename_axis("Team").reset_index(name="Shots")
        if not shots.empty
        else pd.DataFrame(columns=["Team", "Shots"])
    )

    summary = pd.merge(pass_counts, possession_df, on="Team", how="outer")
    summary = pd.merge(summary, shots_counts, on="Team", how="outer")

    for col in [c for c in summary.columns if c.startswith("Passes_")]:
        summary.drop(columns=[col], inplace=True)
    for col in [c for c in summary.columns if c.startswith("Shots_")]:
        summary.drop(columns=[col], inplace=True)

    summary["Passes"] = summary.get("Passes", 0).fillna(0).astype(int)
    summary["Shots"] = summary.get("Shots", 0).fillna(0).astype(int)
    summary["Possession"] = summary.get("Possession", 0.0).fillna(0.0)

    return summary.sort_values("Team").reset_index(drop=True)


def run_inference_stage(
    video: Path,
    model: Path,
    *,
    threshold: float,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    raw_name: str = "detecciones_raw.csv",
) -> Path:
    from ..inference.video_inference import VideoInference
    video_dir = ensure_video_dir(output_root, video)
    inference_dir = stage_dir(video_dir, INFERENCE_STAGE)

    inference = VideoInference(str(model))
    detections = inference.process(str(video), threshold=threshold)

    formatted = detections.copy()
    formatted["BBox"] = formatted["BBox"].apply(format_bbox_list)
    csv_path = inference_dir / raw_name
    formatted.to_csv(csv_path, index=False)
    return csv_path


def run_clustering_stage(
    video: Path,
    detections_csv: Path,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    team_labels: Optional[Sequence[str]] = None,
    random_state: int = 0,
    assignments_name: str = "assignments.csv",
    labeled_name: str = "detecciones_con_equipos.csv",
    show_progress: bool = False,
    batch_size: int = 32,
    enable_faulthandler: bool = False,
    segments: int = 1,
) -> Dict[str, Path]:
    video_dir = ensure_video_dir(output_root, video)
    clustering_dir = stage_dir(video_dir, CLUSTER_STAGE)

    detections = load_detections(detections_csv)

    if enable_faulthandler:
        import faulthandler

        faulthandler.enable()

    cluster = Cluster(
        random_state=random_state,
        show_progress=show_progress,
        batch_size=batch_size,
    )

    total_segments = max(1, segments)
    frame_min = int(detections["Frame"].min())
    frame_max = int(detections["Frame"].max())
    total_frames = frame_max - frame_min + 1
    segment_size = max(1, (total_frames + total_segments - 1) // total_segments)

    ranges: List[Tuple[int, int]] = []
    for idx in range(total_segments):
        start = frame_min + idx * segment_size
        end = min(frame_min + (idx + 1) * segment_size, frame_max + 1)
        if start >= end:
            continue
        ranges.append((start, end))

    assignments_parts: List[pd.DataFrame] = []
    detections_parts: List[pd.DataFrame] = []
    segment_paths: List[Dict[str, Path]] = []

    for seg_idx, (start, end) in enumerate(ranges, start=1):
        segment_mask = (detections["Frame"] >= start) & (detections["Frame"] < end)
        segment_rows = detections.loc[segment_mask].copy()
        if segment_rows.empty:
            print(f"[Segmento {seg_idx}/{len(ranges)}] Sin detecciones en frames {start}-{end - 1}. Se omite.")
            continue
        segment_assign_path = clustering_dir / f"{labeled_name}_segment_{seg_idx}_of_{len(ranges)}_assignments.csv"
        segment_detect_path = clustering_dir / f"{labeled_name}_segment_{seg_idx}_of_{len(ranges)}.csv"

        if segment_assign_path.exists() and segment_detect_path.exists():
            print(
                f"[Segmento {seg_idx}/{len(ranges)}] Archivos existentes detectados "
                f"({segment_detect_path.name}). Se reutiliza la salida previa."
            )
            assignments_clean = load_dataframe_with_bbox(segment_assign_path)
            segment_rows = load_dataframe_with_bbox(segment_detect_path)
            assignments_parts.append(assignments_clean)
            detections_parts.append(segment_rows)
            segment_paths.append(
                {
                    "assignments": segment_assign_path,
                    "detecciones": segment_detect_path,
                    "index": seg_idx,
                }
            )
            continue

        print(
            f"[Segmento {seg_idx}/{len(ranges)}] Clustering frames {start}-{end - 1} "
            f"({segment_rows['Frame'].nunique()} frames con jugadores)."
        )
        cluster_result = cluster.cluster_players(segment_rows[["Frame", "ClassName", "BBox"]], str(video))
        assignments = cluster_result.assignments.copy()
        mapping = map_cluster_labels(assignments, team_labels)
        assignments["Team"] = assignments["Team"].map(mapping)

        assignments["BBoxKey"] = assignments["BBox"].apply(format_bbox_list)
        team_lookup: Dict[tuple[int, str], str] = {
            (int(row.Frame), str(row.BBoxKey)): str(row.Team)
            for row in assignments.itertuples(index=False)
        }

        segment_rows["Team"] = segment_rows.apply(
            lambda row: team_lookup.get((int(row["Frame"]), format_bbox_list(row["BBox"]))), axis=1
        )

        assignments_clean = assignments.drop(columns=["BBoxKey"])
        assignments_parts.append(assignments_clean)
        detections_parts.append(segment_rows)
        assignments_clean.to_csv(segment_assign_path, index=False)
        segment_to_save = segment_rows.copy()
        segment_to_save["BBox"] = segment_to_save["BBox"].apply(format_bbox_list)
        segment_to_save.to_csv(segment_detect_path, index=False)
        segment_paths.append(
            {
                "assignments": segment_assign_path,
                "detecciones": segment_detect_path,
                "index": seg_idx,
            }
        )

    if not assignments_parts or not detections_parts:
        raise RuntimeError(
            "No se generaron asignaciones. Verifica que el CSV contenga detecciones de jugadores."
        )

    assignments_df = pd.concat(assignments_parts, ignore_index=True).sort_values("Frame").reset_index(drop=True)
    detections_df = pd.concat(detections_parts, ignore_index=True).sort_values("Frame").reset_index(drop=True)

    assignments_path = clustering_dir / assignments_name
    assignments_df.to_csv(assignments_path, index=False)

    detections_to_save = detections_df.copy()
    detections_to_save["BBox"] = detections_to_save["BBox"].apply(format_bbox_list)
    detections_path = clustering_dir / labeled_name
    detections_to_save.to_csv(detections_path, index=False)

    final_result = {"assignments": assignments_path, "detecciones": detections_path}
    for entry in segment_paths:
        final_result[f"segment_{entry['index']}_assignments"] = entry["assignments"]
        final_result[f"segment_{entry['index']}_detecciones"] = entry["detecciones"]

    return final_result


def run_stats_stage(
    video: Path,
    detections_with_team_csv: Path,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    pass_distance_threshold: float = 110.0,
    pass_min_possession: int = 1,
    pass_max_gap: int = 18,
    ball_max_interp: int = 24,
    shot_min_duration: int = 1,
    shot_goal_distance: float = 120.0,
    summary_name: str = "resumen_equipos.csv",
) -> Dict[str, Path]:
    video_dir = ensure_video_dir(output_root, video)
    stats_dir = stage_dir(video_dir, STATS_STAGE)

    detections = load_detections(detections_with_team_csv)

    pass_analyzer = PassAnalyzer(
        distance_threshold=pass_distance_threshold,
        min_possession_frames=pass_min_possession,
        max_gap_frames=pass_max_gap,
        ball_max_interp_gap=ball_max_interp,
    )
    pass_result = pass_analyzer.analyze(detections)

    passes_path = stats_dir / "pases_detectados.csv"
    possession_path = stats_dir / "posesion_estimada.csv"
    control_path = stats_dir / "control_timeline.csv"

    pass_result.passes.to_csv(passes_path, index=False)
    pass_result.possession.to_csv(possession_path, index=False)
    pass_result.control_timeline.to_csv(control_path, index=False)

    shots = detect_shots(
        detections,
        ball_class="balon",
        goal_class="arco",
        player_class="jugador",
        min_duration=shot_min_duration,
        goal_distance_threshold=shot_goal_distance,
    )
    shots_path = stats_dir / "tiros_detectados.csv"
    shots.to_csv(shots_path, index=False)

    summary = summarize_metrics(pass_result.passes, pass_result.possession, shots)
    summary_path = stats_dir / summary_name
    summary.to_csv(summary_path, index=False)

    frames_total = (
        int(pass_result.control_timeline["Frame"].max() + 1)
        if not pass_result.control_timeline.empty
        else 0
    )
    metadata = {
        "video": str(video),
        "frames": frames_total,
        "teams": summary["Team"].tolist(),
        "detections_csv": str(detections_with_team_csv),
    }
    metadata_path = video_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "passes": passes_path,
        "possession": possession_path,
        "control": control_path,
        "shots": shots_path,
        "summary": summary_path,
        "metadata": metadata_path,
    }
