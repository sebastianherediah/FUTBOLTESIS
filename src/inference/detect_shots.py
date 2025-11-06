"""CLI heurístico para detectar tiros en un partido utilizando homografía y dinámica del balón."""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..analytics import PassAnalyzer, ShotAnalyzer
from ..homography.config import IMAGE_HEIGHT, IMAGE_WIDTH
from ..homography.field_layout import DEFAULT_FIELD_LAYOUT
from ..homography.homography_estimator import HomographyEstimator
from ..homography.predictor import HomographyKeypointPredictor, Keypoint


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detecta intentos de tiro empleando homografía y la trayectoria del balón.")
    parser.add_argument("--video", type=Path, default=Path("VideoPruebaTesis.mp4"))
    parser.add_argument("--detections", type=Path, default=Path("outputs/detecciones_con_equipos.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/Homografia.pth"))
    parser.add_argument("--output-shots", type=Path, default=Path("outputs/tiros_detectados.csv"))
    parser.add_argument("--output-trajectory", type=Path, default=Path("outputs/ball_trayectoria_campo.csv"))

    parser.add_argument("--distance-threshold", type=float, default=110.0)
    parser.add_argument("--min-possession-frames", type=int, default=1)
    parser.add_argument("--max-gap-frames", type=int, default=18)
    parser.add_argument("--team-link-max-distance", type=float, default=160.0)
    parser.add_argument("--team-link-max-gap", type=int, default=20)
    parser.add_argument("--ball-max-interp-gap", type=int, default=24)
    parser.add_argument("--control-smoothing-window", type=int, default=3)
    parser.add_argument("--control-smoothing-min-votes", type=int, default=1)

    parser.add_argument("--shot-speed-threshold", type=float, default=12.0, help="Velocidad mínima (m/s) para considerar un tiro.")
    parser.add_argument("--shot-direction-threshold", type=float, default=2.0, help="Velocidad mínima en el eje X coherente con la dirección de la portería.")
    parser.add_argument("--shot-min-frames", type=int, default=3, help="Frames consecutivos para consolidar el tiro.")
    parser.add_argument("--shot-cooldown", type=int, default=8, help="Frames de tolerancia para finalizar un evento de tiro.")
    parser.add_argument("--shot-penalty-margin", type=float, default=3.0, help="Margen alrededor del área penal (en metros).")
    parser.add_argument("--shot-gap-tolerance", type=int, default=6, help="Máximo de frames sin balón para considerar continuidad.")

    parser.add_argument("--clip-output", type=Path, default=None, help="Directorio donde guardar clips .mp4 de cada tiro detectado.")
    parser.add_argument("--clip-window", type=float, default=1.5, help="Segundos antes y después del frame del tiro para cada clip.")
    parser.add_argument("--confidence", type=float, default=0.05, help="Confianza mínima para considerar un keypoint de homografía.")
    parser.add_argument("--min-matches", type=int, default=6, help="Keypoints mínimos para estimar la homografía.")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo para el modelo de homografía (cpu/cuda).")
    parser.add_argument("--max-frames", type=int, default=None, help="Limita el número de frames procesados (útil para pruebas).")
    return parser.parse_args(argv)


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


def _collect_ball_detections(detections: pd.DataFrame) -> Dict[int, Tuple[float, float, float, float]]:
    ball_rows = detections[detections["ClassName"].str.lower() == "balon"].copy()
    if ball_rows.empty:
        return {}
    ball_rows = ball_rows.sort_values("Frame")

    parsed = []
    for _, row in ball_rows.iterrows():
        bbox = _parse_bbox(row["BBox"])
        if bbox is None:
            continue
        parsed.append((int(row["Frame"]), bbox))

    return {frame: bbox for frame, bbox in parsed}


def _project_ball_positions(
    video_path: Path,
    *,
    predictor: HomographyKeypointPredictor,
    estimator: HomographyEstimator,
    ball_detections: Dict[int, Tuple[float, float, float, float]],
    confidence_threshold: float,
    min_matches: int,
    max_frames: Optional[int] = None,
) -> Tuple[pd.DataFrame, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total_frames, max_frames) if max_frames is not None else total_frames

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_x = IMAGE_WIDTH / max(original_width, 1)
    scale_y = IMAGE_HEIGHT / max(original_height, 1)

    progress = tqdm(total=limit, desc="Proyectando balón", unit="frame")
    records: List[Dict[str, float]] = []
    last_h: Optional[np.ndarray] = None
    frame_idx = 0

    try:
        while frame_idx < limit:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, _ = predictor.predict(frame, confidence_threshold=confidence_threshold)
            keypoint_dict = _keypoints_to_dict(keypoints)

            try:
                homography = estimator.estimate(keypoint_dict, min_matches=min_matches)
                last_h = homography.matrix
            except RuntimeError:
                pass  # mantener la última homografía válida

            record = {"Frame": frame_idx, "FieldX": np.nan, "FieldY": np.nan}

            bbox = ball_detections.get(frame_idx)
            if bbox and last_h is not None:
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2.0) * scale_x
                cy = ((y1 + y2) / 2.0) * scale_y
                projected = estimator.project_points(last_h, [(cx, cy)])
                if projected.size == 2:
                    record["FieldX"] = float(projected[0, 0])
                    record["FieldY"] = float(projected[0, 1])

            records.append(record)
            frame_idx += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()

    return pd.DataFrame(records), float(fps)


def _generate_clips(
    video_path: Path,
    shots_df: pd.DataFrame,
    output_dir: Path,
    *,
    fps: float,
    window: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for event in shots_df.itertuples(index=False):
        frame_center = getattr(event, "start_frame")
        center_time = frame_center / fps
        start_time = max(center_time - window, 0.0)
        duration = window * 2.0
        team = getattr(event, "team", None) or "UNKNOWN"
        side = getattr(event, "side", "NA")
        filename = f"shot_{int(frame_center):05d}_{team}_{side}.mp4"
        output_path = output_dir / filename

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
        subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    device = args.device.lower() if isinstance(args.device, str) else None
    if device not in (None, "cpu", "cuda"):
        raise ValueError("El parámetro --device debe ser 'cpu' o 'cuda'.")

    detections = pd.read_csv(args.detections)
    if "Frame" not in detections.columns or "ClassName" not in detections.columns or "BBox" not in detections.columns:
        raise ValueError("El CSV de detecciones requiere las columnas Frame, ClassName y BBox.")

    pass_analyzer = PassAnalyzer(
        distance_threshold=args.distance_threshold,
        min_possession_frames=args.min_possession_frames,
        max_gap_frames=args.max_gap_frames,
        team_link_max_distance=args.team_link_max_distance,
        team_link_max_gap=args.team_link_max_gap,
        ball_max_interp_gap=args.ball_max_interp_gap,
        control_smoothing_window=args.control_smoothing_window,
        control_smoothing_min_votes=args.control_smoothing_min_votes,
    )
    pass_result = pass_analyzer.analyze(detections)

    predictor = HomographyKeypointPredictor(args.checkpoint, device=device)
    estimator = HomographyEstimator(DEFAULT_FIELD_LAYOUT)

    ball_detections = _collect_ball_detections(detections)
    ball_positions, fps = _project_ball_positions(
        args.video,
        predictor=predictor,
        estimator=estimator,
        ball_detections=ball_detections,
        confidence_threshold=args.confidence,
        min_matches=args.min_matches,
        max_frames=args.max_frames,
    )

    shot_analyzer = ShotAnalyzer(
        fps=fps,
        speed_threshold=args.shot_speed_threshold,
        direction_threshold=args.shot_direction_threshold,
        min_event_frames=args.shot_min_frames,
        cooldown_frames=args.shot_cooldown,
        penalty_margin=args.shot_penalty_margin,
        ball_gap_tolerance=args.shot_gap_tolerance,
    )
    shot_result = shot_analyzer.analyze(ball_positions, pass_result.control_timeline)

    args.output_shots.parent.mkdir(parents=True, exist_ok=True)
    args.output_trajectory.parent.mkdir(parents=True, exist_ok=True)
    shot_result.shots.to_csv(args.output_shots, index=False)
    shot_result.ball_metrics.to_csv(args.output_trajectory, index=False)

    print(f"Tiros detectados: {len(shot_result.shots)} (exportados en {args.output_shots})")
    print(f"Trayectoria del balón exportada en: {args.output_trajectory}")

    if args.clip_output and len(shot_result.shots) > 0:
        _generate_clips(
            args.video,
            shot_result.shots,
            args.clip_output,
            fps=fps,
            window=args.clip_window,
        )
        print(f"Clips generados en: {args.clip_output}")


if __name__ == "__main__":
    main()
