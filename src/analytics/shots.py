"""Heurísticas para detectar intentos de tiro utilizando la homografía y la dinámica del balón."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..homography.field_layout import DEFAULT_FIELD_LAYOUT  # type: ignore


@dataclass(frozen=True)
class ShotEvent:
    """Representa un intento de tiro detectado."""

    StartFrame: int
    EndFrame: int
    Team: Optional[str]
    Side: str
    Frames: int
    MaxSpeed: float
    MeanSpeed: float


@dataclass
class ShotAnalysisResult:
    """Resultado del análisis de tiros."""

    shots: pd.DataFrame
    ball_metrics: pd.DataFrame


class ShotAnalyzer:
    """Detecta tiros aproximando la trayectoria del balón en el plano del campo."""

    def __init__(
        self,
        *,
        fps: float,
        speed_threshold: float = 12.0,
        direction_threshold: float = 2.0,
        min_event_frames: int = 3,
        cooldown_frames: int = 8,
        penalty_margin: float = 3.0,
        ball_gap_tolerance: int = 6,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("fps debe ser > 0")
        if speed_threshold <= 0.0:
            raise ValueError("speed_threshold debe ser > 0")
        if min_event_frames < 1:
            raise ValueError("min_event_frames debe ser >= 1")
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames debe ser >= 0")
        if penalty_margin < 0.0:
            raise ValueError("penalty_margin debe ser >= 0")
        if ball_gap_tolerance < 0:
            raise ValueError("ball_gap_tolerance debe ser >= 0")

        self.fps = float(fps)
        self.speed_threshold = float(speed_threshold)
        self.direction_threshold = float(direction_threshold)
        self.min_event_frames = int(min_event_frames)
        self.cooldown_frames = int(cooldown_frames)
        self.penalty_margin = float(penalty_margin)
        self.ball_gap_tolerance = int(ball_gap_tolerance)

        layout = DEFAULT_FIELD_LAYOUT
        self.left_penalty_bounds = self._penalty_bounds(prefix="L_", margin=self.penalty_margin, layout=layout)
        self.right_penalty_bounds = self._penalty_bounds(prefix="R_", margin=self.penalty_margin, layout=layout)
        self.left_goal_x = layout.keypoints["TL_PITCH_CORNER"][0]
        self.right_goal_x = layout.keypoints["TR_PITCH_CORNER"][0]

    @staticmethod
    def _penalty_bounds(prefix: str, *, margin: float, layout) -> Tuple[float, float, float, float]:
        tl = layout.keypoints[f"{prefix}PENALTY_AREA_TL_CORNER"]
        br = layout.keypoints[f"{prefix}PENALTY_AREA_BR_CORNER"]
        x_min = min(tl[0], br[0]) - margin
        x_max = max(tl[0], br[0]) + margin
        y_min = min(tl[1], br[1]) - margin
        y_max = max(tl[1], br[1]) + margin
        return x_min, x_max, y_min, y_max

    def analyze(self, ball_positions: pd.DataFrame, control_timeline: pd.DataFrame) -> ShotAnalysisResult:
        """Analiza las posiciones proyectadas del balón y devuelve los tiros detectados."""

        if ball_positions.empty:
            raise ValueError("ball_positions está vacío; se requiere al menos una posición del balón.")
        if "Frame" not in ball_positions.columns:
            raise ValueError("ball_positions requiere la columna 'Frame'.")
        if "FieldX" not in ball_positions.columns or "FieldY" not in ball_positions.columns:
            raise ValueError("ball_positions requiere las columnas 'FieldX' y 'FieldY'.")

        metrics = self._prepare_ball_metrics(ball_positions)
        metrics = metrics.merge(
            control_timeline[["Frame", "Team"]] if "Team" in control_timeline.columns else control_timeline,
            on="Frame",
            how="left",
        )
        metrics["Team"] = metrics["Team"].astype("string").where(metrics["Team"].notna(), None)

        metrics["Side"] = metrics.apply(self._frame_side, axis=1)
        metrics["ShotCandidate"] = self._is_shot_candidate(metrics)

        events = self._group_events(metrics)
        shots_df = (
            pd.DataFrame([vars(event) for event in events])
            if events
            else pd.DataFrame(columns=["StartFrame", "EndFrame", "Team", "Side", "Frames", "MaxSpeed", "MeanSpeed"])
        )

        return ShotAnalysisResult(shots=shots_df, ball_metrics=metrics)

    def _prepare_ball_metrics(self, positions: pd.DataFrame) -> pd.DataFrame:
        metrics = positions.sort_values("Frame").reset_index(drop=True).copy()

        metrics["FieldX"] = metrics["FieldX"].astype(float)
        metrics["FieldY"] = metrics["FieldY"].astype(float)

        metrics["PrevFrame"] = metrics["Frame"].shift(1)
        metrics["PrevX"] = metrics["FieldX"].shift(1)
        metrics["PrevY"] = metrics["FieldY"].shift(1)

        frame_delta = metrics["Frame"] - metrics["PrevFrame"]
        time_delta = frame_delta / self.fps

        dx = metrics["FieldX"] - metrics["PrevX"]
        dy = metrics["FieldY"] - metrics["PrevY"]

        time_delta = time_delta.where((frame_delta.notna()) & (frame_delta > 0), np.nan)
        valid_delta = time_delta.notna() & (time_delta > 0)

        metrics["Vx"] = np.where(valid_delta, dx / time_delta, np.nan)
        metrics["Vy"] = np.where(valid_delta, dy / time_delta, np.nan)
        metrics["Speed"] = np.where(valid_delta, np.sqrt(metrics["Vx"] ** 2 + metrics["Vy"] ** 2), np.nan)

        # Invalidar saltos largos donde no hay seguimiento fiable
        metrics.loc[frame_delta > self.ball_gap_tolerance, ["Vx", "Vy", "Speed"]] = np.nan

        metrics["InLeftPenalty"] = metrics.apply(
            lambda row: self._in_bounds(row["FieldX"], row["FieldY"], self.left_penalty_bounds), axis=1
        )
        metrics["InRightPenalty"] = metrics.apply(
            lambda row: self._in_bounds(row["FieldX"], row["FieldY"], self.right_penalty_bounds), axis=1
        )

        return metrics

    @staticmethod
    def _in_bounds(x: float, y: float, bounds: Tuple[float, float, float, float]) -> bool:
        if np.isnan(x) or np.isnan(y):
            return False
        x_min, x_max, y_min, y_max = bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _frame_side(self, row: pd.Series) -> Optional[str]:
        if row["InLeftPenalty"]:
            return "left"
        if row["InRightPenalty"]:
            return "right"
        return None

    def _is_shot_candidate(self, metrics: pd.DataFrame) -> pd.Series:
        speed_ok = metrics["Speed"] >= self.speed_threshold
        left_dir = (metrics["Vx"] <= -abs(self.direction_threshold)) & metrics["InLeftPenalty"]
        right_dir = (metrics["Vx"] >= abs(self.direction_threshold)) & metrics["InRightPenalty"]
        return speed_ok & (left_dir | right_dir)

    def _group_events(self, metrics: pd.DataFrame) -> List[ShotEvent]:
        events: List[ShotEvent] = []
        current: Optional[dict] = None
        cooldown = 0

        for row in metrics.itertuples(index=False):
            frame = int(row.Frame)
            if not getattr(row, "ShotCandidate"):
                if current is not None:
                    cooldown += 1
                    if cooldown > self.cooldown_frames:
                        events.append(self._finalize_event(current))
                        current = None
                        cooldown = 0
                continue

            side = getattr(row, "Side")
            team = getattr(row, "Team")
            if isinstance(team, str):
                team = team.strip() or None

            speed = float(getattr(row, "Speed", np.nan))

            if current is None:
                current = {
                    "start": frame,
                    "end": frame,
                    "team": team,
                    "side": side,
                    "frames": 1,
                    "speeds": [speed] if not np.isnan(speed) else [],
                }
                cooldown = 0
                continue

            if (
                frame - current["end"] <= self.ball_gap_tolerance
                and current["side"] == side
                and (current["team"] == team or current["team"] is None or team is None)
            ):
                current["end"] = frame
                current["frames"] += 1
                if not np.isnan(speed):
                    current["speeds"].append(speed)
                cooldown = 0
            else:
                if current["frames"] >= self.min_event_frames:
                    events.append(self._finalize_event(current))
                current = {
                    "start": frame,
                    "end": frame,
                    "team": team,
                    "side": side,
                    "frames": 1,
                    "speeds": [speed] if not np.isnan(speed) else [],
                }
                cooldown = 0

        if current is not None and current["frames"] >= self.min_event_frames:
            events.append(self._finalize_event(current))

        return events

    def _finalize_event(self, data: dict) -> ShotEvent:
        speeds = data.get("speeds") or []
        max_speed = float(np.max(speeds)) if speeds else float("nan")
        mean_speed = float(np.mean(speeds)) if speeds else float("nan")
        team = data.get("team")
        if isinstance(team, str):
            team = team.strip() or None
        return ShotEvent(
            StartFrame=int(data["start"]),
            EndFrame=int(data["end"]),
            Team=team,
            Side=str(data["side"]),
            Frames=int(data["frames"]),
            MaxSpeed=max_speed,
            MeanSpeed=mean_speed,
        )


@dataclass(frozen=True)
class ShotCountTimeline:
    """Contador acumulado de tiros por frame."""

    shots: pd.DataFrame
    counts: pd.DataFrame
    teams: Sequence[str]

    def counts_for_frame(self, frame: int) -> Dict[str, int]:
        if self.counts.empty:
            return {team: 0 for team in self.teams}
        frames = self.counts["Frame"].to_numpy()
        idx = int(np.searchsorted(frames, frame, side="right") - 1)
        if idx < 0:
            return {team: 0 for team in self.teams}
        row = self.counts.iloc[idx]
        return {team: int(row.get(team, 0)) for team in self.teams}


def build_shot_count_timeline(
    shots: pd.DataFrame,
    *,
    total_frames: Optional[int] = None,
    frame_range: Optional[Iterable[int]] = None,
    team_order: Optional[Sequence[str]] = None,
) -> ShotCountTimeline:
    """Construye un DataFrame con el conteo acumulado de tiros."""

    if shots.empty:
        teams = list(team_order) if team_order else []
        frames = _build_frame_range(total_frames, frame_range)
        counts = pd.DataFrame([{"Frame": frame, **{team: 0 for team in teams}} for frame in frames])
        return ShotCountTimeline(shots=shots, counts=counts, teams=teams)

    normalized = shots.copy()
    normalized["Team"] = normalized["Team"].fillna("UNKNOWN").astype(str).str.upper()

    if team_order:
        teams = [team for team in team_order if team in normalized["Team"].unique()]
        missing = [team for team in team_order if team not in teams]
        teams.extend(missing)
    else:
        teams = list(dict.fromkeys(normalized["Team"]))

    frames = _build_frame_range(total_frames, frame_range)
    counts: Dict[str, int] = {team: 0 for team in teams}
    rows: List[Dict[str, int]] = []

    events = normalized.sort_values("StartFrame")
    event_frames = events["StartFrame"].to_numpy(dtype=int)
    event_teams = events["Team"].to_numpy(dtype=str)
    event_idx = 0

    for frame in frames:
        while event_idx < len(event_frames) and frame >= event_frames[event_idx]:
            team = event_teams[event_idx]
            counts[team] = counts.get(team, 0) + 1
            event_idx += 1
        row = {"Frame": frame}
        for team in teams:
            row[team] = counts.get(team, 0)
        rows.append(row)

    counts_df = pd.DataFrame(rows)
    return ShotCountTimeline(shots=normalized, counts=counts_df, teams=teams)


def _build_frame_range(total_frames: Optional[int], frame_range: Optional[Iterable[int]]) -> List[int]:
    if frame_range is not None:
        frames = sorted({int(frame) for frame in frame_range})
        if not frames:
            raise ValueError("frame_range no puede estar vacío")
        return frames
    if total_frames is not None:
        return list(range(int(total_frames)))
    raise ValueError("Debe proporcionarse total_frames o frame_range para construir el contador.")
