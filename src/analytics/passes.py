"""Detección heurística de pases y cálculo de posesión a partir de detecciones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PassEvent:
    """Representa un pase detectado en la secuencia."""

    start_frame: int
    end_frame: int
    from_player: int
    to_player: int
    team: str
    gap_frames: int


@dataclass
class PassAnalysisResult:
    """Resultado del análisis de posesión y pases."""

    passes: pd.DataFrame
    possession: pd.DataFrame
    control_timeline: pd.DataFrame


class PassAnalyzer:
    """Identifica cambios de posesión para estimar pases y posesión por equipo.

    Este analizador espera un ``DataFrame`` con las detecciones frame a frame. Los
    registros de jugadores deben incluir las columnas ``Team`` (nombre del equipo) y
    ``TrackId`` (identificador persistente del jugador) además de ``BBox`` y ``Frame``.
    El balón debe aparecer como otra detección con ``ClassName == ball_class``. El
    algoritmo aproxima la posesión determinando en cada frame qué jugador se encuentra
    más próximo al balón y filtrando segmentos cortos para evitar falsos positivos.

    Parameters
    ----------
    distance_threshold:
        Distancia máxima (en píxeles) entre el centro del balón y el centro del jugador
        para considerar que existe control del balón.
    min_possession_frames:
        Número mínimo de frames consecutivos que debe durar una posesión para que se
        tenga en cuenta al identificar un pase.
    max_gap_frames:
        Máxima separación en frames entre dos posesiones consecutivas para considerarlas
        parte del mismo flujo de la jugada (y, por tanto, elegibles para formar un pase).
    """

    def __init__(
        self,
        *,
        distance_threshold: float = 75.0,
        min_possession_frames: int = 3,
        max_gap_frames: int = 10,
    ) -> None:
        if min_possession_frames < 1:
            raise ValueError("min_possession_frames debe ser >= 1")
        if max_gap_frames < 0:
            raise ValueError("max_gap_frames debe ser >= 0")

        self.distance_threshold = float(distance_threshold)
        self.min_possession_frames = int(min_possession_frames)
        self.max_gap_frames = int(max_gap_frames)

    def analyze(
        self,
        detections: pd.DataFrame,
        *,
        player_class: str = "jugador",
        ball_class: str = "balon",
    ) -> PassAnalysisResult:
        """Ejecuta el análisis y devuelve los pases detectados y la posesión agregada."""

        required_columns = {"Frame", "ClassName", "BBox"}
        missing = required_columns - set(detections.columns)
        if missing:
            raise ValueError(
                f"El DataFrame de detecciones no contiene las columnas obligatorias: {missing}"
            )

        players = detections[
            detections["ClassName"].str.lower() == player_class.lower()
        ].copy()
        if players.empty:
            raise ValueError("No se encontraron detecciones de jugadores en el DataFrame")

        for column in ("TrackId", "Team"):
            if column not in players.columns:
                raise ValueError(
                    f"Las detecciones de jugadores requieren la columna '{column}' para el análisis"
                )

        ball = detections[
            detections["ClassName"].str.lower() == ball_class.lower()
        ].copy()
        if ball.empty:
            raise ValueError("No se encontraron detecciones del balón en el DataFrame")

        ball_frames = ball.groupby("Frame")
        player_groups = players.groupby("Frame")
        frames = sorted(set(players["Frame"]) | set(ball["Frame"]))
        control_rows: List[Dict[str, object]] = []

        for frame in frames:
            frame_players = player_groups.get_group(frame) if frame in player_groups.indices else pd.DataFrame(columns=players.columns)
            if frame_players.empty:
                control_rows.append(
                    {"Frame": int(frame), "PlayerId": None, "Team": None, "Distance": np.nan}
                )
                continue

            ball_info = ball_frames.get_group(frame) if frame in ball_frames.indices else None
            if ball_info is None or ball_info.empty:
                control_rows.append(
                    {"Frame": int(frame), "PlayerId": None, "Team": None, "Distance": np.nan}
                )
                continue

            player_id, team, distance = self._nearest_player(ball_info, frame_players)
            if player_id is None or distance is None:
                control_rows.append(
                    {"Frame": int(frame), "PlayerId": None, "Team": None, "Distance": np.nan}
                )
                continue

            if distance > self.distance_threshold:
                control_rows.append(
                    {"Frame": int(frame), "PlayerId": None, "Team": None, "Distance": float(distance)}
                )
                continue

            control_rows.append(
                {
                    "Frame": int(frame),
                    "PlayerId": int(player_id),
                    "Team": str(team),
                    "Distance": float(distance),
                }
            )

        if not control_rows:
            raise RuntimeError("No se pudo construir una línea de posesión a partir de las detecciones")

        control_df = pd.DataFrame(control_rows).sort_values("Frame").reset_index(drop=True)
        segments = self._build_possession_segments(control_df)
        passes = self._detect_passes(segments)
        possession = self._compute_possession(passes)

        passes_df = (
            pd.DataFrame(
                [
                    {
                        "StartFrame": event.start_frame,
                        "EndFrame": event.end_frame,
                        "FromPlayer": event.from_player,
                        "ToPlayer": event.to_player,
                        "Team": event.team,
                        "GapFrames": event.gap_frames,
                    }
                    for event in passes
                ]
            )
            if passes
            else pd.DataFrame(
                columns=[
                    "StartFrame",
                    "EndFrame",
                    "FromPlayer",
                    "ToPlayer",
                    "Team",
                    "GapFrames",
                ]
            )
        )

        if possession:
            counts = {team: sum(1 for event in passes if event.team == team) for team in possession}
            possession_df = pd.DataFrame(
                [
                    {
                        "Team": team,
                        "Passes": counts[team],
                        "Possession": possession[team],
                    }
                    for team in sorted(possession)
                ]
            )
        else:
            possession_df = pd.DataFrame(columns=["Team", "Passes", "Possession"])

        return PassAnalysisResult(
            passes=passes_df,
            possession=possession_df,
            control_timeline=control_df,
        )

    def _nearest_player(
        self,
        ball_rows: pd.DataFrame,
        players: pd.DataFrame,
    ) -> Tuple[Optional[int], Optional[str], Optional[float]]:
        """Encuentra el jugador más cercano al balón en un frame."""

        ball_row = ball_rows.iloc[0]
        ball_center = self._bbox_center(ball_row["BBox"])
        if ball_center is None:
            return None, None, None

        best_player: Optional[int] = None
        best_team: Optional[str] = None
        best_distance: Optional[float] = None

        for _, player_row in players.iterrows():
            player_center = self._bbox_center(player_row["BBox"])
            if player_center is None:
                continue
            distance = self._euclidean(ball_center, player_center)
            if best_distance is None or distance < best_distance:
                best_player = int(player_row["TrackId"])
                best_team = str(player_row["Team"])
                best_distance = float(distance)

        return best_player, best_team, best_distance

    def _build_possession_segments(
        self, control_df: pd.DataFrame
    ) -> List[Dict[str, object]]:
        """Agrupa frames consecutivos con el mismo jugador en posesión."""

        segments: List[Dict[str, object]] = []
        current_player: Optional[int] = None
        current_team: Optional[str] = None
        start_frame: Optional[int] = None
        last_frame: Optional[int] = None
        frame_count = 0

        for row in control_df.itertuples(index=False):
            player_id = getattr(row, "PlayerId")
            team = getattr(row, "Team")
            frame = int(getattr(row, "Frame"))

            if player_id is None or team is None:
                if current_player is not None and frame_count >= self.min_possession_frames:
                    segments.append(
                        {
                            "PlayerId": current_player,
                            "Team": current_team,
                            "StartFrame": start_frame,
                            "EndFrame": last_frame,
                            "Length": frame_count,
                        }
                    )
                current_player = None
                current_team = None
                start_frame = None
                last_frame = None
                frame_count = 0
                continue

            if player_id == current_player:
                last_frame = frame
                frame_count += 1
                continue

            if current_player is not None and frame_count >= self.min_possession_frames:
                segments.append(
                    {
                        "PlayerId": current_player,
                        "Team": current_team,
                        "StartFrame": start_frame,
                        "EndFrame": last_frame,
                        "Length": frame_count,
                    }
                )

            current_player = player_id
            current_team = team
            start_frame = frame
            last_frame = frame
            frame_count = 1

        if current_player is not None and frame_count >= self.min_possession_frames:
            segments.append(
                {
                    "PlayerId": current_player,
                    "Team": current_team,
                    "StartFrame": start_frame,
                    "EndFrame": last_frame,
                    "Length": frame_count,
                }
            )

        return segments

    def _detect_passes(self, segments: List[Dict[str, object]]) -> List[PassEvent]:
        """Detecta transiciones válidas entre segmentos para contabilizar pases."""

        events: List[PassEvent] = []
        if len(segments) < 2:
            return events

        for prev, curr in zip(segments, segments[1:]):
            if prev["Team"] != curr["Team"]:
                continue
            if prev["PlayerId"] == curr["PlayerId"]:
                continue

            gap = curr["StartFrame"] - prev["EndFrame"]
            if gap < 0 or gap > self.max_gap_frames:
                continue

            events.append(
                PassEvent(
                    start_frame=int(prev["EndFrame"]),
                    end_frame=int(curr["StartFrame"]),
                    from_player=int(prev["PlayerId"]),
                    to_player=int(curr["PlayerId"]),
                    team=str(prev["Team"]),
                    gap_frames=int(gap),
                )
            )

        return events

    def _compute_possession(self, passes: List[PassEvent]) -> Dict[str, float]:
        """Calcula la posesión como fracción de pases por equipo."""

        if not passes:
            return {}

        counts: Dict[str, int] = {}
        for event in passes:
            counts[event.team] = counts.get(event.team, 0) + 1

        total = sum(counts.values())
        if total == 0:
            return {team: 0.0 for team in counts}

        return {team: counts[team] / total for team in counts}

    @staticmethod
    def _bbox_center(box: Iterable[float]) -> Optional[Tuple[float, float]]:
        """Calcula el centro de una bounding box."""

        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except (TypeError, ValueError):
            return None
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Distancia euclidiana entre dos puntos."""

        return float(np.linalg.norm(np.subtract(a, b)))
