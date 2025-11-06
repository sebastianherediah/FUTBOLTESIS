"""Detección heurística de pases y cálculo de posesión a partir de detecciones."""

from __future__ import annotations

from collections import Counter
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
    ``BBox`` además de ``Frame``. Si existe la columna ``TrackId`` se utilizará para
    distinguir jugadores; en caso contrario, se generan identificadores heurísticos
    basados en la posición para aproximar los cambios de posesión.
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
    team_link_max_distance:
        Distancia máxima entre posiciones consecutivas de control del mismo equipo
        para asumir que se trata del mismo jugador cuando no hay ``TrackId`` disponible.
    team_link_max_gap:
        Separación máxima en frames para reutilizar el identificador heurístico del mismo
        jugador cuando se trabaja sin ``TrackId``.
    ball_max_interp_gap:
        Huecos máximos (en frames) que se interpolarán para el balón utilizando una
        trayectoria lineal. Se utiliza para mantener la posesión cuando el detector
        falla durante periodos cortos.
    control_smoothing_window:
        Tamaño de la ventana (en frames) usada para suavizar la asignación de posesión
        mediante voto mayoritario.
    control_smoothing_min_votes:
        Número mínimo de votos dentro de la ventana de suavizado requeridos para
        reemplazar el jugador/equipo asignado en un frame.
    """

    def __init__(
        self,
        *,
        distance_threshold: float = 90.0,
        min_possession_frames: int = 2,
        max_gap_frames: int = 12,
        team_link_max_distance: float = 160.0,
        team_link_max_gap: int = 20,
        ball_max_interp_gap: int = 12,
        control_smoothing_window: int = 5,
        control_smoothing_min_votes: int = 2,
    ) -> None:
        if min_possession_frames < 1:
            raise ValueError("min_possession_frames debe ser >= 1")
        if max_gap_frames < 0:
            raise ValueError("max_gap_frames debe ser >= 0")
        if team_link_max_distance < 0:
            raise ValueError("team_link_max_distance debe ser >= 0")
        if team_link_max_gap < 0:
            raise ValueError("team_link_max_gap debe ser >= 0")
        if ball_max_interp_gap < 0:
            raise ValueError("ball_max_interp_gap debe ser >= 0")
        if control_smoothing_window < 1:
            raise ValueError("control_smoothing_window debe ser >= 1")
        if control_smoothing_min_votes < 1:
            raise ValueError("control_smoothing_min_votes debe ser >= 1")

        self.distance_threshold = float(distance_threshold)
        self.min_possession_frames = int(min_possession_frames)
        self.max_gap_frames = int(max_gap_frames)
        self.team_link_max_distance = float(team_link_max_distance)
        self.team_link_max_gap = int(team_link_max_gap)
        self.ball_max_interp_gap = int(ball_max_interp_gap)
        self.control_smoothing_window = int(control_smoothing_window)
        self.control_smoothing_min_votes = int(control_smoothing_min_votes)

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

        if "Team" not in players.columns:
            raise ValueError(
                "Las detecciones de jugadores requieren la columna 'Team' para el análisis"
            )

        has_track_ids = bool(
            "TrackId" in players.columns and not players["TrackId"].isna().all()
        )

        ball = detections[
            detections["ClassName"].str.lower() == ball_class.lower()
        ].copy()
        if ball.empty:
            raise ValueError("No se encontraron detecciones del balón en el DataFrame")
        ball = self._interpolate_ball(ball)

        ball_frames = ball.groupby("Frame")
        player_groups = players.groupby("Frame")
        frames = sorted(set(players["Frame"]) | set(ball["Frame"]))
        control_rows: List[Dict[str, object]] = []

        for frame in frames:
            frame_players = (
                player_groups.get_group(frame)
                if frame in player_groups.indices
                else pd.DataFrame(columns=players.columns)
            )
            if frame_players.empty:
                control_rows.append(
                    {
                        "Frame": int(frame),
                        "PlayerId": None,
                        "Team": None,
                        "Distance": np.nan,
                        "CenterX": np.nan,
                        "CenterY": np.nan,
                    }
                )
                continue

            ball_info = ball_frames.get_group(frame) if frame in ball_frames.indices else None
            if ball_info is None or ball_info.empty:
                control_rows.append(
                    {
                        "Frame": int(frame),
                        "PlayerId": None,
                        "Team": None,
                        "Distance": np.nan,
                        "CenterX": np.nan,
                        "CenterY": np.nan,
                    }
                )
                continue

            player_id, team, distance, center = self._nearest_player(
                ball_info, frame_players, use_track_ids=has_track_ids
            )
            if team is None or distance is None or center is None:
                control_rows.append(
                    {
                        "Frame": int(frame),
                        "PlayerId": None,
                        "Team": None,
                        "Distance": np.nan,
                        "CenterX": np.nan,
                        "CenterY": np.nan,
                    }
                )
                continue

            if distance > self.distance_threshold:
                control_rows.append(
                    {
                        "Frame": int(frame),
                        "PlayerId": None,
                        "Team": None,
                        "Distance": float(distance),
                        "CenterX": float(center[0]),
                        "CenterY": float(center[1]),
                    }
                )
                continue

            control_rows.append(
                {
                    "Frame": int(frame),
                    "PlayerId": int(player_id) if player_id is not None else None,
                    "Team": str(team),
                    "Distance": float(distance),
                    "CenterX": float(center[0]),
                    "CenterY": float(center[1]),
                }
            )

        if not control_rows:
            raise RuntimeError("No se pudo construir una línea de posesión a partir de las detecciones")

        control_df = (
            pd.DataFrame(control_rows).sort_values("Frame").reset_index(drop=True)
        )
        control_df = self._smooth_control(control_df)

        if not has_track_ids:
            control_df = self._assign_team_based_ids(control_df)

        segments = self._build_possession_segments(control_df)
        passes = self._detect_passes(segments)
        possession = self._compute_possession(control_df)

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

        team_keys = sorted(set(possession) | {event.team for event in passes})
        if team_keys:
            counts = {team: 0 for team in team_keys}
            for event in passes:
                counts[event.team] = counts.get(event.team, 0) + 1
            possession_df = pd.DataFrame(
                [
                    {
                        "Team": team,
                        "Passes": counts.get(team, 0),
                        "Possession": possession.get(team, 0.0),
                    }
                    for team in team_keys
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
        *,
        use_track_ids: bool,
    ) -> Tuple[Optional[int], Optional[str], Optional[float], Optional[Tuple[float, float]]]:
        """Encuentra el jugador más cercano al balón en un frame."""

        ball_row = ball_rows.iloc[0]
        ball_center = self._bbox_center(ball_row["BBox"])
        if ball_center is None:
            return None, None, None, None

        best_player: Optional[int] = None
        best_team: Optional[str] = None
        best_distance: Optional[float] = None
        best_center: Optional[Tuple[float, float]] = None

        for _, player_row in players.iterrows():
            team_val = player_row.get("Team")
            if pd.isna(team_val):
                continue
            player_center = self._bbox_center(player_row["BBox"])
            if player_center is None:
                continue
            distance = self._euclidean(ball_center, player_center)
            if best_distance is None or distance < best_distance:
                player_id_value: Optional[int] = None
                if use_track_ids and "TrackId" in player_row.index:
                    track_val = player_row["TrackId"]
                    if pd.notna(track_val):
                        try:
                            player_id_value = int(track_val)
                        except (TypeError, ValueError):
                            player_id_value = None
                best_player = player_id_value
                best_team = str(team_val)
                best_distance = float(distance)
                best_center = player_center

        return best_player, best_team, best_distance, best_center

    def _assign_team_based_ids(self, control_df: pd.DataFrame) -> pd.DataFrame:
        """Asigna IDs heurísticos por equipo cuando no hay TrackId disponible."""

        next_id = 0
        last_state: Dict[str, Tuple[int, int, float, float]] = {}
        assigned: List[Optional[int]] = []

        for row in control_df.itertuples(index=False):
            team = getattr(row, "Team")
            frame = int(getattr(row, "Frame"))
            center_x = getattr(row, "CenterX")
            center_y = getattr(row, "CenterY")

            if team is None or pd.isna(team) or pd.isna(center_x) or pd.isna(center_y):
                assigned.append(None)
                continue

            team_str = str(team)
            previous = last_state.get(team_str)
            candidate_id: Optional[int] = None

            if previous is not None:
                prev_id, prev_frame, prev_x, prev_y = previous
                frame_gap = frame - prev_frame
                distance = self._euclidean((center_x, center_y), (prev_x, prev_y))
                if frame_gap <= self.team_link_max_gap and distance <= self.team_link_max_distance:
                    candidate_id = prev_id

            if candidate_id is None:
                candidate_id = next_id
                next_id += 1

            assigned.append(candidate_id)
            last_state[team_str] = (candidate_id, frame, float(center_x), float(center_y))

        enriched = control_df.copy()
        enriched["PlayerId"] = assigned
        return enriched

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

            if player_id is None or pd.isna(player_id) or team is None or pd.isna(team):
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
            player_id_int = int(player_id)
            team_str = str(team)

            if player_id_int == current_player:
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

            current_player = player_id_int
            current_team = team_str
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

    def _compute_possession(self, control_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula la posesión como fracción de frames controlados por equipo."""

        if "Team" not in control_df.columns or "PlayerId" not in control_df.columns:
            return {}

        mask = control_df["Team"].notna() & control_df["PlayerId"].notna()
        if not mask.any():
            return {}

        counts = control_df.loc[mask, "Team"].astype(str).value_counts()
        total = counts.sum()
        if total == 0:
            return {}

        return {team: float(count) / float(total) for team, count in counts.items()}

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

    def _interpolate_ball(self, ball: pd.DataFrame) -> pd.DataFrame:
        """Interpola detecciones del balón para cubrir huecos cortos."""

        if self.ball_max_interp_gap <= 0 or len(ball) < 2:
            return ball
        processed: List[Dict[str, object]] = []
        sorted_ball = (
            ball.groupby("Frame", as_index=False)
            .head(1)
            .sort_values("Frame")
            .reset_index(drop=True)
        )

        previous_frame: Optional[int] = None
        previous_box: Optional[List[float]] = None
        previous_class: Optional[str] = None

        for row in sorted_ball.itertuples(index=False):
            current_frame = int(getattr(row, "Frame"))
            current_box_raw = getattr(row, "BBox")
            if not isinstance(current_box_raw, (list, tuple)) or len(current_box_raw) != 4:
                previous_frame = current_frame
                previous_box = None
                previous_class = getattr(row, "ClassName")
                processed.append(
                    {
                        "Frame": current_frame,
                        "ClassName": previous_class,
                        "BBox": current_box_raw,
                    }
                )
                continue

            current_box = [float(v) for v in current_box_raw]

            if previous_frame is not None and previous_box is not None:
                gap = current_frame - previous_frame
                if gap > 1:
                    steps = min(gap - 1, self.ball_max_interp_gap)
                    if steps > 0:
                        prev_arr = np.asarray(previous_box, dtype=float)
                        curr_arr = np.asarray(current_box, dtype=float)
                        for step in range(1, steps + 1):
                            alpha = float(step) / float(gap)
                            interp = prev_arr + alpha * (curr_arr - prev_arr)
                            processed.append(
                                {
                                    "Frame": previous_frame + step,
                                    "ClassName": previous_class or getattr(row, "ClassName"),
                                    "BBox": interp.tolist(),
                                }
                            )

            processed.append(
                {
                    "Frame": current_frame,
                    "ClassName": getattr(row, "ClassName"),
                    "BBox": current_box,
                }
            )
            previous_frame = current_frame
            previous_box = current_box
            previous_class = getattr(row, "ClassName")

        enriched = pd.DataFrame(processed, columns=["Frame", "ClassName", "BBox"]).drop_duplicates(
            subset=["Frame"], keep="last"
        )
        return enriched.sort_values("Frame").reset_index(drop=True)

    def _smooth_control(self, control_df: pd.DataFrame) -> pd.DataFrame:
        """Aplica una ventana deslizante para suavizar la posesión por frame."""

        if control_df.empty or self.control_smoothing_window <= 1:
            return control_df

        window = self.control_smoothing_window
        half = window // 2
        smoothed_players: List[Optional[object]] = []
        smoothed_teams: List[Optional[object]] = []

        frames = control_df["Frame"].to_numpy()
        players = control_df["PlayerId"].to_numpy(copy=True)
        teams = control_df["Team"].to_numpy(copy=True)

        for idx in range(len(control_df)):
            left = max(0, idx - half)
            right = min(len(control_df), idx + half + 1)
            window_players = players[left:right]
            window_teams = teams[left:right]

            candidates = [
                (window_players[i], window_teams[i])
                for i in range(len(window_players))
                if pd.notna(window_players[i]) and pd.notna(window_teams[i])
            ]

            if not candidates:
                smoothed_players.append(players[idx])
                smoothed_teams.append(teams[idx])
                continue

            counts = Counter(candidates)
            (chosen_player, chosen_team), freq = counts.most_common(1)[0]
            if freq >= self.control_smoothing_min_votes:
                smoothed_players.append(int(chosen_player))
                smoothed_teams.append(str(chosen_team))
            else:
                smoothed_players.append(players[idx])
                smoothed_teams.append(teams[idx])

        enriched = control_df.copy()
        enriched["PlayerId"] = smoothed_players
        enriched["Team"] = smoothed_teams
        return enriched
