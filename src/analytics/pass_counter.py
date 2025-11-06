"""Herramientas para construir un contador acumulado de pases por equipo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .passes import PassAnalysisResult, PassAnalyzer


@dataclass(frozen=True)
class PassCountTimeline:
    """Almacena el resultado del análisis y el conteo acumulado por frame."""

    analysis: PassAnalysisResult
    counts: pd.DataFrame
    teams: Sequence[str]

    def counts_for_frame(self, frame: int) -> Dict[str, int]:
        """Devuelve el conteo acumulado de pases para cada equipo hasta ``frame``."""

        if self.counts.empty:
            return {team: 0 for team in self.teams}

        frames = self.counts["Frame"].to_numpy()
        idx = int(np.searchsorted(frames, frame, side="right") - 1)
        if idx < 0:
            return {team: 0 for team in self.teams}

        row = self.counts.iloc[idx]
        return {team: int(row.get(team, 0)) for team in self.teams}


def build_pass_count_timeline(
    detections: pd.DataFrame,
    *,
    analyzer: Optional[PassAnalyzer] = None,
    team_order: Optional[Sequence[str]] = None,
    frame_range: Optional[Iterable[int]] = None,
) -> PassCountTimeline:
    """Ejecuta el análisis de pases y genera el contador acumulado por frame.

    Parameters
    ----------
    detections:
        DataFrame con las detecciones enriquecidas. Debe contener al menos las
        columnas utilizadas por :class:`PassAnalyzer.analyze`.
    analyzer:
        Instancia opcional de :class:`PassAnalyzer`. Si no se proporciona se crea
        una con los parámetros por defecto.
    team_order:
        Secuencia con el orden deseado de los equipos en el contador. Si se omite,
        se utiliza el orden natural de los equipos detectados en los pases.
    frame_range:
        Iterable con los frames a considerar para el contador (por ejemplo
        ``range(total_frames)``). Si se omite, se utiliza el rango mínimo que cubre
        los frames presentes en ``detections``.

    Returns
    -------
    PassCountTimeline
        Objeto con el resultado del análisis de pases y el contador acumulado.
    """

    if analyzer is None:
        analyzer = PassAnalyzer()

    result = analyzer.analyze(detections)

    if result.passes.empty:
        teams: List[str]
        if team_order:
            teams = list(team_order)
        else:
            team_series = detections.get("Team")
            if team_series is None or team_series.dropna().empty:
                teams = []
            else:
                teams = list(dict.fromkeys(team_series.dropna().astype(str)))

        frame_values = _infer_frame_range(detections, frame_range)
        counts_df = _build_empty_counts(frame_values, teams)
        return PassCountTimeline(analysis=result, counts=counts_df, teams=teams)

    if team_order:
        teams = [team for team in team_order if team in set(result.passes["Team"])]
        missing = [team for team in team_order if team not in teams]
        # Incluimos equipos que no registraron pases para mantener el orden deseado.
        teams.extend(missing)
    else:
        teams = list(dict.fromkeys(result.passes["Team"].astype(str)))

    frame_values = _infer_frame_range(detections, frame_range)
    counts = {team: 0 for team in teams}
    rows: List[Dict[str, int]] = []

    events = result.passes.sort_values("EndFrame")
    event_frames = events["EndFrame"].to_numpy(dtype=int)
    event_teams = events["Team"].astype(str).to_numpy()
    event_idx = 0

    for frame in frame_values:
        while event_idx < len(event_frames) and frame >= event_frames[event_idx]:
            team = event_teams[event_idx]
            counts[team] = counts.get(team, 0) + 1
            event_idx += 1

        row = {"Frame": frame}
        for team in teams:
            row[team] = counts.get(team, 0)
        rows.append(row)

    counts_df = pd.DataFrame(rows)
    return PassCountTimeline(analysis=result, counts=counts_df, teams=teams)


def _infer_frame_range(
    detections: pd.DataFrame, frame_range: Optional[Iterable[int]]
) -> List[int]:
    if frame_range is not None:
        frames = [int(frame) for frame in frame_range]
        if not frames:
            raise ValueError("frame_range no puede ser una secuencia vacía")
        return sorted(set(frames))

    if "Frame" not in detections:
        raise ValueError(
            "El DataFrame de detecciones no contiene la columna 'Frame' necesaria para calcular el rango."
        )

    frame_series = detections["Frame"].dropna().astype(int)
    if frame_series.empty:
        raise ValueError(
            "El DataFrame de detecciones no tiene valores válidos en la columna 'Frame'."
        )

    start = frame_series.min()
    end = frame_series.max()
    return list(range(int(start), int(end) + 1))


def _build_empty_counts(frames: List[int], teams: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, int]] = []
    running = {team: 0 for team in teams}
    for frame in frames:
        row = {"Frame": frame}
        for team in teams:
            row[team] = running[team]
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["PassCountTimeline", "build_pass_count_timeline"]
