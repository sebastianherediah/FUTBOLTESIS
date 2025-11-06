"""Detecta tiros de forma heurística usando intersección balón-arco."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detecta intentos de tiro cuando el balón intersecta la detección del arco. "
            "Si el jugador más cercano pertenece a Colombia se incrementa el contador de Brasil y viceversa."
        )
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path("outputs/detecciones_con_equipos.csv"),
        help="CSV con las detecciones (Frame, ClassName, BBox, Team).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tiros_simple.csv"),
        help="CSV donde se guardarán los tiros detectados.",
    )
    parser.add_argument(
        "--ball-class",
        type=str,
        default="balon",
        help="Nombre de la clase utilizada para el balón.",
    )
    parser.add_argument(
        "--goal-class",
        type=str,
        default="arco",
        help="Nombre de la clase utilizada para los arcos.",
    )
    parser.add_argument(
        "--player-class",
        type=str,
        default="jugador",
        help="Nombre de la clase utilizada para los jugadores.",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=1,
        help="Frames mínimos consecutivos para consolidar un tiro.",
    )
    parser.add_argument(
        "--team-order",
        nargs="+",
        default=["COLOMBIA", "BRASIL"],
        help="Equipos esperados en el video (se usa para validar etiquetas).",
    )
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


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_bbox(point: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.subtract(a, b)))


def _resolve_attacking_team(defending_team: Optional[str]) -> str:
    normalized = (defending_team or "").strip().upper()
    if normalized == "COLOMBIA":
        return "BRASIL"
    if normalized == "BRASIL":
        return "COLOMBIA"
    return "UNKNOWN"


def _load_detections(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Frame", "ClassName", "BBox"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El CSV de detecciones carece de columnas obligatorias: {missing}")
    df["BBox"] = df["BBox"].apply(_parse_bbox)
    df = df[df["BBox"].notna()].copy()
    return df


def detect_shots(
    detections: pd.DataFrame,
    *,
    ball_class: str,
    goal_class: str,
    player_class: str,
    min_duration: int,
) -> pd.DataFrame:
    frames = sorted(detections["Frame"].unique())
    grouped = detections.groupby("Frame")

    events: List[Dict[str, object]] = []
    active = False
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    defending_teams: List[str] = []

    for frame in frames:
        frame_rows = grouped.get_group(frame)
        ball_rows = frame_rows[frame_rows["ClassName"].str.lower() == ball_class.lower()]
        goal_rows = frame_rows[frame_rows["ClassName"].str.lower() == goal_class.lower()]
        player_rows = frame_rows[frame_rows["ClassName"].str.lower() == player_class.lower()]

        if ball_rows.empty or goal_rows.empty:
            inside = False
        else:
            ball_center = _bbox_center(ball_rows.iloc[0]["BBox"])
            inside = any(_point_in_bbox(ball_center, bbox) for bbox in goal_rows["BBox"])

        if not inside:
            if active and current_start is not None and current_end is not None:
                duration = current_end - current_start + 1
                if duration >= min_duration:
                    defending_team = Counter(defending_teams).most_common(1)[0][0] if defending_teams else "UNKNOWN"
                    events.append(
                        {
                            "StartFrame": current_start,
                            "EndFrame": current_end,
                            "Frames": duration,
                            "DefendingTeam": defending_team,
                            "AttackingTeam": _resolve_attacking_team(defending_team),
                        }
                    )
            active = False
            current_start = None
            current_end = None
            defending_teams = []
            continue

        # inside goal
        ball_center = _bbox_center(ball_rows.iloc[0]["BBox"])
        defending_team = "UNKNOWN"
        if not player_rows.empty:
            best_distance = None
            best_team = None
            for _, prow in player_rows.iterrows():
                team = prow.get("Team")
                if pd.isna(team):
                    continue
                team_str = str(team).strip().upper()
                player_center = _bbox_center(prow["BBox"])
                distance = _euclidean(ball_center, player_center)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_team = team_str
            if best_team:
                defending_team = best_team

        if not active:
            active = True
            current_start = frame
            current_end = frame
            defending_teams = [defending_team]
        else:
            current_end = frame
            defending_teams.append(defending_team)

    # finalize if video ended inside goal
    if active and current_start is not None and current_end is not None:
        duration = current_end - current_start + 1
        if duration >= min_duration:
            defending_team = Counter(defending_teams).most_common(1)[0][0] if defending_teams else "UNKNOWN"
            events.append(
                {
                    "StartFrame": current_start,
                    "EndFrame": current_end,
                    "Frames": duration,
                    "DefendingTeam": defending_team,
                    "AttackingTeam": _resolve_attacking_team(defending_team),
                }
            )

    columns = ["StartFrame", "EndFrame", "Frames", "DefendingTeam", "AttackingTeam"]
    return pd.DataFrame(events, columns=columns)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    detections = _load_detections(args.detections)
    shots = detect_shots(
        detections,
        ball_class=args.ball_class,
        goal_class=args.goal_class,
        player_class=args.player_class,
        min_duration=args.min_duration,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shots.to_csv(args.output, index=False)

    counts = defaultdict(int)
    for team in shots["AttackingTeam"]:
        counts[str(team).upper()] += 1

    print(f"Total de tiros detectados: {len(shots)}")
    for team in args.team_order:
        print(f"Tiros {team}: {counts.get(team.upper(), 0)}")


if __name__ == "__main__":
    main()
