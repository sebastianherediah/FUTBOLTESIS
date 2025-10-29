"""Ejecuta el clustering de equipos sobre las detecciones y anota el DataFrame resultante."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

if __package__ in (None, ""):
    # Permitir ejecución directa `python src/clustering/run_local.py`.
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.clustering.cluster import Cluster  # type: ignore
else:
    from .cluster import Cluster

BASE_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCH_HOME", str(BASE_DIR / ".cache" / "torch"))
DEFAULT_DETECTIONS = BASE_DIR / "outputs" / "detecciones_locales.csv"
DEFAULT_VIDEO = BASE_DIR / "VideoPruebaTesis.mp4"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "detecciones_con_equipos.csv"


def _parse_bbox(raw: object) -> Optional[List[float]]:
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        return None


def _bbox_key(values: Sequence[float]) -> Tuple[int, int, int, int]:
    """Genera una clave determinística redondeando a píxeles."""

    x1, y1, x2, y2 = values
    return (
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
    )


def run(random_state: int = 0) -> Path:
    """Clasifica cada detección en uno de los dos equipos disponibles.

    Parameters
    ----------
    random_state:
        Semilla para el algoritmo de clustering (default: 0).

    Returns
    -------
    Path
        Ruta al CSV anotado con la columna ``Team``.
    """

    if not DEFAULT_DETECTIONS.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de detecciones en {DEFAULT_DETECTIONS}. "
            "Ejecuta primero `python -m src.inference.run_local`."
        )

    if not DEFAULT_VIDEO.exists():
        raise FileNotFoundError(
            f"No se encontró el video esperado en {DEFAULT_VIDEO}. "
            "Verifica que 'VideoPruebaTesis.mp4' esté en la raíz del repositorio."
        )

    detections = pd.read_csv(DEFAULT_DETECTIONS)
    required = {"Frame", "ClassName", "BBox"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"El CSV de detecciones carece de las columnas: {missing}")

    detections["BBoxParsed"] = detections["BBox"].apply(_parse_bbox)
    if detections["BBoxParsed"].isnull().any():
        raise ValueError("Algunas detecciones contienen bounding boxes inválidas.")

    detections["BBoxKey"] = detections["BBoxParsed"].apply(_bbox_key)

    player_mask = detections["ClassName"].str.lower() == "jugador"
    players = detections.loc[player_mask, ["Frame", "ClassName", "BBoxParsed"]].copy()
    if players.empty:
        raise ValueError("No se encontraron detecciones de jugadores para clúster.")

    players.rename(columns={"BBoxParsed": "BBox"}, inplace=True)

    cluster = Cluster(random_state=random_state)
    result = cluster.cluster_players(players, str(DEFAULT_VIDEO))

    assignments = result.assignments.copy()
    assignments["BBoxParsed"] = assignments["BBox"].apply(_parse_bbox)
    assignments["BBoxKey"] = assignments["BBoxParsed"].apply(_bbox_key)

    team_labels = assignments["Team"].unique()
    if len(team_labels) != 2:
        raise RuntimeError(
            f"Se esperaban exactamente dos equipos, pero se obtuvieron: {team_labels}"
        )

    sorted_labels = sorted(team_labels)
    team_mapping: Dict[str, str] = {
        sorted_labels[0]: "BRASIL",
        sorted_labels[1]: "COLOMBIA",
    }

    team_by_frame_bbox: Dict[tuple, str] = {
        (int(row.Frame), row.BBoxKey): team_mapping[str(row.Team)]
        for row in assignments.itertuples(index=False)
    }

    def _lookup_team(row: pd.Series) -> Optional[str]:
        return team_by_frame_bbox.get((int(row["Frame"]), row["BBoxKey"]))

    detections["Team"] = detections.apply(_lookup_team, axis=1)

    detections.drop(columns=["BBoxParsed", "BBoxKey"], inplace=True)

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    detections.to_csv(DEFAULT_OUTPUT, index=False)
    return DEFAULT_OUTPUT


if __name__ == "__main__":
    output_path = run()
    print(f"Detecciones con equipos guardadas en: {output_path}")
