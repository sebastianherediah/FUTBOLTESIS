"""Reference coordinates for the soccer pitch keypoints.

Actual mappings must be supplied according to the trained keypoint ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class FieldLayout:
    """Defines the 2D coordinates (in meters) for each semantic keypoint."""

    keypoints: Dict[str, Tuple[float, float]]

    @property
    def coordinate_pairs(self) -> Dict[str, Tuple[float, float]]:
        return self.keypoints


# Placeholder layout – replace with real mapping according to the dataset used to train Homografia.pth.
DEFAULT_FIELD_LAYOUT = FieldLayout(
    keypoints={
        # Ejemplo: "kp_00": (0.0, 0.0),
        # Rellena este diccionario con las coordenadas reales del campo para cada keypoint.
    }
)


__all__ = ["FieldLayout", "DEFAULT_FIELD_LAYOUT"]
