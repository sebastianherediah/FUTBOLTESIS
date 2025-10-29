"""Rendering utilities to draw projected detections on a 2D minimap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .field_layout import FieldLayout


@dataclass
class PlayerPoint:
    x: float
    y: float
    team: Optional[str] = None


class MinimapRenderer:
    """Draws a bird-eye view of the pitch with projected points."""

    def __init__(
        self,
        layout: FieldLayout,
        *,
        output_size: Tuple[int, int] = (640, 240),
        team_colors: Optional[dict[str, Tuple[int, int, int]]] = None,
    ) -> None:
        if not layout.keypoints:
            raise ValueError(
                "El layout de la cancha no define keypoints. Completa DEFAULT_FIELD_LAYOUT antes de renderizar."
            )

        self.layout = layout
        self.output_width, self.output_height = output_size

        coords = np.array(list(layout.keypoints.values()), dtype=np.float32)
        self.min_x, self.min_y = coords.min(axis=0)
        self.max_x, self.max_y = coords.max(axis=0)

        self.team_colors = {
            "BRASIL": (0, 215, 255),
            "COLOMBIA": (0, 255, 0),
        }
        if team_colors:
            self.team_colors.update(team_colors)

    def _to_canvas(self, x: float, y: float) -> Tuple[int, int]:
        norm_x = (x - self.min_x) / max(self.max_x - self.min_x, 1e-6)
        norm_y = (y - self.min_y) / max(self.max_y - self.min_y, 1e-6)
        canvas_x = int(norm_x * (self.output_width - 1))
        canvas_y = int((1.0 - norm_y) * (self.output_height - 1))
        return canvas_x, canvas_y

    def render(
        self,
        players: Iterable[PlayerPoint],
        *,
        ball_position: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        canvas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        # Draw pitch outline using reference keypoints (simple rectangle fallback)
        top_left = self._to_canvas(self.min_x, self.max_y)
        bottom_right = self._to_canvas(self.max_x, self.min_y)
        cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), 2)
        mid_x = int((top_left[0] + bottom_right[0]) / 2)
        cv2.line(canvas, (mid_x, top_left[1]), (mid_x, bottom_right[1]), (255, 255, 255), 1)
        center_circle_radius = int(0.0915 * self.output_width)  # 9.15m approx scaled
        center_point = (mid_x, int((top_left[1] + bottom_right[1]) / 2))
        cv2.circle(canvas, center_point, center_circle_radius, (255, 255, 255), 1)

        for player in players:
            px, py = self._to_canvas(player.x, player.y)
            color = self.team_colors.get(player.team or "", (255, 255, 0))
            cv2.circle(canvas, (px, py), 5, color, -1)

        if ball_position is not None:
            bx, by = self._to_canvas(*ball_position)
            cv2.circle(canvas, (bx, by), 4, (0, 140, 255), -1)

        return canvas

    @staticmethod
    def overlay(frame: np.ndarray, minimap: np.ndarray, *, margin: int = 16) -> np.ndarray:
        frame_h, frame_w = frame.shape[:2]
        map_h, map_w = minimap.shape[:2]

        scale = min((frame_w - 2 * margin) / map_w, (frame_h // 3) / map_h)
        resized = cv2.resize(minimap, (int(map_w * scale), int(map_h * scale)), interpolation=cv2.INTER_LINEAR)

        x_offset = (frame_w - resized.shape[1]) // 2
        y_offset = frame_h - resized.shape[0] - margin

        result = frame.copy()
        roi = result[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]]
        blended = cv2.addWeighted(roi, 0.4, resized, 0.6, 0)
        result[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = blended
        return result


__all__ = ["MinimapRenderer", "PlayerPoint"]
