"""Rendering utilities to display projected detections on a 2D minimap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from .field_layout import DEFAULT_FIELD_LAYOUT, FieldLayout, PITCH


@dataclass
class PlayerPoint:
    x: float
    y: float
    team: Optional[str] = None


class MinimapRenderer:
    def __init__(
        self,
        output_size: Tuple[int, int] = (420, 210),
        *,
        layout: FieldLayout = DEFAULT_FIELD_LAYOUT,
        team_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        flip_x: bool = False,
        flip_y: bool = False,
    ) -> None:
        if not layout.keypoints:
            raise ValueError("El layout de la cancha debe contener keypoints.")

        self.layout = layout
        self.width, self.height = output_size
        self.flip_x = flip_x
        self.flip_y = flip_y

        coords = np.array(list(layout.keypoints.values()), dtype=np.float32)
        self.min_x, self.min_y = coords.min(axis=0)
        self.max_x, self.max_y = coords.max(axis=0)

        self.team_colors = {
            "local": (0, 215, 255),
            "visitante": (0, 255, 0),
            "BRASIL": (0, 215, 255),
            "COLOMBIA": (0, 255, 0),
        }
        if team_colors:
            self.team_colors.update(team_colors)

    def _to_canvas(self, x: float, y: float) -> Tuple[int, int]:
        norm_x = (x - self.min_x) / max(self.max_x - self.min_x, 1e-6)
        norm_y = (y - self.min_y) / max(self.max_y - self.min_y, 1e-6)
        if self.flip_x:
            norm_x = 1.0 - norm_x
        if self.flip_y:
            norm_y = 1.0 - norm_y
        canvas_x = int(norm_x * (self.width - 1))
        canvas_y = int((1.0 - norm_y) * (self.height - 1))
        return canvas_x, canvas_y

    def _draw_field(self, canvas: np.ndarray) -> None:
        color = (200, 200, 200)

        def draw_line(a: str, b: str, thickness: int = 2) -> None:
            x1, y1 = self._to_canvas(*self.layout.keypoints[a])
            x2, y2 = self._to_canvas(*self.layout.keypoints[b])
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=thickness)

        # Outer rectangle
        draw_line("TL_PITCH_CORNER", "TR_PITCH_CORNER")
        draw_line("TR_PITCH_CORNER", "BR_PITCH_CORNER")
        draw_line("BR_PITCH_CORNER", "BL_PITCH_CORNER")
        draw_line("BL_PITCH_CORNER", "TL_PITCH_CORNER")

        # Midfield line
        draw_line("T_TOUCH_AND_HALFWAY_LINES_INTERSECTION", "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION", thickness=1)

        # Centre circle
        center = self._to_canvas(*self.layout.keypoints["CENTER_MARK"])
        radius_px = int(PITCH.centre_circle_radius * self.width / (self.max_x - self.min_x))
        cv2.circle(canvas, center, radius_px, color, 1)
        cv2.circle(canvas, center, 3, color, -1)

        # Penalty areas and goal areas
        for prefix in ("L_", "R_"):
            draw_line(f"{prefix}PENALTY_AREA_TL_CORNER", f"{prefix}PENALTY_AREA_TR_CORNER")
            draw_line(f"{prefix}PENALTY_AREA_TR_CORNER", f"{prefix}PENALTY_AREA_BR_CORNER")
            draw_line(f"{prefix}PENALTY_AREA_BR_CORNER", f"{prefix}PENALTY_AREA_BL_CORNER")
            draw_line(f"{prefix}PENALTY_AREA_BL_CORNER", f"{prefix}PENALTY_AREA_TL_CORNER")

            draw_line(f"{prefix}GOAL_AREA_TL_CORNER", f"{prefix}GOAL_AREA_TR_CORNER")
            draw_line(f"{prefix}GOAL_AREA_TR_CORNER", f"{prefix}GOAL_AREA_BR_CORNER")
            draw_line(f"{prefix}GOAL_AREA_BR_CORNER", f"{prefix}GOAL_AREA_BL_CORNER")
            draw_line(f"{prefix}GOAL_AREA_BL_CORNER", f"{prefix}GOAL_AREA_TL_CORNER")

            penalty_mark = self._to_canvas(*self.layout.keypoints[f"{prefix}PENALTY_MARK"])
            cv2.circle(canvas, penalty_mark, 3, color, -1)

        # Approximate penalty arcs
        def arc_points(center_name: str, start_angle: float, end_angle: float) -> np.ndarray:
            cx, cy = self.layout.keypoints[center_name]
            radius = PITCH.centre_circle_radius
            angles = np.linspace(start_angle, end_angle, 30)
            pts = [self._to_canvas(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
            return np.array(pts, dtype=np.int32)

        left_arc = arc_points("L_PENALTY_MARK", np.radians(-60), np.radians(60))
        right_arc = arc_points("R_PENALTY_MARK", np.radians(120), np.radians(240))
        cv2.polylines(canvas, [left_arc], False, color, 1)
        cv2.polylines(canvas, [right_arc], False, color, 1)

    def render(
        self,
        players: Iterable[PlayerPoint],
        *,
        ball_position: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_field(canvas)

        player_radius = 6
        for player in players:
            px, py = self._to_canvas(player.x, player.y)
            color = self.team_colors.get((player.team or "").upper(), (0, 255, 255))
            cv2.circle(canvas, (px, py), player_radius, color, -1, lineType=cv2.LINE_AA)

        if ball_position is not None:
            bx, by = self._to_canvas(*ball_position)
            ball_radius = max(2, player_radius // 2)
            cv2.circle(canvas, (bx, by), ball_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        return canvas

    @staticmethod
    def overlay(frame: np.ndarray, minimap: np.ndarray, *, margin: int = 12) -> np.ndarray:
        frame_h, frame_w = frame.shape[:2]
        map_h, map_w = minimap.shape[:2]

        scale = min((frame_w - 2 * margin) / map_w, (frame_h // 4) / map_h)
        resized = cv2.resize(minimap, (int(map_w * scale), int(map_h * scale)), interpolation=cv2.INTER_LINEAR)

        x_offset = frame_w - resized.shape[1] - margin
        y_offset = frame_h - resized.shape[0] - margin

        result = frame.copy()
        roi = result[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]]
        blended = cv2.addWeighted(roi, 0.3, resized, 0.7, 0.0)
        result[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = blended
        return result
