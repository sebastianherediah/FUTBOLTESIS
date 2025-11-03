"""Canonical layout for the soccer pitch used in the homography keypoint model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class FieldLayout:
    """Defines the 2D coordinates (in meters) for each semantic keypoint."""

    keypoints: Dict[str, Tuple[float, float]]


class _PitchSpec:
    length = 105.0  # meters
    width = 68.0
    centre_circle_radius = 9.15
    penalty_area_length = 16.5
    penalty_area_width = 40.32
    goal_area_length = 5.5
    goal_area_width = 18.32
    goal_line_to_penalty_mark = 11.0
    goal_width = 7.32
    goal_height = 2.44


PITCH = _PitchSpec()


def _build_layout() -> Dict[str, Tuple[float, float]]:
    half_length = PITCH.length / 2.0
    half_width = PITCH.width / 2.0
    layout: Dict[str, Tuple[float, float]] = {}

    def add(name: str, x: float, y: float) -> None:
        layout[name] = (x, y)

    # Corners
    add("TL_PITCH_CORNER", -half_length, -half_width)
    add("TR_PITCH_CORNER", half_length, -half_width)
    add("BL_PITCH_CORNER", -half_length, half_width)
    add("BR_PITCH_CORNER", half_length, half_width)

    # Halfway line & centre
    add("T_TOUCH_AND_HALFWAY_LINES_INTERSECTION", 0.0, -half_width)
    add("B_TOUCH_AND_HALFWAY_LINES_INTERSECTION", 0.0, half_width)
    add("CENTER_MARK", 0.0, 0.0)

    # Penalty areas
    left_x = -half_length
    right_x = half_length

    add("L_PENALTY_AREA_TL_CORNER", left_x, -PITCH.penalty_area_width / 2.0)
    add("L_PENALTY_AREA_TR_CORNER", left_x + PITCH.penalty_area_length, -PITCH.penalty_area_width / 2.0)
    add("L_PENALTY_AREA_BL_CORNER", left_x, PITCH.penalty_area_width / 2.0)
    add("L_PENALTY_AREA_BR_CORNER", left_x + PITCH.penalty_area_length, PITCH.penalty_area_width / 2.0)
    add("L_PENALTY_MARK", left_x + PITCH.goal_line_to_penalty_mark, 0.0)

    add("R_PENALTY_AREA_TL_CORNER", right_x - PITCH.penalty_area_length, -PITCH.penalty_area_width / 2.0)
    add("R_PENALTY_AREA_TR_CORNER", right_x, -PITCH.penalty_area_width / 2.0)
    add("R_PENALTY_AREA_BL_CORNER", right_x - PITCH.penalty_area_length, PITCH.penalty_area_width / 2.0)
    add("R_PENALTY_AREA_BR_CORNER", right_x, PITCH.penalty_area_width / 2.0)
    add("R_PENALTY_MARK", right_x - PITCH.goal_line_to_penalty_mark, 0.0)

    # Goal areas
    add("L_GOAL_AREA_TL_CORNER", left_x, -PITCH.goal_area_width / 2.0)
    add("L_GOAL_AREA_TR_CORNER", left_x + PITCH.goal_area_length, -PITCH.goal_area_width / 2.0)
    add("L_GOAL_AREA_BL_CORNER", left_x, PITCH.goal_area_width / 2.0)
    add("L_GOAL_AREA_BR_CORNER", left_x + PITCH.goal_area_length, PITCH.goal_area_width / 2.0)

    add("R_GOAL_AREA_TL_CORNER", right_x - PITCH.goal_area_length, -PITCH.goal_area_width / 2.0)
    add("R_GOAL_AREA_TR_CORNER", right_x, -PITCH.goal_area_width / 2.0)
    add("R_GOAL_AREA_BL_CORNER", right_x - PITCH.goal_area_length, PITCH.goal_area_width / 2.0)
    add("R_GOAL_AREA_BR_CORNER", right_x, PITCH.goal_area_width / 2.0)

    # Goal posts (top/bottom refer to pitch orientation)
    goal_half = PITCH.goal_width / 2.0
    add("L_GOAL_TL_POST", left_x, -goal_half)
    add("L_GOAL_TR_POST", left_x, goal_half)
    add("L_GOAL_BL_POST", left_x, -goal_half)  # bases coincide with TL/BR for planar mapping
    add("L_GOAL_BR_POST", left_x, goal_half)

    add("R_GOAL_TL_POST", right_x, -goal_half)
    add("R_GOAL_TR_POST", right_x, goal_half)
    add("R_GOAL_BL_POST", right_x, -goal_half)
    add("R_GOAL_BR_POST", right_x, goal_half)

    # Centre circle intersections
    add("T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION", 0.0, -PITCH.centre_circle_radius)
    add("B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION", 0.0, PITCH.centre_circle_radius)

    diag = PITCH.centre_circle_radius / math.sqrt(2.0)
    add("CENTER_CIRCLE_R", PITCH.centre_circle_radius, 0.0)
    add("CENTER_CIRCLE_L", -PITCH.centre_circle_radius, 0.0)
    add("CENTER_CIRCLE_TR", diag, -diag)
    add("CENTER_CIRCLE_TL", -diag, -diag)
    add("CENTER_CIRCLE_BR", diag, diag)
    add("CENTER_CIRCLE_BL", -diag, diag)

    # Tangents from touchline intersections
    def circle_tangent_points(cx: float, cy: float, radius: float, px: float, py: float):
        dx = px - cx
        dy = py - cy
        dist_sq = dx * dx + dy * dy
        if dist_sq <= radius * radius:
            return None
        dist = math.sqrt(dist_sq)
        angle = math.atan2(dy, dx)
        alpha = math.acos(radius / dist)
        t1 = (cx + radius * math.cos(angle + alpha), cy + radius * math.sin(angle + alpha))
        t2 = (cx + radius * math.cos(angle - alpha), cy + radius * math.sin(angle - alpha))
        return t1, t2

    top_tangents = circle_tangent_points(
        0.0,
        0.0,
        PITCH.centre_circle_radius,
        layout["T_TOUCH_AND_HALFWAY_LINES_INTERSECTION"][0],
        layout["T_TOUCH_AND_HALFWAY_LINES_INTERSECTION"][1],
    )
    bottom_tangents = circle_tangent_points(
        0.0,
        0.0,
        PITCH.centre_circle_radius,
        layout["B_TOUCH_AND_HALFWAY_LINES_INTERSECTION"][0],
        layout["B_TOUCH_AND_HALFWAY_LINES_INTERSECTION"][1],
    )

    if top_tangents:
        # ensure TL has smaller x than TR
        t_left, t_right = sorted(top_tangents, key=lambda p: p[0])
        add("CENTER_CIRCLE_TANGENT_TL", *t_left)
        add("CENTER_CIRCLE_TANGENT_TR", *t_right)
    if bottom_tangents:
        b_left, b_right = sorted(bottom_tangents, key=lambda p: p[0])
        add("CENTER_CIRCLE_TANGENT_BL", *b_left)
        add("CENTER_CIRCLE_TANGENT_BR", *b_right)

    # Penalty arcs
    dx_arc = PITCH.penalty_area_length - PITCH.goal_line_to_penalty_mark
    arc_offset = math.sqrt(max(PITCH.centre_circle_radius**2 - dx_arc**2, 0.0))
    add(
        "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
        left_x + PITCH.penalty_area_length,
        -arc_offset,
    )
    add(
        "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
        left_x + PITCH.penalty_area_length,
        arc_offset,
    )
    add(
        "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
        right_x - PITCH.penalty_area_length,
        -arc_offset,
    )
    add(
        "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
        right_x - PITCH.penalty_area_length,
        arc_offset,
    )

    # Additional circle points around penalty marks
    add("LEFT_CIRCLE_R", layout["L_PENALTY_MARK"][0] + PITCH.centre_circle_radius, 0.0)
    add("RIGHT_CIRCLE_L", layout["R_PENALTY_MARK"][0] - PITCH.centre_circle_radius, 0.0)
    add("L_MIDDLE_PENALTY", left_x + PITCH.penalty_area_length, 0.0)
    add("R_MIDDLE_PENALTY", right_x - PITCH.penalty_area_length, 0.0)

    def circle_tangent_from_mark(mark: Tuple[float, float], reference: Tuple[float, float], pref: str) -> Tuple[float, float] | None:
        tangents = circle_tangent_points(mark[0], mark[1], PITCH.centre_circle_radius, reference[0], reference[1])
        if not tangents:
            return None
        if pref == "Top":
            return min(tangents, key=lambda p: p[1])
        if pref == "Bottom":
            return max(tangents, key=lambda p: p[1])
        return tangents[0]

    left_top = circle_tangent_from_mark(
        layout["L_PENALTY_MARK"],
        layout["L_PENALTY_AREA_TR_CORNER"],
        "Top",
    )
    left_bottom = circle_tangent_from_mark(
        layout["L_PENALTY_MARK"],
        layout["L_PENALTY_AREA_BR_CORNER"],
        "Bottom",
    )
    right_top = circle_tangent_from_mark(
        layout["R_PENALTY_MARK"],
        layout["R_PENALTY_AREA_TL_CORNER"],
        "Top",
    )
    right_bottom = circle_tangent_from_mark(
        layout["R_PENALTY_MARK"],
        layout["R_PENALTY_AREA_BL_CORNER"],
        "Bottom",
    )

    if left_top:
        add("LEFT_CIRCLE_TANGENT_T", *left_top)
    if left_bottom:
        add("LEFT_CIRCLE_TANGENT_B", *left_bottom)
    if right_top:
        add("RIGHT_CIRCLE_TANGENT_T", *right_top)
    if right_bottom:
        add("RIGHT_CIRCLE_TANGENT_B", *right_bottom)

    return layout


DEFAULT_FIELD_LAYOUT = FieldLayout(keypoints=_build_layout())

__all__ = ["FieldLayout", "DEFAULT_FIELD_LAYOUT", "PITCH"]
