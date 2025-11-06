"""Herramientas de inferencia para el proyecto FUTBOLTESIS."""

from typing import TYPE_CHECKING

__all__ = ["VideoInference", "render_pass_counter_video"]

if TYPE_CHECKING:  # pragma: no cover - solo para type checkers
    from .pass_counter_video import render_pass_counter_video
    from .video_inference import VideoInference


def __getattr__(name: str):
    if name == "VideoInference":
        from .video_inference import VideoInference

        return VideoInference
    if name == "render_pass_counter_video":
        from .pass_counter_video import render_pass_counter_video

        return render_pass_counter_video
    raise AttributeError(f"module 'src.inference' has no attribute '{name}'")
