"""CLI para ejecutar únicamente la etapa de inferencia."""

from __future__ import annotations

import argparse
from pathlib import Path

from .match_steps import DEFAULT_OUTPUT_ROOT, run_inference_stage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera las detecciones base de un video usando RF-DETR.")
    parser.add_argument("--video", required=True, type=Path, help="Ruta al video a procesar.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ModeloRF.pth"),
        help="Checkpoint del detector RF-DETR.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directorio donde se creará la carpeta por video (por defecto outputs/match_runs).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral de confianza para conservar detecciones.",
    )
    parser.add_argument(
        "--raw-name",
        type=str,
        default="detecciones_raw.csv",
        help="Nombre del CSV generado en la carpeta 1_inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"No se encontró el video: {args.video}")
    if not args.model.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {args.model}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold debe estar en el rango [0, 1]")

    detections_csv = run_inference_stage(
        args.video,
        args.model,
        threshold=args.threshold,
        output_root=args.output_root,
        raw_name=args.raw_name,
    )
    print(f"Detecciones guardadas en: {detections_csv}")


if __name__ == "__main__":
    main()
