"""Herramienta para validar rápidamente el clustering en un subconjunto de frames."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

from .match_steps import load_detections
from ..clustering import Cluster
from ..inference.visualize_detections import Detection, _annotate_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el clustering solo en un subconjunto pequeño de frames para validar que la asignación de equipos sea correcta."
        )
    )
    parser.add_argument("--video", required=True, type=Path, help="Ruta al video original.")
    parser.add_argument(
        "--detections",
        required=True,
        type=Path,
        help="CSV de detecciones con columnas Frame/ClassName/BBox.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame específico a inspeccionar. Si se omite, se elige uno aleatorio.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Número de frames adicionales a cada lado para incluir en la muestra.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Semilla para la inicialización de K-Means.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Número de recortes procesados simultáneamente por el encoder.",
    )
    parser.add_argument(
        "--player-class",
        type=str,
        default="jugador",
        help="Nombre de la clase que representa a los jugadores dentro del CSV.",
    )
    parser.add_argument(
        "--min-players",
        type=int,
        default=6,
        help="Cantidad mínima de jugadores requerida para ejecutar el clustering.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Ruta para guardar una imagen anotada del frame objetivo (se genera solo si la ruta es distinta de None).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"No se encontró el video: {args.video}")
    if not args.detections.exists():
        raise FileNotFoundError(f"No se encontró el CSV de detecciones: {args.detections}")

    detections = load_detections(args.detections)
    player_mask = detections["ClassName"].str.lower() == args.player_class.lower()
    players = detections.loc[player_mask].copy()
    if players.empty:
        raise ValueError(f"No se encontraron filas con ClassName == {args.player_class!r}")

    unique_frames = sorted(players["Frame"].unique())
    if not unique_frames:
        raise ValueError("No hay frames disponibles con jugadores para el clustering.")

    target_frame: int
    if args.frame is not None:
        target_frame = int(args.frame)
        if target_frame not in unique_frames:
            raise ValueError(f"El frame solicitado ({target_frame}) no contiene detecciones de jugadores.")
    else:
        target_frame = random.choice(unique_frames)

    start = max(target_frame - max(0, args.window), unique_frames[0])
    end = target_frame + max(0, args.window)
    subset = players[(players["Frame"] >= start) & (players["Frame"] <= end)].copy()
    if len(subset) < args.min_players:
        raise ValueError(
            f"La muestra seleccionada solo tiene {len(subset)} jugadores. "
            "Aumenta el parámetro --window o elige un frame con más jugadores."
        )

    cluster = Cluster(random_state=args.random_state, batch_size=max(1, args.batch_size), show_progress=False)
    result = cluster.cluster_players(subset[["Frame", "ClassName", "BBox"]], str(args.video))

    assignments = result.assignments.copy()
    counts = assignments["Team"].value_counts().rename_axis("Cluster").reset_index(name="Players")
    print(f"Frame objetivo: {target_frame} (ventana [{start}, {end}])")
    print("Resumen de clusters detectados:")
    print(counts.to_string(index=False))

    assignments["BBoxStr"] = assignments["BBox"].apply(lambda box: ",".join(f"{float(v):.2f}" for v in box))
    sample = assignments.sort_values(["Frame", "BBoxStr"]).head(10)
    print("\nPrimeras filas anotadas:")
    print(sample[["Frame", "Team", "BBox"]].to_string(index=False))

    if args.output_image:
        assignments["BBoxKey"] = assignments["BBox"].apply(lambda box: tuple(round(float(v), 2) for v in box))
        subset["BBoxKey"] = subset["BBox"].apply(lambda box: tuple(round(float(v), 2) for v in box))
        subset_with_team = subset.merge(
            assignments[["Frame", "BBoxKey", "Team"]].rename(columns={"Team": "ClusterTeam"}),
            on=["Frame", "BBoxKey"],
            how="left",
        )
        subset_with_team["BBox"] = subset_with_team["BBox"].apply(lambda box: [float(v) for v in box])
        frame_rows = subset_with_team[subset_with_team["Frame"] == target_frame]

        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = cap.read()
        cap.release()
        if not success or frame is None:
            raise RuntimeError(f"No se pudo leer el frame {target_frame} desde el video.")

        detections_list = []
        for row in frame_rows.itertuples(index=False):
            bbox = [int(round(v)) for v in row.BBox]
            detections_list.append(
                Detection(
                    frame=int(row.Frame),
                    class_name=str(row.ClassName),
                    bbox=tuple(bbox),
                    team=str(row.ClusterTeam) if pd.notna(row.ClusterTeam) else None,
                )
            )

        annotated = _annotate_frame(frame, detections_list)
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output_image), annotated)
    print(f"Imagen anotada guardada en: {args.output_image}")


if __name__ == "__main__":
    main()
