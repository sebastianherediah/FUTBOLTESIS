"""Conveniencia para ejecutar inferencia con los artefactos locales del proyecto."""

from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    # Permitir ejecución directa `python src/inference/run_local.py`.
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.inference.video_inference import VideoInference  # type: ignore
else:
    from .video_inference import VideoInference

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = BASE_DIR / "ModeloRF.pth"
DEFAULT_VIDEO = BASE_DIR / "VideoPruebaTesis.mp4"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "detecciones_locales.csv"


def run(threshold: float = 0.5) -> Path:
    """Procesa el video local usando el modelo preentrenado disponible en el repositorio.

    Parameters
    ----------
    threshold:
        Confianza mínima para conservar detecciones. Rango [0, 1].

    Returns
    -------
    Path
        Ruta al archivo CSV generado con las detecciones.
    """

    if not DEFAULT_MODEL.exists():
        raise FileNotFoundError(
            f"No se encontró el checkpoint esperado en {DEFAULT_MODEL}. "
            "Verifica que 'ModeloRF.pth' esté en la raíz del repositorio."
        )

    if not DEFAULT_VIDEO.exists():
        raise FileNotFoundError(
            f"No se encontró el video esperado en {DEFAULT_VIDEO}. "
            "Verifica que 'VideoPruebaTesis.mp4' esté en la raíz del repositorio."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold debe estar en el rango [0, 1]")

    inference = VideoInference(str(DEFAULT_MODEL))
    detections = inference.process(str(DEFAULT_VIDEO), threshold=threshold)

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    detections.to_csv(DEFAULT_OUTPUT, index=False)

    return DEFAULT_OUTPUT


if __name__ == "__main__":
    output_path = run()
    print(f"Detecciones generadas en: {output_path}")
