"""Clustering de jugadores en dos equipos usando embeddings de apariencia."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch import nn

from PIL import Image
from tqdm import tqdm

try:  # pragma: no cover
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:  # pragma: no cover - import guard
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError as exc:  # pragma: no cover - explicit error for missing optional deps
    raise ImportError(
        "Se requiere torchvision para utilizar Cluster. Instálalo antes de continuar."
    ) from exc


@dataclass
class ClusterResult:
    """Resultado del proceso de clustering."""

    assignments: pd.DataFrame
    embeddings: np.ndarray
    model: KMeans


class Cluster:
    """Agrupa jugadores detectados en dos equipos (local y visitante).

    El flujo parte de las detecciones frame a frame generadas por :class:`VideoInference`
    o cualquier otro detector que produzca un :class:`pandas.DataFrame` con columnas
    ``Frame``, ``ClassName`` y ``BBox``. Para cada bounding box de la clase ``jugador``
    se recorta la imagen correspondiente, se calcula un embedding de apariencia con un
    encoder convolucional preentrenado en ImageNet y finalmente se ejecuta un algoritmo
    de *clustering* (K-Means) para separar a los jugadores en dos grupos.

    Parameters
    ----------
    encoder_name:
        Nombre del encoder soportado. Actualmente solo ``"resnet18"``.
    batch_size:
        Número de recortes procesados simultáneamente al generar embeddings.
    device:
        Dispositivo sobre el que ejecutar el encoder. Por defecto detecta CUDA si está
        disponible y en caso contrario utiliza CPU.
    random_state:
        Semilla utilizada por K-Means para garantizar reproducibilidad.
    """

    SUPPORTED_ENCODERS = ("resnet18",)

    def __init__(
        self,
        encoder_name: str = "resnet18",
        *,
        batch_size: int = 32,
        device: Optional[str] = None,
        random_state: int = 0,
        show_progress: bool = False,
    ) -> None:
        if encoder_name not in self.SUPPORTED_ENCODERS:
            raise ValueError(
                f"Encoder '{encoder_name}' no soportado. Opciones: {self.SUPPORTED_ENCODERS}"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = max(1, batch_size)
        self.random_state = random_state
        self.show_progress = bool(show_progress)
        self._psutil_process = psutil.Process() if (self.show_progress and psutil is not None) else None
        if self._psutil_process is not None:
            self._psutil_process.cpu_percent(None)

        self.encoder: nn.Module
        self.transform: transforms.Compose
        self.encoder, self.transform = self._build_encoder(encoder_name)
        self.encoder.to(self.device)
        self.encoder.eval()

    def _build_encoder(
        self, encoder_name: str
    ) -> Tuple[nn.Module, transforms.Compose]:
        """Crea el backbone de extracción de embeddings y sus transformaciones."""

        if encoder_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            model.fc = nn.Identity()
            transform = weights.transforms()
            return model, transform

        raise AssertionError("Encoder inesperado. Esta línea no debería alcanzarse")

    def cluster_players(
        self,
        detections: pd.DataFrame,
        video_path: str,
        *,
        player_class: str = "jugador",
    ) -> ClusterResult:
        """Genera embeddings para jugadores y aplica K-Means con ``k=2``.

        Parameters
        ----------
        detections:
            DataFrame con las columnas ``Frame``, ``ClassName`` y ``BBox``.
        video_path:
            Ruta al video de origen utilizado para recortar los jugadores.
        player_class:
            Nombre de la clase que identifica a los jugadores. La comparación es
            insensible a mayúsculas/minúsculas.

        Returns
        -------
        ClusterResult
            Objeto con el DataFrame de asignaciones, la matriz de embeddings y el
            modelo de K-Means entrenado.
        """

        player_rows = detections[
            detections["ClassName"].str.lower() == player_class.lower()
        ].copy()
        if player_rows.empty:
            raise ValueError(
                "El DataFrame de detecciones no contiene ejemplos de la clase especificada"
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        embeddings: List[np.ndarray] = []
        metadata: List[Dict[str, object]] = []
        progress_bar = (
            tqdm(
                total=int(player_rows["Frame"].nunique()),
                desc="Clustering jugadores",
                unit="frame",
            )
            if (self.show_progress and not player_rows.empty)
            else None
        )

        try:
            for frame_idx, rows in player_rows.groupby("Frame"):
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx)):
                    raise RuntimeError(
                        f"No se pudo posicionar el video en el frame {frame_idx}"
                    )

                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError(
                        f"No se pudo leer el frame {frame_idx} desde el video"
                    )

                crops: List[np.ndarray] = []
                frame_metadata: List[Dict[str, object]] = []
                for bbox, class_name in zip(rows["BBox"], rows["ClassName"]):
                    crop = self._extract_crop(frame, bbox)
                    if crop is None:
                        continue
                    crops.append(crop)
                    frame_metadata.append(
                        {
                            "Frame": int(frame_idx),
                            "ClassName": str(class_name),
                            "BBox": bbox,
                        }
                    )

                if not crops:
                    continue

                crop_embeddings = self._encode_crops(crops)

                embeddings.extend(crop_embeddings)
                metadata.extend(frame_metadata)
                if progress_bar is not None:
                    progress_bar.set_postfix(self._progress_postfix(crops=len(crops)), refresh=False)
                    progress_bar.update(1)
        finally:
            cap.release()
            if progress_bar is not None:
                progress_bar.close()

        if not embeddings:
            raise RuntimeError("No fue posible generar embeddings a partir de los recortes")

        feature_matrix = np.vstack(embeddings)

        kmeans = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)

        ordering = np.argsort(kmeans.cluster_centers_.sum(axis=1))
        team_names = {
            ordering[0]: "local",
            ordering[1]: "visitante",
        }
        teams = [team_names[label] for label in labels]

        assignments = pd.DataFrame(metadata)
        assignments["Team"] = teams
        assignments["ClusterId"] = labels

        return ClusterResult(assignments=assignments, embeddings=feature_matrix, model=kmeans)
    def _progress_postfix(self, *, crops: int) -> Dict[str, str]:
        info: Dict[str, str] = {"crops": str(crops)}
        if self._psutil_process is not None:
            mem_gb = self._psutil_process.memory_info().rss / (1024**3)
            info["mem_gb"] = f"{mem_gb:.2f}"
        return info

    def _extract_crop(
        self, frame: np.ndarray, box: Iterable[float]
    ) -> Optional[np.ndarray]:
        """Recorta una caja y devuelve la imagen en BGR si es válida."""

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._sanitize_box(box, width, height)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _sanitize_box(
        self, box: Iterable[float], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        """Convierte la caja a coordenadas enteras válidas dentro del frame."""

        x1, y1, x2, y2 = map(float, box)
        x1 = max(int(np.floor(x1)), 0)
        y1 = max(int(np.floor(y1)), 0)
        x2 = min(int(np.ceil(x2)), width - 1)
        y2 = min(int(np.ceil(y2)), height - 1)
        return x1, y1, x2, y2

    def _encode_crops(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Obtiene embeddings normalizados para cada recorte."""

        if not crops:
            return []

        tensors: List[torch.Tensor] = []
        for crop in crops:
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            tensor = self.transform(pil_image)
            tensors.append(tensor)

        embeddings: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[start : start + self.batch_size]).to(self.device)
                feats = self.encoder(batch)
                feats = nn.functional.normalize(feats, p=2, dim=1)
                embeddings.extend(feats.cpu().numpy())
        return embeddings
