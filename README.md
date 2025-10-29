# FUTBOLTESIS

Repositorio de la tesis de maestría orientada a extraer métricas avanzadas de fútbol combinando detección de objetos, clustering cromático y homografía para proyectar jugadores y balón sobre un minimapa 2D.

El pipeline actual permite:

- Detectar jugadores, árbitros y balón frame a frame a partir de un checkpoint RF-DETR preentrenado.
- Separar automáticamente a los equipos mediante embeddings de apariencia.
- Estimar 57 keypoints de homografía (modelo HRNet) en cada frame.
- Proyectar detecciones al plano métrico del campo y renderizar un minimapa superpuesto al video.

---

## 1. Requisitos y configuración

### 1.1 Dependencias principales

- Python 3.10 o superior.
- PyTorch 2.x (utiliza la URL acorde a tu versión de CUDA).
- Bibliotecas utilizadas: `torch`, `torchvision`, `opencv-python`, `albumentations`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`, `ffmpeg-python` (opcional).

### 1.2 Entorno virtual recomendado

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt  # si prefieres una instalación centralizada
```

Si no usas `requirements.txt`, instala al menos:

```bash
pip install rfdetr opencv-python albumentations pandas numpy tqdm scikit-learn pillow matplotlib
```

---

## 2. Artefactos de trabajo (no versionados)

| Recurso | Descripción |
| --- | --- |
| `ModeloRF.pth` | Checkpoint RF-DETR entrenado para detectar `jugador`, `arbitro`, `balon`. |
| `VideoPruebaTesis.mp4` | Video de referencia para ejecutar el pipeline extremo a extremo. |
| `outputs/Homografia.pth` | Modelo HRNet (57 keypoints + canal de fondo) entrenado para homografía. |
| `outputs/detecciones_con_equipos.csv` | Ejemplo de detecciones con la columna `Team`. |

> Estos archivos viven en rutas ignoradas (`*.pth`, `*.mp4`, `*.csv`, `outputs/`).  
> Descárgalos desde el almacenamiento compartido o reprodúcelos ejecutando los scripts descritos en la sección 5.

---

## 3. Metodología general

1. Ejecutar inferencia sobre el video para generar detecciones frame a frame.
2. Etiquetar cada jugador como `local` o `visitante` mediante clustering de uniformes.
3. Estimar keypoints de homografía usando HRNet y obtener la matriz de proyección por frame.
4. Proyectar jugadores y balón al plano del campo.
5. Renderizar un minimapa 2D y superponerlo sobre el video original.
6. Re-empacar el video con códecs compatibles (H.264 + yuv420p) para facilitar la reproducción.

---

## 4. Estructura relevante

- `src/inference/`: inferencia RF-DETR, visualización y utilitarios.
- `src/clustering/`: embeddings + K-Means para etiquetar equipos.
- `src/homography/`: modelo HRNet, estimación de homografía y renderizado de minimapa.
- `outputs/`: carpeta de trabajo para CSVs, videos procesados y checkpoints locales (ignorados por Git).

---

## 5. Pipeline detallado

### 5.1 Inferencia sobre el video base

```bash
python -m src.inference.video_inference \
  --model ModeloRF.pth \
  --video VideoPruebaTesis.mp4 \
  --output outputs/detecciones_locales.csv \
  --threshold 0.5
```

- Genera un CSV con columnas `Frame`, `ClassName`, `BBox`, `Confidence`.
- Como atajo, `python -m src.inference.run_local` utiliza los artefactos incluidos y guarda el resultado en `outputs/detecciones_locales.csv`.

### 5.2 Clustering de uniformes

```bash
python -m src.clustering.run_local \
  --video VideoPruebaTesis.mp4 \
  --detections outputs/detecciones_locales.csv \
  --output outputs/detecciones_con_equipos.csv
```

- Recorta cada `jugador`, genera embeddings ResNet18 y aplica K-Means (`k=2`).
- El CSV agrega la columna `Team` (`BRASIL`/`COLOMBIA`); otras clases quedan vacías.

### 5.3 Visualización rápida de cajas

```bash
python -m src.inference.visualize_detections \
  --video VideoPruebaTesis.mp4 \
  --detections outputs/detecciones_con_equipos.csv \
  --output outputs/detecciones_visualizadas.mp4 \
  --use-ffmpeg
```

- Dibuja cajas y etiquetas de equipos para validar detecciones.
- Usa `--minimap` si tu CSV ya contiene columnas `FieldX`/`FieldY` (por ejemplo, tras proyectar posiciones).
- `--use-ffmpeg` exporta con `libx264`, lo que garantiza compatibilidad con VS Code y navegadores.

### 5.4 Probar homografía en un frame

```bash
python -m src.homography.render_minimap_frame \
  --checkpoint outputs/Homografia.pth \
  --video VideoPruebaTesis.mp4 \
  --frame-index 660 \
  --flip-y \
  --output outputs/minimap_frame660.png
```

- Permite depurar un frame aislado: dibuja keypoints detectados y minimapa renderizado.
- Usa `--flip-y` para alinear el minimapa con la orientación del video (validado manualmente).

### 5.5 Renderizar minimapa para todo el video

```bash
python -m src.homography.render_minimap_video \
  --checkpoint outputs/Homografia.pth \
  --video VideoPruebaTesis.mp4 \
  --detections outputs/detecciones_con_equipos.csv \
  --output outputs/videofinalconhomografia.mp4 \
  --flip-y \
  --confidence 0.05 \
  --min-matches 8
```

- Para cada frame: predice keypoints, estima homografía, proyecta detecciones y superpone minimapa.
- `--flip-y` evita invertir horizontalmente; valores positivos de correlación se corrigen automáticamente.
- `--min-matches` exige un mínimo de keypoints válidos antes de aceptar la homografía (sugerido ≥8).
- Exporta a resolución `960x540` (tamaño del modelo HRNet).

### 5.6 Re-encode compatible con VS Code

```bash
ffmpeg -y -i outputs/videofinalconhomografia.mp4 \
  -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p \
  outputs/videofinalconhomografia_vscode.mp4
```

- El reproductor integrado de VS Code requiere H.264 + `yuv420p`.
- El proceso es reproducible y debe documentarse como paso final del entregable.

---

## 6. Componentes clave del módulo de homografía

- `src/homography/config.py`: metadatos (dimensiones de entrada, stride, mapeo índice→nombre).
- `src/homography/field_layout.py`: definición métrica de los 57 puntos del campo profesional.
- `src/homography/model.py`: implementación de HRNet para inferencia (backbone + cabeza de heatmaps).
- `src/homography/predictor.py`: wrapper de carga del checkpoint `Homografia.pth` y ordenamiento por confianza.
- `src/homography/homography_estimator.py`: estima la matriz 3×3 (RANSAC) y proyecta puntos al plano del campo.
- `src/homography/minimap.py`: renderizador parametrizable (flip vertical/horizontal, colores por equipo, escala).
- `src/homography/render_minimap_frame.py` y `src/homography/render_minimap_video.py`: CLIs para depuración y producción.

---

## 7. Consideraciones metodológicas

- **Orientación del campo:** la cámara del video base obliga a invertir el eje Y (`--flip-y`) para que los movimientos coincidan con la realidad.
- **Keypoints mínimos:** si un frame no alcanza el umbral configurado, se reutiliza la última homografía válida para mantener continuidad.
- **Control de calidad:** inspecciona frames representativos (`render_minimap_frame`) antes de procesar el video completo.
- **Rendimiento:** procesar 2 245 frames tarda ~5 min en CPU; usa `--device cuda` cuando haya GPU disponible.
- **Reanudación segura:** si cancelas el script, elimina el MP4 parcial antes de volver a ejecutarlo para evitar archivos corruptos.

---

## 8. Flujo resumido (end-to-end)

```bash
# 1) Detecciones base
python -m src.inference.run_local

# 2) Clustering de equipos
python -m src.clustering.run_local

# 3) Minimapa completo
python -m src.homography.render_minimap_video \
  --checkpoint outputs/Homografia.pth \
  --video VideoPruebaTesis.mp4 \
  --detections outputs/detecciones_con_equipos.csv \
  --output outputs/videofinalconhomografia.mp4 \
  --flip-y

# 4) Re-encoder para VS Code
ffmpeg -y -i outputs/videofinalconhomografia.mp4 \
  -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p \
  outputs/videofinalconhomografia_vscode.mp4
```

---

## 9. Trabajo futuro

- Integrar un tracker (ByteTrack/DeepSORT) para mantener IDs persistentes y habilitar métricas de posesión.
- Añadir tests unitarios que validen la estimación de homografía y la orientación del minimapa.
- Extender el renderizador con heatmaps de ocupación o trayectorias individuales.
- Evaluar el modelo HRNet frente a diferentes ángulos de cámara y diseñar estrategias de refinamiento.

---

## 10. Referencias

- Chen et al., “RF-DETR: End-to-End Object Detection with Relation Fusion.” 2023.
- Sun et al., “High-Resolution Representations for Labeling Pixels and Regions.” (HRNet) 2019.
- Zhang et al., “ByteTrack: Multi-Object Tracking by Associating Every Detection Box.” 2021.
- FIFA Laws of the Game 2023/24 (dimensiones estándar del terreno).
- Hartley & Zisserman, “Multiple View Geometry in Computer Vision.” (homografías y DLT).
