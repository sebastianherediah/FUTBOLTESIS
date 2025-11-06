# FUTBOLTESIS

Repositorio de la tesis de maestría orientada a extraer métricas avanzadas de fútbol combinando detección de objetos, clustering cromático y homografía para proyectar jugadores y balón sobre un minimapa 2D.

El pipeline actual permite:

- Detectar jugadores, árbitros y balón frame a frame a partir de un checkpoint RF-DETR preentrenado.
- Separar automáticamente a los equipos mediante embeddings de apariencia.
- Estimar 57 keypoints de homografía (modelo HRNet) en cada frame.
- Proyectar detecciones al plano métrico del campo y superponer un minimapa informativo sobre el video.

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

> Estos archivos se almacenan en rutas ignoradas (`*.pth`, `*.mp4`, `*.csv`, `outputs/`). Descárgalos desde el almacenamiento compartido o reprodúcelos ejecutando los scripts de la sección 5.

---

## 3. Metodología general

1. Ejecutar inferencia sobre el video para generar detecciones frame a frame.
2. Etiquetar cada jugador como `local` o `visitante` mediante clustering de uniformes.
3. Estimar keypoints de homografía usando HRNet y obtener la matriz de proyección por frame.
4. Proyectar jugadores y balón al plano métrico del campo.
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

- Permite depurar un frame aislado: dibuja keypoints detectados y el minimapa generado.
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

- Para cada frame: predice keypoints, estima homografía, proyecta detecciones y superpone el minimapa.
- `--flip-y` evita inversiones horizontales; se corrige automáticamente la correlación negativa en eje X.
- `--min-matches` exige un mínimo de keypoints válidos antes de aceptar la homografía (sugerido ≥8).
- Exporta a resolución `960x540` (tamaño esperado por el modelo HRNet).

### 5.6 Re-encode compatible con VS Code

```bash
ffmpeg -y -i outputs/videofinalconhomografia.mp4 \
  -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p \
  outputs/videofinalconhomografia_vscode.mp4
```

- El reproductor integrado de VS Code requiere H.264 + `yuv420p`.
- Este paso debe documentarse como parte del entregable final para asegurar la compatibilidad.

---

## 6. Componentes clave del módulo de homografía

- `src/homography/config.py`: metadatos (dimensiones de entrada, stride del heatmap, mapeo índice→nombre).
- `src/homography/field_layout.py`: definición métrica de los 57 puntos del campo profesional.
- `src/homography/model.py`: implementación de HRNet para inferencia (backbone + cabeza de heatmaps).
- `src/homography/predictor.py`: wrapper que carga el checkpoint `Homografia.pth` y ordena por confianza.
- `src/homography/homography_estimator.py`: estima la matriz 3×3 (RANSAC) y proyecta puntos al plano del campo.
- `src/homography/minimap.py`: renderizador parametrizable (flip en ejes, colores según equipo, escala).
- `src/homography/render_minimap_frame.py` y `src/homography/render_minimap_video.py`: CLIs para depuración y producción.

---

## 7. Consideraciones metodológicas

- **Orientación del campo:** la cámara del video base obliga a invertir el eje Y (`--flip-y`) para alinear el minimapa con la jugada real.
- **Keypoints mínimos:** si un frame no alcanza el umbral establecido, se reutiliza la última homografía válida para mantener continuidad.
- **Control de calidad:** inspecciona frames representativos (`render_minimap_frame`) antes de procesar el video completo.
- **Rendimiento:** procesar 2 245 frames tarda ~5 minutos en CPU; usa `--device cuda` si hay GPU disponible.
- **Reanudación segura:** elimina el MP4 parcial antes de relanzar `render_minimap_video` si interrumpes la ejecución.

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

# 4) Re-encoder compatible VS Code
ffmpeg -y -i outputs/videofinalconhomografia.mp4 \
  -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p \
  outputs/videofinalconhomografia_vscode.mp4
```

---

## 9. Trabajo futuro

- Integrar un tracker (ByteTrack/DeepSORT) para mantener IDs persistentes y habilitar métricas de posesión y pases.
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
## Metodología acordada

Algoritmo de extracción de estadísticas por medio de técnicas de computer vision en un partido de futbol completo.

1. **Entrenamiento de detectores (YOLO y RF-DETR)**
   - Realizar fine-tuning de YOLO y RF-DETR sobre un dataset público que incluya anotaciones de jugadores, árbitros y balón.
   - Alinear esquemas de anotación entre ambos modelos para obtener inferencias consistentes frame a frame.
   - Documentar hiperparámetros relevantes (épocas, tasa de aprendizaje, augmentations) para garantizar reproducibilidad.
   - Utilizar el script `src/training/rfdetr_finetuner.py` para automatizar la descarga del dataset, el fine-tuning de RF-DETR y la exportación del modelo resultante.
2. **Clustering por uniformes para separar equipos**
   - Filtrar detecciones de la clase *jugador* y extraer descriptores visuales (color promedio del uniforme, embeddings de apariencia o features CLIP).
   - Utilizar la clase [`Cluster`](src/clustering/cluster.py) para generar embeddings consistentes y agrupar automáticamente en dos equipos (*local* y *visitante*).
   - Implementar mecanismos de corrección manual o reglas basadas en posición inicial para resolver posibles ambigüedades.
3. **Identificación individual de jugadores**
   - Emparejar detecciones sucesivas usando un tracker multi-objeto (ByteTrack, DeepSORT) para mantener IDs temporales.
   - Incorporar un módulo de re-identificación apoyado en uniformes y características físicas para estabilizar la identidad por todo el video.
   - Registrar un diccionario jugador-ID persistente que permita consultar historial y estadísticas individuales.
4. **Homografía 2D del campo**
   - Calibrar la cámara identificando puntos de control (esquinas, punto penal, círculo central) y estimar la homografía plano-cámara.
   - Proyectar las detecciones de jugadores y balón al plano 2D del campo para obtener coordenadas normalizadas.
   - Visualizar las trayectorias resultantes sobre una representación 2D para validar la consistencia geométrica.
5. **Extracción de estadísticas clave**
   - Analizar trayectorias y posesión del balón para contabilizar pases completados, tiempo de posesión por equipo y tiros al arco.
   - Detectar eventos combinando reglas basadas en proximidad/velocidad con clasificadores entrenados sobre secuencias etiquetadas.
   - Almacenar las métricas en una base estructurada que facilite consultas y visualizaciones posteriores.

### Identificación heurística de pases y posesión

El módulo [`src/analytics/passes.py`](src/analytics/passes.py) implementa la clase `PassAnalyzer`, pensada para trabajar con el DataFrame enriquecido tras el clustering y la asignación de IDs de jugador:

- Espera columnas `Frame`, `ClassName`, `BBox` y `Team` para las detecciones de jugadores, además de las detecciones del balón (`ClassName == "balon"`). Si existe `TrackId`, se utilizará para mejorar la continuidad entre frames; en caso contrario, el analizador crea identificadores heurísticos basados en la posición.
- Construye una línea temporal de posesión determinando qué jugador se encuentra más próximo al balón en cada frame y filtrando segmentos cortos en función del umbral `min_possession_frames`.
- Detecta pases cuando la posesión cambia entre jugadores diferentes del mismo equipo dentro de una ventana temporal definida por `max_gap_frames`.
- Devuelve un `PassAnalysisResult` con el listado de pases identificados, la posesión normalizada por equipo (fracción de frames con control) y la línea temporal de control para depuración.

```python
from analytics import PassAnalyzer

analyzer = PassAnalyzer(distance_threshold=70, min_possession_frames=3, max_gap_frames=8)
result = analyzer.analyze(detections)

print(result.passes.head())        # DataFrame con los pases detectados
print(result.possession)           # Posesión por equipo como fracción de frames con control
```

El enfoque es heurístico y sirve como punto de partida para iterar sobre umbrales o incorporar reglas adicionales (por ejemplo, validar que el balón recorra cierta distancia entre jugadores o combinarlo con homografía para descartar pases irreales).

### Renderizar el contador acumulado de pases

Para inspeccionar rápidamente el conteo de pases por equipo sin superponer bounding boxes ni homografía, utiliza el script [`src/inference/pass_counter_video.py`](src/inference/pass_counter_video.py). Genera un video con un recuadro en la esquina inferior izquierda mostrando el total acumulado para cada equipo:

```bash
python -m src.inference.pass_counter_video \
  --video VideoPruebaTesis.mp4 \
  --detections outputs/detecciones_con_equipos.csv \
  --output outputs/passes_counter.mp4 \
  --team-order COLOMBIA BRASIL
```

El CSV de detecciones debe incluir las columnas `Frame`, `ClassName`, `BBox` y `Team` (el campo `TrackId` es opcional) para que el analizador pueda identificar posesiones y pases válidos.

6. **Evaluación contra ground truth**
   - Comparar las estadísticas calculadas con registros oficiales o anotaciones manuales para validar precisión.
   - Calcular métricas de desempeño (precisión, recall, F1, error absoluto medio) por tipo de estadística.
   - Iterar sobre el pipeline ajustando umbrales y modelos según los resultados de evaluación.

## Plan de trabajo propuesto

1. **Definición de objetivos y métricas**
   - Precisar las estadísticas que se desean extraer (posesión, pases completados, tiros, recuperaciones, etc.).
   - Determinar las métricas de evaluación para cada estadística y los requisitos de precisión.
2. **Preparación de datos**
   - Recolectar partidos adicionales etiquetados para validar los modelos YOLO/RF-DETR y generar ground truth.
   - Anotar manualmente eventos clave (pases, tiros, goles) en un subconjunto para supervisión.
   - Generar homografías del campo para distintos estadios si se requiere normalizar posiciones.
3. **Pipeline de percepción**
   - Aplicar RF-DETR o YOLO al video para detectar jugadores, balón y árbitros en cada frame.
   - Ejecutar seguimiento multi-objeto (por ejemplo, ByteTrack o DeepSORT) usando embeddings extraídos del detector.
   - Realizar asociación temporal y resolver identidades de jugadores mediante reidentificación y uniformes.
4. **Estimación de pose y ubicación en el campo**
   - Calcular keypoints del balón y jugadores si el modelo lo soporta, o integrar un módulo de pose adicional.
   - Estimar la proyección al plano del campo mediante homografía para obtener coordenadas métricas.
   - Filtrar trayectorias con técnicas de suavizado (Kalman/particle filters) y corregir oclusiones.
5. **Detección de eventos**
   - Definir reglas o modelos de aprendizaje supervisado para eventos (pases, tiros, recuperaciones).
   - Usar la dinámica del balón y proximidad de jugadores para clasificar posesiones.
   - Identificar goles combinando detección de balón cruzando la línea y señalización del árbitro.
6. **Cálculo de estadísticas**
   - Integrar posesión por intervalos de tiempo y equipo.
   - Contabilizar tiros a puerta, pases completados, recuperaciones y duelos.
   - Derivar mapas de calor y métricas de distancia recorrida por jugador.
7. **Validación y evaluación**
   - Comparar las estadísticas generadas contra anotaciones manuales.
   - Calcular precisión, recall y F1 de cada evento detectado.
   - Ejecutar pruebas de robustez frente a diferentes condiciones de iluminación y ángulos de cámara.
8. **Optimización e implementación**
   - Optimizar inferencia del modelo (cuantización, TensorRT) para procesar el video en tiempo razonable.
   - Paralelizar el pipeline para aprovechar GPUs/CPUs.
   - Diseñar scripts reproducibles para procesamiento batch de partidos completos.
9. **Visualización y reporte**
   - Crear dashboards o reportes que muestren las estadísticas agregadas y clips destacados.
   - Guardar anotaciones sobre el video con bounding boxes y trayectorias para análisis cualitativo.
10. **Documentación y despliegue**
    - Documentar cada módulo (detección, tracking, eventos, métricas).
    - Preparar manual de uso e instrucciones para correr el pipeline con nuevos videos.

## Próximos pasos inmediatos

1. Validar rendimiento comparativo de los modelos fine-tuneados (YOLO y RF-DETR) en un subconjunto del video y analizar fallos.
2. Seleccionar e integrar un tracker multi-objeto robusto para jugadores y balón.
3. Diseñar un esquema inicial de detección de eventos simples (posesión, pases) para iterar rápidamente.

## Entrenamiento automatizado con Roboflow y RF-DETR

El script [`src/training/rfdetr_finetuner.py`](src/training/rfdetr_finetuner.py) plasma paso a paso el flujo compartido para descargar el dataset, entrenar y exportar RF-DETR:

1. **Descarga del dataset**: se crea un cliente de Roboflow con la API key `dcZKRj2xqwPU94CoAUcS`, se accede al workspace `sebas-xxi8x`, al proyecto `dataset-millonarios` y se descarga la versión 4 en formato COCO.
2. **Entrenamiento**: se instancia `RFDETRBase` y se ejecuta `model.train()` indicando la ruta del dataset, 15 épocas, batch size de 4, `grad_accum_steps=1` y tasa de aprendizaje `1e-4`.
3. **Visualización de métricas**: se abre la imagen `/content/output/metrics_plot.png` con `PIL.Image` para inspeccionar la curva de entrenamiento generada por RF-DETR.
4. **Exportación del mejor checkpoint**: se vuelve a instanciar `RFDETRBase` cargando los pesos `checkpoint_best_total.pth` almacenados en `/content/output/` y se ejecuta `model.export()` para producir el archivo ONNX listo para despliegue.

> **Nota:** El script corre estas instrucciones secuencialmente al importarse. Asegúrate de haber instalado `roboflow`, `rfdetr` y `Pillow`, además de contar con credenciales válidas sobre el workspace de Roboflow.

Para personalizar el flujo, modifica parámetros del proyecto, versión, hiperparámetros de entrenamiento o rutas de salida directamente en el script.

## Inferencia de video con RF-DETR

La clase [`VideoInference`](src/inference/video_inference.py) permite generar un dataset de detecciones a partir de un video completo:

- Carga un checkpoint de RF-DETR preservando los nombres de clases almacenados en el metadata del entrenamiento.
- Procesa frame a frame el video indicado, invocando `model.predict` con el umbral de confianza configurado.
- Devuelve un `pandas.DataFrame` con columnas `Frame`, `ClassName` y `BBox` listo para su análisis posterior.

El método `process` muestra una barra de progreso usando `tqdm` y asegura que las bounding boxes se serialicen como listas `[x1, y1, x2, y2]`, facilitando su almacenamiento en disco o en bases de datos.

## Clustering automático de equipos

El módulo [`src/clustering/cluster.py`](src/clustering/cluster.py) implementa la clase `Cluster`, diseñada para tomar las detecciones generadas por `VideoInference` y separar la clase *jugador* en los dos equipos presentes en el partido:

- Extrae los recortes de cada `BBox` perteneciente a la clase objetivo directamente sobre el video original.
- Genera embeddings de apariencia mediante un encoder `ResNet18` preentrenado en ImageNet y normaliza los vectores resultantes.
- Agrupa los embeddings con K-Means (`k=2`) y asigna etiquetas semánticas `local` y `visitante`, devolviendo un `DataFrame` con las columnas originales más la asignación de equipo y el identificador de clúster.

El método `cluster_players` retorna un objeto `ClusterResult` que incluye el `DataFrame` anotado, la matriz completa de embeddings y el modelo de K-Means utilizado, permitiendo inspeccionar los centroides o reutilizar el agrupamiento en pasos posteriores del pipeline.

## Uso con Visual Studio Code

Si prefieres que el flujo de trabajo se ejecute dentro de Visual Studio Code puedes conectar el repositorio utilizando las extensiones remotas oficiales. Las dos opciones más comunes son:

- **Remote - SSH:** abre VS Code en tu máquina, instala la extensión *Remote - SSH* y define una conexión al servidor donde resides este repositorio (por ejemplo, tu propia máquina o una VM en la nube). Una vez establecida la conexión, selecciona "Open Folder" y apunta a la carpeta `FUTBOLTESIS`. Así podrás editar archivos, ejecutar scripts en el terminal integrado y lanzar depuración directamente sobre el entorno remoto.
- **Dev Containers / Codespaces:** si trabajas con contenedores Docker, instala *Dev Containers* y abre el repositorio en un contenedor. Desde VS Code podrás acceder al mismo filesystem y ejecutar `python -m compileall src` u otros comandos tal como lo harías localmente.

En ambos casos recuerda configurar las mismas dependencias listadas anteriormente (`torch`, `opencv-python`, `pandas`, `tqdm`, `rfdetr`, etc.) en el entorno remoto. Al abrir el terminal integrado dentro de VS Code podrás seguir los pasos descritos en las secciones de entrenamiento, inferencia y análisis sin modificaciones adicionales.

## Referencias útiles

- RF-DETR: Chen et al., "RF-DETR: End-to-End Object Detection with Relation Fusion." 2023.
- YOLOv8: Jocher et al., "YOLOv8: A foundation model for real-time object detection." 2023.
- Seguimiento multi-objeto: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." 2021.
- Homografía de campos deportivos: DLT y calibración manual basada en puntos conocidos del terreno de juego.
