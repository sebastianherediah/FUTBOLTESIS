# FUTBOLTESIS

Algoritmo de extracción de estadísticas por medio de técnicas de computer vision en un partido de futbol completo.

## Metodología acordada

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

## Referencias útiles

- RF-DETR: Chen et al., "RF-DETR: End-to-End Object Detection with Relation Fusion." 2023.
- YOLOv8: Jocher et al., "YOLOv8: A foundation model for real-time object detection." 2023.
- Seguimiento multi-objeto: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." 2021.
- Homografía de campos deportivos: DLT y calibración manual basada en puntos conocidos del terreno de juego.
