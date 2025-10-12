# FUTBOLTESIS

Algoritmo de extracción de estadísticas por medio de técnicas de computer vision en un partido de futbol completo.

## Metodología acordada

1. **Entrenamiento de detectores (YOLO y RF-DETR)**
   - Realizar fine-tuning de YOLO y RF-DETR sobre un dataset público que incluya anotaciones de jugadores, árbitros y balón.
   - Alinear esquemas de anotación entre ambos modelos para obtener inferencias consistentes frame a frame.
   - Documentar hiperparámetros relevantes (épocas, tasa de aprendizaje, augmentations) para garantizar reproducibilidad.
2. **Clustering por uniformes para separar equipos**
   - Filtrar detecciones de la clase *jugador* y extraer descriptores visuales (color promedio del uniforme, embeddings de apariencia o features CLIP).
   - Aplicar clustering no supervisado (por ejemplo, K-Means o GMM con K=2) para asignar cada jugador al equipo local o visitante.
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

## Referencias útiles

- RF-DETR: Chen et al., "RF-DETR: End-to-End Object Detection with Relation Fusion." 2023.
- YOLOv8: Jocher et al., "YOLOv8: A foundation model for real-time object detection." 2023.
- Seguimiento multi-objeto: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." 2021.
- Homografía de campos deportivos: DLT y calibración manual basada en puntos conocidos del terreno de juego.
