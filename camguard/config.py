"""Perillas de la demo: valores que se pueden ajustar sin tocar la lógica."""

# Cuadros analizados por segundo. Una persona caminando no cruza una zona en menos de 0,2 s,
# así que alcanza y es 5-6 veces más rápido que analizar los 25-30 cuadros del video.
# ponytail: CPU-only (~0.5 s/cuadro en 1080p); si es lento, exportar RF-DETR a ONNX +
# onnxruntime-directml para usar la GPU AMD.
FPS = 5

# Segundos de video que se analizan como máximo, para que en CPU no tarde una eternidad.
MAX_SECONDS = 60

# Confianza mínima para contar una detección. Más bajo aparecen "personas" falsas;
# más alto se pierden las que están lejos. Con cámaras altas el modelo da confianzas bajas
# incluso a personas claras: en el video de ejemplo, 0.4 detectaba al 70% de la gente dentro
# de la zona y 0.25 al 96%, sin sumar falsos (medido contra RF-DETR Large como referencia).
CONF = 0.25

# Si alguien "desaparece" de la zona menos que esto (el detector lo pierde un cuadro, alguien
# le pasa por delante), sigue siendo el mismo evento en vez de partirse en varios.
MAX_GAP_S = 1.0

# Id de "person" entre las 80 clases COCO que reconoce RF-DETR.
PERSON_ID = 1
