"""CamGuard: detecta personas que entran a una zona restringida de un video.

Todo corre local, en CPU: el video nunca sale de la máquina.

Módulos:
    config: perillas ajustables (fps analizados, tope de duración, confianza).
    zone: de lo que pinta el usuario a una máscara, y quién está sobre ella.
    events: de cuadros sueltos a eventos de entrada y salida.
    video: lectura submuestreada y escritura H.264.
    detection: detección (RF-DETR) y seguimiento (ByteTrack) de personas.
    annotate: dibujo de la zona y los recuadros sobre cada cuadro.
    pipeline: orquesta todo el análisis; no depende de la interfaz.
    ui: la página web (Gradio).
"""
