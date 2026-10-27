"""Dibujo sobre cada cuadro: la zona en rojo translúcido y un recuadro por persona."""

import numpy as np
import supervision as sv

ZONE_COLOR_BGR = np.array([0, 0, 255])  # rojo (OpenCV usa BGR)
ZONE_OPACITY = 0.4


def annotate(
    frame_bgr: np.ndarray, detections: sv.Detections, inside: np.ndarray, zone: np.ndarray
) -> np.ndarray:
    """Dibuja la zona, las personas de afuera en verde y los intrusos en rojo.

    Args:
        frame_bgr: Cuadro original en BGR. No se modifica.
        detections: Personas detectadas, con ``tracker_id``.
        inside: Array booleano: True para las personas que están sobre la zona.
        zone: Máscara booleana de la zona.

    Returns:
        Una copia del cuadro, anotada, en BGR.
    """
    out = frame_bgr.copy()
    # Mezcla en vez de pintar sólido, para que se vea lo que hay debajo de la zona.
    out[zone] = ((1 - ZONE_OPACITY) * out[zone] + ZONE_OPACITY * ZONE_COLOR_BGR).astype(np.uint8)
    groups = (
        (~inside, sv.Color.GREEN, "persona #{}"),
        (inside, sv.Color.RED, "INTRUSO #{}"),
    )
    for mask, color, label in groups:
        group = detections[mask]
        out = sv.BoxAnnotator(color=color, thickness=3).annotate(out, group)
        labels = [label.format(track_id) for track_id in group.tracker_id]
        out = sv.LabelAnnotator(color=color).annotate(out, group, labels=labels)
    return out
