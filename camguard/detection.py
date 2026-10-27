"""Detección y seguimiento de personas.

RF-DETR (Roboflow, Apache-2.0) encuentra personas en cada cuadro; ByteTrack les asigna un número
que se mantiene entre cuadros, para poder decir "el #3 entró a los 12 s y salió a los 18 s".
Este es el único módulo que conoce el modelo: cambiarlo (por ejemplo a una versión ONNX) no toca
el resto del código.
"""

from functools import cache

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from camguard.config import CONF, PERSON_ID


@cache
def load_model():
    """Carga RF-DETR Small una sola vez por proceso.

    La primera vez descarga los pesos (~368 MB) a ``~/.roboflow/models``.
    """
    # Import diferido: rfdetr arrastra PyTorch (~14 s de import) y los tests no lo necesitan.
    from rfdetr import RFDETRSmall

    return RFDETRSmall()


class PersonTracker:
    """Detecta personas en cada cuadro y les asigna un número de seguimiento estable.

    Se crea uno nuevo por cada video analizado, para que la numeración arranque de cero.

    Args:
        fps: Cuadros por segundo que va a recibir (los analizados, no los del archivo). ByteTrack
            lo usa para saber cuánto tiempo "recordar" a una persona que dejó de verse.
        conf: Confianza mínima de detección.
    """

    def __init__(self, fps: float, conf: float = CONF):
        self._model = load_model()
        self._conf = conf
        # El umbral de activación por defecto (0.7) tardaría en "confirmar" a las personas,
        # y minimum_consecutive_frames=1 les da número desde el primer cuadro en que aparecen.
        self._tracker = ByteTrackTracker(
            frame_rate=fps, track_activation_threshold=conf, minimum_consecutive_frames=1
        )

    def update(self, frame_bgr: np.ndarray) -> sv.Detections:
        """Procesa el siguiente cuadro del video.

        Args:
            frame_bgr: Cuadro en BGR, como lo entrega OpenCV.

        Returns:
            Las personas detectadas, con ``tracker_id`` asignado. Las que el tracker todavía
            no confirmó (``tracker_id == -1``) se descartan.
        """
        # El modelo espera RGB; .copy() porque la vista invertida tiene strides negativos.
        detections = self._model.predict(frame_bgr[:, :, ::-1].copy(), threshold=self._conf)
        detections = self._tracker.update(detections[detections.class_id == PERSON_ID])
        return detections[detections.tracker_id != -1]
