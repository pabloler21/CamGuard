"""Lectura y escritura de video.

Junta los detalles de bajo nivel en un solo lugar: OpenCV entrega los cuadros en BGR (no RGB),
H.264 exige ancho y alto pares, y saltear cuadros con ``grab()`` es más rápido que decodificarlos.
Si mañana se lee de una cámara en vivo (RTSP), este es el único módulo que cambia.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

UNREADABLE_MSG = "No pude leer el video."


@dataclass(frozen=True)
class VideoInfo:
    """Datos básicos de un archivo de video.

    Attributes:
        fps: Cuadros por segundo del archivo.
        width: Ancho en píxeles.
        height: Alto en píxeles.
        frame_count: Cantidad de cuadros (0 si el archivo no lo informa).
    """

    fps: float
    width: int
    height: int
    frame_count: int


def probe(path: str | Path) -> VideoInfo:
    """Lee los datos básicos de un video.

    Raises:
        ValueError: Si el archivo no se puede abrir como video.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(UNREADABLE_MSG)
        return VideoInfo(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def first_frame(path: str | Path) -> np.ndarray:
    """Devuelve el primer cuadro del video en RGB (el orden que espera el navegador).

    Raises:
        ValueError: Si no se puede leer ningún cuadro.
    """
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(UNREADABLE_MSG)
    return frame[:, :, ::-1]


def sampling_step(source_fps: float, target_fps: float) -> int:
    """Cada cuántos cuadros del archivo hay que analizar uno para acercarse a ``target_fps``."""
    return max(1, round(source_fps / target_fps))


def frame_limit(info: VideoInfo, max_seconds: float) -> int:
    """Cuántos cuadros del archivo recorrer como máximo para no pasar de ``max_seconds``."""
    limit = math.floor(max_seconds * info.fps)
    return min(info.frame_count, limit) if info.frame_count > 0 else limit


def sample_frames(path: str | Path, step: int, max_frames: int) -> Iterator[np.ndarray]:
    """Recorre el video y devuelve uno de cada ``step`` cuadros, en BGR.

    Args:
        path: Archivo de video.
        step: Se devuelve el cuadro 0, el ``step``, el ``2*step``... (ver :func:`sampling_step`).
        max_frames: Cuadros del archivo a recorrer como máximo (ver :func:`frame_limit`).

    Yields:
        Cuadros BGR (alto, ancho, 3).
    """
    cap = cv2.VideoCapture(str(path))
    try:
        for idx in range(max_frames):
            if idx % step:
                # grab() avanza sin decodificar la imagen completa: mucho más rápido que read().
                if not cap.grab():
                    return
                continue
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


class H264Writer:
    """Escribe cuadros BGR en un mp4 H.264, el formato que reproducen todos los navegadores.

    El codec por defecto de OpenCV muchas veces no se reproduce en el navegador; PyAV trae
    libx264 incluido, así que no hace falta instalar ffmpeg. Se usa con ``with``::

        with H264Writer("salida.mp4", fps=5, width=1920, height=1080) as writer:
            writer.write(frame)
    """

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        # El formato de color yuv420p exige ancho y alto pares: si son impares se recorta 1 px.
        self._width, self._height = width // 2 * 2, height // 2 * 2
        self._container = av.open(str(path), "w")
        self._stream = self._container.add_stream(
            "libx264", rate=Fraction(fps).limit_denominator(1000)
        )
        self._stream.width, self._stream.height = self._width, self._height
        self._stream.pix_fmt = "yuv420p"

    def write(self, frame_bgr: np.ndarray) -> None:
        """Agrega un cuadro BGR al video."""
        frame = av.VideoFrame.from_ndarray(
            frame_bgr[: self._height, : self._width], format="bgr24"
        )
        self._container.mux(self._stream.encode(frame))

    def close(self) -> None:
        """Vacía los cuadros que el encoder tiene en buffer y cierra el archivo."""
        self._container.mux(self._stream.encode())
        self._container.close()

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
