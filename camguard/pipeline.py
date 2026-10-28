"""El análisis completo de un video: lee, detecta, decide quién es intruso, dibuja y resume.

No depende de la interfaz: informa errores con ``ValueError`` y va devolviendo resultados
parciales a medida que avanza, así se puede usar desde Gradio, desde un script o desde los tests
(con un tracker falso).
"""

import math
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import supervision as sv

from camguard.annotate import annotate
from camguard.config import FPS, MAX_SECONDS
from camguard.detection import PersonTracker
from camguard.events import Event, build_events
from camguard.video import H264Writer, frame_limit, probe, sample_frames, sampling_step
from camguard.zone import points_in_zone, zone_from_layers


class Tracker(Protocol):
    """Lo que el pipeline necesita de un tracker: personas con número, cuadro a cuadro."""

    def update(self, frame_bgr: np.ndarray) -> sv.Detections:
        """Devuelve las personas del cuadro, con ``tracker_id``."""
        ...


# Aire que se deja alrededor de una persona al recortarla, como fracción de su alto y ancho.
CROP_MARGIN = 0.1


@dataclass(frozen=True)
class Snapshot:
    """Foto de una persona en el momento en que aparece por primera vez dentro de la zona.

    Attributes:
        image: La persona recortada del cuadro original (sin dibujos encima), en RGB.
        track_id: Número de seguimiento de la persona.
        time_s: Segundo del video.
    """

    image: np.ndarray
    track_id: int
    time_s: float


@dataclass(frozen=True)
class AnalysisResult:
    """Resultado de analizar un video, parcial mientras el análisis avanza.

    Attributes:
        progress: Avance, entre 0 y 1.
        events: Cada estadía de una persona en la zona, ordenadas por entrada. Mientras se
            analiza, la salida de quien sigue adentro todavía no es la definitiva.
        snapshots: Una foto por persona: la de su primera entrada.
        video_path: mp4 anotado (H.264), en una carpeta temporal. ``None`` hasta terminar.
    """

    progress: float
    events: list[Event]
    snapshots: list[Snapshot]
    video_path: Path | None = None


def crop_person(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Recorta a una persona del cuadro, con algo de aire alrededor."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
    h, w = frame.shape[:2]
    top, bottom = max(0, int(y1 - my)), min(h, int(y2 + my))
    left, right = max(0, int(x1 - mx)), min(w, int(x2 + mx))
    return frame[top:bottom, left:right]


def analyze(
    video_path: str | Path,
    zone_layers: list[np.ndarray],
    make_tracker: Callable[[float], Tracker] = PersonTracker,
    out_dir: Path | None = None,
) -> Iterator[AnalysisResult]:
    """Analiza un video y detecta quién entra a la zona restringida.

    Se analizan ``FPS`` cuadros por segundo y hasta ``MAX_SECONDS`` segundos (ver ``config``).
    Es un generador: devuelve el resultado parcial después de cada cuadro, para mostrar a los
    intrusos a medida que aparecen, y al final el completo, con el video.

    Args:
        video_path: Video a analizar.
        zone_layers: Capas RGBA pintadas por el usuario (ver :func:`zone_from_layers`).
        make_tracker: Crea el tracker a partir de los fps analizados. Por defecto el real
            (RF-DETR + ByteTrack); los tests pasan uno falso.
        out_dir: Dónde guardar el video anotado (en una subcarpeta propia). Por defecto, la
            carpeta temporal del sistema. La app pasa una que vacía al arrancar.

    Yields:
        Los resultados parciales y, último, el completo (con ``video_path``).

    Raises:
        ValueError: Si el video no se puede leer o no hay zona pintada.
    """
    info = probe(video_path)
    zone = zone_from_layers(zone_layers, (info.height, info.width))
    step = sampling_step(info.fps, FPS)
    fps = info.fps / step  # cuadros analizados por segundo
    max_frames = frame_limit(info, MAX_SECONDS)
    expected = max(1, math.ceil(max_frames / step))

    tracker = make_tracker(fps)
    # Carpeta temporal propia: el archivo no queda abierto (en Windows eso bloquea la escritura)
    # y se puede llamar "resultado.mp4", que es el nombre que ve el usuario si lo descarga.
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(tempfile.mkdtemp(prefix="camguard-", dir=out_dir)) / "resultado.mp4"
    frames_in_zone: dict[int, list[int]] = {}
    snapshots: list[Snapshot] = []

    with H264Writer(out_path, fps, info.width, info.height) as writer:
        for i, frame in enumerate(sample_frames(video_path, step, max_frames)):
            detections = tracker.update(frame)
            # Intruso = el centro de su cuerpo está sobre la zona (ver points_in_zone).
            centers = detections.get_anchors_coordinates(sv.Position.CENTER)
            inside = points_in_zone(centers, zone)
            annotated = annotate(frame, detections, inside, zone)
            for box, track_id in zip(
                detections.xyxy[inside], detections.tracker_id[inside], strict=True
            ):
                if track_id not in frames_in_zone:
                    photo = crop_person(frame, box)[:, :, ::-1]
                    snapshots.append(Snapshot(photo, int(track_id), i / fps))
                frames_in_zone.setdefault(int(track_id), []).append(i)
            writer.write(annotated)
            progress = min(1.0, (i + 1) / expected)
            yield AnalysisResult(progress, build_events(frames_in_zone, fps), list(snapshots))

    yield AnalysisResult(1.0, build_events(frames_in_zone, fps), snapshots, out_path)
