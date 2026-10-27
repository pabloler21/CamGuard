"""El análisis completo de un video: lee, detecta, decide quién es intruso, dibuja y resume.

No depende de la interfaz: informa errores con ``ValueError`` y el progreso con un callback,
así se puede usar desde Gradio, desde un script o desde los tests (con un tracker falso).
"""

import math
import tempfile
from collections.abc import Callable
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


@dataclass(frozen=True)
class Snapshot:
    """Captura del momento en que una persona aparece por primera vez dentro de la zona.

    Attributes:
        image: Cuadro anotado, en RGB (listo para mostrar).
        track_id: Número de seguimiento de la persona.
        time_s: Segundo del video.
    """

    image: np.ndarray
    track_id: int
    time_s: float


@dataclass(frozen=True)
class AnalysisResult:
    """Resultado de analizar un video.

    Attributes:
        video_path: mp4 anotado (H.264), en una carpeta temporal.
        events: Cada estadía de una persona en la zona, ordenadas por entrada.
        snapshots: Una captura por persona: la de su primera entrada.
    """

    video_path: Path
    events: list[Event]
    snapshots: list[Snapshot]


def analyze(
    video_path: str | Path,
    zone_layers: list[np.ndarray],
    make_tracker: Callable[[float], Tracker] = PersonTracker,
    progress: Callable[[float], None] | None = None,
    out_dir: Path | None = None,
) -> AnalysisResult:
    """Analiza un video y detecta quién entra a la zona restringida.

    Se analizan ``FPS`` cuadros por segundo y hasta ``MAX_SECONDS`` segundos (ver ``config``).

    Args:
        video_path: Video a analizar.
        zone_layers: Capas RGBA pintadas por el usuario (ver :func:`zone_from_layers`).
        make_tracker: Crea el tracker a partir de los fps analizados. Por defecto el real
            (RF-DETR + ByteTrack); los tests pasan uno falso.
        progress: Se llama después de cada cuadro con el avance, entre 0 y 1.
        out_dir: Dónde guardar el video anotado (en una subcarpeta propia). Por defecto, la
            carpeta temporal del sistema. La app pasa una que vacía al arrancar.

    Returns:
        El video anotado, los eventos y las capturas.

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
            for track_id in detections.tracker_id[inside].tolist():
                if track_id not in frames_in_zone:
                    snapshots.append(Snapshot(annotated[:, :, ::-1], track_id, i / fps))
                frames_in_zone.setdefault(track_id, []).append(i)
            writer.write(annotated)
            if progress:
                progress(min(1.0, (i + 1) / expected))

    return AnalysisResult(out_path, build_events(frames_in_zone, fps), snapshots)
