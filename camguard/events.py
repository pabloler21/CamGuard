"""Eventos de intrusión: de "en qué cuadros estuvo cada persona en la zona" a entradas y salidas."""

from dataclasses import dataclass

from camguard.config import MAX_GAP_S


@dataclass(frozen=True)
class Event:
    """Una estadía continua de una persona dentro de la zona.

    Attributes:
        track_id: Número de seguimiento de la persona (el "#N" del video).
        enter_s: Segundo del video en que entró (o en que se la vio adentro por primera vez).
        leave_s: Segundo del video en que salió.
    """

    track_id: int
    enter_s: float
    leave_s: float

    @property
    def duration_s(self) -> float:
        """Segundos que estuvo dentro de la zona."""
        return round(self.leave_s - self.enter_s, 1)


def build_events(
    frames_in_zone: dict[int, list[int]], fps: float, max_gap_s: float = MAX_GAP_S
) -> list[Event]:
    """Agrupa los cuadros en que cada persona estuvo en la zona en eventos de entrada y salida.

    Mientras el salto entre dos cuadros consecutivos sea de ``max_gap_s`` o menos, es el mismo
    evento: así un cuadro en que el detector "pierde" a la persona no parte una entrada en varias.
    Si el salto es mayor, la persona salió y volvió: son dos eventos.

    Args:
        frames_in_zone: Para cada persona, los índices (crecientes) de los cuadros analizados
            en que estuvo sobre la zona.
        fps: Cuadros analizados por segundo, para pasar de índices a segundos.
        max_gap_s: Tolerancia en segundos antes de considerar que la persona salió.

    Returns:
        Los eventos, ordenados por segundo de entrada.
    """
    max_gap = max(1, round(max_gap_s * fps))
    events = []
    for track_id, frames in frames_in_zone.items():
        start = prev = frames[0]
        for frame in [*frames[1:], None]:  # el None final cierra el último evento
            if frame is not None and frame - prev <= max_gap:
                prev = frame
                continue
            # prev + 1: si estuvo un solo cuadro, igual estuvo 1/fps segundos, no 0.
            events.append(Event(track_id, round(start / fps, 1), round((prev + 1) / fps, 1)))
            start = prev = frame
    return sorted(events, key=lambda e: e.enter_s)
