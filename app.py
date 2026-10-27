"""CamGuard: detecta personas que entran a una zona restringida de un video.

Todo corre local, en CPU. El video nunca sale de la máquina.
"""

import numpy as np

MAX_GAP_S = 1.0  # si alguien "desaparece" menos que esto, sigue siendo el mismo evento


def feet_in_zone(feet_xy, zone):
    """True para cada punto (x, y) que cae dentro de la máscara de zona."""
    h, w = zone.shape
    x = np.clip(feet_xy[:, 0].astype(int), 0, w - 1)
    y = np.clip(feet_xy[:, 1].astype(int), 0, h - 1)
    return zone[y, x]


def build_events(frames_in_zone, fps, max_gap_s=MAX_GAP_S):
    """{id: [índices de cuadro dentro de la zona]} -> filas [id, entró, salió, duración] en segundos."""
    max_gap = max(1, round(max_gap_s * fps))
    rows = []
    for tid, frames in frames_in_zone.items():
        start = prev = frames[0]
        for f in frames[1:] + [None]:
            if f is not None and f - prev <= max_gap:
                prev = f
                continue
            enter, leave = start / fps, (prev + 1) / fps
            rows.append([tid, round(enter, 1), round(leave, 1), round(leave - enter, 1)])
            start = prev = f
    return sorted(rows, key=lambda r: r[1])
