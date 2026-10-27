"""Zona restringida: de lo que pinta el usuario a una máscara, y quién la pisa.

La zona es una máscara booleana del tamaño del video (True = restringido) en vez de un
polígono, porque el usuario la pinta a mano: puede tener cualquier forma o ser varias zonas.
"""

import cv2
import numpy as np

NO_ZONE_MSG = "Pintá la zona restringida sobre la imagen antes de analizar."


def zone_from_layers(layers: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Combina las capas pintadas en el editor en una sola máscara del tamaño del video.

    Args:
        layers: Capas RGBA del editor. Un píxel es zona si su canal alfa es mayor que 0.
        shape: (alto, ancho) del video.

    Returns:
        Máscara booleana de forma ``shape``.

    Raises:
        ValueError: Si no hay nada pintado.
    """
    if not layers:
        raise ValueError(NO_ZONE_MSG)
    mask = np.any([layer[..., 3] > 0 for layer in layers], axis=0)
    if not mask.any():
        raise ValueError(NO_ZONE_MSG)
    if mask.shape != shape:
        # INTER_NEAREST mantiene los bordes nítidos (sí/no) en vez de difuminarlos.
        size = (shape[1], shape[0])  # OpenCV pide (ancho, alto)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return mask


def feet_in_zone(feet_xy: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Indica qué personas tienen los pies dentro de la zona.

    Se usan los pies (centro del borde inferior del recuadro) y no el recuadro entero porque,
    con la cámara en perspectiva, alguien que pasa delante de una puerta la tapa con el cuerpo
    sin estar en ella: los pies dicen dónde está parado de verdad.

    Args:
        feet_xy: Array (N, 2) con las coordenadas (x, y) de los pies de cada persona.
        zone: Máscara booleana (alto, ancho).

    Returns:
        Array booleano (N,): True si los pies de esa persona caen en la zona.
    """
    h, w = zone.shape
    # clip: un punto justo en el borde (x == ancho) se saldría del array al indexar.
    x = np.clip(feet_xy[:, 0].astype(int), 0, w - 1)
    y = np.clip(feet_xy[:, 1].astype(int), 0, h - 1)
    return zone[y, x]
