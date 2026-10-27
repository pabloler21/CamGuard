"""Zona restringida: de lo que pinta el usuario a una máscara, y quién la pisa.

La zona es una máscara booleana del tamaño del video (True = restringido) en vez de un
polígono, porque el usuario la pinta a mano: puede tener cualquier forma o ser varias zonas.
"""

import cv2
import numpy as np

NO_ZONE_MSG = "Pintá la zona restringida sobre la imagen antes de analizar."

# Hueco máximo (en píxeles) que se cierra en un contorno dibujado a mano antes de rellenarlo.
# También acota el costo: cerrar con un radio grande sobre un área pintada tarda minutos.
MAX_GAP_PX = 60


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
    return fill_outlines(mask)


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def fill_outlines(mask: np.ndarray) -> np.ndarray:
    """Si la zona se marcó dibujando su contorno, devuelve el área encerrada.

    Lo natural es marcar un área dibujando su borde, pero solo lo pintado cuenta como zona:
    con un contorno, quien tocaba la línea por afuera era "intruso" y quien estaba adentro no.
    Medido en el video de ejemplo, con contorno se detectaba solo al ~15% de los de adentro.

    Pasos:
        1. Cierra huecos de hasta ~2 grosores de trazo, con un máximo de ``MAX_GAP_PX`` (un
           contorno a mano rara vez cierra justo).
        2. Rellena lo que quedó encerrado.
        3. Recorta medio trazo hacia adentro: el trazo se dibuja centrado sobre el borde que
           se quiso marcar, así que su mitad externa ya está afuera.

    Si no hay nada encerrado (se pintó el área entera), devuelve la máscara sin cambios.

    Args:
        mask: Máscara booleana con lo pintado.

    Returns:
        Máscara booleana de la zona.
    """
    painted = mask.astype(np.uint8)
    # Medio grosor del trazo: la mayor distancia de un píxel pintado al borde de lo pintado.
    # ponytail: asume trazos de grosor parejo; un contorno junto a un área grande pintada
    # sobreestima el grosor (si pasa, medir el grosor solo sobre los trazos finos).
    half_stroke = max(1, int(cv2.distanceTransform(painted, cv2.DIST_L2, 5).max()))
    closed = cv2.morphologyEx(painted, cv2.MORPH_CLOSE, _disk(min(2 * half_stroke, MAX_GAP_PX)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(painted)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    if filled.sum() == closed.sum():  # nada encerrado: se pintó el área entera
        return mask
    return cv2.erode(filled, _disk(half_stroke)).astype(bool)


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
