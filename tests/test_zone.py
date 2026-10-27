import time

import cv2
import numpy as np
import pytest

from camguard.zone import feet_in_zone, fill_outlines, zone_from_layers


def outline(gap=0):
    """Máscara 200x200 con el contorno de un cuadrado (50,50)-(150,150), trazo de 10 px.

    `gap` deja un hueco de ese ancho en el lado de arriba, como un trazo a mano que no cierra.
    """
    mask = np.zeros((200, 200), np.uint8)
    cv2.rectangle(mask, (50, 50), (150, 150), 1, 10)
    if gap:
        mask[40:60, 100 : 100 + gap] = 0
    return mask.astype(bool)


@pytest.mark.parametrize("gap", [pytest.param(0, id="cerrado"), pytest.param(15, id="con-hueco")])
def test_fill_outlines_fills_the_drawn_contour(gap):
    zone = fill_outlines(outline(gap))
    assert zone[100, 100]  # centro del cuadrado: adentro
    assert zone[100, 53] and zone[100, 147]  # justo adentro del borde marcado
    # justo afuera: la mitad externa del trazo no cuenta como zona
    assert not zone[100, 46] and not zone[100, 154]


def test_fill_outlines_keeps_a_painted_area_as_is():
    mask = np.zeros((200, 200), bool)
    mask[50:150, 50:150] = True
    assert np.array_equal(fill_outlines(mask), mask)


def test_fill_outlines_is_fast_on_a_large_painted_area():
    # Regresión: con un área grande el "grosor de trazo" estimado es enorme y cerrar huecos con
    # ese radio tardaba ~100 s en 1080p.
    mask = np.zeros((1080, 1920), bool)
    mask[594:, 960:] = True
    start = time.perf_counter()
    assert np.array_equal(fill_outlines(mask), mask)
    assert time.perf_counter() - start < 2


def test_zone_from_layers_fills_a_drawn_contour():
    layer = np.zeros((200, 200, 4), np.uint8)
    layer[outline()] = (255, 0, 0, 255)
    assert zone_from_layers([layer], (200, 200))[100, 100]


def make_layer(shape, painted=None):
    """Capa RGBA del editor; `painted` es la región (slice) pintada."""
    layer = np.zeros((*shape, 4), np.uint8)
    if painted is not None:
        layer[painted] = (255, 0, 0, 255)
    return layer


@pytest.fixture
def zone():
    """Imagen de 10x10 con el cuarto inferior derecho restringido."""
    mask = np.zeros((10, 10), bool)
    mask[5:, 5:] = True
    return mask


@pytest.mark.parametrize(
    ("feet", "expected"),
    [
        pytest.param((7, 7), True, id="adentro"),
        pytest.param((2, 2), False, id="afuera"),
        pytest.param((7, 2), False, id="misma-columna-pero-arriba"),
        pytest.param((5, 5), True, id="borde-de-la-zona"),
        pytest.param((9.9, 9.9), True, id="decimales"),
        pytest.param((50, 50), True, id="fuera-de-la-imagen-se-lleva-al-borde"),
    ],
)
def test_feet_in_zone(zone, feet, expected):
    assert feet_in_zone(np.array([feet]), zone).tolist() == [expected]


def test_feet_in_zone_without_people(zone):
    assert feet_in_zone(np.empty((0, 2)), zone).tolist() == []


def test_zone_from_layers_combines_all_layers():
    layers = [make_layer((4, 4), np.s_[:2, :2]), make_layer((4, 4), np.s_[2:, 2:])]
    mask = zone_from_layers(layers, (4, 4))
    assert mask.sum() == 8
    assert mask[0, 0] and mask[3, 3] and not mask[0, 3]


@pytest.mark.parametrize(
    "layers",
    [pytest.param([], id="sin-capas"), pytest.param([make_layer((4, 4))], id="capa-vacia")],
)
def test_zone_from_layers_requires_paint(layers):
    with pytest.raises(ValueError, match="Pintá la zona"):
        zone_from_layers(layers, (4, 4))


def test_zone_from_layers_resizes_to_video_size():
    mask = zone_from_layers([make_layer((4, 4), np.s_[2:, :])], (8, 6))
    assert mask.shape == (8, 6)
    assert mask[4:].all()
    assert not mask[:4].any()
