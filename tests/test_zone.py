import numpy as np
import pytest

from camguard.zone import feet_in_zone, zone_from_layers


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
