import numpy as np
import pytest
import supervision as sv

from camguard.config import PERSON_ID
from camguard.events import Event
from camguard.pipeline import analyze
from camguard.video import H264Writer, probe

SIZE = 100  # video de 100x100; la zona es el cuarto inferior derecho


def last(results):
    """El resultado final de analyze, que va devolviendo parciales."""
    *_, result = results
    return result


class FakeTracker:
    """Reemplaza al modelo: una persona (#1) camina afuera, entra en los cuadros 2 a 5 y sale."""

    def __init__(self, fps):
        self.frame = 0

    def update(self, frame_bgr):
        x = 75 if 2 <= self.frame <= 5 else 25  # centro del cuerpo en (x, 65)
        self.frame += 1
        # Como el tracker real: solo personas (class_id = PERSON_ID) y con número de seguimiento.
        return sv.Detections(
            xyxy=np.array([[x - 5, 55.0, x + 5, 75.0]]),
            class_id=np.array([PERSON_ID]),
            tracker_id=np.array([1]),
        )


@pytest.fixture
def video(tmp_path):
    """2 s de video gris a 5 fps (10 cuadros): se analizan todos."""
    path = tmp_path / "video.mp4"
    with H264Writer(path, fps=5, width=SIZE, height=SIZE) as writer:
        for _ in range(10):
            writer.write(np.full((SIZE, SIZE, 3), 128, np.uint8))
    return path


@pytest.fixture
def zone_layers():
    layer = np.zeros((SIZE, SIZE, 4), np.uint8)
    layer[50:, 50:] = (255, 0, 0, 255)
    return [layer]


def test_analyze_detects_the_intrusion(video, zone_layers):
    *partials, result = analyze(video, zone_layers, make_tracker=FakeTracker)

    assert result.events == [Event(track_id=1, enter_s=0.4, leave_s=1.2)]
    assert [(s.track_id, s.time_s) for s in result.snapshots] == [(1, 0.4)]
    assert probe(result.video_path).frame_count == 10
    # Un parcial por cuadro, sin video; el intruso aparece desde el cuadro en que entra.
    assert [p.progress for p in partials] == [(i + 1) / 10 for i in range(10)]
    assert all(p.video_path is None for p in partials)
    assert [len(p.events) for p in partials] == [0, 0] + [1] * 8


def test_snapshot_is_the_person_cropped(video, zone_layers):
    photo = last(analyze(video, zone_layers, make_tracker=FakeTracker)).snapshots[0].image
    # Recuadro de 10x20 en (70, 55), con 10% de aire: 1 px a los costados, 2 arriba y abajo.
    assert photo.shape == (24, 12, 3)


class StraddlingTracker:
    """Una persona parada sobre el borde de arriba: pies apenas adentro, cuerpo afuera."""

    def __init__(self, fps):
        pass

    def update(self, frame_bgr):
        return sv.Detections(
            xyxy=np.array([[65.0, 20.0, 85.0, 55.0]]),  # centro en y=37 (afuera); pies en y=55
            class_id=np.array([PERSON_ID]),
            tracker_id=np.array([1]),
        )


def test_person_counts_by_body_center_not_feet(video, zone_layers):
    # En la imagen se la ve afuera de la zona, que es lo que importa a quien mira el video.
    result = last(analyze(video, zone_layers, make_tracker=StraddlingTracker))
    assert result.events == []


def test_analyze_writes_the_result_inside_out_dir(video, zone_layers, tmp_path):
    # Así la app puede borrar todos los videos generados (son grabaciones reales).
    out_dir = tmp_path / "resultados"
    result = last(analyze(video, zone_layers, make_tracker=FakeTracker, out_dir=out_dir))
    assert out_dir in result.video_path.parents


def test_analyze_respects_max_seconds(video, zone_layers, monkeypatch):
    monkeypatch.setattr("camguard.pipeline.MAX_SECONDS", 1)
    result = last(analyze(video, zone_layers, make_tracker=FakeTracker))
    assert probe(result.video_path).frame_count == 5


def test_analyze_without_zone_raises(video):
    with pytest.raises(ValueError, match="Pintá la zona"):
        next(analyze(video, [], make_tracker=FakeTracker))
