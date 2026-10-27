import numpy as np
import pytest
import supervision as sv

from camguard.config import PERSON_ID
from camguard.events import Event
from camguard.pipeline import analyze
from camguard.video import H264Writer, probe

SIZE = 100  # video de 100x100; la zona es el cuarto inferior derecho


class FakeTracker:
    """Reemplaza al modelo: una persona (#1) camina afuera, entra en los cuadros 2 a 5 y sale."""

    def __init__(self, fps):
        self.frame = 0

    def update(self, frame_bgr):
        x = 75 if 2 <= self.frame <= 5 else 25  # pies en (x, 75)
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
    progress = []
    result = analyze(video, zone_layers, make_tracker=FakeTracker, progress=progress.append)

    assert result.events == [Event(track_id=1, enter_s=0.4, leave_s=1.2)]
    assert [(s.track_id, s.time_s) for s in result.snapshots] == [(1, 0.4)]
    assert result.snapshots[0].image.shape == (SIZE, SIZE, 3)
    assert probe(result.video_path).frame_count == 10
    assert len(progress) == 10
    assert progress[-1] == 1.0


def test_analyze_respects_max_seconds(video, zone_layers, monkeypatch):
    monkeypatch.setattr("camguard.pipeline.MAX_SECONDS", 1)
    result = analyze(video, zone_layers, make_tracker=FakeTracker)
    assert probe(result.video_path).frame_count == 5


def test_analyze_without_zone_raises(video):
    with pytest.raises(ValueError, match="Pintá la zona"):
        analyze(video, [], make_tracker=FakeTracker)
