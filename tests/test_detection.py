import numpy as np
import supervision as sv

from camguard.config import PERSON_ID
from camguard.detection import new_bytetrack


def person(confidence):
    return sv.Detections(
        xyxy=np.array([[10.0, 10.0, 30.0, 60.0]]),
        confidence=np.array([confidence]),
        class_id=np.array([PERSON_ID]),
    )


def test_tracks_people_with_confidence_between_conf_and_06():
    # ByteTrack trae un umbral interno (0.6) para crear seguimientos: si no se alinea con CONF,
    # una persona detectada con 0.5 nunca recibe número y no aparece (se perdía ~31%).
    tracker = new_bytetrack(fps=5, conf=0.4)
    for _ in range(3):
        tracked = tracker.update(person(0.5))
    assert tracked.tracker_id.tolist() != [-1]


def test_ignores_people_below_conf():
    tracker = new_bytetrack(fps=5, conf=0.4)
    for _ in range(3):
        tracked = tracker.update(person(0.3))
    assert all(tid == -1 for tid in tracked.tracker_id)
