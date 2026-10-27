import pytest

from camguard.events import Event, build_events

# Todos los casos a 5 cuadros por segundo: cada cuadro son 0,2 s y la tolerancia
# (MAX_GAP_S = 1 s) son 5 cuadros.


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        pytest.param([0, 1, 2, 3, 4], [(0.0, 1.0)], id="una-estadia"),
        pytest.param([10, 12], [(2.0, 2.6)], id="hueco-corto-no-corta"),
        pytest.param([10, 30], [(2.0, 2.2), (6.0, 6.2)], id="hueco-largo-son-dos-eventos"),
        pytest.param([7], [(1.4, 1.6)], id="un-solo-cuadro-dura-un-cuadro"),
    ],
)
def test_build_events_one_person(frames, expected):
    events = build_events({1: frames}, fps=5)
    assert [(e.enter_s, e.leave_s) for e in events] == expected


def test_gap_tolerance_boundary():
    assert len(build_events({1: [0, 5]}, fps=5)) == 1  # hueco de 1 s exacto: mismo evento
    assert len(build_events({1: [0, 6]}, fps=5)) == 2  # un cuadro más: salió y volvió


def test_events_sorted_by_entry_across_people():
    events = build_events({1: [20], 2: [5]}, fps=5)
    assert [e.track_id for e in events] == [2, 1]


def test_no_people_no_events():
    assert build_events({}, fps=5) == []


def test_event_duration():
    assert Event(track_id=1, enter_s=2.0, leave_s=2.6).duration_s == 0.6
