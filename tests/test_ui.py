import gradio as gr
import numpy as np
import pytest

from camguard.ui import blank_zone, on_analyze
from camguard.video import H264Writer

# Que falte un paso (sin video, sin zona) no es un error de la app: tiene que mostrarse como
# aviso y dejar los resultados como estaban, en vez de lanzar gr.Error.


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "video.mp4"
    with H264Writer(path, fps=5, width=64, height=48) as writer:
        writer.write(np.zeros((48, 64, 3), np.uint8))
    return str(path)


# Fuera de la página, gr.Warning se emite como UserWarning: así se verifica que el aviso aparece.


def test_analyze_without_video_warns_and_skips_outputs():
    with pytest.warns(UserWarning, match="Primero subí un video"):
        assert on_analyze(None, None) == (gr.skip(),) * 3


def test_analyze_without_zone_warns_and_skips_outputs(video):
    with pytest.warns(UserWarning, match="Pintá la zona"):
        assert on_analyze(video, {"layers": []}) == (gr.skip(),) * 3


def test_clear_zone_without_video_warns_and_skips_editor():
    with pytest.warns(UserWarning, match="Primero subí un video"):
        assert blank_zone(None) == gr.skip()
