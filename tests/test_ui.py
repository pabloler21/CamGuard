import gradio as gr
import numpy as np
import pytest

from camguard.ui import on_analyze, use_video
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
        assert list(on_analyze(None, None)) == [(gr.skip(),) * 3]


def test_analyze_without_zone_warns_and_skips_outputs(video):
    with pytest.warns(UserWarning, match="Pintá la zona"):
        assert list(on_analyze(video, {"layers": []})) == [(gr.skip(),) * 3]


def test_unreadable_upload_warns_and_keeps_everything(tmp_path):
    path = tmp_path / "roto.mp4"
    path.write_bytes(b"no es un video")
    with pytest.warns(UserWarning):
        assert use_video(str(path)) == (gr.skip(),) * 4
