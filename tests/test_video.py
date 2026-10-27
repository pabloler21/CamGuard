import numpy as np
import pytest

from camguard.video import (
    H264Writer,
    VideoInfo,
    first_frame,
    frame_limit,
    probe,
    sample_frames,
    sampling_step,
)


def make_video(path, n_frames, fps=25, width=64, height=48):
    """Video sintético donde cada cuadro tiene un brillo distinto."""
    with H264Writer(path, fps, width, height) as writer:
        for i in range(n_frames):
            writer.write(np.full((height, width, 3), 4 * i, np.uint8))
    return path


@pytest.fixture
def video(tmp_path):
    """2 s de video a 25 fps: 50 cuadros."""
    return make_video(tmp_path / "video.mp4", n_frames=50)


def test_written_video_can_be_read_back(video):
    assert probe(video) == VideoInfo(fps=25.0, width=64, height=48, frame_count=50)


def test_writer_crops_odd_size_to_even(tmp_path):
    path = make_video(tmp_path / "impar.mp4", n_frames=3, width=65, height=49)
    info = probe(path)
    assert (info.width, info.height) == (64, 48)


def test_sample_frames_takes_one_every_step(video):
    every_frame = list(sample_frames(video, step=1, max_frames=50))
    sampled = list(sample_frames(video, step=5, max_frames=50))
    assert len(every_frame) == 50
    assert len(sampled) == 10
    # Los cuadros salteados con grab() tienen que ser exactamente los cuadros 0, 5, 10...
    for got, expected in zip(sampled, every_frame[::5], strict=True):
        assert np.array_equal(got, expected)


def test_sample_frames_stops_at_max_frames(video):
    assert len(list(sample_frames(video, step=5, max_frames=20))) == 4


def test_sample_frames_stops_at_end_of_video(video):
    assert len(list(sample_frames(video, step=5, max_frames=1000))) == 10


@pytest.mark.parametrize(
    ("source_fps", "step"),
    [(25, 5), (30, 6), (29.97, 6), (3, 1)],
)
def test_sampling_step(source_fps, step):
    assert sampling_step(source_fps, target_fps=5) == step


@pytest.mark.parametrize(
    ("frame_count", "limit"),
    [
        pytest.param(100, 50, id="video-largo-se-corta"),
        pytest.param(10, 10, id="video-corto-entero"),
        pytest.param(0, 50, id="sin-cantidad-de-cuadros-usa-el-tope"),
    ],
)
def test_frame_limit(frame_count, limit):
    info = VideoInfo(fps=25.0, width=64, height=48, frame_count=frame_count)
    assert frame_limit(info, max_seconds=2) == limit


def test_first_frame_is_rgb(tmp_path):
    path = tmp_path / "azul.mp4"
    with H264Writer(path, fps=25, width=64, height=48) as writer:
        writer.write(np.full((48, 64, 3), (255, 0, 0), np.uint8))  # azul en BGR
    frame = first_frame(path)
    assert frame[..., 2].mean() > 200  # en RGB el azul es el último canal
    assert frame[..., 0].mean() < 50


def test_unreadable_file_raises(tmp_path):
    path = tmp_path / "roto.mp4"
    path.write_text("esto no es un video")
    with pytest.raises(ValueError, match="No pude leer"):
        probe(path)
    with pytest.raises(ValueError, match="No pude leer"):
        first_frame(path)
