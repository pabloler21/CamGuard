"""La página web (Gradio): subir un video, pintar la zona, analizar y ver el resultado.

Es la única parte que conoce Gradio: traduce los ``ValueError`` del pipeline en avisos para
el usuario y los resultados en tabla y galería.
"""

import gradio as gr
import numpy as np

from camguard.config import FPS, MAX_SECONDS
from camguard.detection import load_model
from camguard.pipeline import analyze
from camguard.video import first_frame

# Zona del video de ejemplo, en fracción de la imagen: de la mitad hacia la derecha
# y del 55 % de la altura hacia abajo.
EXAMPLE_ZONE_LEFT = 0.5
EXAMPLE_ZONE_TOP = 0.55

INSTRUCTIONS = f"""\
# CamGuard — Intruso en zona
**Para probar rápido:** ya viene cargado un video de ejemplo con la zona marcada. \
Tocá **Analizar**.

**Con tu propio video:** tocá la ✕ del video, subí el tuyo, pintá encima de la imagen \
el área restringida y tocá **Analizar**.

Se analizan {FPS} cuadros por segundo y hasta {MAX_SECONDS} s de video. Todo corre en esta PC.
"""


def editor_value(background: np.ndarray, layers: list[np.ndarray] | None = None) -> dict:
    """Arma el valor del editor de Gradio: imagen de fondo y capas pintadas."""
    return {"background": background, "layers": layers or [], "composite": None}


def load_example() -> tuple[str | None, dict | None]:
    """Descarga el video de ejemplo (solo la primera vez) y le pinta una zona por defecto.

    El video es de Roboflow y no tiene una licencia clara, por eso se descarga en vez de
    incluirlo en el repo.

    Returns:
        (ruta del video, valor del editor), o (None, None) si no hay internet.
    """
    from supervision.assets import VideoAssets, download_assets

    try:
        path = download_assets(VideoAssets.PEOPLE_WALKING, directory="examples")
    except Exception:  # sin internet: la app arranca igual, sin ejemplo
        return None, None
    background = first_frame(path)
    h, w = background.shape[:2]
    layer = np.zeros((h, w, 4), np.uint8)
    layer[int(h * EXAMPLE_ZONE_TOP) :, int(w * EXAMPLE_ZONE_LEFT) :] = (255, 0, 0, 255)
    return path, editor_value(background, [layer])


def on_upload(video_path: str) -> dict:
    """Al subir un video, pone su primer cuadro en el editor para pintar la zona."""
    try:
        return editor_value(first_frame(video_path))
    except ValueError as e:
        raise gr.Error(str(e)) from e


def on_analyze(video_path: str | None, editor: dict | None, progress=gr.Progress()):  # noqa: B008
    """Corre el análisis y lo convierte en lo que muestran los componentes de la página.

    Returns:
        (video anotado, filas de la tabla de eventos, imágenes de la galería con su leyenda).
    """
    if not video_path:
        raise gr.Error("Primero subí un video.")
    layers = (editor or {}).get("layers") or []
    try:
        result = analyze(video_path, layers, progress=lambda p: progress(p, desc="Analizando"))
    except ValueError as e:
        raise gr.Error(str(e)) from e

    if not result.events:
        gr.Info("No hubo intrusiones en la zona.")
    rows = [[e.track_id, e.enter_s, e.leave_s, e.duration_s] for e in result.events]
    gallery = [(s.image, f"Intruso #{s.track_id} — {s.time_s:.1f} s") for s in result.snapshots]
    return str(result.video_path), rows, gallery


def build_demo() -> gr.Blocks:
    """Arma la página: video y editor lado a lado, el botón, y los tres resultados."""
    example_video, example_editor = load_example()
    with gr.Blocks(title="CamGuard") as demo:
        gr.Markdown(INSTRUCTIONS)
        with gr.Row():
            video = gr.Video(value=example_video, label="Video", sources=["upload"])
            editor = gr.ImageEditor(
                value=example_editor,
                label="Zona restringida (pintá encima)",
                type="numpy",
                sources=[],  # la imagen siempre es el primer cuadro del video
                brush=gr.Brush(default_size=120, colors=["#ff0000"], color_mode="fixed"),
            )
        run = gr.Button("Analizar", variant="primary")
        result = gr.Video(label="Resultado")
        table = gr.Dataframe(
            headers=["ID", "Entró (s)", "Salió (s)", "Duración (s)"], label="Eventos"
        )
        gallery = gr.Gallery(label="Momento de entrada de cada intruso", columns=4)

        video.upload(on_upload, video, editor)
        run.click(on_analyze, [video, editor], [result, table, gallery])
    return demo


def main() -> None:
    """Carga el modelo, arma la página y la abre en el navegador.

    Solo escucha en 127.0.0.1: nadie de la red ni de internet puede entrar (son videos reales).
    """
    load_model()  # al arrancar, para que el primer análisis no espere la carga
    build_demo().launch(inbrowser=True)
