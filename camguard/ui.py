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

# Grosor del pincel, en píxeles del video (1920 de ancho en el ejemplo).
BRUSH_SIZE = 60

INSTRUCTIONS = f"""\
# CamGuard — Intruso en zona
1. Ya viene cargado un video de ejemplo. Si querés usar uno propio, tocá la ✕ del video y \
subí el tuyo.
2. Pintá encima de la imagen de la derecha el área restringida: el pincel ya está activo, \
arrastrá directamente sobre la imagen. Si te equivocás, tocá **Borrar zona** y volvé a pintar.
3. Tocá **Analizar**.

Se analizan {FPS} cuadros por segundo y hasta {MAX_SECONDS} s de video. Todo corre en esta PC.
"""


def editor_value(background: np.ndarray, layers: list[np.ndarray] | None = None) -> dict:
    """Arma el valor del editor de Gradio: imagen de fondo y capas pintadas."""
    return {"background": background, "layers": layers or [], "composite": None}


def load_example() -> tuple[str | None, dict | None]:
    """Descarga el video de ejemplo (solo la primera vez), sin zona: la pinta quien prueba.

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
    return path, editor_value(first_frame(path))


def warn(message: str) -> None:
    """Muestra un aviso amarillo para cuando al usuario le falta un paso.

    Se usa en vez de ``gr.Error`` porque no es una falla de la app: ``gr.Error`` muestra un
    cartel rojo "Error", marca el resultado como fallido e imprime un traceback en la consola.
    """
    gr.Warning(message)


def blank_zone(video_path: str | None):
    """Pone el primer cuadro del video en el editor, sin nada pintado.

    Se usa al subir un video y al tocar "Borrar zona".

    Returns:
        El valor del editor, o ``gr.skip()`` (no lo cambia) si no hay un video legible.
    """
    if not video_path:
        warn("Primero subí un video.")
        return gr.skip()
    try:
        return editor_value(first_frame(video_path))
    except ValueError as e:
        warn(str(e))
        return gr.skip()


def on_analyze(video_path: str | None, editor: dict | None, progress=gr.Progress()):  # noqa: B008
    """Corre el análisis y lo convierte en lo que muestran los componentes de la página.

    Returns:
        (video anotado, filas de la tabla de eventos, imágenes de la galería con su leyenda), o
        ``gr.skip()`` en las tres (deja los resultados como estaban) si falta el video o la zona.
    """
    if not video_path:
        warn("Primero subí un video.")
        return (gr.skip(),) * 3
    layers = (editor or {}).get("layers") or []
    try:
        result = analyze(video_path, layers, progress=lambda p: progress(p, desc="Analizando"))
    except ValueError as e:  # video ilegible o sin zona pintada
        warn(str(e))
        return (gr.skip(),) * 3

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
            with gr.Column():
                editor = gr.ImageEditor(
                    value=example_editor,
                    label="Zona restringida (pintá encima)",
                    type="numpy",
                    sources=[],  # la imagen siempre es el primer cuadro del video
                    brush=gr.Brush(default_size=BRUSH_SIZE, colors=["#ff0000"], color_mode="fixed"),
                )
                clear = gr.Button("Borrar zona", size="sm")
        run = gr.Button("Analizar", variant="primary")
        result = gr.Video(label="Resultado")
        table = gr.Dataframe(
            headers=["ID", "Entró (s)", "Salió (s)", "Duración (s)"], label="Eventos"
        )
        gallery = gr.Gallery(label="Momento de entrada de cada intruso", columns=4)

        video.upload(blank_zone, video, editor)
        clear.click(blank_zone, video, editor)
        run.click(on_analyze, [video, editor], [result, table, gallery])
    return demo


def main() -> None:
    """Carga el modelo, arma la página y la abre en el navegador.

    Solo escucha en 127.0.0.1: nadie de la red ni de internet puede entrar (son videos reales).
    """
    load_model()  # al arrancar, para que el primer análisis no espere la carga
    build_demo().launch(inbrowser=True)
