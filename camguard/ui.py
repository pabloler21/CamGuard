"""La página web (Gradio): subir un video, pintar la zona, analizar y ver el resultado.

Es la única parte que conoce Gradio: traduce los ``ValueError`` del pipeline en avisos para
el usuario y los resultados en tabla y galería.
"""

import os
import shutil

import gradio as gr
import numpy as np
import spaces

from camguard.config import FPS, MAX_SECONDS, MAX_UPLOAD, TEMP_DIR
from camguard.detection import load_model
from camguard.pipeline import analyze
from camguard.video import first_frame

# Grosor del pincel, en píxeles del video (1920 de ancho en el ejemplo).
BRUSH_SIZE = 60

# Gradio borra sus copias de videos subidos y resultados: cada media hora, las de más de 1 hora.
CACHE_CLEANUP = (1800, 3600)

# Arregla el editor de Gradio (6.30) cuando el zoom del navegador no es 100%: su lienzo queda
# más ancho que el recuadro (1853 px en uno de 916 con zoom 90%), centra la imagen sobre ese
# ancho y se ve corrida a la derecha, cortada, sin poder pintar encima. Ajusta el lienzo al
# recuadro y vuelve a encuadrar la imagen; se repite unos segundos para cubrir la carga lenta de
# la imagen, y al cambiar el zoom o el tamaño de la ventana. window.editor es interno de Gradio:
# si deja de existir, no hace nada.
REFIT_EDITOR_JS = """() => {
    if (!window.camguardRefit) {
        window.camguardRefit = () => {
            try {
                const editor = window.editor;
                const box = editor.target_element.getBoundingClientRect();
                editor.resize_canvas(box.width, box.height);
                editor.tools.get("zoom").set_zoom("fit");
            } catch {}
        };
        window.addEventListener("resize", () => setTimeout(window.camguardRefit, 100));
    }
    for (const ms of [300, 1000, 2500, 5000]) setTimeout(window.camguardRefit, ms);
}"""

# Hugging Face define SPACE_ID en todo Space; en la PC no existe.
WHERE = (
    "> 🔒 **El video se procesa en servidores de Hugging Face.** No subas grabaciones reales o "
    "sensibles: para eso está la versión que corre en tu PC (ver el README)."
    if os.getenv("SPACE_ID")
    else "Todo corre en esta PC."
)

INSTRUCTIONS = f"""\
# CamGuard — Intruso en zona
1. Ya viene cargado un video de ejemplo. Si querés usar uno propio, tocá la ✕ del video y \
subí el tuyo.
2. Pintá encima de la imagen de la derecha el área restringida: el pincel ya está activo, \
arrastrá directamente sobre la imagen. Si te equivocás, tocá **Borrar zona** y volvé a pintar.
3. Tocá **Analizar**.

Se analizan {FPS} cuadros por segundo y hasta {MAX_SECONDS} s de video.

{WHERE}

> ⚠️ **Es una demo.** Puede no detectar a algunas personas (lejanas, tapadas, agachadas o con \
poca luz) y no reemplaza la vigilancia humana. Una entrada a los 0,2 s puede significar que la \
persona ya estaba en la zona cuando empezó el video.
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


# En el Space de Hugging Face pide una GPU (ZeroGPU) mientras dura el análisis; en la PC no hace
# nada. Va sobre on_analyze y no sobre analyze porque ZeroGPU lo corre en otro proceso y convierte
# cualquier excepción que no sea gr.Error en un "ZeroGPU worker error": acá los ValueError ya
# se volvieron avisos.
@spaces.GPU
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
        result = analyze(
            video_path,
            layers,
            progress=lambda p: progress(p, desc="Analizando"),
            out_dir=TEMP_DIR / "resultados",
        )
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
    # analytics_enabled=False: Gradio manda telemetría de uso por defecto (no los videos, pero
    # una app que maneja grabaciones de seguridad no debería hablar con nadie).
    with gr.Blocks(title="CamGuard", analytics_enabled=False, delete_cache=CACHE_CLEANUP) as demo:
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

        demo.load(None, js=REFIT_EDITOR_JS)
        video.upload(blank_zone, video, editor).then(None, js=REFIT_EDITOR_JS)
        clear.click(blank_zone, video, editor).then(None, js=REFIT_EDITOR_JS)
        run.click(on_analyze, [video, editor], [result, table, gallery])
    return demo


def main() -> None:
    """Borra los videos de la sesión anterior, carga el modelo y abre la página.

    Los videos subidos y los resultados son grabaciones reales y quedan en ``TEMP_DIR``. Al
    cerrar la ventana el proceso termina de golpe y no alcanza a limpiar, por eso se vacía al
    arrancar; mientras la app está abierta, ``delete_cache`` borra lo que tiene más de una hora.
    En la PC solo escucha en 127.0.0.1: nadie de la red ni de internet puede entrar. En el Space
    de Hugging Face, Gradio escucha en 0.0.0.0 porque así lo indica ``GRADIO_SERVER_NAME``.
    """
    # ponytail: los videos de la última sesión quedan en disco hasta el próximo arranque.
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    # Al arrancar, para que el primer análisis no espere la carga. En ZeroGPU además es
    # obligatorio: el modelo tiene que pasar a la GPU antes de entrar a una función @spaces.GPU.
    load_model()
    build_demo().launch(inbrowser=True, max_file_size=MAX_UPLOAD)
