"""La página web (Gradio): subir un video, pintar la zona, analizar y ver el resultado.

Es la única parte que conoce Gradio: traduce los ``ValueError`` del pipeline en avisos para
el usuario y los resultados en una tabla (con la foto de cada intruso) y el video anotado.
"""

import base64
import shutil

import cv2
import gradio as gr
import numpy as np
import spaces

from camguard.config import MAX_UPLOAD, TEMP_DIR
from camguard.detection import load_model
from camguard.pipeline import AnalysisResult, analyze
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
# si deja de existir, no hace nada. Encuadra sin el margen de 30 px que el editor deja alrededor:
# sin sus barras de herramientas no hace falta, y la imagen se veía chica.
REFIT_EDITOR_JS = """() => {
    if (!window.camguardRefit) {
        window.camguardRefit = () => {
            try {
                const editor = window.editor;
                const box = editor.target_element.getBoundingClientRect();
                editor.resize_canvas(box.width, box.height);
                const zoom = editor.tools.get("zoom");
                zoom.border_padding = 0;
                zoom.set_zoom("fit");
            } catch {}
        };
        window.addEventListener("resize", () => setTimeout(window.camguardRefit, 100));
        // El editor se queda con la rueda del mouse: con el puntero encima, la página no
        // scrolleaba. Se corta antes de que le llegue (el zoom no hace falta).
        document.getElementById("zona")?.addEventListener(
            "wheel", (e) => e.stopPropagation(), true
        );
    }
    for (const ms of [300, 1000, 2500, 5000]) setTimeout(window.camguardRefit, ms);
}"""

# El lienzo de la herramienta de recorte del mismo editor mide el doble de su recuadro aunque no
# se use: se sale y la página queda desplazable a los costados; al arrastrar cerca del borde se
# corre entera y el trazo cae fuera de la imagen. Recortar lo que sobresale lo evita.
# Las barras del editor (zoom, pincel, color, tamaño) se ocultan: solo se pinta, y deshacer es
# un botón de abajo. El recuadro toma la proporción de un video común (16:9) para medir
# lo mismo que el video de al lado; si el video es de otra proporción, quedan bordes.
EDITOR_CSS = """
.pixi-target, .pixi-target-crop { overflow: hidden; }
#zona .top-panel, #zona .toolbar-wrap { display: none !important; }
#zona .image-container { aspect-ratio: 16 / 9; }
#camguard-hint {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    pointer-events: none; z-index: 10; transition: opacity .3s;
}
#camguard-hint.hidden { opacity: 0; }
#camguard-hint svg { width: 40%; max-width: 260px; filter: drop-shadow(0 1px 3px #0008); }
"""

# Sin texto, la imagen de la zona no dice que hay que pintar encima: un pincel pinta en loop un
# área sobre la imagen. Se va al primer trazo y vuelve con un video nuevo. Va encima del editor
# y deja pasar los clics (pointer-events: none).
ZONE_HINT_JS = """() => {
    const block = document.getElementById("zona");
    if (!block) return;
    let hint = document.getElementById("camguard-hint");
    if (!hint) {
        // Ida y vuelta, con trazos que se tocan: se ve un área pintada, no un contorno.
        const path = "M40,30 H180 V55 H40 V80 H180 V105 H40";
        hint = document.createElement("div");
        hint.id = "camguard-hint";
        hint.innerHTML = `<svg viewBox="0 0 220 130">
            <path d="${path}" pathLength="1" fill="none" stroke="#ff0000" stroke-opacity=".7"
                stroke-width="28" stroke-linecap="round" stroke-linejoin="round"
                stroke-dasharray="1">
                <animate attributeName="stroke-dashoffset" values="1;0;0" keyTimes="0;.7;1"
                    dur="3s" repeatCount="indefinite"/>
            </path>
            <circle r="9" fill="#ff0000" stroke="#fff" stroke-width="3">
                <animateMotion path="${path}" keyPoints="0;1;1" keyTimes="0;.7;1"
                    calcMode="linear" dur="3s" repeatCount="indefinite"/>
            </circle></svg>`;
        block.appendChild(hint);
        block.addEventListener("pointerdown", (e) => {
            if (e.target.closest(".pixi-target")) hint.classList.add("hidden");
        }, true);
    }
    hint.classList.remove("hidden");
}"""

# Al cargar la página y cada vez que la zona vuelve a quedar en blanco.
ZONE_RESET_JS = f"() => {{ ({REFIT_EDITOR_JS})(); ({ZONE_HINT_JS})(); }}"

HEADERS = ["ID", "Entró (s)", "Salió (s)", "Duración (s)", "Foto"]

# Alto con que se ve la foto de cada intruso en la tabla.
PHOTO_PX = 120


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


def use_video(video_path: str):
    """Cambia al video subido con "Subir mi video": zona en blanco y sin los resultados del otro.

    Returns:
        (video, editor, tabla, video analizado), o ``gr.skip()`` en los cuatro (deja todo como
        estaba) si el video no se puede leer.
    """
    try:
        frame = first_frame(video_path)
    except ValueError as e:
        warn(str(e))
        return (gr.skip(),) * 4
    return video_path, editor_value(frame), gr.Dataframe(visible=False), gr.Video(visible=False)


def photo_html(image_rgb: np.ndarray) -> str:
    """Convierte la foto de un intruso en un <img> embebido, para la columna "Foto"."""
    _, jpg = cv2.imencode(".jpg", image_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(jpg).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" style="height:{PHOTO_PX}px;width:auto">'


def table_rows(result: AnalysisResult) -> list[list]:
    """Una fila por entrada a la zona. Salida y duración quedan vacías hasta terminar."""
    photos = {s.track_id: photo_html(s.image) for s in result.snapshots}
    done = result.video_path is not None
    return [
        [
            e.track_id,
            e.enter_s,
            e.leave_s if done else "",
            e.duration_s if done else "",
            photos[e.track_id],
        ]
        for e in result.events
    ]


# En el Space de Hugging Face pide una GPU (ZeroGPU) mientras dura el análisis; en la PC no hace
# nada. Va sobre on_analyze y no sobre analyze porque ZeroGPU lo corre en otro proceso y convierte
# cualquier excepción que no sea gr.Error en un "ZeroGPU worker error": acá los ValueError ya
# se volvieron avisos.
@spaces.GPU
def on_analyze(video_path: str | None, editor: dict | None):
    """Corre el análisis y va mostrando sus resultados en la página.

    Yields:
        (filas de la tabla, video anotado, texto del botón): al empezar (y se oculta el video
        anterior), cada vez que entra alguien o avanza un 1%, y al final con el video. Si falta
        el video o la zona, una sola vez ``gr.skip()`` en los tres: deja todo como estaba.
    """
    if not video_path:
        warn("Primero subí un video.")
        yield (gr.skip(),) * 3
        return
    layers = (editor or {}).get("layers") or []
    results = analyze(video_path, layers, out_dir=TEMP_DIR / "resultados")
    try:
        result = next(results)  # el primer paso valida el video y la zona
    except ValueError as e:  # video ilegible o sin zona pintada
        warn(str(e))
        yield (gr.skip(),) * 3
        return

    yield (
        gr.Dataframe(table_rows(result), visible=True),
        gr.Video(value=None, visible=False),
        gr.skip(),
    )
    shown, label = len(result.events), ""
    for result in results:
        if len(result.events) != shown:
            shown = len(result.events)
            yield table_rows(result), gr.skip(), gr.skip()
        text = f"Analizando… {result.progress:.0%}"
        if text != label:
            label = text
            yield gr.skip(), gr.skip(), label

    if not result.events:
        gr.Info("No hubo intrusiones en la zona.")
    yield table_rows(result), gr.Video(value=str(result.video_path), visible=True), gr.skip()


def build_demo() -> gr.Blocks:
    """Arma la página: video y editor lado a lado, el botón, la tabla de intrusos y el video."""
    example_video, example_editor = load_example()
    # analytics_enabled=False: Gradio manda telemetría de uso por defecto (no los videos, pero
    # una app que maneja grabaciones de seguridad no debería hablar con nadie).
    with gr.Blocks(title="CamGuard", analytics_enabled=False, delete_cache=CACHE_CLEANUP) as demo:
        gr.Markdown(
            "# CamGuard\nPablo Lerner\n\n"
            "Detecta quién entra a una zona restringida de tus cámaras, cuándo y por cuánto tiempo."
        )
        with gr.Row():
            with gr.Column():
                # Solo se reproduce: se cambia con el botón, más visible que la ✕ del componente.
                video = gr.Video(value=example_video, label="Video", interactive=False)
                upload = gr.UploadButton("Subir mi video", file_types=["video"], size="sm")
            with gr.Column():
                editor = gr.ImageEditor(
                    value=example_editor,
                    label="Zona restringida",
                    elem_id="zona",  # ahí se dibuja ZONE_HINT_JS
                    type="numpy",
                    sources=[],  # la imagen siempre es el primer cuadro del video
                    brush=gr.Brush(default_size=BRUSH_SIZE, colors=["#ff0000"], color_mode="fixed"),
                    eraser=False,
                    transforms=(),
                    layers=False,
                    height="auto",  # lo da la proporción de EDITOR_CSS
                )
                undo = gr.Button("Deshacer", size="sm")
        run = gr.Button("Analizar", variant="primary")
        table = gr.Dataframe(
            headers=HEADERS,
            datatype=["number", "number", "number", "number", "html"],
            label="Intrusos",
            visible=False,  # hasta analizar: vacía era ruido
            interactive=False,
            wrap=True,
            max_height="none",  # la página crece con la tabla, en vez de un scroll adentro
        )
        # Oculto hasta que termina el análisis: vacío parecía roto.
        result = gr.Video(label="Video analizado", visible=False)

        demo.load(None, js=ZONE_RESET_JS)
        upload.upload(use_video, upload, [video, editor, table, result]).then(
            None, js=ZONE_RESET_JS
        )
        undo.click(None, js="() => window.editor?.undo()")
        # El avance se ve en el botón, que queda deshabilitado mientras analiza. Las barras de
        # progreso de Gradio iban una por resultado, y la de la tabla tapaba las filas nuevas.
        analysis = run.click(
            lambda: gr.Button("Analizando…", interactive=False), None, run, queue=False
        ).then(on_analyze, [video, editor], [table, result, run], show_progress="hidden")
        # .then debería correr también si on_analyze falla, pero en Gradio 6.30 no (probado): el
        # botón quedaba trabado en "Analizando…". Por eso se restablece por éxito y por falla.
        for after in (analysis.success, analysis.failure):
            after(lambda: gr.Button("Analizar", interactive=True), None, run, queue=False)
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
    build_demo().launch(inbrowser=True, max_file_size=MAX_UPLOAD, css=EDITOR_CSS)
