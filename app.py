"""CamGuard: detecta personas que entran a una zona restringida de un video.

Todo corre local, en CPU. El video nunca sale de la máquina.
"""

import tempfile
from fractions import Fraction

import av
import cv2
import gradio as gr
import numpy as np
import supervision as sv

FPS = 5  # cuadros analizados por segundo
MAX_SECONDS = 60  # tope de video analizado
CONF = 0.4  # confianza mínima para contar una detección
MAX_GAP_S = 1.0  # si alguien "desaparece" menos que esto, sigue siendo el mismo evento
PERSON_ID = 1  # id COCO de "person" en RF-DETR
# ponytail: CPU-only (~0.5 s/cuadro en 1080p); si es lento, exportar RF-DETR a ONNX + onnxruntime-directml

model = None  # se carga en main(), así los tests no bajan el modelo


def feet_in_zone(feet_xy, zone):
    """True para cada punto (x, y) que cae dentro de la máscara de zona."""
    h, w = zone.shape
    x = np.clip(feet_xy[:, 0].astype(int), 0, w - 1)
    y = np.clip(feet_xy[:, 1].astype(int), 0, h - 1)
    return zone[y, x]


def build_events(frames_in_zone, fps, max_gap_s=MAX_GAP_S):
    """{id: [índices de cuadro dentro de la zona]} -> filas [id, entró, salió, duración] en segundos."""
    max_gap = max(1, round(max_gap_s * fps))
    rows = []
    for tid, frames in frames_in_zone.items():
        start = prev = frames[0]
        for f in frames[1:] + [None]:
            if f is not None and f - prev <= max_gap:
                prev = f
                continue
            enter, leave = start / fps, (prev + 1) / fps
            rows.append([tid, round(enter, 1), round(leave, 1), round(leave - enter, 1)])
            start = prev = f
    return sorted(rows, key=lambda r: r[1])


def zone_from_editor(editor, shape):
    layers = (editor or {}).get("layers") or []
    mask = np.any([layer[..., 3] > 0 for layer in layers], axis=0) if layers else None
    if mask is None or not mask.any():
        raise gr.Error("Pintá la zona restringida sobre la imagen antes de analizar.")
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def first_frame(video_path):
    ok, frame = cv2.VideoCapture(video_path).read()
    if not ok:
        raise gr.Error("No pude leer el video.")
    return {"background": frame[:, :, ::-1], "layers": [], "composite": None}


def example():
    """Video de ejemplo (se baja la primera vez) con una zona ya pintada abajo a la derecha."""
    from supervision.assets import VideoAssets, download_assets

    try:
        path = download_assets(VideoAssets.PEOPLE_WALKING, directory="examples")
    except Exception:  # sin internet: arranca sin ejemplo
        return None, None
    editor = first_frame(path)
    h, w = editor["background"].shape[:2]
    layer = np.zeros((h, w, 4), np.uint8)
    layer[int(h * 0.55) :, w // 2 :] = (255, 0, 0, 255)
    editor["layers"] = [layer]
    return path, editor


def draw(frame, det, inside, zone):
    out = frame.copy()
    out[zone] = (0.6 * out[zone] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)  # rojo (BGR)
    for mask, color, text in ((~inside, sv.Color.GREEN, "persona #{}"), (inside, sv.Color.RED, "INTRUSO #{}")):
        d = det[mask]
        out = sv.BoxAnnotator(color=color, thickness=3).annotate(out, d)
        out = sv.LabelAnnotator(color=color).annotate(out, d, labels=[text.format(t) for t in d.tracker_id])
    return out


def analyze(video_path, editor, progress=gr.Progress()):
    from trackers import ByteTrackTracker

    if not video_path:
        raise gr.Error("Primero subí un video.")
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / FPS))
    fps = src_fps / step
    total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1e9, MAX_SECONDS * src_fps))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2
    zone = zone_from_editor(editor, (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))

    tracker = ByteTrackTracker(frame_rate=fps, track_activation_threshold=CONF, minimum_consecutive_frames=1)
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    container = av.open(out_path, "w")
    stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"

    in_zone, snapshots, i = {}, [], 0
    for idx in progress.tqdm(range(total), desc="Analizando"):
        if idx % step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        det = model.predict(frame[:, :, ::-1].copy(), threshold=CONF)
        det = tracker.update(det[det.class_id == PERSON_ID])
        det = det[det.tracker_id != -1]
        inside = feet_in_zone(det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER), zone)
        out = draw(frame, det, inside, zone)
        for tid in det.tracker_id[inside].tolist():
            if tid not in in_zone:
                snapshots.append((out[:, :, ::-1], f"Intruso #{tid} — {i / fps:.1f} s"))
            in_zone.setdefault(tid, []).append(i)
        for packet in stream.encode(av.VideoFrame.from_ndarray(out[:h, :w], format="bgr24")):
            container.mux(packet)
        i += 1
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    cap.release()

    rows = build_events(in_zone, fps)
    if not rows:
        gr.Info("No hubo intrusiones en la zona.")
    return out_path, rows, snapshots


def main():
    global model
    from rfdetr import RFDETRSmall

    model = RFDETRSmall()
    example_video, example_zone = example()
    with gr.Blocks(title="CamGuard") as demo:
        gr.Markdown(
            "# CamGuard — Intruso en zona\n"
            "**Para probar rápido:** ya viene cargado un video de ejemplo con la zona marcada. Tocá **Analizar**.\n\n"
            "**Con tu propio video:** tocá la ✕ del video, subí el tuyo, pintá encima de la imagen el área "
            "restringida y tocá **Analizar**.\n\n"
            f"Se analizan {FPS} cuadros por segundo y hasta {MAX_SECONDS} s de video. Todo corre en esta PC."
        )
        with gr.Row():
            video = gr.Video(value=example_video, label="Video", sources=["upload"])
            editor = gr.ImageEditor(
                value=example_zone,
                label="Zona restringida (pintá encima)",
                type="numpy",
                sources=[],
                brush=gr.Brush(default_size=120, colors=["#ff0000"], color_mode="fixed"),
            )
        run = gr.Button("Analizar", variant="primary")
        result = gr.Video(label="Resultado")
        table = gr.Dataframe(headers=["ID", "Entró (s)", "Salió (s)", "Duración (s)"], label="Eventos")
        gallery = gr.Gallery(label="Momento de entrada de cada intruso", columns=4)

        video.upload(first_frame, video, editor)
        run.click(analyze, [video, editor], [result, table, gallery])
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
