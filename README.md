# CamGuard: demo "Intruso en zona"

Subís un video de una cámara, pintás el área restringida y la demo marca quién entró, cuándo y por cuánto tiempo.
**Todo corre en tu computadora: el video no se sube a ningún lado.**

## Cómo usarla

1. Descargá el proyecto: en GitHub, botón verde **Code** → **Download ZIP**. Descomprimilo.
2. Abrilo:
   - **Windows:** doble clic en `run.bat`.
   - **Mac / Linux:** abrí una terminal en la carpeta y ejecutá `sh run.sh`.

   La primera vez tarda unos minutos, porque descarga Python, las librerías y el modelo (~1 GB en total). Después arranca en segundos.
3. Se abre el navegador en `http://127.0.0.1:7860`. Si no se abre solo, copiá esa dirección.
4. Ya viene cargado un **video de ejemplo con la zona marcada**: tocá **Analizar** y en menos de un minuto ves el resultado.
5. Para probar con un video propio: tocá la ✕ del video, subí el tuyo, pintá en rojo la zona restringida sobre la imagen de la derecha y tocá **Analizar**.

Para cerrarla, cerrá la ventana negra (la terminal).

## Qué devuelve

- El video anotado: la zona en rojo, las personas en verde y los intrusos en rojo, con un número de seguimiento.
- Una tabla con cada entrada a la zona: ID, segundo de entrada, segundo de salida y duración.
- Una captura del momento en que entró cada intruso.

## Límites de esta versión

- Analiza 5 cuadros por segundo y como máximo 60 s de video. Corre en CPU: unos 2 a 3 minutos para 60 s en 1080p.
- Una persona cuenta como intrusa cuando **sus pies** están dentro de la zona.
- Las personas muy lejanas, muy tapadas o en escenas oscuras pueden no detectarse. Si dos personas se cruzan, pueden intercambiar el número.
- Solo detecta "persona". No distingue guardias de civiles ni detecta humo: eso queda para la próxima versión.

## Para desarrolladores

Modelo: [RF-DETR Small](https://github.com/roboflow/rf-detr) (Apache-2.0). Tracking: [trackers](https://github.com/roboflow/trackers) ByteTrack. UI: Gradio.

```sh
uv run python app.py      # levantar la app
uv run pytest             # pruebas (~3 s, no descargan el modelo)
uv run ruff check .       # linter
uv run ruff format .      # formatear
```

`app.py` es solo el punto de entrada; el código está en el paquete `camguard/`:

| Módulo | Qué hace |
|---|---|
| `config.py` | Perillas ajustables: `FPS`, `MAX_SECONDS`, `CONF`, `MAX_GAP_S` |
| `zone.py` | Convierte lo pintado en una máscara y decide quién tiene los pies adentro |
| `events.py` | Agrupa cuadros en eventos de entrada y salida (`Event`) |
| `video.py` | Lectura submuestreada y escritura H.264 |
| `detection.py` | RF-DETR + ByteTrack: personas con número de seguimiento |
| `annotate.py` | Dibuja la zona y los recuadros |
| `pipeline.py` | `analyze()`: orquesta todo; no depende de Gradio |
| `ui.py` | La página web y el video de ejemplo |
