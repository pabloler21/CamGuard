---
title: CamGuard
emoji: 📹
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12.12
app_file: app.py
short_description: Detecta intrusos en una zona restringida de un video
---

# CamGuard: demo "Intruso en zona"

Subís un video de una cámara, pintás el área restringida y la demo marca quién entró, cuándo y por cuánto tiempo.

## Cómo usarla

1. Abrí **https://huggingface.co/spaces/Pabloler21/CamGuard**. Si dice que está durmiendo, esperá a que arranque (un par de minutos).
2. Ya viene cargado un **video de ejemplo**. Si preferís uno propio, tocá la ✕ del video y subí el tuyo.
3. Marcá en rojo la zona restringida sobre la imagen de la derecha: podés pintarla entera o solo dibujar su contorno (se rellena sola). Si te equivocás, tocá **Borrar zona** (o la flecha ↶ para deshacer el último trazo) y volvé a pintar.
4. Tocá **Analizar**.

Corre en una GPU gratuita compartida de Hugging Face (ZeroGPU), con una cuota diaria por persona: unos 2 minutos de GPU sin cuenta y 5 con una cuenta gratis de Hugging Face. Si se agota, esperá al día siguiente o iniciá sesión.

## Qué devuelve

- El video anotado: la zona en rojo, las personas en verde y los intrusos en rojo, con un número de seguimiento.
- Una tabla con cada entrada a la zona: ID, segundo de entrada, segundo de salida y duración.
- Una captura del momento en que entró cada intruso.

## Privacidad

- **En el link de Hugging Face, el video se procesa en sus servidores.** No subas grabaciones reales o sensibles ahí.
- Para grabaciones reales, usá la versión que corre en tu PC: descargá el proyecto (en GitHub, **Code** → **Download ZIP**), descomprimilo y hacé doble clic en `run.bat` (Windows) o ejecutá `sh run.sh` (Mac / Linux). La primera vez descarga ~1 GB. Se abre en `http://127.0.0.1:7860`, solo se puede abrir desde esa computadora y el video no sale de ella.
- En las dos versiones, los videos subidos y los resultados se guardan temporalmente (`%TEMP%\camguard` en Windows, `/tmp/camguard` en Linux / Mac): se borran a la hora y, por completo, cada vez que la app arranca.
- No se envía telemetría.

## Límites de esta versión

- **Es una demo: no reemplaza la vigilancia humana.** Puede no detectar a personas lejanas, tapadas, agachadas o con poca luz, y no está probada contra alguien que intente evitar la cámara a propósito.
- Videos de hasta 1 GB.

- Analiza 5 cuadros por segundo y como máximo 60 s de video. En la versión local corre en CPU: unos 2 a 3 minutos para 60 s en 1080p.
- Una persona cuenta como intrusa cuando **el centro de su cuerpo** está sobre la zona, tal como se ve en la imagen. En el video de ejemplo detecta al ~95% de la gente dentro de la zona y ~97% de las marcas son correctas.
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
| `zone.py` | Convierte lo pintado en una máscara (rellena contornos) y decide quién está adentro |
| `events.py` | Agrupa cuadros en eventos de entrada y salida (`Event`) |
| `video.py` | Lectura submuestreada y escritura H.264 |
| `detection.py` | RF-DETR + ByteTrack: personas con número de seguimiento |
| `annotate.py` | Dibuja la zona y los recuadros |
| `pipeline.py` | `analyze()`: orquesta todo; no depende de Gradio |
| `ui.py` | La página web y el video de ejemplo |
