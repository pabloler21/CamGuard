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
2. Ya viene cargado un **video de ejemplo**. Si preferís uno propio, tocá **Subir mi video**.
3. Marcá en rojo la zona restringida sobre la imagen de la derecha: podés pintarla entera o solo dibujar su contorno (se rellena sola). Si te equivocás, tocá **Deshacer**.
4. Tocá **Analizar**.

Corre en una GPU gratuita compartida de Hugging Face (ZeroGPU), con una cuota diaria por persona: unos 2 minutos de GPU sin cuenta y 5 con una cuenta gratis de Hugging Face. Si se agota, esperá al día siguiente o iniciá sesión.

## Qué devuelve

- Una tabla con cada entrada a la zona: ID, segundo de entrada, segundo de salida, duración y la foto del intruso al entrar. Las filas aparecen mientras se analiza.
- Abajo, al terminar, el video anotado: la zona en rojo, las personas en verde y los intrusos en rojo, con un número de seguimiento.

## Privacidad

- **El video se procesa en servidores de Hugging Face.** No subas grabaciones reales o sensibles.
- Los videos subidos y los resultados se guardan temporalmente en el servidor: se borran a la hora y, por completo, cada vez que la app arranca.
- No se envía telemetría.

## Límites de esta versión

- **Es una demo: no reemplaza la vigilancia humana.** Puede no detectar a personas lejanas, tapadas, agachadas o con poca luz, y no está probada contra alguien que intente evitar la cámara a propósito.
- Videos de hasta 1 GB.

- Analiza 5 cuadros por segundo y como máximo 60 s de video.
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
