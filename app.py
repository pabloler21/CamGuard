"""Punto de entrada de CamGuard: lo ejecuta el Space; en la PC, ``uv run python app.py``."""

import os

from camguard.config import TEMP_DIR

# Antes de importar Gradio: lee GRADIO_TEMP_DIR al importarse. Así sus copias de los videos
# quedan en TEMP_DIR, que la app vacía al arrancar (ver camguard.ui.main).
os.environ.setdefault("GRADIO_TEMP_DIR", str(TEMP_DIR / "gradio"))

from camguard.ui import main  # noqa: E402

if __name__ == "__main__":
    main()
