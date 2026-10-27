"""Perillas de la demo: valores que se pueden ajustar sin tocar la lógica."""

import tempfile
from pathlib import Path

# Carpeta donde quedan los videos subidos y los resultados. Son grabaciones reales: la app la
# vacía cada vez que arranca y Gradio borra sus copias a la hora (ver ui.main).
TEMP_DIR = Path(tempfile.gettempdir()) / "camguard"

