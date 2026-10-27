import tempfile

import pytest


@pytest.fixture(autouse=True)
def _temp_files_inside_pytest(tmp_path, monkeypatch):
    """Los archivos temporales que crea el código bajo prueba van a la carpeta de pytest.

    Sin esto, cada corrida dejaba videos de resultado sueltos en la carpeta temporal del sistema.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
