"""F7 · Lógica de vuelos (web/vuelos.js): se prueba con Node si está instalado."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js no está instalado")
def test_logica_de_vuelos():
    r = subprocess.run(["node", str(Path(__file__).with_name("vuelos.test.cjs"))], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or r.stdout
