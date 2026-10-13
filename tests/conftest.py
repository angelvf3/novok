import pytest

from novok import aeropuertos


@pytest.fixture(autouse=True)
def catalogo_aislado(tmp_path, monkeypatch):
    """Las pruebas no dependen del config/aeropuertos.json de cada máquina: solo LEAS con sus datos de F5."""
    monkeypatch.setattr(aeropuertos, "FICHERO", tmp_path / "no_existe.json")
    aeropuertos.usar(None)
    monkeypatch.delenv("NOVOK_ESTACION", raising=False)
