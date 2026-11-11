from novok import alertas, instantanea


def test_foto_con_rutas_en_formato_routeset(monkeypatch):
    monkeypatch.setattr(alertas, "posiciones", lambda cfg: [{"hex": "abc", "flight": "AEA7233 ", "lat": 43, "lon": -8}])
    monkeypatch.setattr(alertas, "rutas", lambda cfg, av: {"AEA7233": {"origen": "LEMD", "destino": "LECO"}})
    f = instantanea.foto({"proveedor": "adsb.lol"})
    assert f["ac"][0]["hex"] == "abc" and "error" not in f
    assert f["rutas"] == [{"callsign": "AEA7233", "airport_codes": "LEMD-LECO"}]
    assert f["now"] > 1e12


def test_foto_vacia_lleva_error(monkeypatch):
    monkeypatch.setattr(alertas, "posiciones", lambda cfg: [])
    f = instantanea.foto({})
    assert f["ac"] == [] and f["error"]
