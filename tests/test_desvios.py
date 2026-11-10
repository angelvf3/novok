import numpy as np
import pandas as pd

from novok import desvios as dv


def _heatmap(entradas):
    """Construye un heatmap de readsb: lista de ('t', ms) o (hex, lat, lon, alt_ft, gs_kt) o ('cs', hex, texto)."""
    b = bytearray()
    for e in entradas:
        if e[0] == "t":
            ms = e[1]
            b += np.array([dv.MARCA_TIEMPO, ms >> 32, ms & 0xFFFFFFFF, 10000], dtype="<u4").astype("<i4").tobytes()
        elif e[0] == "cs":
            b += np.array([e[1], (1 << 30) | 1000], dtype="<i4").tobytes() + e[2].ljust(8).encode()
        else:
            hexa, lat, lon, alt, gs = e
            b += np.array([hexa, round(lat * 1e6), round(lon * 1e6)], dtype="<i4").tobytes()
            b += np.array([-123 if alt is None else alt // 25, round(gs * 10)], dtype="<i2").tobytes()
    return bytes(b)


def test_leer_heatmap():
    t0 = 1_768_900_000_000
    datos = _heatmap([("t", t0), ("cs", 0x3443C1, "VOE2104"), (0x3443C1, 43.6, -6.0, 2500, 140),
                      (0xABCDEF, 50.0, 2.0, 30000, 450),          # fuera de la caja
                      ("t", t0 + 10_000), (0x3443C1, 43.56, -6.03, None, 20)])
    df = dv.leer_heatmap(datos)
    assert len(df) == 2
    assert list(df["t"]) == [t0, t0 + 10_000]
    assert list(df["alt_ft"]) == [2500, 0]                     # -123 = en tierra
    assert df["indicativo"].iat[0] == "VOE2104"
    assert abs(df["lat"].iat[0] - 43.6) < 1e-5 and df["gs_kt"].iat[0] == 140


def _pista(hexa, ind, puntos, t0=1_768_900_000_000):
    return pd.DataFrame([{"t": t0 + i * 60_000, "hex": hexa, "lat": la, "lon": lo, "alt_ft": al, "gs_kt": 200,
                          "indicativo": ind} for i, (la, lo, al) in enumerate(puntos)])


def test_aproximacion_frustrada_y_desvio():
    leas, lexj = dv.RED["LEAS"], dv.RED["LEXJ"]
    crucero = [(42.5 + i * 0.1, -5.0, 30000) for i in range(8)]
    aprox = [(leas[1] + 0.05, leas[2] + 0.05, 2500), (leas[1] + 0.01, leas[2], 900)]     # baja en Asturias...
    escape = [(leas[1] + 0.2, leas[2] + 0.5, 6000), (43.5, -4.8, 8000), (43.45, -4.2, 5000)]
    final = [(lexj[1] + 0.03, lexj[2] - 0.05, 1500), (lexj[1], lexj[2], 0)]            # ...y aterriza en Santander
    v = dv.vuelos_del_dia(_pista(0x3443C1, "VOE2104", crucero + aprox + escape + final))
    assert v["aterriza"].iat[0] == "LEXJ"
    assert [a["icao"] for a in v["aproximaciones"].iat[0]] == ["LEAS"]


def test_vuelo_normal_sin_aproximacion_extra():
    leas = dv.RED["LEAS"]
    crucero = [(42.0 + i * 0.2, -5.5, 30000) for i in range(8)]
    final = [(leas[1] - 0.05, leas[2] + 0.1, 2000), (leas[1], leas[2], 0)]
    v = dv.vuelos_del_dia(_pista(0x111111, "IBE0532", crucero + final))
    assert v["aterriza"].iat[0] == "LEAS" and v["aproximaciones"].iat[0] == []


def test_web_resumen_por_aeropuerto(tmp_path, monkeypatch):
    import json
    informe, raiz = tmp_path / "informe", tmp_path / "raiz"
    informe.mkdir(); (raiz / "web").mkdir(parents=True)
    monkeypatch.setattr(dv, "DIR_INFORME", informe)
    monkeypatch.setattr(dv, "RAIZ", raiz)
    pd.DataFrame([
        {"dia": "2026-01-10", "indicativo": "AEA7233", "previsto": "LECO", "aterriza": "LEST", "hora_prevista_utc": "2026-01-10T07:00:00+00:00",
         "bajo_minimos_obs": True, "aviso_6h": True, "persistencia_6h": False, "en_prueba": True},
        {"dia": "2026-02-03", "indicativo": "IBE512", "previsto": "LECO", "aterriza": "LEVX", "hora_prevista_utc": "2026-02-03T08:00:00+00:00",
         "bajo_minimos_obs": True, "aviso_6h": False, "persistencia_6h": True, "en_prueba": True},
        {"dia": "2026-02-04", "indicativo": "VLG1", "previsto": "LECO", "aterriza": "LEST", "hora_prevista_utc": "2026-02-04T08:00:00+00:00",
         "bajo_minimos_obs": False, "aviso_6h": False, "persistencia_6h": False, "en_prueba": True},
    ]).to_csv(informe / "desvios_evaluados.csv", index=False)
    out = dv.web()
    x = out["aeropuertos"]["LECO"]
    assert (x["casos"], x["evaluados"], x["avisados"], x["persistencia"]) == (2, 2, 1, 1)
    assert x["ultimos"][0] == {"fecha": "03/02", "indicativo": "IBE512", "aterriza": "LEVX", "avisado": False}
    assert out["periodo"] == "enero a febrero de 2026"
    assert json.loads((raiz / "web" / "desvios.json").read_text())["aeropuertos"]["LECO"]["casos"] == 2
