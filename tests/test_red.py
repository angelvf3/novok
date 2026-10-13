"""F6 · Red de aeropuertos: catálogo de OurAirports, aeropuerto activo, viento con varias pistas y resumen de red."""

import numpy as np
import pandas as pd

from novok import aeropuertos as ae
from novok.config import dir_modelos, dir_proc
from novok.data.grabar_taf_vivo import separar
from novok.data.viento import viento_en_pista
from novok.red import categoria, resumen_red

AIRPORTS = (
    "id,ident,type,name,latitude_deg,longitude_deg,elevation_ft,continent,iso_country,iso_region,municipality\n"
    "1,LEXJ,medium_airport,Santander Airport,43.4271,-3.82,16,EU,ES,ES-S,Santander\n"
    "2,LEBB,large_airport,Bilbao Airport,43.3011,-2.9106,138,EU,ES,ES-PV,Bilbao\n"
)
RUNWAYS = (
    "id,airport_ref,airport_ident,length_ft,width_ft,surface,lighted,closed,le_ident,le_latitude_deg,"
    "le_longitude_deg,le_elevation_ft,le_heading_degT,le_displaced_threshold_ft,he_ident,he_latitude_deg,"
    "he_longitude_deg,he_elevation_ft,he_heading_degT,he_displaced_threshold_ft\n"
    "1,1,LEXJ,7808,148,ASP,1,0,11,43.43,-3.837,,107,,29,43.424,-3.803,,,\n"       # sin rumbo en 29: se deduce
    "2,2,LEBB,8530,148,ASP,1,0,12,43.31,-2.93,,122,,30,43.29,-2.89,,302,\n"
    "3,2,LEBB,6562,148,ASP,1,0,10,43.30,-2.93,,,,28,43.30,-2.90,,,\n"              # sin rumbos: por la designación
    "4,2,LEBB,500,50,ASP,1,0,H1,43.3,-2.9,,,,,,,,,\n"                              # helipuerto: fuera
    "5,2,LEBB,3000,50,ASP,1,1,05,43.3,-2.9,,50,,23,43.3,-2.9,,230,\n"              # cerrada: fuera
)


def test_catalogo_desde_ourairports():
    d = ae.construir(AIRPORTS, RUNWAYS, ["LEXJ", "LEBB", "LEZZ"])
    assert set(d) == {"LEXJ", "LEBB"}
    lexj = {p["ident"]: p["rumbo"] for p in d["LEXJ"]["pistas"]}
    assert lexj == {"11": 107.0, "29": 287.0}
    lebb = {p["ident"]: p["rumbo"] for p in d["LEBB"]["pistas"]}
    assert lebb == {"12": 122.0, "30": 302.0, "10": 100.0, "28": 280.0}
    assert d["LEXJ"]["nombre"] == "Santander" and d["LEXJ"]["pistas"][0]["longitud_m"] == 2380


def test_aeropuerto_activo_y_rutas(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "FICHERO", tmp_path / "aeropuertos.json")  # catálogo propio de la prueba
    ae.FICHERO.write_text(__import__("json").dumps(ae.construir(AIRPORTS, RUNWAYS, ["LEXJ"])))
    assert ae.activa().icao == "LEAS"            # por defecto
    with ae.estacion("lexj") as a:
        assert a.icao == "LEXJ" and a.rumbos == {"11": 107.0, "29": 287.0}
        assert dir_proc().name == "LEXJ" and dir_modelos().name == "LEXJ"
    assert ae.activa().icao == "LEAS"            # se restaura al salir


def test_viento_con_dos_pistas():
    h = pd.DataFrame({"viento_dir": [120.0, 290.0], "viento_kt": [10.0, 10.0], "racha_kt": [np.nan, np.nan]})
    v = viento_en_pista(h, {"12": 122.0, "30": 302.0, "10": 100.0, "28": 280.0})
    assert list(v["pista_favorable"]) == ["12", "28"]
    assert v.loc[0, "cruzado_kt"] < 0.5


def test_categoria_de_vuelo():
    assert categoria(150, 100) == "LIFR"
    assert categoria(3000, None) == "IFR"
    assert categoria(10000, 2500) == "MVFR"
    assert categoria(10000, None) == "VFR"
    assert categoria(None, None) is None


def test_taf_de_la_red_por_estacion():
    g = separar(["TAF LEAS 071013Z 0710/0809 30009KT 9999=", "TAF AMD LEXJ 071030Z 0710/0809 VRB02KT CAVOK=",
                 "TAF COR LEBB 071040Z 0710/0809 33010KT 9999 FEW020="])
    assert set(g) == {"LEAS", "LEXJ", "LEBB"}


def test_resumen_de_red():
    b = {"estacion": "LEAS", "aeropuerto": {"nombre": "Asturias", "lat": 43.56, "lon": -6.03},
         "calidad": {"pronostico_emitido": True, "metar_ultimo_utc": "2025-06-30T04:30Z"}, "version_modelo": "v1",
         "ahora": {"vis_m": 150.0, "techo_ft": 100.0, "viento_texto": "300° 4 kt",
                   "niebla": True, "bajo_minimos": True},
         "pronostico": [{"horizonte_h": L, "bajo_minimos": {"p": p, "nivel": n, "alerta": a},
                         "niebla": {"alerta": False}}
                        for L, p, n, a in [(1, .8, "alto", True), (3, .5, "alto", True), (6, .2, "moderado", False)]],
         "episodio": {"disipacion": {"despeja_utc_q50": "2025-06-30T09:00Z"}},
         "resumen": ["Ahora: ...", "Riesgo alto de bajo mínimos tipo CAT I a +1 h y +3 h."]}
    r = resumen_red(b)
    assert r["categoria"] == "LIFR" and r["riesgo"] == {"p": .8, "nivel": "alto", "horizonte_h": 1, "alerta": True}
    assert r["despeja_utc"] == "2025-06-30T09:00Z" and r["resumen"].startswith("Riesgo alto")


def test_relieve_rejilla_y_codificacion(tmp_path):
    from PIL import Image

    from novok.red import guardar_relieve, rejilla_relieve

    def tesela(z, x, y):  # rampa este-oeste: la altura crece hacia el este, 1 m por píxel global
        return np.tile(np.arange(256, dtype=float) + x * 256, (256, 1))

    alt = rejilla_relieve(43.56, -6.03, alcance_m=5000, n=33, tesela=tesela)
    assert alt.shape == (33, 33)
    assert (np.diff(alt, axis=1) > 0).all() and np.allclose(alt[0], alt[-1])   # crece al este, igual en N y S
    alt = alt - alt.min() - 50                                                # de -50 m a unos cientos
    guardar_relieve(alt, tmp_path / "r.png")
    p = np.asarray(Image.open(tmp_path / "r.png"), dtype=float)
    leido = p[..., 0] * 256 + p[..., 1] + p[..., 2] / 256 - 32768
    assert np.abs(leido - alt).max() < 0.01                          # ida y vuelta, incluso bajo cero
