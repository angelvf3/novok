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
         "pronostico": [{"horizonte_h": L, "hora_utc": f"2025-06-30T0{4 + L}:00Z",
                         "bajo_minimos": {"p": p, "nivel": n, "alerta": a}, "niebla": {"p": .1, "alerta": False}}
                        for L, p, n, a in [(1, .8, "alto", True), (3, .5, "alto", True), (6, .2, "moderado", False)]],
         "episodio": {"disipacion": {"despeja_utc_q50": "2025-06-30T09:00Z"}},
         "resumen": ["Ahora: ...", "Riesgo alto de bajo mínimos tipo CAT I a +1 h y +3 h."]}
    r = resumen_red(b)
    assert r["categoria"] == "LIFR" and r["riesgo"] == {"p": .8, "nivel": "alto", "horizonte_h": 1, "alerta": True}
    assert r["despeja_utc"] == "2025-06-30T09:00Z" and r["resumen"].startswith("Riesgo alto")
    assert [x["p_bajo_minimos"] for x in r["serie"]] == [.8, .5, .2]


def test_relieve_rejilla_y_codificacion(tmp_path):
    from PIL import Image

    from novok.cartografia import guardar_alturas, rejilla_relieve

    def tesela(z, x, y):  # rampa este-oeste: la altura crece hacia el este, 1 m por píxel global
        return np.tile(np.arange(256, dtype=float) + x * 256, (256, 1))

    alt = rejilla_relieve(43.56, -6.03, alcance_m=5000, n=33, tesela=tesela)
    assert alt.shape == (33, 33)
    assert (np.diff(alt, axis=1) > 0).all() and np.allclose(alt[0], alt[-1])   # crece al este, igual en N y S
    alt = alt - alt.min() - 50                                                # de -50 m a unos cientos
    guardar_alturas(alt, tmp_path / "r.png")
    p = np.asarray(Image.open(tmp_path / "r.png"), dtype=float)
    leido = p[..., 0] * 256 + p[..., 1] + p[..., 2] / 256 - 32768
    assert np.abs(leido - alt).max() < 0.01                          # ida y vuelta, incluso bajo cero


def test_mapa_relieve_teselas(tmp_path):
    from PIL import Image

    from novok.cartografia import colorear, mapa_relieve

    def tesela(z, x, y):  # media tesela de mar y media de montaña
        a = np.zeros((256, 256))
        a[:, 128:] = np.linspace(0, 1500, 128)
        return a
    n = mapa_relieve(caja=(-6.5, 43.0, -5.5, 43.8), zmin=6, zmax=7, destino=tmp_path, tesela=tesela)
    assert n >= 2 and (tmp_path / "info.json").exists()
    img = np.asarray(Image.open(next(tmp_path.glob("7/*/*.jpg"))))
    assert img.shape == (256, 256, 3)
    rgb = colorear(np.array([[0.0, 0.0], [800.0, 800.0]]), 100)
    assert tuple(rgb[0, 0]) == (170, 211, 236) and rgb[1, 0, 2] < 200       # mar azul; tierra en tonos cálidos


def test_ortofoto_por_trozos():
    import io

    from PIL import Image

    from novok.cartografia import pnoa
    pedidas = []

    def pedir(params):
        pedidas.append(params["BBOX"])
        b = io.BytesIO()
        Image.new("RGB", (64, 64), (len(pedidas) * 40, 0, 0)).save(b, "JPEG")
        return b.getvalue()
    img = pnoa(43.56, -6.03, 3000, px=128, trozo=64, pedir=pedir)
    assert img.size == (128, 128) and len(pedidas) == 4
    sur, oeste, norte, este = map(float, pedidas[0].split(","))
    assert norte > sur and este > oeste and abs(norte - 43.56 - 3000 / 111320) < 1e-6   # primer trozo: esquina NO


def test_codificacion_terrarium_ida_y_vuelta():
    import numpy as np

    from novok.cartografia import a_terrarium
    alt = np.array([[-12.5, 0.0], [845.25, 2648.0]])
    rgb = a_terrarium(alt).astype(float)
    vuelta = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
    assert np.allclose(vuelta, alt, atol=1 / 256)


def test_mapa_existente_se_reutiliza(tmp_path, monkeypatch):
    from novok import cartografia
    cabecera = bytearray(127)
    cabecera[:7], cabecera[7], cabecera[101] = b"PMTiles", 3, 6
    f = tmp_path / "mundo.pmtiles"
    f.write_bytes(bytes(cabecera) + b"datos")
    monkeypatch.setattr(cartografia, "herramienta_pmtiles", lambda: (_ for _ in ()).throw(AssertionError("descarga")))
    assert cartografia.zoom_maximo(f) == 6
    assert cartografia._recortar(f, None, (4, 5, 6, 7)) == 6
    assert '"maxzoom": 6' in (tmp_path / "mundo.json").read_text()
