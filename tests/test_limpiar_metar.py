import numpy as np
import pandas as pd
import pytest

from novok.data.limpiar_metar import a_horario, calcular_techo_ft, limpiar, rvr_min_desde_metar, vis_desde_metar


def fila_iem(valid, vsby="6.21", skyc=("", "", "", ""), skyl=("M", "M", "M", "M"),
             wx="M", tmpf="50.0", dwpf="48.0", metar=None):
    d = {"station": "LEAS", "valid": valid, "tmpf": tmpf, "dwpf": dwpf, "relh": "93",
         "drct": "180", "sknt": "3", "gust": "M", "alti": "30.00", "vsby": vsby,
         "wxcodes": wx, "metar": metar or f"LEAS {valid}"}
    for i in range(4):
        d[f"skyc{i + 1}"] = skyc[i]
        d[f"skyl{i + 1}"] = skyl[i]
    return d


def test_techo_toma_la_capa_bkn_ovc_mas_baja():
    fila = pd.Series({"skyc1": "FEW", "skyl1": 300.0, "skyc2": "BKN", "skyl2": 800.0,
                      "skyc3": "OVC", "skyl3": 400.0, "skyc4": "", "skyl4": np.nan})
    assert calcular_techo_ft(fila) == 400.0


def test_cielo_invisible_vv_cuenta_como_techo():
    fila = pd.Series({"skyc1": "VV", "skyl1": 100.0})
    assert calcular_techo_ft(fila) == 100.0


def test_sin_techo_devuelve_nan():
    fila = pd.Series({"skyc1": "FEW", "skyl1": 2000.0, "skyc2": "SCT", "skyl2": 3000.0})
    assert np.isnan(calcular_techo_ft(fila))


def test_conversion_de_unidades():
    obs = limpiar(pd.DataFrame([fila_iem("2024-01-01 06:00", vsby="0.12", tmpf="32.0", dwpf="32.0")]))
    assert obs.loc[0, "temp_c"] == pytest.approx(0.0)
    assert obs.loc[0, "dif_rocio_c"] == pytest.approx(0.0)
    assert obs.loc[0, "vis_m"] == 200  # 193 m de IEM, redondeado al paso METAR de 50 m


def test_visibilidad_maxima_recortada_a_10_km():
    obs = limpiar(pd.DataFrame([fila_iem("2024-01-01 06:00", vsby="10.0")]))
    assert obs.loc[0, "vis_m"] == 10_000


def test_horario_toma_la_peor_visibilidad_y_marca_umbrales():
    crudo = pd.DataFrame([
        fila_iem("2024-01-01 06:00", vsby="6.21"),
        fila_iem("2024-01-01 06:20", vsby="0.19", skyc=("VV", "", "", ""), skyl=("100", "M", "M", "M"), wx="FG"),
        fila_iem("2024-01-01 07:00", vsby="6.21"),
    ])
    h = a_horario(limpiar(crudo)).set_index("fecha_utc")
    h6 = h.loc[pd.Timestamp("2024-01-01 06:00", tz="UTC")]
    assert h6["vis_m"] == 300
    assert h6["n_obs"] == 2
    assert h6["vis_bajo_niebla"] and h6["vis_bajo_aprox_cat1"] and not h6["vis_bajo_aprox_cat2"]
    assert h6["techo_bajo_aprox_cat1"] and h6["techo_bajo_aprox_cat2"]  # VV001 = 100 ft
    assert h6["niebla_reportada"]

    h7 = h.loc[pd.Timestamp("2024-01-01 07:00", tz="UTC")]
    assert not h7["vis_bajo_niebla"]
    assert not h7["techo_bajo_techo_bajo"]  # sin techo = ilimitado


def test_valores_ausentes_no_rompen():
    obs = limpiar(pd.DataFrame([fila_iem("2024-01-01 06:00", vsby="M", tmpf="M")]))
    assert np.isnan(obs.loc[0, "vis_m"]) and np.isnan(obs.loc[0, "temp_c"])


@pytest.mark.parametrize("metar, esperado", [
    ("LEAS 151038Z 00000KT 0900 R29/1100N FG OVC002 08/08 Q1025", 900),
    ("LEAS 010030Z AUTO 22009KT 190V250 CAVOK 10/02 Q1018", 10_000),
    ("LEAS 010300Z AUTO 23008KT 200V260 9999 FEW086 10/02 Q1018", 10_000),
    ("LEAS 051317Z 30018G31KT 270V330 0800 R29/0800V1200U RA BCFG", 800),
    ("LEAS 010000Z AUTO 23008KT 1000NDV BR 10/09 Q1018", 1000),
    ("LEAS 121930Z AUTO 1/2SM 14/14 A3006", None),
])
def test_visibilidad_desde_texto(metar, esperado):
    v = vis_desde_metar(metar)
    assert np.isnan(v) if esperado is None else v == esperado


def test_metar_1000_m_no_cuenta_como_niebla():
    obs = limpiar(pd.DataFrame([fila_iem("2024-01-01 06:00", vsby="0.62",
                                         metar="LEAS 010600Z 22005KT 1000 BR 10/09 Q1018")]))
    h = a_horario(obs)
    assert obs.loc[0, "vis_m"] == 1000 and not h.loc[0, "vis_bajo_niebla"]


def test_rvr_minimo():
    assert rvr_min_desde_metar("LEAS 180700Z 30011KT 0050 R29/0175N R11/0125N FG VV001") == 125
    assert np.isnan(rvr_min_desde_metar("LEAS 010000Z AUTO 23008KT CAVOK 10/02 Q1018"))
