import numpy as np
import pandas as pd
import pytest

from novok.data.viento import componentes, viento_en_pista


def test_viento_de_cara_puro():
    cru, cara = componentes(290, 15, 290)
    assert cru == pytest.approx(0, abs=1e-9)
    assert cara == pytest.approx(15)


def test_viento_cruzado_puro():
    cru, cara = componentes(200, 12, 290)
    assert cru == pytest.approx(12)
    assert cara == pytest.approx(0, abs=1e-9)


def test_viento_de_cola():
    _, cara = componentes(110, 10, 290)
    assert cara == pytest.approx(-10)


def horas(viento_dir, viento_kt, racha_kt=np.nan):
    return pd.DataFrame({"viento_dir": [viento_dir], "viento_kt": [viento_kt], "racha_kt": [racha_kt]})


def test_elige_cabecera_con_viento_de_cara():
    h = viento_en_pista(horas(270, 10))
    assert h.loc[0, "pista_favorable"] == "29"
    assert h.loc[0, "cruzado_kt"] == pytest.approx(10 * np.sin(np.deg2rad(20)))


def test_racha_y_aviso_de_cruzado():
    h = viento_en_pista(horas(200, 18, racha_kt=30))
    assert h.loc[0, "cruzado_kt"] == pytest.approx(18)
    assert h.loc[0, "cruzado_racha_kt"] == pytest.approx(30)
    assert h.loc[0, "aviso_cruzado"]


def test_calma_sin_direccion_es_cero():
    h = viento_en_pista(horas(np.nan, 0))
    assert h.loc[0, "cruzado_kt_11"] == 0 and h.loc[0, "cara_kt_29"] == 0


def test_direccion_variable_no_rompe():
    h = viento_en_pista(horas(np.nan, 5))
    assert h.loc[0, "pista_favorable"] is None
    assert np.isnan(h.loc[0, "cruzado_kt"])
    assert not h.loc[0, "aviso_cruzado"]


def test_aviso_de_cola_por_cabecera():
    h = viento_en_pista(horas(110, 15))  # viento del 110: de cara en la 11, de cola en la 29
    assert h.loc[0, "pista_favorable"] == "11"
    assert h.loc[0, "aviso_cola_29"] and not h.loc[0, "aviso_cola_11"]
