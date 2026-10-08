import numpy as np
import pandas as pd
import pytest

from novok.episodios.disipacion import climatologia_condicionada, elevacion_solar
from novok.episodios.episodios import clasificar, detectar


def rejilla(vis, viento=5.0, techo=np.nan, fen="", inicio="2024-06-10 00:00"):
    idx = pd.date_range(inicio, periods=len(vis), freq="1h", tz="UTC")
    n = len(vis)
    def col(v):
        return v if isinstance(v, list) else [v] * n
    return pd.DataFrame({"vis_m": [float(v) for v in vis], "hay_obs": True, "viento_kt": col(viento),
                         "techo_ft": col(techo), "fenomenos": col(fen)}, index=idx)


def test_episodio_simple_y_union_de_huecos_de_una_hora():
    h = rejilla([9999, 300, 300, 9999, 400, 9999, 9999, 500, 9999])
    ep = detectar(h)
    # 300,300,(hueco 1 h),400 -> un episodio de 4 h; 500 aislado tras 2 h despejadas -> otro
    assert list(ep["duracion_h"]) == [4, 1]
    assert ep.loc[0, "inicio"] == h.index[1] and ep.loc[0, "fin"] == h.index[5]
    assert ep["completo"].all()
    assert ep.loc[0, "vis_min_m"] == 300


def test_episodio_al_borde_no_es_completo():
    ep = detectar(rejilla([300, 300, 9999]))
    assert not ep.loc[0, "completo"]


@pytest.mark.parametrize("kwargs, esperado", [
    ({"fen": ["", "-DZ", "FG", "", ""]}, "llovizna o lluvia"),
    ({"techo": [800.0, 500.0, 100.0, np.nan, np.nan]}, "descenso de nubes"),
    ({"viento": 2.0}, "radiación"),
    ({"viento": 9.0}, "advección"),
])
def test_clasificacion(kwargs, esperado):
    # inicio a las 03 UTC (05 local): de noche
    h = rejilla([9999, 9999, 200, 200, 9999], inicio="2024-06-10 01:00", **kwargs)
    ep = detectar(h)
    assert clasificar(ep, h).iloc[0] == esperado


def test_radiacion_no_de_dia():
    h = rejilla([9999, 9999, 200, 200, 9999], viento=2.0, inicio="2024-06-10 11:00")  # 13-14 h local
    assert clasificar(detectar(h), h).iloc[0] == "advección"


def test_elevacion_solar():
    t = pd.DatetimeIndex(["2024-06-21 12:30", "2024-06-21 00:00", "2024-12-21 12:30"], tz="UTC")
    e = elevacion_solar(t)
    assert 66 < e[0] < 71        # mediodía solar de verano en Asturias ~70°
    assert e[1] < 0              # medianoche
    assert 20 < e[2] < 25        # mediodía de invierno ~23°


def test_climatologia_condicionada_usa_respaldo():
    tr = pd.DataFrame({"transcurridas_h": [1, 1, 1, 2], "verano": [True, True, True, True],
                       "restantes_h": [0, 2, 4, 1]})
    te = pd.DataFrame({"transcurridas_h": [1, 2], "verano": [True, False]})
    c = climatologia_condicionada(tr, te)
    assert c.loc[0, "q50"] == 2
    assert c.loc[1, "q50"] == 1   # (2, invierno) no existe -> respaldo por horas transcurridas
