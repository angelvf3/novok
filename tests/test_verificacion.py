import numpy as np
import pandas as pd
import pytest

from novok.verificacion import baselines as bl
from novok.verificacion.metricas import brier, brier_skill, deterministas


def test_metricas_deterministas():
    obs = pd.Series([1, 1, 0, 0, 1, 0], dtype=bool)
    pred = pd.Series([1, 0, 1, 0, 1, 0], dtype=bool)
    m = deterministas(obs, pred)
    assert (m["a"], m["b"], m["c"], m["d"]) == (2, 1, 1, 2)
    assert m["POD"] == pytest.approx(2 / 3)
    assert m["FAR"] == pytest.approx(1 / 3)
    assert m["CSI"] == pytest.approx(0.5)
    assert m["sesgo"] == pytest.approx(1.0)


def test_brier_y_skill():
    obs = pd.Series([1, 0, 0, 0], dtype=bool)
    assert brier(obs, pd.Series([1.0, 0, 0, 0])) == 0
    bs_clim = brier(obs, pd.Series([0.25] * 4))
    assert brier_skill(brier(obs, pd.Series([0.5, 0, 0, 0])), bs_clim) > 0


def rejilla(valores, inicio="2020-01-01 00:00"):
    idx = pd.date_range(inicio, periods=len(valores), freq="1h", tz="UTC")
    return pd.DataFrame({"obs_niebla": pd.Series(valores, index=idx, dtype=float)})


def test_persistencia_usa_solo_horas_completas_antes_de_emitir():
    h = rejilla([0, 0, 0, 1, 1, 0, 0])
    p = bl.persistencia(h, "niebla", 1)
    # Para t=05:00 con L=1 se emite a las 04:00 y la última hora completa es 03:00 (=1)
    assert p.iloc[5] == 1
    # Para t=04:00 se emite a las 03:00; la última hora completa es 02:00 (=0): no ve la niebla de 03:00
    assert p.iloc[4] == 0


def test_climatologia_deja_fuera_el_propio_anio(monkeypatch):
    monkeypatch.setattr(bl, "ANIO_INICIO_FIABLE", 2020)
    idx = pd.DatetimeIndex([pd.Timestamp(f"{a}-06-15 04:00", tz="UTC") for a in (2020, 2021, 2022)])
    h = pd.DataFrame({"obs_niebla": [1.0, 0.0, 0.0]}, index=idx)
    p = bl.climatologia(h, "niebla")
    assert p.iloc[0] == 0          # 2020 solo ve 2021 y 2022 (sin niebla)
    assert p.iloc[1] == pytest.approx(0.5)


def test_taf_para_horizonte_no_usa_tafs_futuros():
    T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    th = pd.DataFrame({
        "emision": [T("2024-06-10 05:00"), T("2024-06-10 11:00")],
        "hora_utc": [T("2024-06-10 14:00"), T("2024-06-10 14:00")],
        "vis_prev_m": [9999.0, 300.0],
    })
    assert bl.taf_para_horizonte(th, 1).loc[T("2024-06-10 14:00"), "vis_prev_m"] == 300
    assert bl.taf_para_horizonte(th, 6).loc[T("2024-06-10 14:00"), "vis_prev_m"] == 9999


def test_evento_techo_sin_techo_no_cuenta():
    vis = pd.Series([9999.0, 9999.0])
    techo = pd.Series([np.nan, 200.0])
    assert bl.evento(vis, techo, None, 200).tolist() == [False, True]
