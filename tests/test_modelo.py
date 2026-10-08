import numpy as np
import pandas as pd

from novok.modelo.entrenar import umbral_csi
from novok.modelo.variables import matriz, variables_base


def tabla(n=60, semilla=0):
    rng = np.random.default_rng(semilla)
    idx = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    vis = rng.choice([300.0, 800.0, 10_000.0], size=n, p=[0.1, 0.1, 0.8])
    h = pd.DataFrame({
        "vis_m": vis, "rvr_min_m": np.nan, "techo_ft": np.where(vis < 1000, 100.0, np.nan),
        "dif_rocio_c": rng.uniform(0, 5, n), "hr_pct": rng.uniform(70, 100, n), "temp_c": rng.uniform(10, 20, n),
        "qnh_hpa": rng.uniform(1010, 1025, n), "viento_kt": rng.uniform(0, 15, n), "racha_kt": np.nan,
        "viento_dir": rng.uniform(0, 360, n), "n_obs": 2.0, "fenomenos": np.where(vis < 1000, "FG", ""),
        "hay_obs": True,
    }, index=idx)
    h["obs_niebla"] = (h["vis_m"] < 1000).astype(float)
    h["obs_bajo_minimos_cat1"] = ((h["vis_m"] < 550) | (h["techo_ft"].fillna(np.inf) <= 200)).astype(float)
    return h


def test_variables_no_miran_al_futuro():
    h = tabla()
    i = 30
    original = variables_base(h).iloc[i]
    alterada = h.copy()
    alterada.iloc[i + 1:, alterada.columns.get_indexer(["vis_m", "dif_rocio_c", "hr_pct", "temp_c", "qnh_hpa",
                                                         "viento_kt", "obs_niebla", "obs_bajo_minimos_cat1"])] = 0.0
    pd.testing.assert_series_equal(original, variables_base(alterada).iloc[i], check_names=False)


def test_objetivo_alineado_con_horizonte():
    h = tabla()
    base = variables_base(h)
    for L in (1, 3, 6):
        x, y = matriz(h, base, "niebla", L)
        assert y.iloc[10] == h["obs_niebla"].iloc[10 + L + 1]
        if "suceso_ayer_misma_hora" in x:
            # t - 24 h, siempre en el pasado respecto a la hora observada
            assert x["suceso_ayer_misma_hora"].iloc[40] == h["obs_niebla"].iloc[40 + L + 1 - 24]


def test_umbral_csi():
    y = np.array([0, 0, 1, 1, 0, 1], bool)
    p = np.array([0.1, 0.2, 0.7, 0.6, 0.3, 0.8])
    assert 0.3 <= umbral_csi(y, p) <= 0.6


def test_variables_nwp_usan_el_pronostico_de_la_hora_objetivo():
    from novok.modelo.variables import variables_nwp

    h = tabla()
    nwp = pd.DataFrame({"nwp_ecmwf_hr": np.arange(len(h), dtype=float),
                        "nwp_ecmwf_temp": 15.0, "nwp_ecmwf_rocio": 14.0}, index=h.index)
    for paso in (2, 7):
        x = variables_nwp(h, nwp, paso)
        assert x["ecmwf_hr_t"].iloc[10] == 10 + paso          # pronóstico válido en t = i + paso
        assert x["ecmwf_hr_cambio"].iloc[10] == paso          # cambio previsto entre i y t
        assert x["ecmwf_hr_error_ahora"].iloc[10] == h["hr_pct"].iloc[10] - 10  # error en i
        assert x["ecmwf_dif_rocio_t"].iloc[10] == 1.0


def test_descargar_nwp_nombres_de_columnas():
    from novok.data.descargar_nwp import a_tabla

    datos = {"hourly": {"time": ["2024-06-10T00:00", "2024-06-10T01:00"],
                        "relative_humidity_2m_previous_day1": [95, None],
                        "cloud_cover_low_previous_day1": [80, 90]}}
    t = a_tabla(datos, "jma_gsm")
    assert list(t.columns) == ["nwp_jma_hr", "nwp_jma_nubes_bajas"]
    assert str(t.index.tz) == "UTC"
