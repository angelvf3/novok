import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from novok.app import briefing as br
from novok.data.metar_texto import leer
from novok.modelo.variables import matriz, variables_base

AHORA = pd.Timestamp("2025-06-30 04:40", tz="UTC")


def metars(n_horas=48, niebla_desde=6, fin=AHORA):
    """METAR cada 30 min; niebla en las últimas `niebla_desde` horas."""
    textos = []
    for t in pd.date_range((fin - pd.Timedelta(hours=n_horas)).floor("30min"), fin, freq="30min"):
        if t > fin - pd.Timedelta(minutes=10):
            continue
        niebla = t >= fin - pd.Timedelta(hours=niebla_desde)
        cuerpo = "30004KT 0150 FG VV001 15/15" if niebla else "30008KT 9999 FEW020 17/12"
        textos.append(f"LEAS {t:%d%H%M}Z {cuerpo} Q1018")
    return textos


class Fuentes:
    def __init__(self, textos, nwp=None, tafs=None):
        self.textos, self._nwp, self._tafs = textos, nwp, tafs or []

    def metar(self):
        return self.textos

    def taf(self):
        return self._tafs

    def nwp(self):
        return self._nwp


def test_leer_metar_texto():
    f = leer("METAR LEAS 300430Z 30004KT 270V330 0150 R29/0450N FG VV001 18/18 Q1018 TEMPO 0400 BCFG", AHORA)
    assert f["valid"] == "2025-06-30 04:30"
    assert f["drct"] == 300 and f["sknt"] == 4
    assert f["skyc1"] == "VV" and f["skyl1"] == 100
    assert f["wxcodes"] == "FG"                       # el TEMPO no cuenta
    assert f["alti"] == pytest.approx(1018 / 33.8639)
    assert f["relh"] == pytest.approx(100, abs=0.1)


def test_sin_metar_reciente_no_se_emite_pronostico():
    viejos = metars(fin=AHORA - pd.Timedelta(hours=3))
    b = br.generar(Fuentes(viejos), AHORA)
    assert not b["calidad"]["pronostico_emitido"]
    assert b["pronostico"] == []
    assert any("min" in m for m in b["calidad"]["mensajes"])
    assert b["resumen"][0].startswith("Sin pronóstico")


@pytest.fixture
def modelos_falsos(tmp_path, monkeypatch):
    """Modelos diminutos con las mismas variables que los reales, para probar el circuito completo."""
    h, _, _ = br._observado(metars(), AHORA)
    base = variables_base(h)
    rng = np.random.default_rng(0)
    umbrales = {}
    for s in br.SUCESOS:
        for L in br.HORIZONTES_H:
            x, _ = matriz(h, base, s, L)
            y = rng.integers(0, 2, len(x))
            lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 1}, lgb.Dataset(x, y),
                      num_boost_round=3).save_model(str(tmp_path / f"novok_v1_{s}_{L}h.txt"))
            umbrales[f"v1_{s}_{L}h"] = 0.3
    (tmp_path / "umbrales.json").write_text(json.dumps(umbrales))
    monkeypatch.setattr(br, "DIR_MODELOS", tmp_path)
    return tmp_path


def test_briefing_completo_con_niebla(modelos_falsos):
    taf = (AHORA - pd.Timedelta(hours=1), "TAF LEAS 300300Z 3003/0103 30005KT 0300 FG VV001 BECMG 3008/3010 9999 NSC=")
    b = br.generar(Fuentes(metars(), tafs=[taf]), AHORA)
    assert b["calidad"]["pronostico_emitido"] and b["version_modelo"] == "v1"
    assert b["calidad"]["hora_base_utc"] == "2025-06-30T04:00Z"
    assert [p["hora_utc"] for p in b["pronostico"]] == ["2025-06-30T06:00Z", "2025-06-30T08:00Z",
                                                         "2025-06-30T11:00Z", "2025-06-30T17:00Z"]
    for p in b["pronostico"]:
        assert 0 <= p["niebla"]["p"] <= 1 and p["bajo_minimos"]["nivel"] in ("bajo", "moderado", "alto")
    assert b["pronostico"][0]["taf"]["vis_predominante_m"] == 300      # 06Z: niebla según el TAF
    assert b["pronostico"][2]["taf"]["vis_predominante_m"] == 10000    # 11Z: tras el BECMG
    assert b["ahora"]["niebla"] and b["episodio"]["tipo"] in ("advección", "radiación", "descenso de nubes",
                                                               "llovizna o lluvia")
    assert b["viento_pista"]["ahora"]["pista_favorable"] == "29"
    assert any("NOVOK v1" in f for f in b["resumen"])                   # sin NWP lo dice
    json.dumps(b, default=str)                                          # serializable


def test_nivel_por_probabilidad_absoluta():
    assert br._nivel(0.05) == "bajo" and br._nivel(0.2) == "moderado" and br._nivel(0.5) == "alto"


def nwp_falso():
    base = AHORA.floor("h")
    idx = pd.date_range(base - pd.Timedelta(days=3), base + pd.Timedelta(days=1), freq="1h")
    return pd.DataFrame({"nwp_ecmwf_hr": 95.0, "nwp_ecmwf_temp": 15.0, "nwp_ecmwf_rocio": 14.5,
                         "nwp_ecmwf_viento_kt": 8.0, "nwp_ecmwf_viento_dir": 200.0}, index=idx)


def test_briefing_v2_con_pronostico_numerico(modelos_falsos):
    h, _, _ = br._observado(metars(), AHORA)
    base = variables_base(h)
    nwp = nwp_falso()
    umbrales = json.loads((modelos_falsos / "umbrales.json").read_text())
    for s in br.SUCESOS:
        for L in br.HORIZONTES_H:
            x, _ = matriz(h, base, s, L, nwp=nwp)
            assert x.loc[h.attrs["i"], "ecmwf_hr_t"] == 95.0      # el pronóstico llega a la hora objetivo
            lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 1},
                      lgb.Dataset(x, np.arange(len(x)) % 2), num_boost_round=3).save_model(
                str(modelos_falsos / f"novok_v2_{s}_{L}h.txt"))
            umbrales[f"v2_{s}_{L}h"] = 0.2
    (modelos_falsos / "umbrales.json").write_text(json.dumps(umbrales))

    b = br.generar(Fuentes(metars(), nwp=nwp), AHORA)
    assert b["version_modelo"] == "v2" and b["calidad"]["nwp"]
    prev = b["viento_pista"]["previsto"]
    assert len(prev) == 12 and prev[0]["fuente"] == "ECMWF"
    assert prev[0]["cruzado_kt"] == pytest.approx(8 * abs(np.sin(np.deg2rad(200 - 290))), abs=0.1)

    # si el pronóstico no cubre las próximas horas, degrada a v1 y lo dice
    corto = nwp.loc[: AHORA - pd.Timedelta(hours=2)]
    b1 = br.generar(Fuentes(metars(), nwp=corto), AHORA)
    assert b1["version_modelo"] == "v1" and not b1["calidad"]["nwp"]
