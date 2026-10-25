import numpy as np
import pandas as pd
import pytest

from novok.data.descargar_taf import extraer_tafs
from novok.taf.parser import SIN_TECHO, a_horario, elementos, parsear, probabilidad, taf_a_horario

REF = pd.Timestamp("2024-06-10 11:00", tz="UTC")
T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def fila(df, hora):
    return df.set_index("hora_utc").loc[T(hora)]


def test_elementos_basicos():
    e = elementos("30008G20KT 0300 FG VV001".split())
    assert e == {"viento_dir": 300, "viento_kt": 8, "racha_kt": 20, "vis_m": 300, "techo_ft": 100}


def test_cavok_y_nsc_sin_techo():
    assert elementos(["CAVOK"]) == {"vis_m": 10_000, "techo_ft": SIN_TECHO}
    assert elementos(["9999", "NSC"])["techo_ft"] == SIN_TECHO
    assert elementos(["FEW005", "SCT010", "BKN030"])["techo_ft"] == 3000


def test_cabecera_y_periodo():
    taf = parsear("TAF LEAS 101100Z 1012/1112 30008KT 9999 FEW020=", REF)
    assert taf.estacion == "LEAS" and not taf.enmienda
    assert taf.emision == T("2024-06-10 11:00")
    assert (taf.inicio, taf.fin) == (T("2024-06-10 12:00"), T("2024-06-11 12:00"))


def test_hora_24_y_cambio_de_mes():
    taf = parsear("TAF LEAS 301700Z 3018/0124 27010KT 9999 SCT020=", pd.Timestamp("2024-06-30 17:00", tz="UTC"))
    assert taf.inicio == T("2024-06-30 18:00")
    assert taf.fin == T("2024-07-02 00:00")


def test_nil_y_cnl_se_ignoran():
    assert parsear("TAF LEAS 101100Z NIL=", REF) is None
    assert parsear("TAF AMD LEAS 101300Z 1012/1112 CNL=", REF) is None


def test_tempo_no_cambia_predominante_pero_si_peor_caso():
    df = taf_a_horario("TAF LEAS 101100Z 1012/1112 30008KT 9999 FEW020 TEMPO 1102/1108 0300 FG VV001=", REF)
    assert len(df) == 24
    f = fila(df, "2024-06-11 04:00")
    assert f.vis_prev_m == 10_000 and f.techo_prev_ft == SIN_TECHO
    assert f.vis_peor_m == 300 and f.techo_peor_ft == 100
    f = fila(df, "2024-06-11 08:00")  # fin del TEMPO (exclusivo)
    assert f.vis_peor_m == 10_000


def test_becmg_se_aplica_al_terminar_la_transicion():
    df = taf_a_horario("TAF LEAS 101100Z 1012/1112 30008KT 9999 SCT030 BECMG 1020/1022 2000 BR BKN004=", REF)
    assert fila(df, "2024-06-10 19:00").vis_prev_m == 10_000
    t = fila(df, "2024-06-10 21:00")  # en transición: predominante antigua, nueva como posible
    assert t.vis_prev_m == 10_000 and t.vis_peor_m == 2000
    f = fila(df, "2024-06-10 22:00")
    assert f.vis_prev_m == 2000 and f.techo_prev_ft == 400
    assert fila(df, "2024-06-11 10:00").techo_prev_ft == 400  # persiste


def test_fm_sustituye_desde_su_hora():
    df = taf_a_horario("TAF LEAS 101100Z 1012/1112 30008KT 9999 FEW020 FM110300 VRB02KT 0200 FG VV001=", REF)
    assert fila(df, "2024-06-11 02:00").vis_prev_m == 10_000
    f = fila(df, "2024-06-11 03:00")
    assert f.vis_prev_m == 200 and f.techo_prev_ft == 100 and np.isnan(f.viento_dir)


def test_prob40_tempo_y_probabilidades():
    df = taf_a_horario(
        "TAF LEAS 101100Z 1012/1112 30008KT 9999 SCT015 "
        "PROB40 TEMPO 1104/1108 0400 FG BKN002 PROB30 1100/1104 1500 BR=", REF)
    p = probabilidad(df, vis_umbral=1000).set_axis(df["hora_utc"])
    assert p[T("2024-06-11 05:00")] == pytest.approx(0.4)
    assert p[T("2024-06-11 01:00")] == 0  # PROB30 con 1500 m no baja de 1000
    assert probabilidad(df, vis_umbral=2000).set_axis(df["hora_utc"])[T("2024-06-11 01:00")] == pytest.approx(0.3)
    assert probabilidad(df, techo_umbral=200).set_axis(df["hora_utc"])[T("2024-06-11 05:00")] == pytest.approx(0.4)


def test_predominante_bajo_umbral_da_probabilidad_1():
    df = taf_a_horario("TAF LEAS 101100Z 1012/1112 00000KT 0300 FG VV001=", REF)
    assert (probabilidad(df, vis_umbral=550) == 1).all()


def test_amd_marca_enmienda():
    taf = parsear("TAF AMD LEAS 100845Z 1009/1018 31005KT 0500 FG VV002 BECMG 1011/1013 9999 NSC=", REF)
    assert taf.enmienda and taf.emision == T("2024-06-10 08:45")
    df = a_horario(taf)
    assert fila(df, "2024-06-10 14:00").techo_prev_ft == SIN_TECHO


def test_extraer_tafs_de_texto_ogimet():
    texto = """<pre>
# TAF from LEAS
202406101100 TAF LEAS 101100Z 1012/1112 30008KT 9999 FEW020
                  TEMPO 1102/1108 0300 FG VV001=
202406101700 TAF LEAS 101700Z 1018/1118 30005KT CAVOK=
</pre>"""
    tafs = extraer_tafs(texto)
    assert len(tafs) == 2
    assert tafs[0] == ("202406101100", "TAF LEAS 101100Z 1012/1112 30008KT 9999 FEW020 TEMPO 1102/1108 0300 FG VV001=")


def test_extraer_tafs_variantes_de_formato():
    texto = """202406101100 LEAS 101100Z 1012/1112 30008KT 9999 FEW020
    202406101400 TAF AMD LEAS 101400Z 1014/1112 30008KT 0300 FG VV001=
    202406101030 METAR LEAS 101030Z 30008KT 9999 FEW020 17/15 Q1015="""
    tafs = extraer_tafs(texto)
    assert [t for _, t in tafs] == [
        "TAF LEAS 101100Z 1012/1112 30008KT 9999 FEW020=",
        "TAF AMD LEAS 101400Z 1014/1112 30008KT 0300 FG VV001=",
    ]


def test_grabar_taf_vivo_sin_duplicados(tmp_path):
    from datetime import UTC, datetime

    from novok.data.grabar_taf_vivo import grabar

    f = tmp_path / "taf_LEAS.txt"
    taf = "TAF AMD LEAS 071013Z 0710/0809 30009KT 9999 FEW005 BKN020 PROB30 TEMPO 0803/0809 3000 RA="
    assert grabar([taf], f, datetime(2026, 10, 7, 10, 37, tzinfo=UTC)) == 1
    assert grabar([taf], f, datetime(2026, 10, 7, 11, 7, tzinfo=UTC)) == 0
    linea = f.read_text().strip()
    assert linea.startswith("202610071037 TAF AMD LEAS")
    assert extraer_tafs(linea) == [("202610071037", taf)]
