"""F4 · ¿Cuándo se levanta la niebla?

Para cada hora con niebla dentro de un episodio, con lo observado hasta esa hora, se predicen las horas
que faltan hasta que despeje (0 = la hora siguiente ya está despejada), con un intervalo del 80 %
(cuantiles 10, 50 y 90 de LightGBM).

Referencia a batir: la climatología condicionada. Mediana y cuantiles de las horas restantes en episodios
pasados con el mismo tiempo transcurrido y la misma época (abril-septiembre u octubre-marzo).

Validación por años con ventana creciente: para el año Y solo se usan episodios de años anteriores.

Uso:
    python -m novok.episodios.disipacion
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from novok.config import ANIO_INICIO_FIABLE, RAIZ, REPORTS
from novok.episodios.episodios import TIPOS, ZONA_LOCAL, episodios
from novok.modelo.variables import variables_base
from novok.verificacion.baselines import cargar_observado

LAT, LON = 43.5636, -6.0346
ANIOS_PRUEBA = list(range(ANIO_INICIO_FIABLE + 3, 2026))
CUANTILES = (0.1, 0.5, 0.9)
PARAMS = {"learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 15, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 42}
N_ARBOLES = 250
DIR_MODELOS = RAIZ / "modelos"


def elevacion_solar(t_utc: pd.DatetimeIndex) -> np.ndarray:
    """Altura del sol sobre el horizonte (grados) en LEAS. Aproximación suficiente para este uso."""
    dia = t_utc.dayofyear.to_numpy()
    hora = (t_utc.hour + t_utc.minute / 60).to_numpy()
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + dia) / 365)
    angulo_horario = np.deg2rad(15 * (hora + LON / 15 - 12))
    lat = np.deg2rad(LAT)
    s = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(angulo_horario)
    return np.rad2deg(np.arcsin(np.clip(s, -1, 1)))


def filas_episodios(h: pd.DataFrame, ep: pd.DataFrame) -> pd.DataFrame:
    """Una fila por hora de niebla de cada episodio completo: variables en esa hora y horas restantes."""
    base = variables_base(h)
    filas = []
    for _, e in ep[ep["completo"]].iterrows():
        horas = pd.date_range(e["inicio"], e["fin"], freq="1h", inclusive="left")
        for k, t in enumerate(horas):
            filas.append({"hora_obs_utc": t, "inicio": e["inicio"], "tipo": e["tipo"],
                          "transcurridas_h": k + 1, "restantes_h": len(horas) - k - 1})
    d = pd.DataFrame(filas).set_index("hora_obs_utc")
    return completar_variables(d, base)


def completar_variables(d: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Añade las variables del modelo a filas con índice = hora observada y columnas inicio, tipo y
    transcurridas_h. Lo usan el entrenamiento y el briefing en directo."""
    d = d.join(base[["vis_log", "rvr_log", "techo_ft", "dif_rocio_c", "hr_pct", "temp_c", "viento_kt", "viento_u",
                     "viento_v", "tend1_vis_log", "tend3_vis_log", "tend1_dif_rocio", "tend3_temp", "tend3_qnh",
                     "fen_lluvia", "fen_llovizna"]])
    sig = d.index + pd.Timedelta(hours=1)
    local = sig.tz_convert(ZONA_LOCAL)
    d["sol_elev_sig"] = elevacion_solar(sig)
    d["sol_sube"] = elevacion_solar(sig + pd.Timedelta(hours=2)) - d["sol_elev_sig"]
    d["hora_sin"] = np.sin(2 * np.pi * local.hour / 24)
    d["hora_cos"] = np.cos(2 * np.pi * local.hour / 24)
    d["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    d["tipo_cod"] = d["tipo"].map({t: i for i, t in enumerate(TIPOS)}).astype(float)
    d["anio"] = d.index.year
    d["verano"] = local.month.isin(range(4, 10))
    return d


def entrenar_final(d: pd.DataFrame) -> None:
    """Modelos de cuantiles con todos los años, para el briefing en directo."""
    DIR_MODELOS.mkdir(exist_ok=True)
    cols = columnas_x(d)
    for q in CUANTILES:
        m = lgb.train({**PARAMS, "objective": "quantile", "alpha": q}, lgb.Dataset(d[cols], d["restantes_h"]),
                      num_boost_round=N_ARBOLES)
        m.save_model(str(DIR_MODELOS / f"disipacion_q{int(q * 100)}.txt"))


def columnas_x(d: pd.DataFrame) -> list[str]:
    return [c for c in d.columns if c not in ("inicio", "tipo", "restantes_h", "anio", "verano")]


def climatologia_condicionada(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """Cuantiles de horas restantes por (transcurridas, época) en el entrenamiento; con respaldo si hay pocos."""
    def tramo(x):
        return np.minimum(x, 6)
    out = pd.DataFrame(index=te.index)
    for q in CUANTILES:
        tabla = tr.groupby([tramo(tr["transcurridas_h"]), tr["verano"]])["restantes_h"].quantile(q)
        global_t = tr.groupby(tramo(tr["transcurridas_h"]))["restantes_h"].quantile(q)
        clave = list(zip(tramo(te["transcurridas_h"]), te["verano"], strict=True))
        v = [tabla.get(c, np.nan) for c in clave]
        out[f"q{int(q * 100)}"] = pd.Series(v, index=te.index).fillna(
            pd.Series(global_t.reindex(tramo(te["transcurridas_h"])).to_numpy(), index=te.index))
    return out


def validacion(d: pd.DataFrame) -> pd.DataFrame:
    cols = columnas_x(d)
    salidas = []
    for Y in ANIOS_PRUEBA:
        tr, te = d[(d["anio"] >= ANIO_INICIO_FIABLE) & (d["anio"] < Y)], d[d["anio"] == Y]
        if te.empty:
            continue
        o = te[["inicio", "tipo", "transcurridas_h", "restantes_h", "anio"]].copy()
        for q in CUANTILES:
            m = lgb.train({**PARAMS, "objective": "quantile", "alpha": q}, lgb.Dataset(tr[cols], tr["restantes_h"]),
                          num_boost_round=N_ARBOLES)
            o[f"m_q{int(q * 100)}"] = np.clip(m.predict(te[cols]), 0, None)
        clim = climatologia_condicionada(tr, te)
        for c in clim:
            o[f"c_{c}"] = clim[c]
        salidas.append(o)
    o = pd.concat(salidas)
    # cuantiles ordenados (q10 <= q50 <= q90)
    o[["m_q10", "m_q50", "m_q90"]] = np.sort(o[["m_q10", "m_q50", "m_q90"]].to_numpy(), axis=1)
    return o


def resumen(o: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for nombre, sel in {"todas las horas": o.index == o.index, "1.ª hora del episodio": o["transcurridas_h"] == 1,
                        "episodios de 3 h o más": (o["transcurridas_h"] + o["restantes_h"]) >= 3}.items():
        d = o[sel]
        for pref, metodo in (("m_", "NOVOK"), ("c_", "climatología")):
            # la verdad son horas enteras: ambas medianas se redondean a la hora para comparar igual
            err = (d[f"{pref}q50"].round() - d["restantes_h"]).abs()
            dentro = (d["restantes_h"] >= np.floor(d[f"{pref}q10"])) & (d["restantes_h"] <= np.ceil(d[f"{pref}q90"]))
            filas.append({"muestra": nombre, "metodo": metodo, "n": len(d), "MAE_h": err.mean(),
                          "acierto_±1h": (err <= 1).mean(), "cobertura_intervalo_80": dentro.mean(),
                          "anchura_intervalo_h": (d[f"{pref}q90"] - d[f"{pref}q10"]).mean()})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ gráficos
FONDO, TINTA, TINTA2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5df"
COLORES_TIPO = dict(zip(TIPOS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], strict=True))  # slots 1-4


def grafico_tipos(ep: pd.DataFrame, destino) -> None:
    c = ep[ep["completo"]]
    anios = c["inicio"].dt.year.nunique()
    tabla = (c.groupby([c["inicio"].dt.tz_convert(ZONA_LOCAL).dt.month, "tipo"]).size().unstack(fill_value=0)
             .reindex(columns=TIPOS, fill_value=0).reindex(range(1, 13), fill_value=0)) / anios
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=FONDO)
    ax.set_facecolor(FONDO)
    abajo = np.zeros(12)
    for tipo in TIPOS:
        ax.bar(meses, tabla[tipo], bottom=abajo, color=COLORES_TIPO[tipo], width=0.7, label=tipo,
               edgecolor=FONDO, linewidth=2)
        abajo += tabla[tipo].to_numpy()
    ax.set_ylabel("Episodios por año", color=TINTA2)
    ax.set_title(f"LEAS · Episodios de niebla por tipo y mes ({c['inicio'].dt.year.min()}–{c['inicio'].dt.year.max()})",
                 loc="left", fontsize=11.5, color=TINTA)
    ax.grid(axis="y", color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=TINTA2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(destino, dpi=150, facecolor=FONDO)
    plt.close(fig)


def main() -> None:
    h = cargar_observado()
    h = h[h.index.year >= ANIO_INICIO_FIABLE]
    ep = episodios(h)
    REPORTS.mkdir(exist_ok=True)
    ep.to_csv(REPORTS / "episodios_niebla.csv", index=False)

    c = ep[ep["completo"]]
    n_anios = c["inicio"].dt.year.nunique()
    tabla_tipos = c.groupby("tipo").agg(episodios_por_anio=("duracion_h", lambda s: len(s) / n_anios),
                                        duracion_mediana_h=("duracion_h", "median"),
                                        duracion_p90_h=("duracion_h", lambda s: s.quantile(0.9)),
                                        vis_min_mediana_m=("vis_min_m", "median")).reindex(TIPOS).round(1)
    print("Tipos de niebla (2017-2025):")
    print(tabla_tipos.to_string())
    tabla_tipos.to_csv(REPORTS / "tipos_niebla.csv")
    grafico_tipos(ep, REPORTS / "tipos_niebla.png")

    d = filas_episodios(h, ep)
    o = validacion(d)
    res = resumen(o)
    res.round(3).to_csv(REPORTS / "disipacion_resultados.csv", index=False)
    entrenar_final(d)
    print("\nDisipación · validación por años 2020-2025 (horas restantes):")
    print(res.round(2).to_string(index=False))
    print(f"\nInformes en {REPORTS}")


if __name__ == "__main__":
    main()
