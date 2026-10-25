"""F3 · Modelo NOVOK: probabilidad de niebla / bajo mínimos por horizonte, con validación temporal estricta.

Para cada año de prueba Y (2020-2025):
  1. Se entrena con los años < Y-1 y se usa Y-1 como validación: parada temprana y umbral de decisión
     (el que maximiza el CSI en Y-1).
  2. Se reentrena con todos los años < Y con ese número de árboles y se predice Y.
Así el modelo nunca ve el año que se evalúa ni años posteriores.

Se compara en las mismas horas con persistencia y climatología (F2).

Uso:
    python -m novok.modelo.entrenar
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from novok.aeropuertos import activa, cli_estacion
from novok.config import ANIO_INICIO_FIABLE, dir_informes, dir_modelos, dir_proc
from novok.modelo.variables import matriz, variables_base
from novok.verificacion.baselines import NOMBRES, cargar_observado, climatologia
from novok.verificacion.metricas import brier, brier_skill, deterministas

SUCESOS = ["niebla", "bajo_minimos_cat1"]
HORIZONTES_H = [1, 3, 6, 12]
ANIOS_PRUEBA = list(range(ANIO_INICIO_FIABLE + 3, 2026))  # 2020-2025: al menos 2 años de entrenamiento

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}
MAX_ARBOLES = 2000


def guardar_umbral(clave: str, valor: float) -> None:
    """Guarda el umbral de decisión de un modelo en modelos/umbrales.json (lo usa el briefing)."""
    import json

    dir_modelos().mkdir(parents=True, exist_ok=True)
    f = dir_modelos() / "umbrales.json"
    datos = json.loads(f.read_text()) if f.exists() else {}
    datos[clave] = round(float(valor), 3)
    f.write_text(json.dumps(dict(sorted(datos.items())), indent=2))


def umbral_csi(y: np.ndarray, p: np.ndarray) -> float:
    """Umbral de probabilidad que maximiza el CSI."""
    mejor, mejor_csi = 0.5, -1.0
    for u in np.arange(0.05, 0.9, 0.01):
        pred = p >= u
        a = np.sum(pred & y)
        csi = a / max(np.sum(pred | y), 1)
        if csi > mejor_csi:
            mejor, mejor_csi = u, csi
    return float(mejor)


def entrenar(x_tr, y_tr, x_va=None, y_va=None, n_arboles=None) -> lgb.Booster:
    dtr = lgb.Dataset(x_tr, y_tr)
    if n_arboles is not None:
        return lgb.train(PARAMS, dtr, num_boost_round=n_arboles)
    dva = lgb.Dataset(x_va, y_va, reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=MAX_ARBOLES, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def validacion_temporal(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Predicciones fuera de muestra para cada año de prueba (ventana creciente)."""
    anio = x.index.year
    salidas = []
    for Y in ANIOS_PRUEBA:
        tr_in = (anio >= ANIO_INICIO_FIABLE) & (anio < Y - 1)
        va = anio == Y - 1
        te = anio == Y
        # Aeropuertos con menos historia: se saltan los años sin datos suficientes para entrenar
        if tr_in.sum() < 2000 or va.sum() < 1000 or te.sum() == 0 or y[tr_in].sum() < 10:
            continue
        m = entrenar(x[tr_in], y[tr_in], x[va], y[va])
        n = max(m.best_iteration, 50)
        umbral = umbral_csi(y[va].to_numpy(bool), m.predict(x[va], num_iteration=n))
        final = entrenar(x[tr_in | va], y[tr_in | va], n_arboles=n)
        salidas.append(pd.DataFrame({"y": y[te].astype(bool), "p": final.predict(x[te]), "umbral": umbral,
                                     "anio": Y, "arboles": n}, index=x.index[te]))
    if not salidas:
        raise SystemExit("Datos insuficientes para validar el modelo: hacen falta al menos 3 años de METAR "
                         f"desde {ANIO_INICIO_FIABLE} con algún caso del suceso.")
    return pd.concat(salidas)


def importancia_shap(x: pd.DataFrame, y: pd.Series, n_arboles: int) -> pd.Series:
    """Importancia media |SHAP| (contribuciones de LightGBM) con un modelo entrenado en todo el periodo."""
    m = entrenar(x, y, n_arboles=n_arboles)
    contrib = m.predict(x, pred_contrib=True)[:, :-1]  # última columna = valor base
    return pd.Series(np.abs(contrib).mean(axis=0), index=x.columns).sort_values(ascending=False), m


def main() -> None:
    cli_estacion()
    h = cargar_observado()
    base = variables_base(h)
    filas, predicciones, importancias = [], [], {}
    dir_modelos().mkdir(parents=True, exist_ok=True)

    for suceso in SUCESOS:
        clim_t = climatologia(h, suceso)  # indexada por hora objetivo t
        for L in HORIZONTES_H:
            x, y = matriz(h, base, suceso, L)
            ok = h["hay_obs"].to_numpy() & y.notna().to_numpy() & (x.index.year >= ANIO_INICIO_FIABLE)
            x, y = x[ok], y[ok].astype(int)
            oos = validacion_temporal(x, y)

            # Baselines en exactamente las mismas horas
            t = oos.index + pd.Timedelta(hours=L + 1)
            oos["persistencia"] = x.loc[oos.index, "obs_bajo_minimos" if suceso != "niebla" else "obs_niebla"]
            oos["climatologia"] = clim_t.reindex(t).to_numpy()
            oos = oos.dropna(subset=["climatologia", "persistencia"])
            yy = oos["y"]
            bs_clim = brier(yy, oos["climatologia"])
            comunes = {"suceso": suceso, "horizonte_h": L, "n_horas": len(oos), "frecuencia_obs": yy.mean()}
            bs_novok = brier(yy, oos["p"])
            filas.append({**comunes, "metodo": "novok", **deterministas(yy, oos["p"] >= oos["umbral"]),
                          "brier": bs_novok, "brier_skill_vs_clima": brier_skill(bs_novok, bs_clim),
                          "umbral_medio": oos["umbral"].mean()})
            filas.append({**comunes, "metodo": "persistencia", **deterministas(yy, oos["persistencia"] == 1),
                          "brier": brier(yy, oos["persistencia"]),
                          "brier_skill_vs_clima": brier_skill(brier(yy, oos["persistencia"]), bs_clim)})
            filas.append({**comunes, "metodo": "climatologia", "brier": bs_clim, "brier_skill_vs_clima": 0.0})
            predicciones.append(oos.assign(suceso=suceso, horizonte_h=L,
                                           hora_obj_utc=oos.index + pd.Timedelta(hours=L + 1)))
            print(f"{suceso:<18} L={L:>2} h  CSI NOVOK {filas[-3]['CSI']:.3f} vs persistencia {filas[-2]['CSI']:.3f}"
                  f"  | BSS NOVOK {filas[-3]['brier_skill_vs_clima']:+.3f}")

            # Modelo final con todos los años (para la app) e importancia de variables
            n_final = int(oos["arboles"].iloc[-1])
            imp, modelo = importancia_shap(x, y, n_final)
            importancias[(suceso, L)] = imp
            modelo.save_model(str(dir_modelos() / f"novok_v1_{suceso}_{L}h.txt"))
            guardar_umbral(f"v1_{suceso}_{L}h", oos["umbral"].iloc[-1])

    res = pd.DataFrame(filas)
    dir_informes().mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(dir_informes() / "modelo_resultados.csv", index=False)
    pd.concat(predicciones).to_parquet(dir_proc() / "predicciones_novok.parquet")
    pd.DataFrame(importancias).round(5).to_csv(dir_informes() / "modelo_importancia_shap.csv")

    graficos(res, pd.concat(predicciones), importancias)
    print(f"\nInformes en {dir_informes()}")


# ------------------------------------------------------------------ gráficos

COLORES = {"persistencia": "#2a78d6", "novok": "#eda100"}  # mismos slots que en F2
ETIQ = {"persistencia": "Persistencia", "novok": "NOVOK"}
TINTA, TINTA2, FONDO, REJILLA = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5df"
NOMBRES_VAR = {
    "hr_pct": "Humedad relativa", "temp_c": "Temperatura", "dif_rocio_c": "Diferencia temperatura–rocío",
    "dif_rocio_min6": "Mín. diferencia T–rocío (6 h)", "doy_obj_sin": "Época del año (sen)",
    "doy_obj_cos": "Época del año (cos)", "hora_obj_sin": "Hora del día (sen)", "hora_obj_cos": "Hora del día (cos)",
    "viento_u": "Viento componente O→E", "viento_v": "Viento componente S→N", "viento_kt": "Velocidad del viento",
    "techo_ft": "Techo de nubes", "vis_log": "Visibilidad actual", "vis_log_min6": "Mín. visibilidad (6 h)",
    "tend3_hr": "Tendencia humedad (3 h)", "tend3_qnh": "Tendencia presión (3 h)", "qnh_hpa": "Presión",
    "tend3_temp": "Tendencia temperatura (3 h)", "tend3_dif_rocio": "Tendencia T–rocío (3 h)",
    "horas_desde_niebla": "Horas desde la última niebla", "niebla_ult24h": "Horas de niebla (24 h)",
    "suceso_ayer_misma_hora": "Niebla ayer a esta hora", "obs_niebla": "Niebla ahora",
    "rvr_log": "RVR actual", "horas_desde_bajo_minimos": "Horas desde bajo mínimos",
}


def _estilo(ax):
    ax.set_facecolor(FONDO)
    ax.grid(axis="y", color=REJILLA, lw=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c9c8c1")
    ax.tick_params(colors=TINTA2)


def graficos(res, pred, importancias) -> None:
    # 1) CSI por antelación
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, facecolor=FONDO)
    for ax, suc in zip(axes, SUCESOS, strict=True):
        _estilo(ax)
        d = res[(res["suceso"] == suc) & res["CSI"].notna()]
        for metodo, color in COLORES.items():
            s = d[d["metodo"] == metodo].sort_values("horizonte_h")
            ax.plot(s["horizonte_h"], s["CSI"], color=color, lw=2, marker="o", ms=8,
                    markeredgecolor=FONDO, markeredgewidth=2, label=ETIQ[metodo])
            u = s.iloc[-1]
            ax.annotate(ETIQ[metodo], (u["horizonte_h"], u["CSI"]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=TINTA2)
        ax.set_title(NOMBRES[suc], fontsize=10.5, loc="left", color=TINTA)
        ax.set_xscale("log")
        ax.set_xticks(HORIZONTES_H, [f"{x} h" for x in HORIZONTES_H])
        ax.minorticks_off()
        ax.set_xlim(0.8, 25)
        ax.set_xlabel("Antelación del pronóstico", color=TINTA2)
    axes[0].set_ylabel("CSI (1 = perfecto)", color=TINTA2)
    axes[0].set_ylim(0, None)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(f"{activa().icao} · NOVOK frente a persistencia (validación por años 2020–2025)", x=0.01, ha="left",
                 fontsize=12, color=TINTA)
    fig.tight_layout()
    fig.savefig(dir_informes() / "modelo_csi.png", dpi=150, facecolor=FONDO)
    plt.close(fig)

    # 2) Importancia de variables (niebla, 3 h)
    imp = importancias[("niebla", 3)].head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=FONDO)
    _estilo(ax)
    ax.grid(axis="x", color=REJILLA, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.barh([NOMBRES_VAR.get(v, v) for v in imp.index], imp.values, color="#2a78d6", height=0.6)
    ax.set_xlabel("Importancia media |SHAP|", color=TINTA2)
    ax.set_title("¿Qué mira NOVOK para predecir niebla a 3 h?", loc="left", fontsize=11, color=TINTA)
    fig.tight_layout()
    fig.savefig(dir_informes() / "modelo_importancia.png", dpi=150, facecolor=FONDO)
    plt.close(fig)

    # 3) Fiabilidad (niebla, 3 h): ¿el 30 % ocurre el 30 % de las veces?
    d = pred[(pred["suceso"] == "niebla") & (pred["horizonte_h"] == 3)]
    bordes = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001])
    grupo = pd.cut(d["p"], bordes, right=False)
    tabla = d.groupby(grupo, observed=True).agg(p=("p", "mean"), f=("y", "mean"), n=("y", "size"))
    tabla = tabla[tabla["n"] >= 20]
    fig, ax = plt.subplots(figsize=(5.2, 5), facecolor=FONDO)
    _estilo(ax)
    ax.plot([0, 1], [0, 1], color="#c9c8c1", lw=1, ls="--")
    ax.plot(tabla["p"], tabla["f"], color="#eda100", lw=2, marker="o", ms=8, markeredgecolor=FONDO,
            markeredgewidth=2)
    for _, r in tabla.iterrows():
        ax.annotate(f"n={int(r['n']):,}".replace(",", "."), (r["p"], r["f"]), xytext=(6, -10),
                    textcoords="offset points", fontsize=7.5, color=TINTA2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidad pronosticada", color=TINTA2)
    ax.set_ylabel("Frecuencia observada", color=TINTA2)
    ax.set_title("Fiabilidad · niebla a 3 h", loc="left", fontsize=11, color=TINTA)
    fig.tight_layout()
    fig.savefig(dir_informes() / "modelo_fiabilidad.png", dpi=150, facecolor=FONDO)
    plt.close(fig)


if __name__ == "__main__":
    main()
