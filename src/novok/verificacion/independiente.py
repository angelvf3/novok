"""F8 · Verificación independiente y valor económico.

La validación por años de F3 es honesta para medir el modelo, pero la elección entre v1 y v2 y los umbrales se
hicieron con los mismos años que luego se publican (2024-2025). Aquí se reserva una temporada entera que no se usa
para nada: ni para entrenar, ni para elegir modelo, ni para fijar umbrales.

    entrenamiento: horas objetivo  < corte - 1 año
    elección:      el año anterior al corte (parada temprana, umbral, v1 o v2)
    reentreno:     todo lo anterior al corte
    prueba:        desde el corte hasta el último METAR (por defecto, 2026)

En la temporada de prueba se compara NOVOK con lo que una aerolínea ya tiene: el TAF de AEMET, la persistencia
("lo que hay ahora seguirá") y la climatología. Además de las métricas meteorológicas (CSI, falsas alarmas, Brier),
se calculan tres cosas que entiende un responsable de operaciones:

1. Episodios avisados: de los episodios que empezaron en la temporada, cuántos se avisaron con 1, 3, 6 y 12 h.
2. Horas de falsa alarma al mes.
3. Valor económico (modelo coste/pérdida): por cada llegada, actuar (preparar contingencia) cuesta C; no actuar y
   que haya bajo mínimos cuesta L (la parte evitable de un desvío o un retraso). Con una probabilidad calibrada lo
   óptimo es actuar si p >= C/L. Se calcula cuánto ahorra cada método por cada 100 llegadas frente a no tener
   previsión, y qué parte del ahorro máximo posible (previsión perfecta) consigue. Los costes los pone cada
   aerolínea en config/costes.json; los que trae el repositorio son un ejemplo.

Simplificación: cada hora con llegadas es una decisión independiente y la llegada a la hora t se ve afectada si a
esa hora hay bajo mínimos. No modela esperas, ni que un episodio afecte a varias llegadas a la vez.

Uso:
    python -m novok.verificacion.independiente [--estacion LEXJ] [--corte 2026-01-01]
    python -m novok.red independiente [--actualizar]        # toda la red
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from novok.aeropuertos import activa, usar
from novok.config import ANIO_INICIO_FIABLE, RAIZ, dir_informes, dir_proc
from novok.modelo.entrenar import FONDO, HORIZONTES_H, REJILLA, SUCESOS, TINTA, TINTA2, _estilo, entrenar, umbral_csi
from novok.modelo.variables import matriz, variables_base
from novok.verificacion.baselines import (
    EVENTOS,
    NOMBRES,
    cargar_observado,
    climatologia,
    evento,
    taf_horario,
    taf_para_horizonte,
)
from novok.verificacion.metricas import brier, brier_skill, deterministas

CORTE_DEF = "2026-01-01"
MIN_DIAS_PRUEBA = 90
MIN_HORAS_TAF = 300   # horas de la temporada con TAF para compararse con AEMET
FICH_COSTES = RAIZ / "config" / "costes.json"
COSTES_EJEMPLO = {
    "comentario": ("EJEMPLO. Cada aerolínea pone sus cifras. coste_actuar_eur: preparar la contingencia para una "
                   "llegada (combustible extra, tripulación o handling de reserva, avisar al pasajero). "
                   "perdida_evitable_eur: lo que se ahorra si se actuó a tiempo y hubo bajo mínimos. Referencia: "
                   "EUROCONTROL estima un desvío de vuelo continental en 1.400-10.500 EUR (precios de 2022)."),
    "coste_actuar_eur": 300,
    "perdida_evitable_eur": 3000,
}
ALFAS = np.round(np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 0.96, 0.05)]), 3)
COLORES = {"novok": "#eda100", "taf": "#eb6834", "persistencia": "#2a78d6"}
ETIQ = {"novok": "NOVOK", "taf": "TAF de AEMET", "persistencia": "Persistencia", "climatologia": "Climatología",
        "novok_v1": "NOVOK v1", "novok_v2": "NOVOK v2", "taf_predominante": "TAF (predominante)"}


def costes() -> dict:
    if FICH_COSTES.exists():
        return json.loads(FICH_COSTES.read_text(encoding="utf-8"))
    return COSTES_EJEMPLO


# ------------------------------------------------------------------ valor económico

def gasto_relativo(h: float, f: float, s: float, alfa: float) -> float:
    """Gasto medio por llegada, en unidades de L, con tasa de acierto h y de falsa alarma f (frecuencia s)."""
    return f * alfa * (1 - s) + h * alfa * s + (1 - h) * s


def valor_relativo(y: np.ndarray, actuar: np.ndarray, alfa: float) -> float:
    """Parte del ahorro de una previsión perfecta que se consigue (1 = perfecto, 0 = como sin previsión)."""
    y, actuar = y.astype(bool), actuar.astype(bool)
    s = y.mean()
    if s == 0 or s == 1:
        return np.nan
    h = (actuar & y).sum() / y.sum()
    f = (actuar & ~y).sum() / (~y).sum()
    sin_prev, perfecta = min(alfa, s), alfa * s
    return float((sin_prev - gasto_relativo(h, f, s, alfa)) / (sin_prev - perfecta))


def ahorro_por_100(y: np.ndarray, actuar: np.ndarray, c: float, perdida: float) -> float:
    """EUR ahorrados por cada 100 llegadas frente a la mejor estrategia sin previsión (actuar siempre o nunca)."""
    y, actuar = y.astype(bool), actuar.astype(bool)
    gasto = np.where(actuar, c, np.where(y, perdida, 0.0)).mean()
    sin_prev = min(c, y.mean() * perdida)
    return float(100 * (sin_prev - gasto))


# ------------------------------------------------------------------ episodios

def inicios(y: pd.Series) -> pd.DatetimeIndex:
    """Horas en que empieza un episodio (suceso ahora y no en la hora anterior observada)."""
    v = y.dropna().astype(bool)
    previo = v.shift(1)
    seguido = (v.index.to_series().diff() == pd.Timedelta(hours=1)).to_numpy()
    return v.index[v.to_numpy() & (previo.to_numpy() == False) & seguido]  # noqa: E712


# ------------------------------------------------------------------ entrenamiento sin mirar la prueba

def modelo_hasta(x: pd.DataFrame, y: pd.Series, objetivo: pd.DatetimeIndex, corte: pd.Timestamp):
    """Entrena con horas objetivo < corte; el año anterior al corte sirve para parar, fijar umbral y comparar."""
    inicio_val = pd.Timestamp(year=corte.year - 1, month=1, day=1, tz="UTC")
    pre = (objetivo < corte) & (x.index.year >= ANIO_INICIO_FIABLE)
    tr, va = pre & (objetivo < inicio_val), pre & (objetivo >= inicio_val)
    if tr.sum() < 2000 or va.sum() < 1000 or y[tr].sum() < 10 or y[va].sum() == 0:
        return None
    m = entrenar(x[tr], y[tr], x[va], y[va])
    n = max(m.best_iteration, 50)
    p_va = m.predict(x[va], num_iteration=n)
    return {"modelo": entrenar(x[pre], y[pre], n_arboles=n), "umbral": umbral_csi(y[va].to_numpy(bool), p_va),
            "brier_val": brier(y[va], pd.Series(p_va)), "arboles": n}


def verificar(corte: pd.Timestamp, nwp: pd.DataFrame | None = None, h: pd.DataFrame | None = None) -> dict:
    """Verificación independiente del aeropuerto activo. Devuelve métricas, episodios, valor y predicciones."""
    h = cargar_observado() if h is None else h
    if (h.index.max() - corte).days < MIN_DIAS_PRUEBA:
        raise SystemExit(f"{activa().icao}: solo hay datos hasta {h.index.max():%Y-%m-%d}; hacen falta al menos "
                         f"{MIN_DIAS_PRUEBA} días después del corte ({corte:%Y-%m-%d}). Actualiza los METAR "
                         "(python -m novok.red independiente --actualizar) o adelanta el corte.")
    base = variables_base(h)
    th = taf_horario()
    k = costes()
    c, perdida = float(k["coste_actuar_eur"]), float(k["perdida_evitable_eur"])
    filas, episodios, valor, series = [], [], [], {}

    for suceso in SUCESOS:
        clim = climatologia(h, suceso)
        uv, ut = EVENTOS[suceso]
        y_obs = h[f"obs_{suceso}"]
        arranques = inicios(y_obs[y_obs.index >= corte])
        for L in HORIZONTES_H:
            x2, y = matriz(h, base, suceso, L, nwp=nwp)
            objetivo = x2.index + pd.Timedelta(hours=L + 1)
            ok = (h["hay_obs"].to_numpy() & y.notna().to_numpy())
            x2, y, objetivo = x2[ok], y[ok].astype(int), objetivo[ok]
            x1 = x2[[col for col in x2.columns if not col.startswith(("ecmwf_", "jma_"))]]

            candidatos = {"novok_v1": (x1, modelo_hasta(x1, y, objetivo, corte))}
            if nwp is not None:
                candidatos["novok_v2"] = (x2, modelo_hasta(x2, y, objetivo, corte))
            candidatos = {v: (x, m) for v, (x, m) in candidatos.items() if m is not None}
            if not candidatos:
                print(f"   {suceso} {L} h: datos insuficientes antes del corte")
                continue
            elegido = min(candidatos, key=lambda v: candidatos[v][1]["brier_val"])

            prueba = objetivo >= corte
            idx_t = objetivo[prueba]
            d = pd.DataFrame({"y": y[prueba].to_numpy(bool)}, index=idx_t)
            for v, (x, m) in candidatos.items():
                d[f"p_{v}"] = m["modelo"].predict(x[prueba])
                d[f"u_{v}"] = m["umbral"]
            d["p_novok"], d["u_novok"] = d[f"p_{elegido}"], d[f"u_{elegido}"]
            d["persistencia"] = h[f"obs_{suceso}"].shift(L + 1).reindex(idx_t).to_numpy()
            d["climatologia"] = clim.reindex(idx_t).to_numpy()
            if th is not None:
                t = taf_para_horizonte(th, L).reindex(idx_t)
                d["taf_predominante"] = evento(t["vis_prev_m"], t["techo_prev_ft"], uv, ut).astype(float)\
                    .where(t["emision"].notna()).to_numpy()
                d["taf"] = t[f"p_{suceso}"].where(t["emision"].notna()).to_numpy()
            d = d.dropna(subset=["persistencia", "climatologia"])
            if d.empty or not d["y"].any():
                print(f"   {suceso} {L} h: sin casos en la temporada de prueba")
                continue
            series[(suceso, L)] = d

            # 1) Todas las horas de la temporada: NOVOK frente a persistencia y climatología
            filas += medir(d, suceso, L, [*candidatos, "novok"], elegido, "todas", c, perdida)
            # 2) Solo las horas con TAF disponible: NOVOK frente al TAF en exactamente las mismas horas
            con_taf = "taf" in d and d["taf"].notna().sum() >= MIN_HORAS_TAF
            if con_taf:
                dt = d[d["taf"].notna()]
                if dt["y"].any():
                    filas += medir(dt, suceso, L, ["novok"], elegido, "horas_con_taf", c, perdida, con_taf=True)
                else:
                    con_taf = False

            muestras = {"todas": (d, ["novok", "persistencia", "climatologia"])}
            if con_taf:
                muestras["horas_con_taf"] = (dt, ["novok", "taf", "persistencia", "climatologia"])
            for muestra, (dd, metodos) in muestras.items():
                columna = {"novok": "p_novok"}
                for metodo in metodos:
                    pr = dd[columna.get(metodo, metodo)].to_numpy()
                    for a in ALFAS:
                        valor.append({"suceso": suceso, "horizonte_h": L, "muestra": muestra, "metodo": metodo,
                                      "alfa": a, "valor_relativo": valor_relativo(dd["y"].to_numpy(), pr >= a, a)})

            # Episodios: ¿se avisó del arranque con L h de antelación?
            for t0 in arranques:
                if t0 not in d.index:
                    continue
                r = d.loc[t0]
                fila = {"suceso": suceso, "inicio_utc": t0.isoformat(), "horizonte_h": L,
                        "novok": bool(r["p_novok"] >= r["u_novok"]), "p_novok": float(r["p_novok"])}
                if con_taf and pd.notna(r["taf"]):
                    fila["taf"] = bool(r["taf_predominante"] == 1) or bool(r["taf"] >= 0.3)
                episodios.append(fila)
            taf = (f"  | mismas {len(dt)} h con TAF: NOVOK {filas_csi(filas, 'novok', 'horas_con_taf'):.3f}"
                   f" TAF {filas_csi(filas, 'taf_predominante', 'horas_con_taf'):.3f}") if con_taf else "  | sin TAF"
            print(f"   {suceso:<18} {L:>2} h  [{elegido}]  CSI NOVOK {filas_csi(filas, 'novok'):.3f}"
                  f"  pers {filas_csi(filas, 'persistencia'):.3f}{taf}")

    return {"metricas": pd.DataFrame(filas), "episodios": pd.DataFrame(episodios), "valor": pd.DataFrame(valor),
            "series": series, "costes": k, "corte": corte}


def medir(d: pd.DataFrame, suceso: str, L: int, versiones: list[str], elegido: str, muestra: str,
          c: float, perdida: float, con_taf: bool = False) -> list[dict]:
    """Métricas de cada método en las horas de d. NOVOK usa su umbral fijado antes del corte."""
    yy = d["y"]
    meses = max((d.index.max() - d.index.min()).days / 30.44, 1)
    comunes = {"suceso": suceso, "horizonte_h": L, "muestra": muestra, "n_horas": len(d),
               "frecuencia_obs": yy.mean(), "desde": f"{d.index.min():%Y-%m-%d}", "hasta": f"{d.index.max():%Y-%m-%d}"}
    bs_clim = brier(yy, d["climatologia"])
    decisiones = {"persistencia": d["persistencia"] == 1}
    probs = {"climatologia": d["climatologia"], "persistencia": d["persistencia"]}
    for v in versiones:
        decisiones[v] = d[f"p_{v}"] >= d[f"u_{v}"]
        probs[v] = d[f"p_{v}"]
    if con_taf:
        decisiones["taf_predominante"] = d["taf_predominante"] == 1
        probs["taf"] = d["taf"]
    filas = []
    for metodo in [*decisiones, *(m for m in probs if m not in decisiones)]:
        fila = {**comunes, "metodo": metodo, "modelo_elegido": elegido if metodo == "novok" else ""}
        if metodo in decisiones:
            det = deterministas(yy, decisiones[metodo])
            fila |= {**det, "falsas_alarmas_h_mes": det["b"] / meses}
        if metodo in probs:
            bs = brier(yy, probs[metodo])
            fila |= {"brier": bs, "brier_skill_vs_clima": brier_skill(bs, bs_clim)}
            # coste/pérdida: con probabilidad calibrada, actuar si p >= C/L (sin ajustar nada en la prueba)
            actuar = probs[metodo].to_numpy() >= c / perdida
            fila["ahorro_eur_100_llegadas"] = ahorro_por_100(yy.to_numpy(), actuar, c, perdida)
            fila["valor_relativo"] = valor_relativo(yy.to_numpy(), actuar, c / perdida)
        filas.append(fila)
    return filas


def filas_csi(filas: list[dict], metodo: str, muestra: str = "todas") -> float:
    for f in reversed(filas):
        if f["metodo"] == metodo and f["muestra"] == muestra:
            return f.get("CSI", np.nan)
    return np.nan


def resumen_episodios(ep: pd.DataFrame) -> pd.DataFrame:
    """% de arranques de episodio avisados por método y antelación."""
    if ep.empty:
        return pd.DataFrame()
    metodos = [m for m in ("novok", "taf") if m in ep]
    r = ep.groupby(["suceso", "horizonte_h"]).agg(episodios=("inicio_utc", "nunique"),
                                                 **{f"avisados_{m}": (m, "mean") for m in metodos})
    return r.reset_index()


# ------------------------------------------------------------------ informe

def guardar(res: dict) -> dict:
    d = dir_informes()
    res["metricas"].round(4).to_csv(d / "independiente_resultados.csv", index=False)
    res["episodios"].to_csv(d / "independiente_episodios.csv", index=False)
    res["valor"].round(4).to_csv(d / "independiente_valor.csv", index=False)
    ep = resumen_episodios(res["episodios"])
    ep.round(3).to_csv(d / "independiente_episodios_resumen.csv", index=False)
    m = res["metricas"]
    resumen = {"aeropuerto": activa().icao, "corte": f"{res['corte']:%Y-%m-%d}", "costes": res["costes"],
               "filas": json.loads(m.round(4).to_json(orient="records")),
               "episodios": json.loads(ep.round(3).to_json(orient="records"))}
    (d / "independiente.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    grafico_valor(res)
    return resumen


def grafico_valor(res: dict, suceso: str = "bajo_minimos_cat1") -> None:
    v = res["valor"]
    if v.empty or suceso not in set(v["suceso"]):
        return
    hs = [L for L in (3, 6, 12) if L in set(v[v["suceso"] == suceso]["horizonte_h"])]
    fig, axes = plt.subplots(1, len(hs), figsize=(4.4 * len(hs), 4.2), sharey=True, facecolor=FONDO, squeeze=False)
    alfa_k = res["costes"]["coste_actuar_eur"] / res["costes"]["perdida_evitable_eur"]
    for ax, L in zip(axes[0], hs, strict=True):
        _estilo(ax)
        d = v[(v["suceso"] == suceso) & (v["horizonte_h"] == L)]
        muestra = "horas_con_taf" if (d["muestra"] == "horas_con_taf").any() else "todas"
        d = d[d["muestra"] == muestra]
        for metodo, color in COLORES.items():
            s = d[d["metodo"] == metodo].sort_values("alfa")
            if s.empty:
                continue
            ax.plot(s["alfa"], s["valor_relativo"].clip(lower=-0.2), color=color, lw=2, label=ETIQ[metodo])
        ax.axvline(alfa_k, color="#c9c8c1", lw=1, ls="--")
        ax.annotate("tus costes", (alfa_k, 0.95), xytext=(4, 0), textcoords="offset points", fontsize=8,
                    color=TINTA2)
        ax.axhline(0, color=REJILLA, lw=1)
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ["1:100", "1:50", "1:20", "1:10", "1:5", "1:2"])
        ax.minorticks_off()
        ax.set_xlim(0.01, 0.95)
        ax.set_ylim(-0.2, 1)
        titulo = f"Con {L} h de antelación" + (" (horas con TAF)" if muestra != "todas" else "")
        ax.set_title(titulo, loc="left", fontsize=10.5, color=TINTA)
        ax.set_xlabel("Coste de actuar / pérdida evitable", color=TINTA2)
    axes[0][0].set_ylabel("Valor (1 = previsión perfecta, 0 = sin previsión)", color=TINTA2)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(f"{activa().icao} · Valor económico en la temporada de prueba ({NOMBRES[suceso].split(' (')[0]}, "
                 f"desde {res['corte']:%d/%m/%Y})", x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    fig.savefig(dir_informes() / "independiente_valor.png", dpi=150, facecolor=FONDO)
    plt.close(fig)


def cargar_nwp() -> pd.DataFrame | None:
    ruta = dir_proc() / "nwp.parquet"
    return pd.read_parquet(ruta) if ruta.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--estacion", default=None)
    ap.add_argument("--corte", default=CORTE_DEF, help="primer día de la temporada de prueba (UTC)")
    args = ap.parse_args()
    if args.estacion:
        usar(args.estacion)
    np.seterr(all="ignore")
    corte = pd.Timestamp(args.corte, tz="UTC")
    print(f"{activa().icao} · verificación independiente desde {corte:%Y-%m-%d}")
    guardar(verificar(corte, cargar_nwp()))
    print(f"Informes en {dir_informes()}")


if __name__ == "__main__":
    main()
