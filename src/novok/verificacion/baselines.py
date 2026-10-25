"""F2 · Baselines: ¿cómo de buenos son la persistencia, la climatología y el TAF?

Para cada hora objetivo t y horizonte L, el pronóstico se "emite" en T = t - L y solo
usa información disponible en T:

- Persistencia: lo observado en la última hora completa (T-1) se mantiene en t.
- Climatología: frecuencia del suceso para ese mes y hora local, calculada SIN el año
  de t (validación dejando un año fuera, para no usar el propio resultado).
- TAF: el último TAF (o enmienda) emitido como muy tarde en T que cubre t.
  Se evalúa su condición predominante, su peor caso (con TEMPO/PROB) y su probabilidad.

Todos los métodos se comparan en la MISMA muestra de horas. La verdad es la peor
visibilidad / techo observados en cada hora (tabla horaria de METAR).

Uso:
    python -m novok.verificacion.baselines
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from novok.aeropuertos import activa, cli_estacion
from novok.config import ANIO_INICIO_FIABLE, dir_informes, dir_proc
from novok.taf.parser import a_horario, parsear, probabilidad
from novok.verificacion.metricas import brier, brier_skill, deterministas

ZONA_LOCAL = "Europe/Madrid"
HORIZONTES_H = [1, 3, 6, 12, 18]  # a 24 h casi ninguna hora tiene un TAF que la cubra

# Suceso: (visibilidad < umbral m, techo <= umbral ft); None = no se usa
EVENTOS = {
    "niebla": (1000, None),
    "vis_cat1": (550, None),
    "techo_cat1": (None, 200),
    "bajo_minimos_cat1": (550, 200),
}
NOMBRES = {
    "niebla": "Niebla (visibilidad < 1000 m)",
    "vis_cat1": "Visibilidad < 550 m",
    "techo_cat1": "Techo ≤ 200 ft",
    "bajo_minimos_cat1": "Bajo mínimos tipo CAT I (vis < 550 m o techo ≤ 200 ft)",
}


def evento(vis, techo, umbral_vis, umbral_techo) -> pd.Series:
    r = pd.Series(False, index=vis.index)
    if umbral_vis is not None:
        r |= vis < umbral_vis
    if umbral_techo is not None:
        r |= techo.fillna(np.inf) <= umbral_techo
    return r


# ---------------------------------------------------------------- observación

def cargar_observado() -> pd.DataFrame:
    """Tabla horaria en rejilla regular (para desplazar exactamente L horas)."""
    return preparar_observado(pd.read_parquet(dir_proc() / "horario.parquet"))


def preparar_observado(horario: pd.DataFrame, hasta: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rejilla horaria regular con los sucesos observados. Lo usan el entrenamiento y el sistema en directo."""
    h = horario.set_index("fecha_utc").sort_index() if "fecha_utc" in horario else horario.sort_index()
    rejilla = pd.date_range(h.index.min(), hasta if hasta is not None else h.index.max(), freq="1h")
    h = h.reindex(rejilla)
    h["hay_obs"] = h["n_obs"].fillna(0) > 0
    for nombre, (uv, ut) in EVENTOS.items():
        h[f"obs_{nombre}"] = evento(h["vis_m"], h["techo_ft"], uv, ut).astype(float).where(h["hay_obs"])
    return h


def persistencia(h: pd.DataFrame, nombre: str, horizonte: int) -> pd.Series:
    """Pronóstico para t = lo observado en la última hora completa antes de emitir (t - L - 1)."""
    return h[f"obs_{nombre}"].shift(horizonte + 1)


def climatologia(h: pd.DataFrame, nombre: str) -> pd.Series:
    """P(suceso | mes, hora local) sin contar el año de la hora objetivo."""
    local = h.index.tz_convert(ZONA_LOCAL)
    df = pd.DataFrame({"y": h[f"obs_{nombre}"], "anio": h.index.year, "mes": local.month, "hora": local.hour},
                      index=h.index)
    df = df[df["anio"] >= ANIO_INICIO_FIABLE]
    valido = df.dropna(subset=["y"])
    por_anio = valido.groupby(["anio", "mes", "hora"])["y"].agg(["sum", "count"])
    total = por_anio.groupby(["mes", "hora"]).sum()
    claves = pd.MultiIndex.from_arrays([df["anio"], df["mes"], df["hora"]])
    propio = por_anio.reindex(claves).fillna(0).to_numpy()
    tot = total.reindex(pd.MultiIndex.from_arrays([df["mes"], df["hora"]])).fillna(0).to_numpy()
    p = (tot[:, 0] - propio[:, 0]) / np.maximum(tot[:, 1] - propio[:, 1], 1)
    return pd.Series(p, index=df.index).reindex(h.index)


# ---------------------------------------------------------------- TAF

def taf_horario() -> pd.DataFrame | None:
    """Expande todos los TAF a filas (emisión, hora) con condiciones y probabilidades por suceso."""
    ruta_taf = dir_proc() / "taf.parquet"
    cache = dir_proc() / "taf_horario.parquet"
    if not ruta_taf.exists():
        return None
    if cache.exists() and cache.stat().st_mtime >= ruta_taf.stat().st_mtime:
        return pd.read_parquet(cache)

    crudos = pd.read_parquet(ruta_taf)
    partes, fallidos = [], 0
    for marca, texto in zip(crudos["marca_ogimet"], crudos["taf"], strict=True):
        taf = parsear(texto, marca)
        if taf is None or taf.estacion != activa().icao:
            fallidos += 1
            continue
        df = a_horario(taf)
        for nombre, (uv, ut) in EVENTOS.items():
            df[f"p_{nombre}"] = probabilidad(df, uv, ut)
        partes.append(df.drop(columns="_posibles"))
    out = pd.concat(partes, ignore_index=True)
    out.to_parquet(cache, index=False)
    print(f"TAF expandidos: {len(partes):,} ({fallidos} NIL/CNL/ilegibles descartados)")
    return out


def taf_para_horizonte(th: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """Para cada hora, el último TAF emitido como muy tarde `horizonte` horas antes."""
    disp = th[th["emision"] <= th["hora_utc"] - pd.Timedelta(hours=horizonte)]
    disp = disp.sort_values("emision")
    return disp.groupby("hora_utc").tail(1).set_index("hora_utc").sort_index()


# ---------------------------------------------------------------- evaluación

def evaluar(h: pd.DataFrame, th: pd.DataFrame | None) -> pd.DataFrame:
    filas = []
    anios_ok = h.index.year >= ANIO_INICIO_FIABLE
    for nombre, (uv, ut) in EVENTOS.items():
        clim = climatologia(h, nombre)
        for L in HORIZONTES_H:
            y = h[f"obs_{nombre}"]
            pers = persistencia(h, nombre, L)
            muestra = anios_ok & y.notna() & pers.notna() & clim.notna()
            preds = {"persistencia": pers.astype(float)}
            probs = {"climatologia": clim, "persistencia": pers.astype(float)}
            if th is not None:
                t = taf_para_horizonte(th, L).reindex(h.index)
                muestra &= t["emision"].notna()
                preds["taf_predominante"] = evento(t["vis_prev_m"], t["techo_prev_ft"], uv, ut).astype(float)
                preds["taf_peor_caso"] = evento(t["vis_peor_m"], t["techo_peor_ft"], uv, ut).astype(float)
                probs["taf_probabilidad"] = t[f"p_{nombre}"]
            yy = y[muestra].astype(bool)
            bs_clim = brier(yy, clim[muestra])
            for metodo, p in preds.items():
                filas.append({"suceso": nombre, "horizonte_h": L, "metodo": metodo, "n_horas": int(muestra.sum()),
                              "frecuencia_obs": yy.mean(), **deterministas(yy, p[muestra] > 0.5)})
            for metodo, p in probs.items():
                bs = brier(yy, p[muestra])
                fila = next((f for f in filas if f["suceso"] == nombre and f["horizonte_h"] == L
                             and f["metodo"] == metodo), None)
                if fila is None:
                    fila = {"suceso": nombre, "horizonte_h": L, "metodo": metodo, "n_horas": int(muestra.sum()),
                            "frecuencia_obs": yy.mean()}
                    filas.append(fila)
                fila["brier"] = bs
                fila["brier_skill_vs_clima"] = brier_skill(bs, bs_clim)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- gráfico

COLORES = {  # paleta categórica de referencia, orden fijo
    "persistencia": "#2a78d6",
    "taf_predominante": "#eb6834",
    "taf_peor_caso": "#1baf7a",
    "novok": "#eda100",
}
ETIQUETAS = {"persistencia": "Persistencia", "taf_predominante": "TAF (predominante)",
             "taf_peor_caso": "TAF (con TEMPO/PROB)", "novok": "NOVOK"}


def grafico_csi(res: pd.DataFrame, destino, sucesos=("niebla", "bajo_minimos_cat1")) -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c1", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"})
    fig, axes = plt.subplots(1, len(sucesos), figsize=(11, 4.3), sharey=True, facecolor="#fcfcfb")
    for ax, suc in zip(np.atleast_1d(axes), sucesos, strict=True):
        ax.set_facecolor("#fcfcfb")
        d = res[(res["suceso"] == suc) & res["CSI"].notna()]
        for metodo, color in COLORES.items():
            s = d[d["metodo"] == metodo].sort_values("horizonte_h")
            if s.empty:
                continue
            ax.plot(s["horizonte_h"], s["CSI"], color=color, lw=2, marker="o", ms=8,
                    markeredgecolor="#fcfcfb", markeredgewidth=2, label=ETIQUETAS[metodo])
            ult = s.iloc[-1]
            ax.annotate(ETIQUETAS[metodo], (ult["horizonte_h"], ult["CSI"]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color="#52514e")
        ax.set_title(NOMBRES[suc], fontsize=10.5, loc="left", color="#0b0b0b")
        ax.set_xscale("log")
        ax.set_xticks(HORIZONTES_H, [f"{h} h" for h in HORIZONTES_H])
        ax.minorticks_off()
        ax.set_xlim(0.8, 45)
        ax.set_xlabel("Antelación del pronóstico")
        ax.grid(axis="y", color="#e6e5df", lw=0.8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("CSI (1 = perfecto)")
    np.atleast_1d(axes)[0].set_ylim(0, None)
    if len(np.atleast_1d(axes)[0].get_lines()) > 1:  # una sola serie: el título basta
        np.atleast_1d(axes)[0].legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.suptitle(f"{activa().icao} · Índice de éxito crítico por antelación (2017–2025)", x=0.01, ha="left",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(destino, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    cli_estacion()
    h = cargar_observado()
    th = taf_horario()
    if th is None:
        print("Sin TAF (data/processed/taf.parquet). Ejecuta: python -m novok.data.descargar_taf\n"
              "Evalúo solo persistencia y climatología.")
    res = evaluar(h, th)
    dir_informes().mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(dir_informes() / "verificacion_baselines.csv", index=False)
    grafico_csi(res, dir_informes() / "verificacion_csi.png")

    cols = ["horizonte_h", "metodo", "n_horas", "POD", "FAR", "CSI", "brier_skill_vs_clima"]
    for suc in ("niebla", "bajo_minimos_cat1"):
        d = res[res["suceso"] == suc]
        print(f"\n{NOMBRES[suc]} · frecuencia observada = {d['frecuencia_obs'].iloc[0]:.2%}")
        print(d[cols].round(3).to_string(index=False))
    print(f"\nInformes en {dir_informes()}")


if __name__ == "__main__":
    main()
