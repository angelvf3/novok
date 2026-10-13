"""F3 · NOVOK v2: METAR + pronósticos numéricos (NWP), comparado con la v1 en las mismas horas.

Requiere data/processed/nwp.parquet (python -m novok.data.descargar_nwp).
Misma validación temporal estricta que la v1. Para cada suceso y antelación se entrenan
la v1 (solo METAR) y la v2 (METAR + NWP) con exactamente las mismas filas.

Los resultados se dan para todo 2020-2025 y por separado para 2024-2025, los únicos años con ECMWF
(antes solo hay JMA). LightGBM trata los huecos de NWP como valores ausentes.

Uso:
    python -m novok.modelo.entrenar_nwp       (~15 min)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from novok.aeropuertos import activa, cli_estacion
from novok.config import ANIO_INICIO_FIABLE, dir_informes, dir_modelos, dir_proc
from novok.modelo.entrenar import (
    FONDO,
    HORIZONTES_H,
    REJILLA,
    SUCESOS,
    TINTA,
    TINTA2,
    _estilo,
    entrenar,
    guardar_umbral,
    importancia_shap,
    validacion_temporal,
)
from novok.modelo.variables import matriz, variables_base
from novok.verificacion.baselines import NOMBRES, cargar_observado, climatologia
from novok.verificacion.metricas import brier, brier_skill, deterministas

ANIOS_ECMWF = (2024, 2025)
COLORES = {"persistencia": "#2a78d6", "novok_v1": "#eda100", "novok_v2": "#e87ba4"}  # slots 1, 4 y 5
ETIQ = {"persistencia": "Persistencia", "novok_v1": "NOVOK v1 (METAR)", "novok_v2": "NOVOK v2 (+ pronóstico)"}


def metricas(oos: pd.DataFrame, columna_p: str, columna_u: str | None, bs_clim: float) -> dict:
    yy = oos["y"]
    p = oos[columna_p]
    det = deterministas(yy, p >= oos[columna_u]) if columna_u else deterministas(yy, p == 1)
    bs = brier(yy, p)
    return {**det, "brier": bs, "brier_skill_vs_clima": brier_skill(bs, bs_clim)}


def main() -> None:
    cli_estacion()
    ruta = dir_proc() / "nwp.parquet"
    if not ruta.exists():
        raise SystemExit("Falta data/processed/nwp.parquet: ejecuta python -m novok.data.descargar_nwp")
    nwp = pd.read_parquet(ruta)
    h = cargar_observado()
    base = variables_base(h)
    filas, preds, importancias = [], [], {}

    for suceso in SUCESOS:
        clim_t = climatologia(h, suceso)
        for L in HORIZONTES_H:
            x2, y = matriz(h, base, suceso, L, nwp=nwp)
            ok = h["hay_obs"].to_numpy() & y.notna().to_numpy() & (x2.index.year >= ANIO_INICIO_FIABLE)
            x2, y = x2[ok], y[ok].astype(int)
            x1 = x2[[c for c in x2.columns if not c.startswith(("ecmwf_", "jma_"))]]

            o1 = validacion_temporal(x1, y)
            o2 = validacion_temporal(x2, y)
            oos = o1.rename(columns={"p": "p_v1", "umbral": "u_v1"})[["y", "p_v1", "u_v1", "anio"]]
            oos[["p_v2", "u_v2"]] = o2[["p", "umbral"]]
            t = oos.index + pd.Timedelta(hours=L + 1)
            oos["persistencia"] = x1.loc[oos.index, "obs_niebla" if suceso == "niebla" else "obs_bajo_minimos"]
            oos["climatologia"] = clim_t.reindex(t).to_numpy()
            oos = oos.dropna(subset=["climatologia", "persistencia"])
            preds.append(oos.assign(suceso=suceso, horizonte_h=L, hora_obj_utc=oos.index + pd.Timedelta(hours=L + 1)))

            for periodo, sel in {"2020-2025": slice(None), "2024-2025": oos["anio"].isin(ANIOS_ECMWF)}.items():
                d = oos[sel]
                bs_clim = brier(d["y"], d["climatologia"])
                comunes = {"periodo": periodo, "suceso": suceso, "horizonte_h": L, "n_horas": len(d),
                           "frecuencia_obs": d["y"].mean()}
                filas.append({**comunes, "metodo": "novok_v1", **metricas(d, "p_v1", "u_v1", bs_clim)})
                filas.append({**comunes, "metodo": "novok_v2", **metricas(d, "p_v2", "u_v2", bs_clim)})
                filas.append({**comunes, "metodo": "persistencia", **metricas(d, "persistencia", None, bs_clim)})
            r = {f["metodo"]: f for f in filas[-3:]}
            print(f"{suceso:<18} L={L:>2} h  [2024-25] CSI v1 {r['novok_v1']['CSI']:.3f}  v2 {r['novok_v2']['CSI']:.3f}"
                  f"  pers {r['persistencia']['CSI']:.3f} | BSS v1 {r['novok_v1']['brier_skill_vs_clima']:+.3f}"
                  f"  v2 {r['novok_v2']['brier_skill_vs_clima']:+.3f}")

            imp, modelo = importancia_shap(x2, y, int(o2["arboles"].iloc[-1]))
            importancias[(suceso, L)] = imp
            dir_modelos().mkdir(parents=True, exist_ok=True)
            modelo.save_model(str(dir_modelos() / f"novok_v2_{suceso}_{L}h.txt"))
            guardar_umbral(f"v2_{suceso}_{L}h", o2["umbral"].iloc[-1])
            # v1 final: respaldo del briefing cuando no hay pronóstico numérico
            entrenar(x1, y, n_arboles=int(o1["arboles"].iloc[-1])).save_model(
                str(dir_modelos() / f"novok_v1_{suceso}_{L}h.txt"))
            guardar_umbral(f"v1_{suceso}_{L}h", o1["umbral"].iloc[-1])

    res = pd.DataFrame(filas)
    res.round(4).to_csv(dir_informes() / "modelo_v2_resultados.csv", index=False)
    pd.concat(preds).to_parquet(dir_proc() / "predicciones_novok_v2.parquet")
    pd.DataFrame(importancias).round(5).to_csv(dir_informes() / "modelo_v2_importancia_shap.csv")
    grafico(res)
    print(f"\nInformes en {dir_informes()}")


def grafico(res: pd.DataFrame, periodo: str = "2024-2025") -> None:
    sucesos = [s for s in SUCESOS if s in set(res["suceso"])]
    fig, axes = plt.subplots(1, len(sucesos), figsize=(5.5 * len(sucesos), 4.3), sharey=True, facecolor=FONDO,
                             squeeze=False)
    axes = axes[0]
    for ax, suc in zip(axes, sucesos, strict=True):
        _estilo(ax)
        d = res[(res["periodo"] == periodo) & (res["suceso"] == suc)]
        for metodo, color in COLORES.items():
            s = d[d["metodo"] == metodo].sort_values("horizonte_h")
            ax.plot(s["horizonte_h"], s["CSI"], color=color, lw=2, marker="o", ms=8,
                    markeredgecolor=FONDO, markeredgewidth=2, label=ETIQ[metodo])
        ax.set_title(NOMBRES[suc], fontsize=10.5, loc="left", color=TINTA)
        ax.set_xscale("log")
        ax.set_xticks(HORIZONTES_H, [f"{x} h" for x in HORIZONTES_H])
        ax.minorticks_off()
        ax.set_xlabel("Antelación del pronóstico", color=TINTA2)
        ax.grid(axis="y", color=REJILLA, lw=0.8)
    axes[0].set_ylabel("CSI (1 = perfecto)", color=TINTA2)
    axes[0].set_ylim(0, None)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(f"{activa().icao} · Efecto de añadir el pronóstico numérico ({periodo.replace('-', '–')})", x=0.01,
                 ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    fig.savefig(dir_informes() / "modelo_v2_csi.png", dpi=150, facecolor=FONDO)
    plt.close(fig)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
