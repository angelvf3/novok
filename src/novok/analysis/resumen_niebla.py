"""Primer análisis: cuánta niebla hay en LEAS y cuándo aparece.

Genera en reports/:
- horas_por_anio.csv: horas al año por debajo de cada umbral
- frecuencia_mes_hora.png: mapa de calor mes x hora local de visibilidad < 1000 m
- cobertura.csv: % de horas con observación por año (calidad del dato)

Uso:
    python -m novok.analysis.resumen_niebla
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from novok.config import ANIO_INICIO_FIABLE, DATA_PROCESSED, REPORTS

ZONA_LOCAL = "Europe/Madrid"


def horas_por_anio(h: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in h.columns if c.startswith(("vis_bajo_", "techo_bajo_", "aviso_"))]
    t = h.assign(anio=h["fecha_utc"].dt.year).groupby("anio")[cols].sum().astype(int)
    t["horas_observadas"] = h.groupby(h["fecha_utc"].dt.year).size()
    return t


def cobertura(h: pd.DataFrame) -> pd.Series:
    conteo = h["fecha_utc"].dt.year.value_counts().sort_index()
    horas_posibles = pd.Series({a: pd.Timestamp(year=a, month=12, day=31).dayofyear * 24 for a in conteo.index})
    return (conteo / horas_posibles * 100).round(1)


def mapa_calor(h: pd.DataFrame, destino, desde: int, hasta: int) -> None:
    local = h["fecha_utc"].dt.tz_convert(ZONA_LOCAL)
    tabla = (h.assign(mes=local.dt.month, hora=local.dt.hour)
              .pivot_table(index="mes", columns="hora", values="vis_bajo_niebla", aggfunc="mean") * 100)

    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(tabla.values, aspect="auto", cmap="Blues", origin="upper")
    ax.set_yticks(range(len(tabla.index)), [meses[m - 1] for m in tabla.index])
    ax.set_xticks(range(0, 24, 2), [f"{x:02d}" for x in range(0, 24, 2)])
    ax.set_xlabel("Hora local")
    ax.set_title(f"LEAS · % de horas con visibilidad < 1000 m ({desde}–{hasta})")
    fig.colorbar(im, ax=ax, label="% de horas")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def main() -> None:
    h = pd.read_parquet(DATA_PROCESSED / "horario.parquet")
    REPORTS.mkdir(parents=True, exist_ok=True)

    t = horas_por_anio(h)
    t["fiable"] = t.index >= ANIO_INICIO_FIABLE
    t.to_csv(REPORTS / "horas_por_anio.csv")
    cobertura(h).rename("pct_horas_con_dato").to_csv(REPORTS / "cobertura.csv")

    # Estadística y mapa solo con años de observación 24 h
    hf = h[h["fecha_utc"].dt.year >= ANIO_INICIO_FIABLE]
    mapa_calor(hf, REPORTS / "frecuencia_mes_hora.png", ANIO_INICIO_FIABLE, hf["fecha_utc"].dt.year.max())

    media = t[t["fiable"]].drop(columns="fiable").mean().round(0).astype(int)
    print(f"Media anual {ANIO_INICIO_FIABLE}-{hf['fecha_utc'].dt.year.max()} de horas por debajo de cada umbral:")
    for col, v in media.items():
        if col != "horas_observadas":
            print(f"  {col:<28} {v:>5} h")
    print(f"Informes en {REPORTS}")


if __name__ == "__main__":
    main()
