"""Limpia el METAR crudo de IEM y genera dos tablas:

- observaciones.parquet: una fila por METAR/SPECI, en unidades métricas.
- horario.parquet: una fila por hora UTC con las peores condiciones de esa hora
  (mínima visibilidad y mínimo techo), que serán las etiquetas del modelo.

Uso:
    python -m novok.data.limpiar_metar [--estacion LEXJ]
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from novok.config import (
    COBERTURAS_TECHO,
    DATA_RAW,
    MILLA_A_METROS,
    UMBRALES_TECHO_FT,
    UMBRALES_VIS_M,
    dir_proc,
)
from novok.data.viento import viento_en_pista

# IEM limita la visibilidad a 6,21 millas (≈10 km, el "9999" del METAR europeo)
VIS_MAX_M = 10_000

# Grupo de viento + (opcional) variación de dirección + visibilidad europea en metros o CAVOK
_RE_VIS = re.compile(
    r"\b(?:\d{3}|VRB|/{3})[\d/]{2,3}(?:G\d{2,3})?(?:KT|MPS)\s+(?:\d{3}V\d{3}\s+)?(CAVOK|\d{4})(?:NDV)?\b"
)
# Alcance visual en pista (RVR): R29/0175N, R11/P2000, R29/0800V1200U...
_RE_RVR = re.compile(r"\bR\d{2}[LCR]?/([PM]?)(\d{4})")


def vis_desde_metar(metar) -> float:
    """Visibilidad (m) leída del texto del METAR. NaN si no se puede leer.

    Es más fiable que la columna vsby de IEM, que pasa los metros a millas y al
    volver a metros deja un "1000" en 998 m (y lo contaría como niebla).
    """
    if not isinstance(metar, str):
        return np.nan
    m = _RE_VIS.search(metar)
    if not m:
        return np.nan
    v = m.group(1)
    return float(VIS_MAX_M) if v in ("CAVOK", "9999") else float(v)


def rvr_min_desde_metar(metar) -> float:
    """RVR mínimo (m) de todas las pistas notificadas. NaN si no hay RVR."""
    if not isinstance(metar, str):
        return np.nan
    valores = [float(v) for _, v in _RE_RVR.findall(metar)]
    return min(valores) if valores else np.nan


def redondear_a_pasos_metar(vis_m: pd.Series) -> pd.Series:
    """Redondea la visibilidad de IEM a los pasos con que se notifica en METAR."""
    paso = np.select([vis_m < 800, vis_m < 5000], [50, 100], 1000)
    return (np.round(vis_m / paso) * paso).clip(upper=VIS_MAX_M)


def _num(serie: pd.Series) -> pd.Series:
    """Convierte a número; 'M' (missing) y 'T' (trace) pasan a NaN."""
    return pd.to_numeric(serie.replace({"M": np.nan, "T": np.nan}), errors="coerce")


def calcular_techo_ft(fila: pd.Series) -> float:
    """Altura (pies) de la capa más baja BKN/OVC/VV. NaN si no hay techo.

    Con CAVOK o cielo despejado no hay techo: devuelve NaN y se interpreta
    como "techo ilimitado" al aplicar umbrales.
    """
    alturas = []
    for i in range(1, 5):
        cob = str(fila.get(f"skyc{i}", "")).strip()
        alt = fila.get(f"skyl{i}")
        if cob in COBERTURAS_TECHO and pd.notna(alt):
            alturas.append(float(alt))
    return min(alturas) if alturas else np.nan


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """De IEM crudo a observaciones limpias en unidades métricas."""
    out = pd.DataFrame()
    out["fecha_utc"] = pd.to_datetime(df["valid"], utc=True)
    out["metar"] = df.get("metar")

    tmpf, dwpf = _num(df["tmpf"]), _num(df["dwpf"])
    out["temp_c"] = (tmpf - 32) * 5 / 9
    out["rocio_c"] = (dwpf - 32) * 5 / 9
    out["dif_rocio_c"] = out["temp_c"] - out["rocio_c"]  # cerca de 0 => saturación
    out["hr_pct"] = _num(df["relh"])
    out["viento_dir"] = _num(df["drct"])
    out["viento_kt"] = _num(df["sknt"])
    out["racha_kt"] = _num(df["gust"])
    out["qnh_hpa"] = _num(df["alti"]) * 33.8639  # inHg -> hPa

    vis_texto = df["metar"].map(vis_desde_metar) if "metar" in df else pd.Series(np.nan, index=df.index)
    vis_iem = redondear_a_pasos_metar(_num(df["vsby"]) * MILLA_A_METROS)
    out["vis_m"] = vis_texto.fillna(vis_iem)
    out["rvr_min_m"] = df["metar"].map(rvr_min_desde_metar) if "metar" in df else np.nan

    for i in range(1, 5):
        out[f"skyc{i}"] = df.get(f"skyc{i}", pd.Series(dtype=str)).fillna("").str.strip()
        out[f"skyl{i}"] = _num(df.get(f"skyl{i}", pd.Series(dtype=str)))
    out["techo_ft"] = out.apply(calcular_techo_ft, axis=1)
    out["fenomenos"] = df.get("wxcodes", pd.Series(dtype=str)).replace("M", "").fillna("")

    out = (out.dropna(subset=["fecha_utc"])
              .drop_duplicates(subset=["fecha_utc", "metar"])
              .sort_values("fecha_utc")
              .reset_index(drop=True))
    return out


def a_horario(obs: pd.DataFrame) -> pd.DataFrame:
    """Agrega a hora UTC: peor visibilidad y techo, y última lectura del resto."""
    o = obs.set_index("fecha_utc")
    peores = o[["vis_m", "rvr_min_m", "techo_ft"]].resample("1h").min()
    ultimos = o[["temp_c", "rocio_c", "dif_rocio_c", "hr_pct", "viento_dir",
                 "viento_kt", "qnh_hpa"]].resample("1h").last()
    racha = o["racha_kt"].resample("1h").max()
    n_obs = o["metar"].resample("1h").count().rename("n_obs")
    fen = o["fenomenos"].resample("1h").agg(lambda s: " ".join(x for x in s if x))

    h = pd.concat([peores, ultimos, racha, n_obs, fen.rename("fenomenos")], axis=1)
    h = h[h["n_obs"] > 0].copy()  # horas sin ninguna observación quedan fuera

    for nombre, umbral in UMBRALES_VIS_M.items():
        h[f"vis_bajo_{nombre}"] = h["vis_m"] < umbral
    techo = h["techo_ft"].fillna(np.inf)  # sin techo = ilimitado
    for nombre, umbral in UMBRALES_TECHO_FT.items():
        h[f"techo_bajo_{nombre}"] = techo <= umbral  # "<=": el METAR da el techo en pasos de 100 ft
    h["niebla_reportada"] = h["fenomenos"].str.contains(r"\bFG\b|FZFG|BCFG|PRFG|MIFG", regex=True)
    h = viento_en_pista(h)
    return h.reset_index().rename(columns={"index": "fecha_utc"})


def main() -> None:
    from novok.aeropuertos import cli_estacion
    procesar(cli_estacion().icao)


def procesar(est: str) -> pd.DataFrame:
    """Limpia los METAR descargados de un aeropuerto y guarda sus tablas en data/processed/<OACI>/."""
    ficheros = sorted(DATA_RAW.glob(f"metar_{est}_*.csv"))
    if not ficheros:
        raise SystemExit("No hay CSV en data/raw. Ejecuta antes: python -m novok.data.descargar_metar")

    crudo = pd.concat((pd.read_csv(f, dtype=str) for f in ficheros), ignore_index=True)
    from novok.aeropuertos import estacion
    with estacion(est):  # las componentes del viento usan las pistas de este aeropuerto
        obs = limpiar(crudo)
        hor = a_horario(obs)

    destino = dir_proc(est)
    destino.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(destino / "observaciones.parquet", index=False)
    hor.to_parquet(destino / "horario.parquet", index=False)
    print(f"{est} · Observaciones: {len(obs):,} | Horas: {len(hor):,} "
          f"| {hor['fecha_utc'].min():%Y-%m-%d} -> {hor['fecha_utc'].max():%Y-%m-%d}")
    return hor


if __name__ == "__main__":
    main()
