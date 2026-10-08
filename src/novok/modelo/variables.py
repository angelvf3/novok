"""Variables del modelo a partir de la tabla horaria de METAR.

Convención temporal (igual que los baselines): para predecir la hora t con antelación L,
el pronóstico se emite en T = t - L y la última hora completa observada es i = T - 1 = t - L - 1.
Cada fila de la matriz corresponde a esa hora i y SOLO usa información de i y anteriores
(desplazamientos hacia atrás y ventanas que terminan en i). El objetivo es el suceso en t = i + L + 1.

Las únicas variables "del futuro" son las de calendario de la hora objetivo (hora del día,
época del año), que se conocen de antemano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ZONA_LOCAL = "Europe/Madrid"

_FENOMENOS = {
    "fen_niebla": r"\bFG\b|FZFG|BCFG|PRFG|MIFG",
    "fen_neblina": r"\bBR\b",
    "fen_lluvia": r"RA",
    "fen_llovizna": r"DZ",
    "fen_chubasco": r"SH",
    "fen_tormenta": r"TS",
}


def _horas_desde(cond: pd.Series, tope: int = 72) -> pd.Series:
    """Horas transcurridas desde la última hora en que se cumplió `cond` (tope si nunca)."""
    idx = np.arange(len(cond))
    ultima = pd.Series(np.where(cond.fillna(False).to_numpy(), idx, np.nan), index=cond.index).ffill()
    return pd.Series(np.minimum(idx - ultima.to_numpy(), tope), index=cond.index).fillna(tope)


def variables_base(h: pd.DataFrame) -> pd.DataFrame:
    """Variables que dependen solo de la hora observada i (no del horizonte)."""
    x = pd.DataFrame(index=h.index)
    vis_log = np.log10(h["vis_m"].clip(lower=25))
    x["vis_log"] = vis_log
    x["rvr_log"] = np.log10(h["rvr_min_m"].clip(lower=25))
    x["techo_ft"] = h["techo_ft"].fillna(5000).clip(upper=5000).where(h["hay_obs"])
    x["dif_rocio_c"] = h["dif_rocio_c"]
    x["hr_pct"] = h["hr_pct"]
    x["temp_c"] = h["temp_c"]
    x["qnh_hpa"] = h["qnh_hpa"]
    x["viento_kt"] = h["viento_kt"]
    x["racha_kt"] = h["racha_kt"]
    rad = np.deg2rad(h["viento_dir"])
    x["viento_u"] = (-h["viento_kt"] * np.sin(rad)).where(h["viento_kt"] > 0, 0.0)  # componente E-O
    x["viento_v"] = (-h["viento_kt"] * np.cos(rad)).where(h["viento_kt"] > 0, 0.0)  # componente N-S
    x["n_obs"] = h["n_obs"].fillna(0)

    for nombre, patron in _FENOMENOS.items():
        x[nombre] = h["fenomenos"].fillna("").str.contains(patron, regex=True).astype(float).where(h["hay_obs"])

    # Tendencias (diferencia con 1 h y 3 h antes)
    for col, serie in {"dif_rocio": h["dif_rocio_c"], "temp": h["temp_c"], "vis_log": vis_log,
                       "hr": h["hr_pct"]}.items():
        x[f"tend1_{col}"] = serie - serie.shift(1)
        x[f"tend3_{col}"] = serie - serie.shift(3)
    x["tend3_qnh"] = h["qnh_hpa"] - h["qnh_hpa"].shift(3)

    # Ventanas recientes (terminan en i)
    x["vis_log_min6"] = vis_log.rolling(6, min_periods=1).min()
    x["dif_rocio_min6"] = h["dif_rocio_c"].rolling(6, min_periods=1).min()
    x["viento_kt_media3"] = h["viento_kt"].rolling(3, min_periods=1).mean()

    # Historial de los sucesos
    x["obs_niebla"] = h["obs_niebla"]
    x["obs_bajo_minimos"] = h["obs_bajo_minimos_cat1"]
    x["obs_niebla_lag1"] = h["obs_niebla"].shift(1)
    x["obs_niebla_lag2"] = h["obs_niebla"].shift(2)
    x["horas_desde_niebla"] = _horas_desde(h["obs_niebla"] == 1)
    x["horas_desde_bajo_minimos"] = _horas_desde(h["obs_bajo_minimos_cat1"] == 1)
    x["niebla_ult24h"] = h["obs_niebla"].rolling(24, min_periods=1).sum()
    return x


def variables_nwp(h: pd.DataFrame, nwp: pd.DataFrame, paso: int) -> pd.DataFrame:
    """Variables a partir del pronóstico numérico emitido ~24 h antes (sufijo _previous_day1).

    Para la hora objetivo t = i + paso se usa lo que el modelo preveía para t. Ese pronóstico se publicó
    unas 18 h antes de t, así que ya existía al emitir NOVOK (antelación <= 12 h): no hay fuga.
    Además: cambio previsto entre i y t, y error actual del modelo frente al METAR en i.
    """
    n = nwp.reindex(h.index)
    partes = [_nwp_un_modelo(h, n, m, paso) for m in sorted({c.split("_")[1] for c in n.columns})]
    return pd.concat(partes, axis=1) if partes else pd.DataFrame(index=h.index)


def _nwp_un_modelo(h: pd.DataFrame, n: pd.DataFrame, m: str, paso: int) -> pd.DataFrame:
    x = pd.DataFrame(index=h.index)

    def ahora(v):  # pronóstico para la hora observada i
        return n.get(f"nwp_{m}_{v}")

    def en_t(v):  # pronóstico para la hora objetivo t = i + paso
        c = ahora(v)
        return None if c is None else c.shift(-paso)

    if ahora("hr") is not None:
        x[f"{m}_hr_t"] = en_t("hr")
        x[f"{m}_hr_cambio"] = en_t("hr") - ahora("hr")
        x[f"{m}_hr_error_ahora"] = h["hr_pct"] - ahora("hr")
    if ahora("temp") is not None and ahora("rocio") is not None:
        dpd = ahora("temp") - ahora("rocio")
        x[f"{m}_dif_rocio_t"] = dpd.shift(-paso)
        x[f"{m}_dif_rocio_cambio"] = dpd.shift(-paso) - dpd
        x[f"{m}_dif_rocio_error_ahora"] = h["dif_rocio_c"] - dpd
    for v in ("nubes_bajas", "nubes", "precip", "presion"):
        if ahora(v) is not None:
            x[f"{m}_{v}_t"] = en_t(v)
    if ahora("vis") is not None:
        x[f"{m}_vis_log_t"] = np.log10(en_t("vis").clip(lower=25))
    if ahora("viento_kt") is not None and ahora("viento_dir") is not None:
        rad = np.deg2rad(en_t("viento_dir"))
        x[f"{m}_viento_u_t"] = -en_t("viento_kt") * np.sin(rad)
        x[f"{m}_viento_v_t"] = -en_t("viento_kt") * np.cos(rad)
    return x


def matriz(h: pd.DataFrame, base: pd.DataFrame, suceso: str, horizonte: int,
           nwp: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Matriz X (filas = hora observada i) y objetivo y (suceso en t = i + horizonte + 1)."""
    paso = horizonte + 1
    x = base.copy()
    if nwp is not None:
        x = x.join(variables_nwp(h, nwp, paso))
    objetivo = h.index + pd.Timedelta(hours=paso)
    local = objetivo.tz_convert(ZONA_LOCAL)
    x["hora_obj_sin"] = np.sin(2 * np.pi * local.hour / 24)
    x["hora_obj_cos"] = np.cos(2 * np.pi * local.hour / 24)
    x["doy_obj_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    x["doy_obj_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    # Lo que pasó ayer a la hora objetivo (t - 24 h), conocido si t - 24 <= i
    if 24 - paso >= 0:
        x["suceso_ayer_misma_hora"] = h[f"obs_{suceso}"].shift(24 - paso)
    y = h[f"obs_{suceso}"].shift(-paso)
    x.index.name = "hora_obs_utc"
    y.index = x.index
    return x, y.rename("y")
