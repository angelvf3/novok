"""Componentes del viento sobre las pistas del aeropuerto activo.

No es un modelo de IA: convierte un viento (observado o pronosticado) en viento
cruzado y de cara/cola para cada cabecera, y elige la cabecera más favorable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from novok.aeropuertos import activa
from novok.config import LIMITE_COLA_KT, LIMITE_CRUZADO_KT


def componentes(direccion_deg, velocidad_kt, rumbo_pista_deg):
    """Devuelve (cruzado_kt, cara_kt) del viento respecto a un rumbo de pista.

    cara_kt > 0 es viento de cara; cara_kt < 0 es viento de cola.
    cruzado_kt es siempre positivo (no distingue izquierda/derecha).
    Funciona con escalares o arrays; dirección desconocida (VRB) -> NaN.
    """
    ang = np.deg2rad(np.asarray(direccion_deg, dtype=float) - rumbo_pista_deg)
    v = np.asarray(velocidad_kt, dtype=float)
    return np.abs(v * np.sin(ang)), v * np.cos(ang)


def viento_en_pista(h: pd.DataFrame, pistas: dict[str, float] | None = None) -> pd.DataFrame:
    """Añade componentes por cabecera, la cabecera favorable y avisos de referencia.

    Usa viento medio y, si existe, la racha (lo que siente el avión en el peor momento).
    Con viento en calma (0 kt) las componentes son 0 aunque falte la dirección.
    """
    pistas = pistas or activa().rumbos
    h = h.copy()
    calma = h["viento_kt"] == 0
    for cab, rumbo in pistas.items():
        cru, cara = componentes(h["viento_dir"], h["viento_kt"], rumbo)
        h[f"cruzado_kt_{cab}"] = np.where(calma, 0.0, cru)
        h[f"cara_kt_{cab}"] = np.where(calma, 0.0, cara)

    cabs = list(pistas)
    caras = h[[f"cara_kt_{c}" for c in cabs]].to_numpy()
    sin_dato = np.isnan(caras).all(axis=1)
    idx = np.argmax(np.nan_to_num(caras, nan=-np.inf), axis=1)
    h["pista_favorable"] = np.where(sin_dato, None, np.array(cabs)[idx])

    rng = np.arange(len(h))
    h["cruzado_kt"] = np.where(sin_dato, np.nan, h[[f"cruzado_kt_{c}" for c in cabs]].to_numpy()[rng, idx])

    racha = h["racha_kt"].fillna(h["viento_kt"]) if "racha_kt" in h else h["viento_kt"]
    factor = (racha / h["viento_kt"]).where(h["viento_kt"] > 0, 1.0)
    h["cruzado_racha_kt"] = h["cruzado_kt"] * factor

    h["aviso_cruzado"] = h["cruzado_racha_kt"] > LIMITE_CRUZADO_KT
    # El viento de cola importa por cabecera: con la favorable es ~0 por definición,
    # pero puede haber que usar una concreta (p. ej. la del ILS con niebla).
    for cab in cabs:
        h[f"aviso_cola_{cab}"] = -h[f"cara_kt_{cab}"] > LIMITE_COLA_KT
    return h
