"""Métricas de verificación para sucesos binarios (niebla / no niebla).

Tabla de contingencia: a = aciertos, b = falsas alarmas, c = fallos, d = negativos correctos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def contingencia(obs: pd.Series, pred: pd.Series) -> dict:
    o, p = obs.astype(bool).to_numpy(), pred.astype(bool).to_numpy()
    return {"a": int((p & o).sum()), "b": int((p & ~o).sum()), "c": int((~p & o).sum()), "d": int((~p & ~o).sum())}


def _div(x, y):
    return x / y if y else np.nan


def deterministas(obs: pd.Series, pred: pd.Series) -> dict:
    """POD (detección), FAR (falsas alarmas), CSI (índice de éxito) y sesgo de frecuencia."""
    t = contingencia(obs, pred)
    a, b, c = t["a"], t["b"], t["c"]
    return {**t,
            "POD": _div(a, a + c),
            "FAR": _div(b, a + b),
            "CSI": _div(a, a + b + c),
            "sesgo": _div(a + b, a + c)}


def brier(obs: pd.Series, prob: pd.Series) -> float:
    """Error cuadrático medio de la probabilidad (0 = perfecto)."""
    return float(np.mean((prob.to_numpy(dtype=float) - obs.to_numpy(dtype=float)) ** 2))


def brier_skill(bs: float, bs_ref: float) -> float:
    """Mejora respecto a la referencia (climatología): 1 = perfecto, 0 = igual, < 0 = peor."""
    return 1 - bs / bs_ref if bs_ref else np.nan
