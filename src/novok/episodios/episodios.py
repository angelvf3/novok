"""F4 · Episodios de niebla y su tipo.

Un episodio es una racha de horas con visibilidad < 1000 m. Se unen rachas separadas por una sola hora
despejada (la niebla "parpadea"). El fin del episodio es la primera hora despejada que no vuelve a
cerrarse a la hora siguiente.

Tipo de niebla por reglas físicas, versión simplificada de la clasificación de Tardif y Rasmussen (2007),
aplicadas en el orden en que se comprueban:

1. llovizna o lluvia: lluvia o llovizna en la hora de inicio o la anterior.
2. descenso de nubes: techo <= 1000 ft en las dos horas previas (estratos que bajan hasta el suelo).
3. radiación: viento <= 3 kt al formarse, entre las 18 y las 09 h local, sin techo bajo antes
   (enfriamiento nocturno con cielo despejado y calma).
4. advección: el resto, normalmente con viento (aire húmedo del mar sobre superficie más fría).

No hay etiquetas reales del tipo: es una clasificación física, no un modelo entrenado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ZONA_LOCAL = "Europe/Madrid"
TIPOS = ["advección", "radiación", "descenso de nubes", "llovizna o lluvia"]


def serie_niebla(h: pd.DataFrame, umbral_m: float = 1000) -> pd.Series:
    """1 = niebla, 0 = despejado, NaN = sin observación (rejilla horaria regular)."""
    return (h["vis_m"] < umbral_m).astype(float).where(h["hay_obs"])


def detectar(h: pd.DataFrame, umbral_m: float = 1000, hueco_max_h: int = 1) -> pd.DataFrame:
    """Lista de episodios con inicio (primera hora con niebla), fin (primera hora despejada) y calidad."""
    f = serie_niebla(h, umbral_m)
    v = f.to_numpy()
    n = len(v)
    episodios = []
    i = 0
    while i < n:
        if v[i] != 1:
            i += 1
            continue
        ini = i
        j = i
        while True:
            while j < n and v[j] == 1:
                j += 1
            # j = primera hora no-niebla; ¿es un hueco corto que vuelve a cerrarse?
            k = j
            while k < n and k - j < hueco_max_h and v[k] == 0:
                k += 1
            if k < n and v[k] == 1 and k - j <= hueco_max_h and not np.isnan(v[j:k]).any():
                j = k
                continue
            break
        fin = j
        completo = (ini > 0 and v[ini - 1] == 0) and (fin < n and v[fin] == 0)
        episodios.append({"inicio": f.index[ini], "fin": f.index[fin] if fin < n else pd.NaT,
                          "duracion_h": fin - ini, "completo": bool(completo)})
        i = fin + 1
    ep = pd.DataFrame(episodios)
    if ep.empty:
        return ep
    vis = h["vis_m"]
    fines = ep["fin"].fillna(ep["inicio"] + pd.Timedelta(hours=1))
    ep["vis_min_m"] = [vis.loc[a:b - pd.Timedelta(hours=1)].min() for a, b in zip(ep["inicio"], fines, strict=True)]
    return ep


def clasificar(ep: pd.DataFrame, h: pd.DataFrame) -> pd.Series:
    """Tipo físico de cada episodio según las condiciones al formarse."""
    tipos = []
    for ini in ep["inicio"]:
        prev = ini - pd.Timedelta(hours=1)
        prev2 = ini - pd.Timedelta(hours=2)
        fen = " ".join(str(h["fenomenos"].get(t, "") or "") for t in (prev, ini))
        techo_prev = [h["techo_ft"].get(t, np.nan) for t in (prev2, prev)]
        viento = np.nanmean([h["viento_kt"].get(t, np.nan) for t in (prev, ini)])
        hora = ini.tz_convert(ZONA_LOCAL).hour
        if "RA" in fen or "DZ" in fen:
            tipos.append("llovizna o lluvia")
        elif all(pd.notna(t) and t <= 1000 for t in techo_prev):
            tipos.append("descenso de nubes")
        elif viento <= 3 and (hora >= 18 or hora <= 9) and not (pd.notna(techo_prev[1]) and techo_prev[1] <= 1000):
            tipos.append("radiación")
        else:
            tipos.append("advección")
    return pd.Series(tipos, index=ep.index, name="tipo")


def episodios(h: pd.DataFrame) -> pd.DataFrame:
    ep = detectar(h)
    ep["tipo"] = clasificar(ep, h)
    return ep
