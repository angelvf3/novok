"""Lector de METAR/SPECI en texto (formato OACI europeo) a las mismas columnas que el archivo de IEM.

Sirve para el sistema en directo: los METAR que llegan como texto se convierten a una tabla idéntica a la
de entrenamiento y pasan por la misma limpieza (limpiar_metar.limpiar). Así el modelo ve exactamente el
mismo tipo de datos en producción que en entrenamiento.

Se ignora el pronóstico de tendencia (TEMPO, BECMG, NOSIG) y los fenómenos recientes (RE..).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from novok.config import MILLA_A_METROS
from novok.taf.parser import _fecha

_RE_HORA = re.compile(r"^(\d{2})(\d{2})(\d{2})Z$")
_RE_VIENTO = re.compile(r"^(\d{3}|VRB|///)(\d{2,3}|//)(?:G(\d{2,3}))?(KT|MPS)$")
_RE_VAR = re.compile(r"^\d{3}V\d{3}$")
_RE_VIS = re.compile(r"^(\d{4})(NDV)?$")
_RE_NUBE = re.compile(r"^(FEW|SCT|BKN|OVC)(\d{3}|///)(CB|TCU|///)?$")
_RE_VV = re.compile(r"^VV(\d{3}|///)$")
_RE_TEMP = re.compile(r"^(M?\d{2})/(M?\d{2})?$")
_RE_QNH = re.compile(r"^Q(\d{4})$")
_RE_WX = re.compile(
    r"^(\+|-|VC)?(MI|BC|PR|DR|BL|SH|TS|FZ)?((DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PO|SQ|FC|SS|DS)+)?$"
)
_FIN = {"TEMPO", "BECMG", "NOSIG", "RMK"}


def _temp(t: str | None) -> float:
    if not t:
        return np.nan
    return -float(t[1:]) if t.startswith("M") else float(t)


def _es_fenomeno(tok: str) -> bool:
    """Fenómeno presente (no reciente RE.., no NSW) con al menos una letra."""
    return not tok.startswith("RE") and tok != "NSW" and any(c.isalpha() for c in tok) and bool(_RE_WX.match(tok))


def humedad_relativa(t_c: float, td_c: float) -> float:
    """HR (%) por la fórmula de Magnus."""
    if np.isnan(t_c) or np.isnan(td_c):
        return np.nan
    a, b = 17.625, 243.04
    return float(100 * np.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c)))


def leer(metar: str, ref: pd.Timestamp) -> dict | None:
    """Un METAR en texto -> fila con columnas de IEM (valid, tmpf, dwpf, relh, drct, sknt, gust, alti,
    vsby, skyc1-4, skyl1-4, wxcodes, metar). `ref` da el mes y el año (el METAR solo trae día y hora)."""
    tokens = metar.replace("=", " ").split()
    if not tokens:
        return None
    i = 1 if tokens[0] in ("METAR", "SPECI") else 0
    if i + 1 >= len(tokens) or not (m := _RE_HORA.match(tokens[i + 1])):
        return None
    valid = _fecha(int(m.group(1)), int(m.group(2)), int(m.group(3)), ref)
    fila = {"station": tokens[i], "valid": valid.strftime("%Y-%m-%d %H:%M"), "metar": " ".join(tokens),
            "tmpf": np.nan, "dwpf": np.nan, "relh": np.nan, "drct": np.nan, "sknt": np.nan, "gust": np.nan,
            "alti": np.nan, "vsby": np.nan, "wxcodes": ""}
    nubes, wx = [], []
    t_c = td_c = np.nan
    for tok in tokens[i + 2:]:
        if tok in _FIN:
            break
        if tok in ("AUTO", "COR", "NIL") or _RE_VAR.match(tok) or tok.startswith("R") and "/" in tok:
            continue
        if (m := _RE_VIENTO.match(tok)):
            factor = 1.94384 if m.group(4) == "MPS" else 1.0
            if m.group(1) not in ("VRB", "///"):
                fila["drct"] = float(m.group(1))
            if m.group(2) != "//":
                fila["sknt"] = round(float(m.group(2)) * factor)
            if m.group(3):
                fila["gust"] = round(float(m.group(3)) * factor)
        elif tok == "CAVOK":
            fila["vsby"] = 10_000 / MILLA_A_METROS
        elif (m := _RE_VIS.match(tok)) and np.isnan(fila["vsby"]):
            fila["vsby"] = (10_000 if m.group(1) == "9999" else int(m.group(1))) / MILLA_A_METROS
        elif (m := _RE_NUBE.match(tok)):
            nubes.append((m.group(1), np.nan if m.group(2) == "///" else int(m.group(2)) * 100.0))
        elif tok in ("NSC", "NCD", "SKC", "CLR"):
            nubes.append((tok, np.nan))
        elif (m := _RE_VV.match(tok)):
            nubes.append(("VV", 0.0 if m.group(1) == "///" else int(m.group(1)) * 100.0))
        elif (m := _RE_TEMP.match(tok)):
            t_c, td_c = _temp(m.group(1)), _temp(m.group(2))
        elif (m := _RE_QNH.match(tok)):
            fila["alti"] = int(m.group(1)) / 33.8639
        elif _es_fenomeno(tok):
            wx.append(tok)
    fila["tmpf"] = t_c * 9 / 5 + 32 if not np.isnan(t_c) else np.nan
    fila["dwpf"] = td_c * 9 / 5 + 32 if not np.isnan(td_c) else np.nan
    fila["relh"] = humedad_relativa(t_c, td_c)
    for k in range(4):
        cob, alt = nubes[k] if k < len(nubes) else ("", np.nan)
        fila[f"skyc{k + 1}"] = cob
        fila[f"skyl{k + 1}"] = alt
    fila["wxcodes"] = " ".join(wx)
    return fila


def leer_varios(textos: list[str], ref: pd.Timestamp) -> pd.DataFrame:
    """Lista de METAR en texto -> tabla con formato IEM, lista para limpiar_metar.limpiar()."""
    filas = [f for t in textos if (f := leer(t, ref)) is not None]
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    return df.sort_values("valid").drop_duplicates(subset=["valid", "metar"])
