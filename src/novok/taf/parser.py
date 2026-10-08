"""Convierte un TAF en un pronóstico hora a hora de visibilidad y techo.

Para cada hora de validez se obtienen:

- Condición predominante (vis_prev_m, techo_prev_ft): grupo base, aplicando los FM
  desde su hora y los BECMG una vez terminado su periodo de transición.
- Peor caso (vis_peor_m, techo_peor_ft): lo peor entre la predominante y los grupos
  TEMPO / PROB30 / PROB40 / transiciones BECMG activos en esa hora.
- Pesos de los grupos "posibles", para traducir el TAF a una probabilidad:
  PROB30 -> 0,3; PROB40 -> 0,4; TEMPO y transición BECMG -> 0,5 (convención propia,
  documentada en el README: el TAF no da una probabilidad numérica para estos grupos).

Techo sin capas BKN/OVC/VV (CAVOK, NSC, solo FEW/SCT) = infinito (sin techo).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VIS_MAX_M = 10_000
SIN_TECHO = np.inf

PESO_TEMPO = 0.5
PESO_BECMG_TRANSICION = 0.5
PESO_PROB = {"PROB30": 0.3, "PROB40": 0.4}

_RE_EMISION = re.compile(r"^(\d{2})(\d{2})(\d{2})Z$")
_RE_PERIODO = re.compile(r"^(\d{2})(\d{2})/(\d{2})(\d{2})$")
_RE_FM = re.compile(r"^FM(\d{2})(\d{2})(\d{2})$")
_RE_VIENTO = re.compile(r"^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT$")
_RE_VIS = re.compile(r"^(\d{4})(?:NDV)?$")
_RE_NUBE = re.compile(r"^(FEW|SCT|BKN|OVC)(\d{3})(?:CB|TCU|///)?$")
_RE_VV = re.compile(r"^VV(\d{3}|///)$")
_INDICADORES = {"BECMG", "TEMPO", "PROB30", "PROB40"}


@dataclass
class Grupo:
    tipo: str                      # BASE, FM, BECMG, TEMPO, PROB
    inicio: pd.Timestamp
    fin: pd.Timestamp
    elementos: dict = field(default_factory=dict)
    peso: float = 1.0


@dataclass
class TAF:
    estacion: str
    emision: pd.Timestamp
    inicio: pd.Timestamp
    fin: pd.Timestamp
    grupos: list[Grupo]
    enmienda: bool = False


def _fecha(dia: int, hora: int, minuto: int, ref: pd.Timestamp) -> pd.Timestamp:
    """Resuelve día/hora del TAF (sin mes ni año) usando la fecha de emisión como referencia.

    Admite "24" como hora (= 00 del día siguiente) y el cambio de mes.
    """
    extra = pd.Timedelta(0)
    if hora == 24:
        hora, extra = 0, pd.Timedelta(days=1)
    candidatos = []
    for desplaz in (-1, 0, 1):
        base = (ref.replace(day=1) + pd.DateOffset(months=desplaz))
        try:
            candidatos.append(base.replace(day=dia, hour=hora, minute=minuto, second=0) + extra)
        except ValueError:  # p. ej. día 31 en un mes de 30
            continue
    return min(candidatos, key=lambda c: abs(c - ref))


def elementos(tokens: list[str]) -> dict:
    """Extrae viento, visibilidad y techo de los tokens de un grupo (solo lo que aparece)."""
    e: dict = {}
    capas_techo, hay_nubes = [], False
    for t in tokens:
        if t == "CAVOK":
            e["vis_m"] = float(VIS_MAX_M)
            e["techo_ft"] = SIN_TECHO
            continue
        if (m := _RE_VIENTO.match(t)):
            e["viento_dir"] = np.nan if m.group(1) == "VRB" else float(m.group(1))
            e["viento_kt"] = float(m.group(2))
            e["racha_kt"] = float(m.group(3)) if m.group(3) else np.nan
        elif (m := _RE_VIS.match(t)):
            v = int(m.group(1))
            e["vis_m"] = float(VIS_MAX_M if v == 9999 else v)
        elif (m := _RE_NUBE.match(t)):
            hay_nubes = True
            if m.group(1) in ("BKN", "OVC"):
                capas_techo.append(int(m.group(2)) * 100)
        elif (m := _RE_VV.match(t)):
            hay_nubes = True
            capas_techo.append(0 if m.group(1) == "///" else int(m.group(1)) * 100)
        elif t in ("NSC", "SKC", "NCD"):
            hay_nubes = True
    if hay_nubes and "techo_ft" not in e:
        e["techo_ft"] = float(min(capas_techo)) if capas_techo else SIN_TECHO
    return e


def parsear(texto: str, ref: pd.Timestamp) -> TAF | None:
    """Parsea un TAF en una línea. Devuelve None si es NIL, CNL o no se puede leer.

    El TAF no lleva mes ni año: se toman de `ref` (la marca de tiempo de Ogimet).
    """
    tokens = texto.replace("=", " ").split()
    if not tokens or tokens[0] != "TAF":
        return None
    i, enmienda = 1, False
    while i < len(tokens) and tokens[i] in ("AMD", "COR", "RTD"):
        enmienda |= tokens[i] == "AMD"
        i += 1
    if i + 2 >= len(tokens):
        return None
    estacion, emi, per = tokens[i], _RE_EMISION.match(tokens[i + 1]), _RE_PERIODO.match(tokens[i + 2])
    if not emi or not per or {"NIL", "CNL"} & set(tokens):
        return None

    d, h, mi = (int(x) for x in emi.groups())
    emision = _fecha(d, h, mi, ref)
    inicio = _fecha(int(per.group(1)), int(per.group(2)), 0, emision)
    fin = _fecha(int(per.group(3)), int(per.group(4)), 0, inicio)
    if fin <= inicio:
        fin += pd.DateOffset(months=1)
    return TAF(estacion, emision, inicio, fin, _grupos(tokens[i + 3:], inicio, fin), enmienda)


def _grupos(cuerpo: list[str], inicio: pd.Timestamp, fin: pd.Timestamp) -> list[Grupo]:
    """Divide el cuerpo del TAF en grupo base y grupos de cambio."""
    grupos: list[Grupo] = []
    actual, toks = Grupo("BASE", inicio, fin), []
    j = 0
    while j < len(cuerpo):
        t = cuerpo[j]
        fm = _RE_FM.match(t)
        if not (t in _INDICADORES or fm):
            toks.append(t)
            j += 1
            continue
        actual.elementos = elementos(toks)
        grupos.append(actual)
        toks = []
        if fm:
            actual = Grupo("FM", _fecha(int(fm.group(1)), int(fm.group(2)), int(fm.group(3)), inicio), fin)
            j += 1
            continue
        tipo, peso = t, {"TEMPO": PESO_TEMPO, "BECMG": PESO_BECMG_TRANSICION}.get(t, 1.0)
        if t in PESO_PROB:
            tipo, peso = "PROB", PESO_PROB[t]
            if j + 1 < len(cuerpo) and cuerpo[j + 1] == "TEMPO":
                j += 1
        ini, fi = inicio, fin
        if j + 1 < len(cuerpo) and (m := _RE_PERIODO.match(cuerpo[j + 1])):
            ini = _fecha(int(m.group(1)), int(m.group(2)), 0, inicio)
            fi = _fecha(int(m.group(3)), int(m.group(4)), 0, ini)
            j += 1
        actual = Grupo(tipo, ini, fi, peso=peso)
        j += 1
    actual.elementos = elementos(toks)
    grupos.append(actual)
    return grupos


def a_horario(taf: TAF) -> pd.DataFrame:
    """Una fila por hora de validez con condición predominante, peor caso y estados posibles."""
    horas = pd.date_range(taf.inicio, taf.fin, freq="1h", inclusive="left")
    base = next(g for g in taf.grupos if g.tipo == "BASE")
    # Cambios permanentes en orden: FM desde su inicio, BECMG desde el final de su transición
    cambios = sorted(
        [(g.inicio, g) for g in taf.grupos if g.tipo == "FM"]
        + [(g.fin, g) for g in taf.grupos if g.tipo == "BECMG"],
        key=lambda x: x[0],
    )
    filas = []
    for h in horas:
        prev = dict(base.elementos)
        for desde, g in cambios:
            if desde <= h:
                prev.update(g.elementos)
        posibles = []
        for g in taf.grupos:
            activo = g.inicio <= h < g.fin
            if activo and g.tipo in ("TEMPO", "PROB", "BECMG"):
                posibles.append(({**prev, **g.elementos}, g.peso))
        vis_prev = prev.get("vis_m", np.nan)
        techo_prev = prev.get("techo_ft", SIN_TECHO)
        vis_peor = min([vis_prev] + [p[0].get("vis_m", vis_prev) for p in posibles])
        techo_peor = min([techo_prev] + [p[0].get("techo_ft", techo_prev) for p in posibles])
        filas.append({
            "emision": taf.emision, "hora_utc": h, "enmienda": taf.enmienda,
            "vis_prev_m": vis_prev, "techo_prev_ft": techo_prev,
            "vis_peor_m": vis_peor, "techo_peor_ft": techo_peor,
            "viento_dir": prev.get("viento_dir", np.nan), "viento_kt": prev.get("viento_kt", np.nan),
            "racha_kt": prev.get("racha_kt", np.nan),
            "_posibles": [(p[0].get("vis_m", vis_prev), p[0].get("techo_ft", techo_prev), p[1]) for p in posibles],
        })
    return pd.DataFrame(filas)


def probabilidad(df: pd.DataFrame, vis_umbral: float | None = None, techo_umbral: float | None = None) -> pd.Series:
    """P(evento) según el TAF: 1 si lo predominante cumple; si no, el mayor peso de los grupos que lo cumplen.

    Evento = vis < vis_umbral, o techo <= techo_umbral, o cualquiera de los dos si se dan ambos.
    """
    def cumple(vis, techo):
        r = False
        if vis_umbral is not None:
            r |= vis < vis_umbral
        if techo_umbral is not None:
            r |= techo <= techo_umbral
        return r

    out = []
    for vis, techo, posibles in zip(df["vis_prev_m"], df["techo_prev_ft"], df["_posibles"], strict=True):
        if cumple(vis, techo):
            out.append(1.0)
            continue
        out.append(max([w for v, t, w in posibles if cumple(v, t)], default=0.0))
    return pd.Series(out, index=df.index, dtype=float)


def taf_a_horario(texto: str, ref: pd.Timestamp) -> pd.DataFrame:
    """Atajo: texto del TAF + fecha de referencia (marca de Ogimet) -> tabla horaria."""
    taf = parsear(texto, ref)
    return a_horario(taf) if taf else pd.DataFrame()
