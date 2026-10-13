"""Fuentes de datos del briefing: en directo (APIs) o repetición de una fecha pasada (datos locales).

Las dos devuelven lo mismo, así que el motor del briefing no sabe de dónde vienen los datos:
- metar(): lista de METAR/SPECI en texto de las últimas ~48 h
- taf():   lista de (marca de tiempo, TAF en texto)
- nwp():   tabla horaria de pronóstico numérico (_previous_day1) alrededor de ahora, o None
"""

from __future__ import annotations

import json

import pandas as pd
import requests

try:  # certificados del sistema (redes con inspección HTTPS)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from novok.aeropuertos import activa
from novok.config import DATA_RAW, dir_proc

AWC = "https://aviationweather.gov/api/data"
CABECERA = {"User-Agent": "novok (https://github.com/Gelan03/novok)"}


class FuentesEnDirecto:
    def __init__(self, ahora: pd.Timestamp | None = None):
        self.ahora = ahora or pd.Timestamp.now(tz="UTC")
        self.icao = activa().icao

    def metar(self) -> list[str]:
        r = requests.get(f"{AWC}/metar", params={"ids": self.icao, "format": "raw", "hours": 48}, timeout=30,
                         headers=CABECERA)
        r.raise_for_status()
        return [ln.strip() for ln in r.text.splitlines() if ln.strip()]

    def taf(self) -> list[tuple[pd.Timestamp, str]]:
        from novok.data.grabar_taf_vivo import tafs_actuales

        return [(self.ahora, t) for t in tafs_actuales(self.icao)]

    def nwp(self) -> pd.DataFrame | None:
        from novok.data.descargar_nwp import FICH_VARIABLES, MODELOS, VARIABLES, a_tabla, pedir

        variables = json.loads(FICH_VARIABLES.read_text()) if FICH_VARIABLES.exists() else {}
        inicio = (self.ahora - pd.Timedelta(days=3)).date().isoformat()
        fin = (self.ahora + pd.Timedelta(days=1)).date().isoformat()
        partes = []
        for modelo in MODELOS:
            vs = variables.get(modelo) or [v for v in VARIABLES if v not in ("cloud_cover_low", "visibility")]
            try:
                partes.append(a_tabla(pedir(modelo, vs, inicio, fin), modelo))
            except Exception as e:  # noqa: BLE001 - sin NWP el briefing degrada a v1
                print(f"Aviso: sin pronóstico numérico de {modelo}: {e}")
        return pd.concat(partes, axis=1, sort=True) if partes else None


class FuentesRepeticion:
    """Reproduce el briefing que se habría emitido en una fecha pasada, con los datos locales.

    Solo usa información anterior a `ahora` (METAR), salvo el pronóstico numérico, que es el emitido
    ~24 h antes de cada hora, igual que en el entrenamiento. No hay TAF histórico salvo el grabado.
    """

    def __init__(self, ahora: pd.Timestamp):
        self.ahora = ahora
        self.icao = activa().icao

    def metar(self) -> list[str]:
        anios = {self.ahora.year, (self.ahora - pd.Timedelta(days=2)).year}
        raw = pd.concat(pd.read_csv(DATA_RAW / f"metar_{self.icao}_{a}.csv", dtype=str) for a in sorted(anios))
        t = pd.to_datetime(raw["valid"], utc=True)
        sel = raw[(t <= self.ahora) & (t > self.ahora - pd.Timedelta(hours=48))]
        return sel["metar"].dropna().tolist()

    def taf(self) -> list[tuple[pd.Timestamp, str]]:
        f = dir_proc(self.icao) / "taf.parquet"
        if not f.exists():
            return []
        d = pd.read_parquet(f)
        d = d[(d["marca_ogimet"] <= self.ahora) & (d["marca_ogimet"] > self.ahora - pd.Timedelta(hours=30))]
        return list(zip(d["marca_ogimet"], d["taf"], strict=True))

    def nwp(self) -> pd.DataFrame | None:
        f = dir_proc(self.icao) / "nwp.parquet"
        if not f.exists():
            return None
        n = pd.read_parquet(f)
        return n.loc[self.ahora - pd.Timedelta(days=3): self.ahora + pd.Timedelta(days=1)]
