"""Descarga pronósticos numéricos (NWP) para LEAS desde Open-Meteo, SIN fugas de información.

Se usa la API "Previous Runs" con el sufijo _previous_day1: para cada hora t devuelve lo que el
modelo pronosticaba ~24 h antes. Ese pronóstico ya estaba publicado cuando NOVOK emite con 1-12 h de
antelación (un modelo tarda unas 6 h en publicarse). La API "Historical Forecast" NO sirve: encadena
las primeras horas de cada pasada, valores que aún no existían en el momento de emitir.

Modelos:
- ecmwf_ifs025: el mejor modelo global; archivo desde 2024.
- jma_gsm: modelo global japonés, más grueso; archivo desde 2018.

Uso:
    python -m novok.data.descargar_nwp --prueba        # 1 día: qué variables tienen datos (1 min)
    python -m novok.data.descargar_nwp                 # descarga completa (año a año, reanudable)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date

import pandas as pd
import requests

try:  # certificados del sistema (redes con inspección HTTPS)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from novok.config import DATA_PROCESSED, DATA_RAW

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAT, LON = 43.5636, -6.0346  # Aeropuerto de Asturias
DIR_NWP = DATA_RAW / "nwp"
FICH_VARIABLES = DIR_NWP / "variables_ok.json"
SUFIJO = "_previous_day1"

MODELOS = {"ecmwf_ifs025": 2024, "jma_gsm": 2018}  # modelo -> primer año con archivo
VARIABLES = [
    "temperature_2m", "dew_point_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
    "cloud_cover", "cloud_cover_low", "visibility", "pressure_msl", "precipitation",
]
CORTO = {  # nombres cortos para las columnas del modelo
    "temperature_2m": "temp", "dew_point_2m": "rocio", "relative_humidity_2m": "hr",
    "wind_speed_10m": "viento_kt", "wind_direction_10m": "viento_dir", "cloud_cover": "nubes",
    "cloud_cover_low": "nubes_bajas", "visibility": "vis", "pressure_msl": "presion", "precipitation": "precip",
}
PAUSA_S = 3


def pedir(modelo: str, variables: list[str], inicio: str, fin: str) -> dict:
    params = {
        "latitude": LAT, "longitude": LON, "models": modelo, "timezone": "GMT", "wind_speed_unit": "kn",
        "hourly": ",".join(v + SUFIJO for v in variables), "start_date": inicio, "end_date": fin,
    }
    r = requests.get(URL, params=params, timeout=120,
                     headers={"User-Agent": "novok (proyecto educativo de prediccion de niebla)"})
    datos = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code != 200 or datos.get("error"):
        raise RuntimeError(datos.get("reason") or f"HTTP {r.status_code}: {r.text[:300]}")
    return datos


def a_tabla(datos: dict, modelo: str) -> pd.DataFrame:
    h = datos["hourly"]
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"})
    df.index = pd.to_datetime(h["time"], utc=True)
    df.columns = [f"nwp_{modelo.split('_')[0]}_{CORTO[c.removesuffix(SUFIJO)]}" for c in df.columns]
    return df


def prueba() -> None:
    """Un día de junio de 2024, variable a variable y modelo a modelo."""
    ok: dict[str, list[str]] = {}
    for modelo in MODELOS:
        ok[modelo] = []
        print(f"\n{modelo}")
        for v in VARIABLES:
            try:
                d = pedir(modelo, [v], "2024-06-10", "2024-06-10")
                valores = d["hourly"][v + SUFIJO]
                n = sum(x is not None for x in valores)
                print(f"  {v:<22} {n:>2}/24 horas con dato   p. ej. {valores[6]}")
                if n:
                    ok[modelo].append(v)
            except Exception as e:  # noqa: BLE001 - queremos ver cualquier fallo en la prueba
                print(f"  {v:<22} ERROR: {e}")
            time.sleep(1)
    DIR_NWP.mkdir(parents=True, exist_ok=True)
    FICH_VARIABLES.write_text(json.dumps(ok, indent=2))
    print(f"\nVariables válidas guardadas en {FICH_VARIABLES}")
    if not any(ok.values()):
        raise SystemExit("Ninguna variable con datos: no lances la descarga completa; pásame esta salida.")


def descargar() -> pd.DataFrame:
    if not FICH_VARIABLES.exists():
        raise SystemExit("Primero ejecuta: python -m novok.data.descargar_nwp --prueba")
    ok = json.loads(FICH_VARIABLES.read_text())
    partes = []
    for modelo, desde in MODELOS.items():
        variables = ok.get(modelo, [])
        if not variables:
            print(f"{modelo}: sin variables válidas, se omite")
            continue
        for anio in range(desde, date.today().year + 1):
            fichero = DIR_NWP / f"{modelo}_{anio}.parquet"
            if not fichero.exists():
                fin = min(date(anio, 12, 31), date.today())
                try:
                    tabla = a_tabla(pedir(modelo, variables, f"{anio}-01-01", fin.isoformat()), modelo)
                except Exception as e:  # noqa: BLE001
                    print(f"{modelo} {anio}: ERROR {e}")
                    continue
                tabla.to_parquet(fichero)
                print(f"{modelo} {anio}: {tabla.notna().any(axis=1).sum():,} horas con dato")
                time.sleep(PAUSA_S)
            partes.append(pd.read_parquet(fichero))
    if not partes:
        raise SystemExit("No se ha descargado nada.")
    por_modelo = {}
    for p in partes:
        clave = p.columns[0].split("_")[1]
        por_modelo.setdefault(clave, []).append(p)
    nwp = pd.concat([pd.concat(v).sort_index() for v in por_modelo.values()], axis=1)
    nwp = nwp[~nwp.index.duplicated()].sort_index()
    nwp.to_parquet(DATA_PROCESSED / "nwp.parquet")
    print(f"\nNWP: {len(nwp):,} horas · columnas: {', '.join(nwp.columns)}")
    return nwp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prueba", action="store_true", help="comprueba variables con 1 día de datos")
    args = p.parse_args()
    prueba() if args.prueba else descargar()


if __name__ == "__main__":
    main()
