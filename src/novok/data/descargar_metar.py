"""Descarga el histórico de METAR de LEAS desde Iowa Environmental Mesonet (IEM).

Descarga año a año (como recomienda IEM) y guarda un CSV crudo por año en data/raw/.
Si el fichero de un año ya existe, lo salta, salvo que se use --forzar.

Uso:
    python -m novok.data.descargar_metar --inicio 2010 --fin 2025
"""

from __future__ import annotations

import argparse
import io
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

try:  # usa los certificados del sistema (necesario tras proxies corporativos con inspección HTTPS)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from novok.aeropuertos import activa
from novok.config import DATA_RAW, IEM_URL

PAUSA_ENTRE_PETICIONES_S = 5  # cortesía con el servidor de IEM
REINTENTOS = 3


def parametros_anio(estacion: str, anio: int) -> list[tuple[str, str | int]]:
    """Parámetros de la petición a IEM para un año completo en UTC."""
    return [
        ("station", estacion),
        ("data", "all"),
        ("year1", anio), ("month1", 1), ("day1", 1),
        ("year2", anio + 1), ("month2", 1), ("day2", 1),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", 3),  # METAR rutinario
        ("report_type", 4),  # SPECI (especiales, clave en niebla)
    ]


def descargar_anio(estacion: str, anio: int, destino: Path, forzar: bool = False) -> Path | None:
    """Descarga un año y lo guarda en CSV. Devuelve la ruta o None si no hay datos."""
    fichero = destino / f"metar_{estacion}_{anio}.csv"
    if fichero.exists() and not forzar:
        print(f"[{anio}] ya existe, se salta ({fichero.name})")
        return fichero

    for intento in range(1, REINTENTOS + 1):
        try:
            r = requests.get(IEM_URL, params=parametros_anio(estacion, anio), timeout=180)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[{anio}] intento {intento}/{REINTENTOS} fallido: {e}")
            if intento == REINTENTOS:
                raise
            time.sleep(PAUSA_ENTRE_PETICIONES_S * intento)

    df = pd.read_csv(io.StringIO(r.text), dtype=str)
    if df.empty:
        print(f"[{anio}] sin observaciones")
        return None

    destino.mkdir(parents=True, exist_ok=True)
    df.to_csv(fichero, index=False)
    print(f"[{anio}] {len(df):,} observaciones -> {fichero.name}")
    return fichero


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--estacion", help="código OACI (por defecto, el aeropuerto activo)")
    p.add_argument("--inicio", type=int, default=2010)
    p.add_argument("--fin", type=int, default=date.today().year)
    p.add_argument("--forzar", action="store_true", help="vuelve a descargar aunque exista")
    args = p.parse_args()
    args.estacion = (args.estacion or activa().icao).upper()

    for anio in range(args.inicio, args.fin + 1):
        descargar_anio(args.estacion, anio, DATA_RAW, args.forzar)
        time.sleep(PAUSA_ENTRE_PETICIONES_S)


if __name__ == "__main__":
    main()
