"""Graba los TAF vigentes de la red de NOVOK desde aviationweather.gov (servicio oficial de la NOAA).

Plan B al histórico de Ogimet: guarda cada TAF nuevo (y cada enmienda) en
data/vivo/taf_<OACI>.txt (un fichero por aeropuerto) con la misma forma que usa el resto del proyecto:

    202610071030 TAF AMD LEAS 071013Z 0710/0809 30009KT 9999 ... =

Pensado para ejecutarse cada 30 min con GitHub Actions (.github/workflows/grabar_taf.yml), que guarda
el fichero en la rama datos-vivo.
Solo añade líneas si el TAF es nuevo, así que se puede lanzar tantas veces como se quiera.

Uso:
    python -m novok.data.grabar_taf_vivo                 # toda la red (una sola petición)
    python -m novok.data.grabar_taf_vivo --estaciones LEAS
"""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from pathlib import Path

import requests

try:  # certificados del sistema (redes con inspección HTTPS)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from novok.aeropuertos import RED
from novok.config import RAIZ

AWC_URL = "https://aviationweather.gov/api/data/taf"
DIR_VIVO = RAIZ / "data" / "vivo"


def tafs_actuales(estacion: str | list[str] = "LEAS") -> list[str]:
    """TAF vigentes de una o varias estaciones, cada uno en una línea y terminado en '='."""
    if not isinstance(estacion, str):
        estacion = ",".join(estacion)
    r = requests.get(AWC_URL, params={"ids": estacion, "format": "raw"}, timeout=30,
                     headers={"User-Agent": "novok (https://github.com/Gelan03/novok)"})
    r.raise_for_status()
    tafs, actual = [], []
    for linea in r.text.splitlines():
        if linea.strip().startswith("TAF") and actual:
            tafs.append(actual)
            actual = []
        if linea.strip():
            actual.append(linea.strip())
    if actual:
        tafs.append(actual)
    return [" ".join(" ".join(t).split()).rstrip("=") + "=" for t in tafs]


def separar(tafs: list[str]) -> dict[str, list[str]]:
    """Agrupa los TAF por estación (el código va tras "TAF", "TAF AMD" o "TAF COR")."""
    salida: dict[str, list[str]] = {}
    for t in tafs:
        m = re.match(r"TAF(?:\s+(?:AMD|COR))*\s+([A-Z]{4})\b", t)
        if m:
            salida.setdefault(m.group(1), []).append(t)
    return salida


def grabar(tafs: list[str], fichero: Path, ahora: datetime | None = None) -> int:
    """Añade al fichero los TAF que no estén ya. Devuelve cuántos ha añadido."""
    ahora = ahora or datetime.now(UTC)
    existentes = set()
    if fichero.exists():
        existentes = {ln.split(" ", 1)[1] for ln in fichero.read_text(encoding="utf-8").splitlines() if " " in ln}
    nuevos = [t for t in tafs if t not in existentes]
    if nuevos:
        fichero.parent.mkdir(parents=True, exist_ok=True)
        with fichero.open("a", encoding="utf-8") as f:
            for t in nuevos:
                f.write(f"{ahora:%Y%m%d%H%M} {t}\n")
    return len(nuevos)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=str(DIR_VIVO), help="carpeta donde se guardan los taf_<OACI>.txt")
    p.add_argument("--estaciones", default=",".join(RED), help="códigos OACI separados por comas")
    args = p.parse_args()
    estaciones = [e.strip().upper() for e in args.estaciones.split(",") if e.strip()]
    por_estacion = separar(tafs_actuales(estaciones))
    for est in estaciones:
        fichero = Path(args.dir) / f"taf_{est}.txt"
        n = grabar(por_estacion.get(est, []), fichero)
        print(f"{est}: {n} TAF nuevos")


if __name__ == "__main__":
    main()
