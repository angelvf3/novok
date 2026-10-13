"""Aeropuertos de la red de NOVOK y aeropuerto activo.

Los datos (coordenadas, elevación, pistas con sus umbrales) salen de OurAirports, una base de datos
abierta de aeropuertos. Se descargan una vez con:

    python -m novok.aeropuertos --descargar

y se guardan en config/aeropuertos.json (se sube al repo). Mientras no exista ese fichero,
NOVOK solo conoce LEAS, con los datos que usaba hasta F5.

El aeropuerto activo decide de dónde se leen los datos y dónde se guardan los modelos.
Se elige con la variable de entorno NOVOK_ESTACION, con --estacion en las órdenes, o en código con
`usar("LEXJ")` o `with estacion("LEXJ"): ...`.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
from dataclasses import dataclass

from novok.config import RAIZ

FICHERO = RAIZ / "config" / "aeropuertos.json"

# Red inicial: aeropuertos del norte de España, donde más pesa la niebla.
RED = {
    "LEAS": "Asturias", "LEXJ": "Santander", "LEBB": "Bilbao", "LESO": "San Sebastián", "LEVT": "Vitoria",
    "LEPP": "Pamplona", "LELO": "Logroño", "LEBG": "Burgos", "LELN": "León", "LEVD": "Valladolid",
    "LEST": "Santiago", "LECO": "A Coruña", "LEVX": "Vigo",
}

_LEAS = {  # lo que usaba NOVOK hasta F5; se sustituye por OurAirports al descargar
    "icao": "LEAS", "nombre": "Asturias", "lat": 43.5636, "lon": -6.0346, "elev_ft": 416,
    "pistas": [
        {"ident": "11", "rumbo": 110.0, "lat": 43.5667, "lon": -6.0465, "longitud_m": 2200},
        {"ident": "29", "rumbo": 290.0, "lat": 43.5605, "lon": -6.0219, "longitud_m": 2200},
    ],
}


@dataclass(frozen=True)
class Pista:
    ident: str                 # designación de la cabecera: "11", "29", "12R"...
    rumbo: float               # rumbo verdadero de aterrizaje, grados
    lat: float | None = None   # umbral
    lon: float | None = None
    longitud_m: float | None = None


@dataclass(frozen=True)
class Aeropuerto:
    icao: str
    nombre: str
    lat: float
    lon: float
    elev_ft: float
    pistas: tuple[Pista, ...]

    @property
    def rumbos(self) -> dict[str, float]:
        return {p.ident: p.rumbo for p in self.pistas}

    def a_dict(self) -> dict:
        return {"icao": self.icao, "nombre": self.nombre, "lat": self.lat, "lon": self.lon, "elev_ft": self.elev_ft,
                "pistas": [p.__dict__ for p in self.pistas]}


def _crear(d: dict) -> Aeropuerto:
    return Aeropuerto(d["icao"], d["nombre"], d["lat"], d["lon"], d["elev_ft"],
                      tuple(Pista(**p) for p in d["pistas"]))


def catalogo() -> dict[str, Aeropuerto]:
    datos = json.loads(FICHERO.read_text(encoding="utf-8")) if FICHERO.exists() else {}
    datos.setdefault("LEAS", _LEAS)
    return {k: _crear(v) for k, v in datos.items()}


_activa: str | None = None


def activa() -> Aeropuerto:
    icao = (_activa or os.environ.get("NOVOK_ESTACION") or "LEAS").upper()
    cat = catalogo()
    if icao not in cat:
        raise SystemExit(f"{icao} no está en {FICHERO.name}. Ejecuta: python -m novok.aeropuertos --descargar")
    return cat[icao]


def usar(icao: str | None) -> None:
    global _activa
    _activa = icao.upper() if icao else None


@contextlib.contextmanager
def estacion(icao: str):
    global _activa
    antes, _activa = _activa, icao.upper()
    try:
        yield activa()
    finally:
        _activa = antes


def cli_estacion() -> Aeropuerto:
    """Lee --estacion de la línea de órdenes (sin molestar a otros argumentos) y lo activa."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--estacion")
    est = p.parse_known_args()[0].estacion
    if est:  # sin --estacion se respeta el aeropuerto ya activo
        usar(est)
    return activa()


# ------------------------------------------------------------------ descarga desde OurAirports

URL = "https://davidmegginson.github.io/ourairports-data/{}.csv"


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def construir(airports_csv: str, runways_csv: str, codigos: list[str]) -> dict:
    aer = {r["ident"]: r for r in csv.DictReader(io.StringIO(airports_csv)) if r["ident"] in codigos}
    salida = {}
    for icao in codigos:
        if icao not in aer:
            print(f"  {icao}: no aparece en OurAirports, se omite")
            continue
        a = aer[icao]
        pistas = []
        for r in csv.DictReader(io.StringIO(runways_csv)):
            if r["airport_ident"] != icao or r["closed"] == "1" or not r["le_ident"] or r["le_ident"].startswith("H"):
                continue
            largo = _num(r["length_ft"])
            for ext, otro in (("le", "he"), ("he", "le")):
                rumbo = _num(r[f"{ext}_heading_degT"])
                if rumbo is None and _num(r[f"{otro}_heading_degT"]) is not None:
                    rumbo = (_num(r[f"{otro}_heading_degT"]) + 180) % 360
                if rumbo is None:  # sin rumbo publicado: el de la designación
                    digitos = "".join(c for c in r[f"{ext}_ident"] if c.isdigit())
                    rumbo = float(int(digitos) * 10) if digitos else None
                if rumbo is None:
                    continue
                pistas.append({"ident": r[f"{ext}_ident"], "rumbo": round(rumbo, 1),
                               "lat": _num(r[f"{ext}_latitude_deg"]), "lon": _num(r[f"{ext}_longitude_deg"]),
                               "longitud_m": round(largo * 0.3048) if largo else None})
        salida[icao] = {"icao": icao, "nombre": RED.get(icao, a["municipality"] or a["name"]),
                        "lat": float(a["latitude_deg"]), "lon": float(a["longitude_deg"]),
                        "elev_ft": _num(a["elevation_ft"]) or 0, "pistas": pistas}
        print(f"  {icao} {salida[icao]['nombre']:<14} pistas: {', '.join(p['ident'] for p in pistas) or 'ninguna'}")
    return salida


def descargar(codigos: list[str]) -> None:
    import requests
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    print("Descargando OurAirports...")
    textos = [requests.get(URL.format(n), timeout=120).text for n in ("airports", "runways")]
    datos = construir(*textos, codigos)
    FICHERO.parent.mkdir(exist_ok=True)
    FICHERO.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {FICHERO}  ({len(datos)} aeropuertos)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--descargar", action="store_true", help="descarga los datos de la red desde OurAirports")
    p.add_argument("codigos", nargs="*", help="códigos OACI (por defecto, toda la red)")
    args = p.parse_args()
    if args.descargar:
        descargar([c.upper() for c in args.codigos] or list(RED))
    else:
        for a in catalogo().values():
            print(f"{a.icao}  {a.nombre:<14} {a.lat:8.4f} {a.lon:8.4f}  {a.elev_ft:5.0f} ft  "
                  f"pistas {'/'.join(p.ident for p in a.pistas)}")


if __name__ == "__main__":
    main()
