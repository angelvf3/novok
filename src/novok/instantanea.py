"""Foto de los vuelos en el aire para la web publicada (sin intermediario en directo).

Los proveedores comunitarios de ADS-B limitan las peticiones que llegan desde Cloudflare, así que la web publicada
no puede pedir los vuelos en directo a través del Worker. En su lugar, cada vez que GitHub publica el briefing
(cada 30 minutos) se guarda una foto de los aviones y de sus rutas en datos/vuelos_ahora.json, y la web la usa
indicando la hora de la foto. En local (python -m novok.servidor) se siguen pidiendo en directo.

Uso:
    python -m novok.instantanea --salida _site/datos/vuelos_ahora.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from novok import alertas
from novok.config import RAIZ


def foto(cfg_vuelos: dict) -> dict:
    ahora = time.time()
    aviones = alertas.posiciones(cfg_vuelos)
    rutas = alertas.rutas(cfg_vuelos, aviones) if aviones else {}
    salida = {
        "now": int(ahora * 1000),
        "generado_utc": datetime.fromtimestamp(ahora, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "proveedor": cfg_vuelos.get("proveedor"),
        "ac": aviones,
        # mismo formato que el routeset de adsb.lol, para que la web lo lea igual
        "rutas": [{"callsign": c, "airport_codes": f"{r['origen']}-{r['destino']}"} for c, r in rutas.items()],
    }
    if not aviones:
        salida["error"] = "el proveedor no devolvió aviones"
    return salida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--salida", type=Path, default=RAIZ / "web" / "datos" / "vuelos_ahora.json")
    a = ap.parse_args()
    cfg = json.loads((RAIZ / "web" / "vuelos.json").read_text(encoding="utf-8"))
    f = foto(cfg)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(f, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{a.salida}: {len(f['ac'])} aviones, {len(f['rutas'])} rutas" + (f" ({f['error']})" if "error" in f else ""))


if __name__ == "__main__":
    main()
