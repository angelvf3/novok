"""Servidor local de la web de NOVOK, con intermediario para los vuelos en directo.

Algunos proveedores de vuelos no aceptan peticiones hechas por el navegador desde otra web (CORS). Este servidor
sirve web/ como `python -m http.server` y, además, reenvía al proveedor las peticiones que llegan a /api/vuelos/,
así el navegador solo habla con el propio servidor.

Solo reenvía a los proveedores definidos aquí (no es un proxy abierto) y guarda las posiciones unos segundos para no
repetir peticiones si hay varias pestañas abiertas.

También atiende peticiones por trozos (cabecera Range), que necesita el mapa en formato PMTiles.

Uso:
    python -m novok.servidor              # http://localhost:8010
    python -m novok.servidor --puerto 8020
"""

from __future__ import annotations

import argparse
import functools
import re
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from novok.config import RAIZ

PREFIJO = "/api/vuelos/"
# Proveedores a los que se reenvía (/api/vuelos/<clave>/...) y rutas permitidas en cada uno
PROVEEDORES = {
    "adsblol": ("https://api.adsb.lol/", ("v2/point/", "api/0/routeset")),
    "adsbfi": ("https://opendata.adsb.fi/api/", ("v3/lat/",)),   # respaldo de pruebas (no comercial)
}
CACHE_S = 5
CABECERA = {"User-Agent": "novok (https://github.com/angelvf3/novok)"}

_cache: dict[str, tuple[float, int, bytes]] = {}
_cerrojo = threading.Lock()


def destino(ruta: str) -> str | None:
    """URL del proveedor para una ruta local /api/vuelos/..., o None si no está permitida."""
    if not ruta.startswith(PREFIJO):
        return None
    clave, _, resto = ruta[len(PREFIJO):].split("?")[0].partition("/")
    if clave not in PROVEEDORES or ".." in resto:
        return None
    base, permitidas = PROVEEDORES[clave]
    return base + resto if resto.startswith(permitidas) else None


class Manejador(SimpleHTTPRequestHandler):
    def _responder(self, estado: int, cuerpo: bytes, tipo: str = "application/json") -> None:
        self.send_response(estado)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _reenviar(self, metodo: str) -> None:
        url = destino(self.path)
        if url is None:
            self._responder(404, b'{"error": "ruta no permitida"}')
            return
        datos = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if metodo == "POST" else None
        if metodo == "GET":
            with _cerrojo:
                guardado = _cache.get(url)
            if guardado and time.time() - guardado[0] < CACHE_S:
                self._responder(guardado[1], guardado[2])
                return
        try:
            r = requests.request(metodo, url, data=datos, timeout=20,
                                 headers={**CABECERA, "Content-Type": "application/json"} if datos else CABECERA)
            estado, cuerpo = r.status_code, r.content
        except requests.RequestException as e:
            estado, cuerpo = 502, f'{{"error": "{e.__class__.__name__}"}}'.encode()
        if metodo == "GET" and estado == 200:
            with _cerrojo:
                _cache[url] = (time.time(), estado, cuerpo)
        self._responder(estado, cuerpo)

    def _rango(self) -> bool:
        """Sirve un trozo de fichero (cabecera Range). Lo necesita el mapa PMTiles. Devuelve False si no aplica."""
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        ruta = Path(self.translate_path(self.path))
        if not m or not ruta.is_file():
            return False
        total = ruta.stat().st_size
        ini, fin = m.groups()
        if ini == "":                                    # "bytes=-N": los N últimos
            ini, fin = max(total - int(fin or 0), 0), total - 1
        else:
            ini, fin = int(ini), min(int(fin) if fin else total - 1, total - 1)
        if ini > fin or ini >= total:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.end_headers()
            return True
        with ruta.open("rb") as f:
            f.seek(ini)
            trozo = f.read(fin - ini + 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(ruta)))
        self.send_header("Content-Range", f"bytes {ini}-{fin}/{total}")
        self.send_header("Content-Length", str(len(trozo)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(trozo)
        return True

    def do_GET(self) -> None:  # noqa: N802 - nombre impuesto por http.server
        if self.path.startswith(PREFIJO):
            self._reenviar("GET")
        elif not ("Range" in self.headers and self._rango()):
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith(PREFIJO):
            self._reenviar("POST")
        else:
            self._responder(405, b'{"error": "metodo no permitido"}')

    def log_message(self, formato, *args) -> None:  # menos ruido: solo errores
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(formato, *args)


class Servidor(ThreadingHTTPServer):
    """El navegador corta conexiones a menudo (al mover el mapa cancela teselas): eso no es un error."""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        import sys
        if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--puerto", type=int, default=8010)
    args = p.parse_args()
    manejador = functools.partial(Manejador, directory=str(RAIZ / "web"))
    with Servidor(("127.0.0.1", args.puerto), manejador) as s:
        print(f"NOVOK en http://localhost:{args.puerto}  (Ctrl+C para parar)")
        s.serve_forever()


if __name__ == "__main__":
    main()
