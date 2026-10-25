"""Cartografía propia de NOVOK, generada una vez y servida desde el propio sitio (sin servicios de mapas en directo).

- Mapa de relieve de España para el mapa 2D: sombreado + tintas por altitud,
  en teselas web/mapa/relieve/{z}/{x}/{y}.jpg.
- Relieve de cada aeropuerto para el visor 3D: web/terreno/<OACI>.png (alturas) y .json.
- Ortofoto de cada aeropuerto para el visor 3D: web/terreno/<OACI>_orto.jpg (24 km) y <OACI>_orto_detalle.jpg (6 km).

Fuentes y licencias (todas permiten el uso comercial citando la fuente):
- Alturas: AWS Terrain Tiles (datos abiertos; atribución en https://github.com/tilezen/joerd/blob/master/docs/attribution.md).
- Ortofoto: PNOA, © Instituto Geográfico Nacional de España, CC BY 4.0 (scne.es).

Uso:
    python -m novok.red mapa        # teselas de relieve de España (unos minutos; una vez)
    python -m novok.red terreno     # relieve y ortofoto de cada aeropuerto (unos minutos; una vez)
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from novok.config import DATA_RAW, RAIZ

URL_TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
URL_PNOA = "https://www.ign.es/wms-inspire/pnoa-ma"
CACHE = DATA_RAW / "teselas_relieve"
DIR_TERRENO = RAIZ / "web" / "terreno"
DIR_RELIEVE = RAIZ / "web" / "mapa" / "relieve"
CAJA_ESPANA = (-10.2, 35.6, 4.6, 44.4)   # oeste, sur, este, norte (península, Baleares y sur de Francia)
ATRIBUCION_PNOA = "Ortofoto PNOA © Instituto Geográfico Nacional de España (CC BY 4.0)"


def _sesion():
    import requests
    try:  # certificados del sistema (redes con inspección HTTPS)
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    s = requests.Session()
    s.headers["User-Agent"] = "novok (https://github.com/angelvf3/novok)"
    return s


# ------------------------------------------------------------------ alturas

def tesela_terrarium(z: int, x: int, y: int) -> np.ndarray:
    """Alturas (m) de una tesela 'terrarium' de 256x256, con caché en disco."""
    from PIL import Image
    f = CACHE / str(z) / str(x) / f"{y}.png"
    if not f.exists():
        r = _sesion().get(URL_TERRARIUM.format(z=z, x=x, y=y), timeout=60)
        r.raise_for_status()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(r.content)
    p = np.asarray(Image.open(f).convert("RGB"), dtype=float)
    return p[..., 0] * 256 + p[..., 1] + p[..., 2] / 256 - 32768


def lon_a_x(lon, z):
    return (np.asarray(lon) + 180) / 360 * 2 ** z


def lat_a_y(lat, z):
    la = np.radians(np.asarray(lat))
    return (1 - np.log(np.tan(la) + 1 / np.cos(la)) / math.pi) / 2 * 2 ** z


def y_a_lat(y, z):
    n = math.pi - 2 * math.pi * np.asarray(y) / 2 ** z
    return np.degrees(np.arctan(np.sinh(n)))


def mosaico(z: int, x0: int, x1: int, y0: int, y1: int, tesela: Callable = tesela_terrarium) -> np.ndarray:
    m = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            m[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = tesela(z, tx, ty)
    return m


def rejilla_relieve(lat0: float, lon0: float, alcance_m: float = 12000, n: int = 321, z: int = 12,
                    tesela: Callable = tesela_terrarium) -> np.ndarray:
    """Alturas en una rejilla n x n de 2*alcance metros centrada en el aeropuerto (fila 0 = norte)."""
    dlat = alcance_m / 111320
    dlon = alcance_m / (111320 * math.cos(math.radians(lat0)))
    lats = lat0 + dlat - 2 * dlat * np.arange(n) / (n - 1)
    lons = lon0 - dlon + 2 * dlon * np.arange(n) / (n - 1)
    px, py = lon_a_x(lons, z) * 256, lat_a_y(lats, z) * 256                  # píxel global (Web Mercator)
    x0, x1 = int(px.min() // 256), int(px.max() // 256) + 1
    y0, y1 = int(py.min() // 256), int(py.max() // 256) + 1
    m = mosaico(z, x0, x1, y0, y1, tesela)
    gx, gy = np.meshgrid(px - x0 * 256 - .5, py - y0 * 256 - .5)
    i, j = np.clip(gx.astype(int), 0, m.shape[1] - 2), np.clip(gy.astype(int), 0, m.shape[0] - 2)
    u, v = np.clip(gx - i, 0, 1), np.clip(gy - j, 0, 1)
    return (m[j, i] * (1 - u) * (1 - v) + m[j, i + 1] * u * (1 - v) + m[j + 1, i] * (1 - u) * v
            + m[j + 1, i + 1] * u * v)


def guardar_alturas(alt: np.ndarray, destino: Path) -> None:
    """PNG con codificación 'terrarium' (la que lee el visor): altura = R*256 + G + B/256 - 32768."""
    from PIL import Image
    v = np.clip(alt, -500, 9000) + 32768
    r = np.floor(v / 256)
    g = np.floor(v - r * 256)
    b = np.round((v - r * 256 - g) * 256).clip(0, 255)
    Image.fromarray(np.dstack([r, g, b]).astype("uint8"), "RGB").save(destino, optimize=True)


# ------------------------------------------------------------------ mapa de relieve (2D)

# Tintas por altitud al estilo de los mapas de relieve habituales: verdes suaves abajo, ocres y grises arriba
TINTAS = [(0, (226, 236, 214)), (200, (214, 229, 199)), (500, (226, 226, 200)), (900, (222, 209, 182)),
          (1500, (206, 190, 168)), (2200, (226, 220, 212)), (3500, (247, 247, 245))]
AGUA = np.array([170, 211, 236], dtype=float)


def colorear(alt: np.ndarray, metros_px: float, exageracion: float = 1.0) -> np.ndarray:
    """Altura -> imagen RGB con tintas hipsométricas y sombreado (luz del noroeste)."""
    gy, gx = np.gradient(alt, metros_px)
    gx, gy = gx * exageracion, gy * exageracion
    az, el = math.radians(315), math.radians(45)
    pendiente = np.arctan(np.hypot(gx, gy))
    orient = np.arctan2(-gx, gy)
    sombra = np.sin(el) * np.cos(pendiente) + np.cos(el) * np.sin(pendiente) * np.cos(az - orient)
    sombra = np.clip(sombra, 0, 1)
    alturas = np.array([a for a, _ in TINTAS], dtype=float)
    colores = np.array([c for _, c in TINTAS], dtype=float)
    a = np.clip(alt, 0, alturas[-1])
    rgb = np.stack([np.interp(a, alturas, colores[:, k]) for k in range(3)], axis=-1)
    luz = (0.62 + 0.45 * sombra)[..., None]                      # llano ~ 0,94: casi sin oscurecer
    rgb = np.clip(rgb * luz, 0, 255)
    mar = alt <= 0
    rgb[mar] = AGUA
    return rgb.astype("uint8")


def mapa_relieve(caja=CAJA_ESPANA, zmin: int = 5, zmax: int = 9, destino: Path = DIR_RELIEVE,
                 tesela: Callable = tesela_terrarium) -> int:
    """Teselas JPEG de relieve para el mapa 2D. Devuelve cuántas ha escrito."""
    from PIL import Image
    oeste, sur, este, norte = caja
    n = 0
    for z in range(zmin, zmax + 1):
        x0, x1 = int(lon_a_x(oeste, z)), int(lon_a_x(este, z))
        y0, y1 = int(lat_a_y(norte, z)), int(lat_a_y(sur, z))
        # con una tesela de margen para que el sombreado no tenga cortes en los bordes
        m = mosaico(z, x0 - 1, x1 + 1, y0 - 1, y1 + 1, tesela)
        lat_media = float(y_a_lat((y0 + y1) / 2 + .5, z))
        metros_px = 156543.03 * math.cos(math.radians(lat_media)) / 2 ** z
        img = colorear(m, metros_px, exageracion=2.5 if z <= 7 else 1.8)
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                f = destino / str(z) / str(tx) / f"{ty}.jpg"
                f.parent.mkdir(parents=True, exist_ok=True)
                fila, col = (ty - y0 + 1) * 256, (tx - x0 + 1) * 256
                Image.fromarray(img[fila:fila + 256, col:col + 256]).save(f, quality=84, optimize=True)
                n += 1
        print(f"  z{z}: {(x1 - x0 + 1) * (y1 - y0 + 1)} teselas")
    (destino / "info.json").write_text(json.dumps({"zmin": zmin, "zmax": zmax, "caja": caja,
                                                   "fuente": "AWS Terrain Tiles (datos abiertos)"}), encoding="utf-8")
    return n


# ------------------------------------------------------------------ mapa vectorial (2D)
#
# El mapa 2D usa teselas vectoriales de OpenStreetMap (calles, pueblos, ríos, bosques...) en un único fichero
# PMTiles alojado en el propio sitio, sin servicios externos. Se recorta del mapa mundial que publica Protomaps
# (datos © OpenStreetMap, licencia ODbL: uso comercial permitido citando la fuente) con su herramienta pmtiles.
# Debajo, el relieve se sombrea en el navegador a partir de teselas de altura (AWS Terrain Tiles).

CAJA_NORTE = (-10.2, 40.8, -0.8, 44.6)          # Galicia, Cantábrico, Castilla y León, Navarra, La Rioja
DIR_MAPA = RAIZ / "web" / "mapa"
DIR_HERRAMIENTAS = DATA_RAW / "herramientas"
URL_BUILDS = "https://build.protomaps.com/{fecha}.pmtiles"
LIMITE_MB = 95                                   # GitHub no admite ficheros de más de 100 MB
CRECIMIENTO = 2.6                                # cada zoom ocupa al menos ~2,6 veces el anterior
ATRIBUCION_OSM = "© OpenStreetMap (ODbL) · Protomaps"


def a_terrarium(alt: np.ndarray) -> np.ndarray:
    """Codifica alturas (m) en RGB terrarium: altura = R·256 + G + B/256 − 32768."""
    v = np.clip(alt.astype(float) + 32768, 0, 65535.996)
    r = np.floor(v / 256)
    g = np.floor(v - r * 256)
    b = np.floor((v - r * 256 - g) * 256)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def dem(caja=CAJA_NORTE, zmin: int = 5, zmax: int = 9, destino: Path = DIR_MAPA / "dem",
        tesela: Callable = tesela_terrarium) -> int:
    """Teselas de altura (PNG terrarium) para que MapLibre sombree el relieve. Devuelve cuántas ha escrito."""
    from PIL import Image
    oeste, sur, este, norte = caja
    n = 0
    for z in range(zmin, zmax + 1):
        for ty in range(int(lat_a_y(norte, z)), int(lat_a_y(sur, z)) + 1):
            for tx in range(int(lon_a_x(oeste, z)), int(lon_a_x(este, z)) + 1):
                f = destino / str(z) / str(tx) / f"{ty}.png"
                f.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(a_terrarium(tesela(z, tx, ty))).save(f, optimize=True)
                n += 1
    info = {"zmin": zmin, "zmax": zmax, "caja": caja, "codificacion": "terrarium",
            "fuente": "AWS Terrain Tiles (datos abiertos)"}
    (destino / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return n


def herramienta_pmtiles() -> Path:
    """Ruta del programa pmtiles (go-pmtiles, BSD-3); si no está, descarga la última versión desde GitHub."""
    import platform
    import shutil
    import tarfile
    import zipfile

    if (en_path := shutil.which("pmtiles")):
        return Path(en_path)
    exe = DIR_HERRAMIENTAS / ("pmtiles.exe" if platform.system() == "Windows" else "pmtiles")
    if exe.exists():
        return exe
    sistema = {"Windows": "windows", "Darwin": "darwin", "Linux": "linux"}[platform.system()]
    arq = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
    s = _sesion()
    rel = s.get("https://api.github.com/repos/protomaps/go-pmtiles/releases/latest", timeout=60).json()
    activo = next(a for a in rel["assets"] if sistema in a["name"].lower() and arq in a["name"].lower())
    print(f"  descargando {activo['name']}")
    DIR_HERRAMIENTAS.mkdir(parents=True, exist_ok=True)
    datos = s.get(activo["browser_download_url"], timeout=300).content
    if activo["name"].endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(datos)) as z:
            nombre = next(n for n in z.namelist() if Path(n).name.startswith("pmtiles"))
            exe.write_bytes(z.read(nombre))
    else:
        with tarfile.open(fileobj=io.BytesIO(datos)) as t:
            m = next(m for m in t.getmembers() if Path(m.name).name == "pmtiles")
            exe.write_bytes(t.extractfile(m).read())
    exe.chmod(0o755)
    return exe


def ultimo_build(dias: int = 14) -> str:
    """URL del mapa mundial más reciente de Protomaps (se publica uno al día y se guardan unos días)."""
    from datetime import date, timedelta
    s = _sesion()
    for d in range(dias):
        url = URL_BUILDS.format(fecha=(date.today() - timedelta(days=d)).strftime("%Y%m%d"))
        if s.head(url, timeout=30).status_code == 200:
            return url
    raise SystemExit("No se encontró ningún mapa reciente en build.protomaps.com; inténtalo más tarde.")


def zoom_maximo(fichero: Path) -> int:
    """Zoom máximo de un fichero PMTiles (byte 101 de la cabecera de la versión 3)."""
    with fichero.open("rb") as f:
        cabecera = f.read(127)
    if cabecera[:7] != b"PMTiles":
        raise ValueError(f"{fichero.name} no es un fichero PMTiles")
    return cabecera[101]


def _registrar(destino: Path, caja, z: int, origen: str | None) -> None:
    destino.with_suffix(".json").write_text(json.dumps({"maxzoom": z, "caja": caja, "origen": origen,
                                                        "atribucion": ATRIBUCION_OSM}), encoding="utf-8")


def _recortar(destino: Path, caja, zooms, rehacer: bool = False) -> int:
    """Recorta el mapa mundial de Protomaps (todo el mundo si caja es None) al mayor zoom que quepa en GitHub.

    Prueba de menos a más zoom y se queda con el último que cabe. Cada nivel ocupa unas 3-4 veces más que el
    anterior: si la previsión ya no cabe, no lo descarga. Si ya existe el fichero, lo reutiliza (salvo rehacer).
    """
    import subprocess
    prueba = destino.with_suffix(".prueba.pmtiles")
    prueba.unlink(missing_ok=True)
    if destino.exists() and not rehacer:
        z = zoom_maximo(destino)
        _registrar(destino, caja, z, None)
        print(f"  ya existe {destino.name} (zoom {z}); para volver a descargarlo usa --rehacer")
        return z
    exe, origen = herramienta_pmtiles(), ultimo_build()
    destino.parent.mkdir(parents=True, exist_ok=True)
    extra = [f"--bbox={','.join(str(c) for c in caja)}"] if caja else []
    elegido, mb = None, 0.0
    for z in zooms:
        if elegido is not None and mb * CRECIMIENTO > LIMITE_MB:
            print(f"  el zoom {z} ocuparía unos {mb * CRECIMIENTO:.0f} MB: no cabe, me quedo con el {elegido}")
            break
        print(f"  recortando {origen.rsplit('/', 1)[-1]} hasta el zoom {z}...")
        try:
            subprocess.run([str(exe), "extract", origen, str(prueba), *extra, f"--maxzoom={z}"], check=True)
        except subprocess.CalledProcessError:
            prueba.unlink(missing_ok=True)
            if elegido is None:
                raise SystemExit("Se cortó la descarga del mapa; vuelve a intentarlo más tarde.") from None
            print(f"  se cortó la descarga del zoom {z}; me quedo con el {elegido}")
            break
        mb = prueba.stat().st_size / 1e6
        print(f"  zoom {z}: {mb:.0f} MB")
        if mb > LIMITE_MB:
            prueba.unlink()
            break
        prueba.replace(destino)
        elegido = z
        _registrar(destino, caja, z, origen)
    if elegido is None:
        raise SystemExit(f"El mapa ocupa más de {LIMITE_MB} MB incluso con zoom {zooms[0]}.")
    return elegido


def mapa_vectorial(caja=CAJA_NORTE, zooms=(10, 11, 12, 13), destino: Path = DIR_MAPA / "norte.pmtiles",
                   rehacer: bool = False) -> Path:
    """Mapa detallado del norte de España (calles, pueblos, bosques)."""
    _recortar(destino, caja, zooms, rehacer)
    return destino


def mapa_mundial(zooms=(4, 5, 6, 7), destino: Path = DIR_MAPA / "mundo.pmtiles", rehacer: bool = False) -> Path:
    """Mapa de todo el mundo con menos detalle (países, ciudades, carreteras principales) para alejar el mapa."""
    _recortar(destino, None, zooms, rehacer)
    return destino


# ------------------------------------------------------------------ ortofoto PNOA (3D)

def pnoa(lat0: float, lon0: float, alcance_m: float, px: int = 2048, trozo: int = 1024,
         pedir: Callable | None = None):
    """Ortofoto PNOA de un cuadrado de 2*alcance metros (px x px), pedida por trozos al WMS del IGN."""
    from PIL import Image
    sesion = None if pedir else _sesion()
    dlat = alcance_m / 111320
    dlon = alcance_m / (111320 * math.cos(math.radians(lat0)))
    lienzo = Image.new("RGB", (px, px))
    k = px // trozo
    for fila in range(k):
        for col in range(k):
            norte = lat0 + dlat - 2 * dlat * fila / k
            sur = lat0 + dlat - 2 * dlat * (fila + 1) / k
            oeste = lon0 - dlon + 2 * dlon * col / k
            este = lon0 - dlon + 2 * dlon * (col + 1) / k
            params = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": "OI.OrthoimageCoverage",
                      "STYLES": "", "CRS": "EPSG:4326", "BBOX": f"{sur},{oeste},{norte},{este}",  # 1.3.0: lat,lon
                      "WIDTH": trozo, "HEIGHT": trozo, "FORMAT": "image/jpeg"}
            if pedir:
                datos = pedir(params)
            else:
                r = sesion.get(URL_PNOA, params=params, timeout=120)
                if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
                    raise RuntimeError(f"el IGN respondió {r.status_code}: {r.text[:200]}")
                datos = r.content
            trozo_img = Image.open(io.BytesIO(datos)).convert("RGB").resize((trozo, trozo))
            lienzo.paste(trozo_img, (col * trozo, fila * trozo))
    return lienzo


def terreno(icao: str, lat: float, lon: float, alcance_m: float = 12000, detalle_m: float = 3000,
            n: int = 321, con_orto: bool = True) -> dict:
    DIR_TERRENO.mkdir(parents=True, exist_ok=True)
    alt = rejilla_relieve(lat, lon, alcance_m, n)
    guardar_alturas(alt, DIR_TERRENO / f"{icao}.png")
    meta = {"alcance_m": alcance_m, "n": n, "fuente_alturas": "AWS Terrain Tiles (datos abiertos)"}
    if con_orto:
        try:
            pnoa(lat, lon, alcance_m).save(DIR_TERRENO / f"{icao}_orto.jpg", quality=80, optimize=True)
            pnoa(lat, lon, detalle_m).save(DIR_TERRENO / f"{icao}_orto_detalle.jpg", quality=82, optimize=True)
            meta.update(orto=True, detalle_m=detalle_m, fuente_orto=ATRIBUCION_PNOA)
        except Exception as e:  # noqa: BLE001 - sin ortofoto el visor usa colores por altitud
            print(f"  {icao}: sin ortofoto ({e.__class__.__name__}: {e})")
    (DIR_TERRENO / f"{icao}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    extra = " + ortofoto PNOA" if meta.get("orto") else ""
    print(f"{icao}: relieve {n}x{n} ({alt.min():.0f} a {alt.max():.0f} m){extra}")
    return meta
