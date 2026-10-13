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
    s.headers["User-Agent"] = "novok (https://github.com/Gelan03/novok)"
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
