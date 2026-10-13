"""Descarga el histórico de TAF de LEAS desde Ogimet (www.ogimet.com).

IEM solo archiva TAF de EE. UU., así que para LEAS la fuente es Ogimet.
Descarga mes a mes y guarda el texto crudo en data/raw/taf/. Es reanudable:
si el fichero de un mes ya existe, lo salta.

Ogimet limita el número de consultas: el script espera entre peticiones y,
si recibe el aviso de cuota, espera varios minutos y reintenta.

Uso:
    python -m novok.data.descargar_taf --prueba 2024-06      # comprobar con un mes (1 min)
    python -m novok.data.descargar_taf --inicio 2017 --fin 2025
    python -m novok.data.descargar_taf --extraer   # solo une los TAF ya descargados
"""

from __future__ import annotations

import argparse
import calendar
import re
import time
from datetime import date

import pandas as pd
import requests

try:  # usa los certificados del sistema (necesario tras proxies corporativos con inspección HTTPS)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from novok.aeropuertos import activa, usar
from novok.config import DATA_RAW, dir_proc

OGIMET_URL = "https://www.ogimet.com/display_metars2.php"
DIR_TAF = DATA_RAW / "taf"
PAUSA_S = 20          # entre peticiones
PAUSA_CUOTA_S = 600   # si Ogimet avisa de cuota
REINTENTOS = 4
TIPOS = ("FT", "FC")  # TAF largo (24-30 h) y corto (9 h)

MAX_MESES_VACIOS = 3  # si tantos meses seguidos no traen TAF, algo va mal: parar y diagnosticar

# Un TAF en el texto de Ogimet: "202406101100 TAF LEAS 101100Z 1012/1112 ... ="
# Tolerante a variantes: palabra TAF opcional, AMD/COR, y TAF sin "=" final.
_RE_TAF = re.compile(
    r"(\d{12})\s+((?:TAF\s+)?(?:AMD\s+|COR\s+)*[A-Z]{4}\s+\d{6}Z\s+.*?)(?:=|(?=\s\d{12}\s)|$)",
    re.DOTALL,
)
_RE_HTML = re.compile(r"<[^>]+>")


def parametros(estacion: str, tipo: str, anio: int, mes: int) -> dict:
    ultimo = calendar.monthrange(anio, mes)[1]
    return {
        "lang": "en", "lugar": estacion, "tipo": tipo, "ord": "DIR", "nil": "NO", "fmt": "txt",
        "ano": anio, "mes": mes, "day": 1, "hora": 0,
        "anof": anio, "mesf": mes, "dayf": ultimo, "horaf": 23, "minf": 59, "enviar": "Ver",
    }


def extraer_tafs(texto: str) -> list[tuple[str, str]]:
    """Devuelve [(marca_ogimet, taf_en_una_linea)] encontrados en un texto de Ogimet.

    El TAF se normaliza a "TAF ... =" aunque Ogimet omita la palabra TAF o el "=".
    """
    limpio = _RE_HTML.sub(" ", texto)
    out = []
    for marca, taf in _RE_TAF.findall(limpio):
        taf = " ".join(taf.split())
        if not taf.startswith("TAF"):
            taf = "TAF " + taf
        if "METAR" in taf or "SPECI" in taf:
            continue
        out.append((marca, taf + "="))
    return out


def diagnostico(texto: str, n: int = 800) -> str:
    """Primeros caracteres útiles de una respuesta (sin HTML) para ver qué devolvió Ogimet."""
    return " ".join(_RE_HTML.sub(" ", texto).split())[:n]


def _es_aviso_cuota(texto: str) -> bool:
    t = texto.lower()
    return "quota" in t or "too many" in t


def descargar_mes(estacion: str, tipo: str, anio: int, mes: int, forzar: bool = False) -> int | None:
    """Descarga un mes. Devuelve cuántos TAF trae, o None si ya estaba descargado o falló."""
    fichero = DIR_TAF / f"taf_{estacion}_{anio}_{mes:02d}_{tipo}.txt"
    if fichero.exists() and not forzar:
        if extraer_tafs(fichero.read_text(encoding="utf-8", errors="replace")):
            return None  # ya descargado y válido: no cuenta como petición
        fichero.unlink()  # descarga anterior vacía (p. ej. página sin datos): se repite

    for intento in range(1, REINTENTOS + 1):
        try:
            r = requests.get(OGIMET_URL, params=parametros(estacion, tipo, anio, mes), timeout=120,
                             headers={"User-Agent": "novok (https://github.com/Gelan03/novok)"})
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  {anio}-{mes:02d} {tipo}: error {e} (intento {intento}/{REINTENTOS})")
            time.sleep(PAUSA_S * intento)
            continue
        if _es_aviso_cuota(r.text):
            print(f"  {anio}-{mes:02d} {tipo}: Ogimet avisa de cuota; espero {PAUSA_CUOTA_S // 60} min")
            time.sleep(PAUSA_CUOTA_S)
            continue
        n = len(extraer_tafs(r.text))
        if n == 0:  # no se guarda: así se reintenta cuando el problema esté resuelto
            vacio = DIR_TAF / "ultima_respuesta_vacia.txt"
            DIR_TAF.mkdir(parents=True, exist_ok=True)
            vacio.write_text(r.text, encoding="utf-8")
            print(f"  {anio}-{mes:02d} {tipo}: 0 TAF (respuesta guardada en {vacio.name})")
            return 0
        DIR_TAF.mkdir(parents=True, exist_ok=True)
        fichero.write_text(r.text, encoding="utf-8")
        print(f"  {anio}-{mes:02d} {tipo}: {n} TAF")
        return n
    print(f"  {anio}-{mes:02d} {tipo}: sin éxito tras {REINTENTOS} intentos; vuelve a lanzar el script más tarde")
    return None


def unir_tafs(estacion: str | None = None) -> pd.DataFrame:
    """Lee todos los ficheros descargados y guarda data/processed/<OACI>/taf.parquet (sin duplicados)."""
    estacion = estacion or activa().icao
    filas = []
    vivo = DATA_RAW.parent / "vivo" / f"taf_{estacion}.txt"  # grabación en directo (grabar_taf_vivo)
    ficheros = sorted(DIR_TAF.glob(f"taf_{estacion}_*.txt")) + ([vivo] if vivo.exists() else [])
    for f in ficheros:
        for marca, taf in extraer_tafs(f.read_text(encoding="utf-8", errors="replace")):
            filas.append({"marca_ogimet": pd.to_datetime(marca, format="%Y%m%d%H%M", utc=True), "taf": taf})
    df = pd.DataFrame(filas, columns=["marca_ogimet", "taf"])
    df = df.drop_duplicates(subset="taf").sort_values("marca_ogimet").reset_index(drop=True)
    dir_proc(estacion).mkdir(parents=True, exist_ok=True)
    df.to_parquet(dir_proc(estacion) / "taf.parquet", index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--estacion", help="código OACI (por defecto, el aeropuerto activo)")
    p.add_argument("--inicio", type=int, default=2017)
    p.add_argument("--fin", type=int, default=date.today().year)
    p.add_argument("--tipos", default=",".join(TIPOS), help="FT (TAF largo), FC (TAF corto)")
    p.add_argument("--forzar", action="store_true")
    p.add_argument("--extraer", action="store_true", help="no descarga; solo une lo ya descargado")
    p.add_argument("--prueba", metavar="AAAA-MM", help="descarga solo ese mes para comprobar que funciona")
    p.add_argument("--diagnostico", action="store_true", help="muestra qué hay en los ficheros descargados")
    args = p.parse_args()
    usar(args.estacion)
    args.estacion = activa().icao

    if args.prueba:
        anio, mes = (int(x) for x in args.prueba.split("-"))
        for tipo in args.tipos.split(","):
            descargar_mes(args.estacion, tipo.strip(), anio, mes, forzar=True)
            time.sleep(PAUSA_S)
        f = DIR_TAF / "ultima_respuesta_vacia.txt"
        if f.exists():
            print(f"\nInicio de la última respuesta vacía:\n{diagnostico(f.read_text(encoding='utf-8'))}")
        return

    if args.diagnostico:
        for f in sorted(DIR_TAF.glob("*.txt"))[:3]:
            t = f.read_text(encoding="utf-8", errors="replace")
            print(f"\n== {f.name} ({len(t):,} caracteres, {len(extraer_tafs(t))} TAF)\n{diagnostico(t)}")
        return

    if not args.extraer:
        hoy = date.today()
        vacios = {t.strip(): 0 for t in args.tipos.split(",")}
        for anio in range(args.inicio, args.fin + 1):
            for mes in range(1, 13):
                if (anio, mes) > (hoy.year, hoy.month):
                    break
                for tipo in list(vacios):
                    n = descargar_mes(args.estacion, tipo, anio, mes, args.forzar)
                    if n is None:
                        continue
                    vacios[tipo] = vacios[tipo] + 1 if n == 0 else 0
                    time.sleep(PAUSA_S)
                if vacios and all(v >= MAX_MESES_VACIOS for v in vacios.values()):
                    f = DIR_TAF / "ultima_respuesta_vacia.txt"
                    raise SystemExit(
                        f"\n{MAX_MESES_VACIOS} meses seguidos sin TAF: paro para no perder tiempo.\n"
                        f"Lo que devolvió Ogimet (inicio):\n{diagnostico(f.read_text(encoding='utf-8'))}\n\n"
                        f"Envía el fichero {f} para revisarlo.")

    df = unir_tafs(args.estacion)
    if df.empty:
        raise SystemExit("No se ha encontrado ningún TAF. Revisa data/raw/taf/ (¿página de error de Ogimet?).")
    print(f"TAF únicos: {len(df):,} | {df['marca_ogimet'].min():%Y-%m-%d} -> {df['marca_ogimet'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
