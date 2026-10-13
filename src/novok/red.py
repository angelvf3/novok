"""NOVOK para toda la red: preparar los aeropuertos y generar los briefings de todos a la vez.

Uso:
    python -m novok.red migrar                       # una vez: mueve los datos y modelos de LEAS a su carpeta
    python -m novok.red preparar LEXJ LEBB           # descarga, limpia, entrena (por defecto, toda la red)
    python -m novok.red preparar --sin-nwp LELO      # solo modelo v1 (sin pronóstico numérico)
    python -m novok.red estado                       # qué tiene cada aeropuerto (datos, modelos, cobertura)
    python -m novok.red elegir                       # v1 o v2 por suceso y antelación, según la validación
    python -m novok.red terreno                      # relieve de cada aeropuerto para el visor 3D (una vez)
    python -m novok.red briefing                     # briefing en directo de toda la red -> web/datos/
    python -m novok.red briefing --repetir 2025-06-30T04:40

Preparar un aeropuerto con pronóstico numérico tarda unos 15-20 min (sobre todo el entrenamiento).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import date
from pathlib import Path

import pandas as pd

from novok.aeropuertos import RED, catalogo, estacion
from novok.config import ANIO_INICIO_FIABLE, DATA_PROCESSED, DATA_RAW, RAIZ, dir_modelos, dir_proc

DIR_WEB_DATOS = RAIZ / "web" / "datos"


# ------------------------------------------------------------------ categoría de vuelo

def categoria(vis_m: float | None, techo_ft: float | None) -> str | None:
    """Categoría de vuelo con los criterios de la FAA (VFR, MVFR, IFR, LIFR)."""
    if vis_m is None:
        return None
    t = float("inf") if techo_ft is None else techo_ft
    if t < 500 or vis_m < 1600:
        return "LIFR"
    if t < 1000 or vis_m < 4800:
        return "IFR"
    if t <= 3000 or vis_m <= 8000:
        return "MVFR"
    return "VFR"


# ------------------------------------------------------------------ migrar LEAS

def migrar() -> None:
    """Hasta F5 todo era de LEAS y estaba en la raíz de cada carpeta; ahora cada aeropuerto tiene la suya."""
    movidos = 0
    for origen, destino, patrones in [
        (DATA_PROCESSED, DATA_PROCESSED / "LEAS", ["*.parquet"]),
        (RAIZ / "modelos", RAIZ / "modelos" / "LEAS", ["*.txt", "umbrales.json"]),
        (DATA_RAW / "nwp", DATA_RAW / "nwp" / "LEAS", ["ecmwf_*.parquet", "jma_*.parquet"]),
    ]:
        for patron in patrones:
            for f in origen.glob(patron):
                destino.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), destino / f.name)
                movidos += 1
                print(f"  {f.relative_to(RAIZ)} -> {(destino / f.name).relative_to(RAIZ)}")
    print(f"{movidos} ficheros movidos." if movidos else "Nada que mover: ya estaba migrado.")


# ------------------------------------------------------------------ preparar

def cobertura(icao: str) -> dict:
    """Qué parte de las horas tiene METAR desde 2017, en total y de madrugada (cuando más niebla hay)."""
    f = dir_proc(icao) / "horario.parquet"
    if not f.exists():
        return {}
    h = pd.read_parquet(f, columns=["fecha_utc"])
    h = h[h["fecha_utc"].dt.year >= ANIO_INICIO_FIABLE]
    if h.empty:
        return {"total": 0.0, "madrugada": 0.0}
    rejilla = pd.date_range(h["fecha_utc"].min().floor("h"), h["fecha_utc"].max().floor("h"), freq="1h")
    con = pd.Index(h["fecha_utc"].dt.floor("h").unique())
    noche = rejilla[rejilla.hour < 7]
    return {"total": round(len(con) / len(rejilla), 3),
            "madrugada": round(con.isin(noche).sum() / max(1, len(noche)), 3)}


def preparar(icao: str, desde: int, con_nwp: bool) -> None:
    from novok.data import descargar_metar, descargar_nwp, limpiar_metar
    from novok.episodios import disipacion
    from novok.modelo import entrenar, entrenar_nwp

    t0 = time.time()
    with estacion(icao) as a:
        print(f"\n===== {icao} · {a.nombre} =====")
        print("1/4 METAR")
        for anio in range(desde, date.today().year + 1):
            try:  # el año en curso se vuelve a descargar para tener lo último
                descargar_metar.descargar_anio(icao, anio, DATA_RAW, forzar=anio == date.today().year)
            except Exception as e:  # noqa: BLE001 - un año que falla no para el resto
                print(f"   [{anio}] no se pudo descargar ({e.__class__.__name__}); se sigue con lo que haya")
            time.sleep(descargar_metar.PAUSA_ENTRE_PETICIONES_S)
        limpiar_metar.procesar(icao)
        c = cobertura(icao)
        print(f"   cobertura desde {ANIO_INICIO_FIABLE}: {c['total']:.0%} de las horas, "
              f"{c['madrugada']:.0%} de madrugada (00-06Z)")
        if c["madrugada"] < 0.6:
            print("   ⚠ Pocas observaciones de madrugada (¿aeropuerto cerrado de noche?): "
                  "la niebla nocturna estará infrarrepresentada y el modelo será menos fiable.")

        argv, sys.argv = sys.argv, [sys.argv[0]]  # los main() leen la línea de órdenes
        try:
            if con_nwp:
                print("2/4 Pronóstico numérico")
                try:
                    descargar_nwp.descargar()
                except SystemExit as e:
                    print(f"   sin pronóstico numérico ({e}); se entrena solo el modelo v1")
                    con_nwp = False
            print("3/4 Modelos" + (" v2 y v1" if con_nwp else " v1"))
            (entrenar_nwp.main if con_nwp else entrenar.main)()
            print("4/4 Disipación")
            disipacion.main()
        finally:
            sys.argv = argv
    print(f"===== {icao} listo en {(time.time() - t0) / 60:.0f} min =====")


def elegir(icao: str) -> dict | None:
    """Escribe modelos/<OACI>/eleccion.json a partir de reports/.../modelo_v2_resultados.csv, sin reentrenar."""
    from novok.config import dir_informes
    from novok.modelo.entrenar_nwp import guardar_eleccion

    f = dir_informes(icao) / "modelo_v2_resultados.csv"
    if not f.exists() or not dir_modelos(icao).exists():
        return None
    r = pd.read_csv(f)
    r = r[r["periodo"] == "2024-2025"]
    salida = {}
    with estacion(icao):
        for (suceso, L), g in r.groupby(["suceso", "horizonte_h"]):
            bss = g.set_index("metodo")["brier_skill_vs_clima"]
            salida[f"{suceso}_{L}h"] = guardar_eleccion(f"{suceso}_{L}h", bss["novok_v1"], bss["novok_v2"])
    return salida


def estado() -> None:
    print(f"{'OACI':<5} {'aeropuerto':<14} {'METAR':>6} {'madrug.':>8} {'NWP':>4} {'modelos':>8}")
    for icao, a in catalogo().items():
        c = cobertura(icao)
        mod = dir_modelos(icao)
        version = "v2" if any(mod.glob("novok_v2_*.txt")) else "v1" if any(mod.glob("novok_v1_*.txt")) else "—"
        print(f"{icao:<5} {a.nombre:<14} {c.get('total', 0):>6.0%} {c.get('madrugada', 0):>8.0%} "
              f"{'sí' if (dir_proc(icao) / 'nwp.parquet').exists() else 'no':>4} {version:>8}")


# ------------------------------------------------------------------ relieve para el visor 3D

DIR_TERRENO = RAIZ / "web" / "terreno"
URL_TERRENO = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"  # AWS Terrain Tiles


def _tesela(z: int, x: int, y: int):
    """Altura (m) de una tesela 'terrarium' de 256x256 píxeles."""
    import io

    import numpy as np
    import requests
    from PIL import Image
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    r = requests.get(URL_TERRENO.format(z=z, x=x, y=y), timeout=60)
    r.raise_for_status()
    p = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=float)
    return p[..., 0] * 256 + p[..., 1] + p[..., 2] / 256 - 32768


def rejilla_relieve(lat0: float, lon0: float, alcance_m: float = 12000, n: int = 321, z: int = 12, tesela=_tesela):
    """Alturas en una rejilla n x n de 2*alcance metros centrada en el aeropuerto (fila 0 = norte)."""
    import math

    import numpy as np
    dlat = alcance_m / 111320
    dlon = alcance_m / (111320 * math.cos(math.radians(lat0)))
    lats = lat0 + dlat - 2 * dlat * np.arange(n) / (n - 1)
    lons = lon0 - dlon + 2 * dlon * np.arange(n) / (n - 1)
    escala = 2 ** z * 256
    px = (lons + 180) / 360 * escala                                            # píxel global (Web Mercator)
    py = (1 - np.log(np.tan(np.radians(lats)) + 1 / np.cos(np.radians(lats))) / math.pi) / 2 * escala
    x0, x1 = int(px.min() // 256), int(px.max() // 256) + 1
    y0, y1 = int(py.min() // 256), int(py.max() // 256) + 1
    mosaico = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            mosaico[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = tesela(z, tx, ty)
    gx, gy = np.meshgrid(px - x0 * 256 - .5, py - y0 * 256 - .5)                # bilineal sobre el mosaico
    i, j = np.clip(gx.astype(int), 0, mosaico.shape[1] - 2), np.clip(gy.astype(int), 0, mosaico.shape[0] - 2)
    u, v = np.clip(gx - i, 0, 1), np.clip(gy - j, 0, 1)
    m = mosaico
    return (m[j, i] * (1 - u) * (1 - v) + m[j, i + 1] * u * (1 - v) + m[j + 1, i] * (1 - u) * v
            + m[j + 1, i + 1] * u * v)


def guardar_relieve(alt, destino: Path) -> None:
    """PNG con codificación 'terrarium' (la que lee el visor): altura = R*256 + G + B/256 - 32768."""
    import numpy as np
    from PIL import Image
    v = np.clip(alt, -500, 9000) + 32768
    r = np.floor(v / 256)
    g = np.floor(v - r * 256)
    b = np.round((v - r * 256 - g) * 256).clip(0, 255)
    Image.fromarray(np.dstack([r, g, b]).astype("uint8"), "RGB").save(destino, optimize=True)


def terreno(icao: str, alcance_m: float = 12000, n: int = 321) -> None:
    a = catalogo()[icao]
    alt = rejilla_relieve(a.lat, a.lon, alcance_m, n)
    DIR_TERRENO.mkdir(parents=True, exist_ok=True)
    guardar_relieve(alt, DIR_TERRENO / f"{icao}.png")
    (DIR_TERRENO / f"{icao}.json").write_text(json.dumps(
        {"alcance_m": alcance_m, "n": n, "fuente": "AWS Terrain Tiles (datos abiertos)"}), encoding="utf-8")
    print(f"{icao}: relieve {n}x{n}, de {alt.min():.0f} a {alt.max():.0f} m -> web/terreno/{icao}.png")


# ------------------------------------------------------------------ briefing de la red

def resumen_red(b: dict) -> dict:
    """Lo que necesita el mapa de la red de cada aeropuerto (el detalle está en <OACI>.json)."""
    a, pr = b.get("ahora") or {}, b.get("pronostico") or []
    peor = max(pr, key=lambda p: p["bajo_minimos"]["p"], default=None)
    ep = b.get("episodio") or {}
    return {
        "icao": b["estacion"], "nombre": b["aeropuerto"]["nombre"],
        "lat": b["aeropuerto"]["lat"], "lon": b["aeropuerto"]["lon"],
        "emitido": b["calidad"]["pronostico_emitido"], "version_modelo": b.get("version_modelo"),
        "metar_ultimo_utc": b["calidad"].get("metar_ultimo_utc"),
        "categoria": categoria(a.get("vis_m"), a.get("techo_ft")),
        "vis_m": a.get("vis_m"), "techo_ft": a.get("techo_ft"), "viento": a.get("viento_texto"),
        "niebla": a.get("niebla"), "bajo_minimos": a.get("bajo_minimos"),
        "riesgo": None if peor is None else {
            "p": peor["bajo_minimos"]["p"], "nivel": peor["bajo_minimos"]["nivel"], "horizonte_h": peor["horizonte_h"],
            "alerta": any(p["bajo_minimos"]["alerta"] or p["niebla"]["alerta"] for p in pr)},
        "despeja_utc": (ep.get("disipacion") or {}).get("despeja_utc_q50"),
        "resumen": next((f for f in b.get("resumen", [])
                         if f.startswith(("Riesgo", "Sin riesgo", "Sin pronóstico"))), None),
    }


def briefing_red(repetir: str | None, salida: Path, solo: list[str] | None = None) -> dict:
    from novok.app.briefing import generar
    from novok.app.fuentes import FuentesEnDirecto, FuentesRepeticion

    ahora = pd.Timestamp(repetir, tz="UTC") if repetir else pd.Timestamp.now(tz="UTC")
    salida.mkdir(parents=True, exist_ok=True)
    filas = []
    for icao, a in catalogo().items():
        if solo and icao not in solo:
            continue
        with estacion(icao):
            try:
                fuentes = FuentesRepeticion(ahora) if repetir else FuentesEnDirecto(ahora)
                b = generar(fuentes, ahora)
            except Exception as e:  # noqa: BLE001 - un aeropuerto caído no tumba la red
                print(f"{icao}: ERROR {e.__class__.__name__}: {e}")
                traceback.print_exc(limit=1)
                filas.append({"icao": icao, "nombre": a.nombre, "lat": a.lat, "lon": a.lon, "emitido": False,
                              "error": f"{e.__class__.__name__}"})
                continue
        (salida / f"{icao}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        filas.append(resumen_red(b))
        r = filas[-1]
        print(f"{icao}: {r['categoria'] or '—':<4} {(r['resumen'] or '')[:90]}")
    red = {"generado_utc": ahora.strftime("%Y-%m-%dT%H:%MZ"), "modo": "repeticion" if repetir else "directo",
           "aeropuertos": filas}
    (salida / "red.json").write_text(json.dumps(red, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"-> {salida}/red.json ({len(filas)} aeropuertos)")
    return red


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="orden", required=True)
    sub.add_parser("migrar")
    sub.add_parser("estado")
    pt = sub.add_parser("terreno", help="genera el relieve del visor 3D (una vez; se sube al repo)")
    pt.add_argument("codigos", nargs="*")
    sub.add_parser("elegir", help="elige v1 o v2 por suceso y antelación con la validación ya hecha")
    pp = sub.add_parser("preparar")
    pp.add_argument("codigos", nargs="*")
    pp.add_argument("--desde", type=int, default=2010, help="primer año de METAR (por defecto 2010)")
    pp.add_argument("--sin-nwp", action="store_true", help="no descarga pronóstico numérico (solo modelo v1)")
    pb = sub.add_parser("briefing")
    pb.add_argument("--repetir", metavar="FECHA_UTC")
    pb.add_argument("--salida", default=str(DIR_WEB_DATOS))
    pb.add_argument("codigos", nargs="*")
    args = p.parse_args()

    if args.orden == "migrar":
        migrar()
    elif args.orden == "estado":
        estado()
    elif args.orden == "terreno":
        for icao in [c.upper() for c in args.codigos] or list(catalogo()):
            try:
                terreno(icao)
            except Exception as e:  # noqa: BLE001
                print(f"{icao}: no se pudo generar el relieve ({e.__class__.__name__}: {e})")
    elif args.orden == "elegir":
        for icao in catalogo():
            e = elegir(icao)
            if e is None:
                print(f"{icao}: sin informe de validación v2 (se usará v2 en todas las antelaciones)")
            else:
                print(f"{icao}: " + "  ".join(f"{k}={v}" for k, v in e.items()))
    elif args.orden == "preparar":
        codigos = [c.upper() for c in args.codigos] or list(RED)
        faltan = [c for c in codigos if c not in catalogo()]
        if faltan:
            raise SystemExit(f"Sin datos de {', '.join(faltan)}. "
                             "Ejecuta antes: python -m novok.aeropuertos --descargar")
        for c in codigos:
            preparar(c, args.desde, not args.sin_nwp)
        estado()
    else:
        briefing_red(args.repetir, Path(args.salida), [c.upper() for c in args.codigos] or None)


if __name__ == "__main__":
    main()
