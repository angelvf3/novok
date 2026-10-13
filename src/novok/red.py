"""NOVOK para toda la red: preparar los aeropuertos y generar los briefings de todos a la vez.

Uso:
    python -m novok.red migrar                       # una vez: mueve los datos y modelos de LEAS a su carpeta
    python -m novok.red preparar LEXJ LEBB           # descarga, limpia, entrena (por defecto, toda la red)
    python -m novok.red preparar --sin-nwp LELO      # solo modelo v1 (sin pronóstico numérico)
    python -m novok.red estado                       # qué tiene cada aeropuerto (datos, modelos, cobertura)
    python -m novok.red elegir                       # v1 o v2 por suceso y antelación, según la validación
    python -m novok.red terreno                      # relieve y ortofoto de cada aeropuerto, visor 3D (una vez)
    python -m novok.red mapa                         # mapa de relieve de España para el mapa 2D (una vez)
    python -m novok.red briefing                     # briefing en directo de toda la red -> web/datos/
    python -m novok.red briefing --repetir 2025-06-30T04:40
    python -m novok.red independiente --actualizar   # verificación en una temporada no usada (2026) y valor en EUR

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


# ------------------------------------------------------------------ cartografía (una vez)

def terreno(icao: str) -> None:
    from novok import cartografia
    a = catalogo()[icao]
    cartografia.terreno(icao, a.lat, a.lon)


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
        # serie para calcular el riesgo a la hora de llegada de cada vuelo (web/vuelos.js)
        "serie": [{"hora_utc": p["hora_utc"], "p_bajo_minimos": p["bajo_minimos"]["p"], "p_niebla": p["niebla"]["p"]}
                  for p in pr],
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


# ------------------------------------------------------------------ verificación independiente (F8)

def actualizar(icao: str) -> None:
    """Vuelve a descargar el año en curso (METAR y pronóstico numérico) sin reentrenar nada."""
    from novok.data import descargar_metar, descargar_nwp, limpiar_metar

    hoy = date.today()
    with estacion(icao):
        descargar_metar.descargar_anio(icao, hoy.year, DATA_RAW, forzar=True)
        limpiar_metar.procesar(icao)
        for f in (descargar_nwp.DIR_NWP / icao).glob(f"*_{hoy.year}.parquet"):
            f.unlink()
        try:
            descargar_nwp.descargar()
        except SystemExit as e:
            print(f"   sin pronóstico numérico ({e})")


def independiente_red(codigos: list[str], corte: str, con_actualizar: bool) -> pd.DataFrame:
    """Verificación independiente de cada aeropuerto y tabla resumen de la red (bajo mínimos CAT I)."""
    import numpy as np

    from novok.verificacion import independiente as ind

    np.seterr(all="ignore")
    t_corte = pd.Timestamp(corte, tz="UTC")
    filas = []
    for icao in codigos:
        print(f"\n===== {icao} =====")
        try:
            if con_actualizar:
                actualizar(icao)
            with estacion(icao):
                res = ind.verificar(t_corte, ind.cargar_nwp())
                ind.guardar(res)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - un aeropuerto que falla no para el resto
            print(f"{icao}: no se pudo verificar ({e})")
            continue
        m = res["metricas"]
        ep = ind.resumen_episodios(res["episodios"])
        for L in (3, 6, 12):
            d = m[(m["suceso"] == "bajo_minimos_cat1") & (m["horizonte_h"] == L)].set_index("metodo")
            if "novok" not in d.index:
                continue
            e = ep[(ep["suceso"] == "bajo_minimos_cat1") & (ep["horizonte_h"] == L)] if not ep.empty else ep
            def dato(met: str, col: str, d: pd.DataFrame = d) -> float | None:
                return float(d.loc[met, col]) if met in d.index and col in d and pd.notna(d.loc[met, col]) else None

            filas.append({
                "aeropuerto": icao, "horizonte_h": L, "horas": int(d.loc["novok", "n_horas"]),
                "desde": d.loc["novok", "desde"], "hasta": d.loc["novok", "hasta"],
                "modelo": d.loc["novok", "modelo_elegido"],
                "csi_novok": dato("novok", "CSI"), "csi_taf": dato("taf_predominante", "CSI"),
                "csi_persistencia": dato("persistencia", "CSI"),
                "falsas_alarmas_h_mes_novok": dato("novok", "falsas_alarmas_h_mes"),
                "falsas_alarmas_h_mes_taf": dato("taf_predominante", "falsas_alarmas_h_mes"),
                "bss_novok": dato("novok", "brier_skill_vs_clima"), "bss_taf": dato("taf", "brier_skill_vs_clima"),
                "episodios": int(e["episodios"].iloc[0]) if len(e) else 0,
                "avisados_novok": float(e["avisados_novok"].iloc[0]) if len(e) else None,
                "avisados_taf": float(e["avisados_taf"].iloc[0]) if len(e) and "avisados_taf" in e else None,
                "ahorro_100_novok": dato("novok", "ahorro_eur_100_llegadas"),
                "ahorro_100_taf": dato("taf", "ahorro_eur_100_llegadas"),
                "ahorro_100_persistencia": dato("persistencia", "ahorro_eur_100_llegadas"),
            })
    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        from novok.config import REPORTS

        tabla.round(3).to_csv(REPORTS / "red_independiente.csv", index=False)
        print("\nBajo mínimos CAT I en la temporada de prueba (CSI y ahorro por 100 llegadas con config/costes.json)")
        for _, r in tabla.iterrows():
            taf = f"{r['csi_taf']:.2f}" if pd.notna(r["csi_taf"]) else " -- "
            print(f"  {r['aeropuerto']} {r['horizonte_h']:>2} h  CSI NOVOK {r['csi_novok']:.2f}  TAF {taf}  "
                  f"pers {r['csi_persistencia']:.2f}  | avisos de arranque {r['avisados_novok'] or 0:.0%}"
                  f"  | ahorro NOVOK {r['ahorro_100_novok']:,.0f} EUR/100 llegadas".replace(",", "."))
        print(f"-> {REPORTS / 'red_independiente.csv'}")
    return tabla


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="orden", required=True)
    sub.add_parser("migrar")
    sub.add_parser("estado")
    pt = sub.add_parser("terreno", help="relieve y ortofoto de cada aeropuerto para el visor 3D (una vez)")
    pt.add_argument("codigos", nargs="*")
    sub.add_parser("mapa", help="teselas de relieve de España para el mapa 2D (una vez)")
    sub.add_parser("elegir", help="elige v1 o v2 por suceso y antelación con la validación ya hecha")
    pp = sub.add_parser("preparar")
    pp.add_argument("codigos", nargs="*")
    pp.add_argument("--desde", type=int, default=2010, help="primer año de METAR (por defecto 2010)")
    pp.add_argument("--sin-nwp", action="store_true", help="no descarga pronóstico numérico (solo modelo v1)")
    pb = sub.add_parser("briefing")
    pb.add_argument("--repetir", metavar="FECHA_UTC")
    pb.add_argument("--salida", default=str(DIR_WEB_DATOS))
    pb.add_argument("codigos", nargs="*")
    pi = sub.add_parser("independiente", help="verificación en una temporada no usada y valor económico")
    pi.add_argument("codigos", nargs="*")
    pi.add_argument("--corte", default="2026-01-01", help="primer día de la temporada de prueba")
    pi.add_argument("--actualizar", action="store_true", help="descarga antes los METAR y NWP del año en curso")
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
    elif args.orden == "mapa":
        from novok import cartografia
        print("Mapa de relieve de España (AWS Terrain Tiles)...")
        print(f"-> {cartografia.mapa_relieve()} teselas en web/mapa/relieve/")
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
    elif args.orden == "independiente":
        independiente_red([c.upper() for c in args.codigos] or list(RED), args.corte, args.actualizar)
    else:
        briefing_red(args.repetir, Path(args.salida), [c.upper() for c in args.codigos] or None)


if __name__ == "__main__":
    main()
