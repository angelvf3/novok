"""Briefing operativo de NOVOK para un aeropuerto de la red.

Genera un JSON con:
- estado actual (último METAR) y controles de calidad de los datos,
- probabilidad de niebla y de bajo mínimos tipo CAT I a 1, 3, 6 y 12 h, junto a lo que dice el TAF en vigor,
- si hay niebla: tipo y hora estimada de disipación con intervalo del 80 %,
- viento en las pistas del aeropuerto ahora y previsto,
- un resumen en texto generado por reglas (trazable: cada frase sale de un dato del propio briefing).

Prototipo de apoyo a la decisión, complementario al TAF de AEMET. NO apto para decisiones de vuelo.

Uso:
    python -m novok.app.briefing --estacion LEAS                              # en directo
    python -m novok.app.briefing --estacion LEAS --repetir 2024-06-12T05:40   # esa fecha (UTC), datos locales
Por defecto escribe web/datos/<OACI>.json. Para toda la red: python -m novok.red briefing
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from novok import __version__
from novok.aeropuertos import activa, usar
from novok.app.fuentes import FuentesEnDirecto, FuentesRepeticion
from novok.config import RAIZ, dir_modelos
from novok.data.limpiar_metar import a_horario, limpiar
from novok.data.metar_texto import leer_varios
from novok.data.viento import viento_en_pista
from novok.episodios.disipacion import CUANTILES, completar_variables
from novok.episodios.episodios import clasificar, detectar
from novok.modelo.variables import matriz, variables_base
from novok.taf.parser import a_horario as taf_a_horario
from novok.taf.parser import parsear, probabilidad
from novok.verificacion.baselines import EVENTOS, preparar_observado

DIR_MODELOS = None  # solo para pruebas: si se fija, se usa en lugar de modelos/<OACI>
ZONA_LOCAL = "Europe/Madrid"
HORIZONTES_H = [1, 3, 6, 12]
SUCESOS = ["niebla", "bajo_minimos_cat1"]
EDAD_MAX_METAR_MIN = 90       # con un METAR más antiguo no se emite pronóstico
HORAS_MIN_ULTIMAS_24 = 18     # horas con observación necesarias en las últimas 24 h
AVISO = ("Prototipo educativo de apoyo a la decisión, complementario al TAF de AEMET. "
         "No apto para decisiones de vuelo ni para uso operativo.")


# ------------------------------------------------------------------ utilidades

def _iso(t) -> str | None:
    return None if t is None or pd.isna(t) else pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")


def _num(v, nd: int = 1):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def _hora(t) -> str:
    """'05:00Z (07:00 local)'."""
    t = pd.Timestamp(t)
    return f"{t.tz_convert('UTC'):%H:%M}Z ({t.tz_convert(ZONA_LOCAL):%H:%M} local)"


def _dir() -> Path:
    return Path(DIR_MODELOS) if DIR_MODELOS else dir_modelos()


def _cargar_modelos(version: str) -> tuple[dict, dict] | None:
    umbrales_f = _dir() / "umbrales.json"
    if not umbrales_f.exists():
        return None
    umbrales = json.loads(umbrales_f.read_text())
    modelos = {}
    for s in SUCESOS:
        for L in HORIZONTES_H:
            f = _dir() / f"novok_{version}_{s}_{L}h.txt"
            clave = f"{version}_{s}_{L}h"
            if not f.exists() or clave not in umbrales:
                return None
            modelos[(s, L)] = lgb.Booster(model_file=str(f))
    return modelos, umbrales


NIVELES = (0.10, 0.30)  # bajo < 10 % <= moderado < 30 % <= alto


def _nivel(p: float) -> str:
    """Nivel por probabilidad absoluta (lo que entiende quien lo lee)."""
    return "alto" if p >= NIVELES[1] else "moderado" if p >= NIVELES[0] else "bajo"


# ------------------------------------------------------------------ piezas

def _observado(textos: list[str], ahora: pd.Timestamp) -> tuple[pd.DataFrame | None, pd.Timestamp | None, str]:
    """METAR en texto -> rejilla horaria preparada (hasta la última hora completa + 13 h vacías)."""
    crudo = leer_varios(textos, ahora)
    if crudo.empty:
        return None, None, ""
    crudo = crudo[pd.to_datetime(crudo["valid"], utc=True) <= ahora]
    obs = limpiar(crudo)
    ultimo = obs["fecha_utc"].max()
    ultimo_texto = obs.loc[obs["fecha_utc"] == ultimo, "metar"].iloc[-1]
    # Última hora completa: si el último METAR es de las HH:30, la hora HH ya tiene sus dos partes
    i = (ultimo - pd.Timedelta(minutes=30)).floor("h")
    hor = a_horario(obs[obs["fecha_utc"] < i + pd.Timedelta(hours=1)])
    h = preparar_observado(hor, hasta=i + pd.Timedelta(hours=13))
    h.attrs["i"] = i
    return h, ultimo, ultimo_texto


def _nwp_util(nwp: pd.DataFrame | None, i: pd.Timestamp) -> pd.DataFrame | None:
    """El NWP solo sirve si cubre las próximas 13 h con humedad prevista de algún modelo."""
    if nwp is None or nwp.empty:
        return None
    futuro = nwp.reindex(pd.date_range(i + pd.Timedelta(hours=1), i + pd.Timedelta(hours=13), freq="1h"))
    cols_hr = [c for c in nwp.columns if c.endswith("_hr")]
    return nwp if cols_hr and futuro[cols_hr].notna().any(axis=1).all() else None


def _taf_vigente(tafs: list[tuple[pd.Timestamp, str]], ahora: pd.Timestamp, objetivos: list[pd.Timestamp]):
    """Último TAF emitido antes de ahora que cubra las horas objetivo -> condiciones por hora."""
    validos = []
    for marca, texto in tafs:
        t = parsear(texto, marca)
        if t is not None and t.emision <= ahora and t.estacion == activa().icao:
            t.texto = texto
            validos.append(t)
    if not validos:
        return None, {}
    taf = max(validos, key=lambda t: t.emision)
    hor = taf_a_horario(taf)
    for nombre, (uv, ut) in EVENTOS.items():
        hor[f"p_{nombre}"] = probabilidad(hor, uv, ut)
    hor = hor.set_index("hora_utc")
    por_hora = {}
    for t in objetivos:
        if t in hor.index:
            f = hor.loc[t]
            por_hora[t] = {"vis_predominante_m": _num(f["vis_prev_m"], 0),
                           "techo_predominante_ft": _num(f["techo_prev_ft"], 0),
                           "vis_peor_m": _num(f["vis_peor_m"], 0), "techo_peor_ft": _num(f["techo_peor_ft"], 0),
                           "p_niebla": _num(f["p_niebla"], 2), "p_bajo_minimos": _num(f["p_bajo_minimos_cat1"], 2)}
    return taf, por_hora


def _disipacion(h: pd.DataFrame, base: pd.DataFrame, i: pd.Timestamp) -> dict | None:
    if h.loc[i, "obs_niebla"] != 1:
        return None
    ep = detectar(h.loc[:i])
    if ep.empty:
        return None
    actual = ep.iloc[[-1]].reset_index(drop=True)
    tipo = clasificar(actual, h).iloc[0]
    inicio = actual.loc[0, "inicio"]
    transcurridas = int((i - inicio) / pd.Timedelta(hours=1)) + 1
    d = pd.DataFrame({"inicio": [inicio], "tipo": [tipo], "transcurridas_h": [transcurridas]},
                     index=pd.DatetimeIndex([i], name="hora_obs_utc"))
    x = completar_variables(d, base)
    out = {"inicio_utc": _iso(inicio), "transcurridas_h": transcurridas, "tipo": tipo, "disipacion": None}
    ficheros = {q: _dir() / f"disipacion_q{int(q * 100)}.txt" for q in CUANTILES}
    if all(f.exists() for f in ficheros.values()):
        qs = []
        for f in ficheros.values():
            m = lgb.Booster(model_file=str(f))
            qs.append(max(0.0, float(m.predict(x.reindex(columns=m.feature_name()))[0])))
        q10, q50, q90 = sorted(qs)
        # restantes = 0 -> la hora siguiente (i + 1) ya está despejada
        a_hora = lambda r: i + pd.Timedelta(hours=1) + pd.Timedelta(hours=round(r))  # noqa: E731
        out["disipacion"] = {"restantes_h_q10": _num(q10), "restantes_h_q50": _num(q50), "restantes_h_q90": _num(q90),
                             "despeja_utc_q10": _iso(a_hora(q10)), "despeja_utc_q50": _iso(a_hora(q50)),
                             "despeja_utc_q90": _iso(a_hora(q90))}
    return out


def _viento(h: pd.DataFrame, nwp: pd.DataFrame | None, i: pd.Timestamp) -> dict:
    ahora = viento_en_pista(h.loc[[i], ["viento_dir", "viento_kt", "racha_kt"]]).iloc[0]

    def fila(f, hora=None):
        return {"hora_utc": _iso(hora), "dir": _num(f["viento_dir"], 0), "kt": _num(f["viento_kt"], 0),
                "racha_kt": _num(f.get("racha_kt"), 0), "pista_favorable": f["pista_favorable"],
                "cruzado_kt": _num(f["cruzado_kt"]), "cruzado_racha_kt": _num(f["cruzado_racha_kt"]),
                **{f"cola_kt_{c}": _num(max(0.0, -f[f"cara_kt_{c}"]) if pd.notna(f[f"cara_kt_{c}"]) else None)
                   for c in activa().rumbos},
                "aviso_cruzado": bool(f["aviso_cruzado"])}

    previsto = []
    if nwp is not None:
        for m in ("ecmwf", "jma"):
            if f"nwp_{m}_viento_kt" in nwp and f"nwp_{m}_viento_dir" in nwp:
                horas = pd.date_range(i + pd.Timedelta(hours=1), i + pd.Timedelta(hours=12), freq="1h")
                v = nwp.reindex(horas)[[f"nwp_{m}_viento_dir", f"nwp_{m}_viento_kt"]]
                v.columns = ["viento_dir", "viento_kt"]
                if v.notna().all(axis=None):
                    v["racha_kt"] = np.nan
                    previsto = [dict(fila(r, t), fuente=m.upper()) for t, r in viento_en_pista(v).iterrows()]
                    break
    return {"ahora": fila(ahora, i), "previsto": previsto}


def _pct(p: float) -> str:
    return f"{p * 100:.0f}\u00a0%"


def _resumen(b: dict) -> list[str]:
    """Frases en lenguaje claro, cada una derivada de un campo del briefing."""
    if not b["calidad"]["pronostico_emitido"]:
        return ["Sin pronóstico: " + "; ".join(b["calidad"]["mensajes"]) + "."]
    a = b["ahora"]
    frases = []
    vis = "más de 10 km" if (a["vis_m"] or 0) >= 10_000 else f"{a['vis_m']:.0f} m"
    techo = "sin techo" if a["techo_ft"] is None else f"techo {a['techo_ft']:.0f} ft"
    frases.append(f"Ahora: visibilidad {vis}, {techo}, viento {a['viento_texto']}.")
    ep = b.get("episodio")
    if ep:
        texto = f"Niebla de {ep['tipo']} en curso desde las {_hora(ep['inicio_utc'])}."
        d = ep.get("disipacion")
        if d:
            texto += (f" Se espera que despeje hacia las {_hora(d['despeja_utc_q50'])}; "
                      f"80 % de probabilidad entre las {pd.Timestamp(d['despeja_utc_q10']):%H:%M}Z "
                      f"y las {pd.Timestamp(d['despeja_utc_q90']):%H:%M}Z.")
        frases.append(texto)
    altos = [p for p in b["pronostico"] if p["bajo_minimos"]["nivel"] == "alto"]
    if altos:
        hs = [f"+{p['horizonte_h']} h" for p in altos]
        hs = hs[0] if len(hs) == 1 else ", ".join(hs[:-1]) + " y " + hs[-1]
        frases.append(f"Riesgo alto de bajo mínimos tipo CAT I a {hs}.")
    else:
        mx = max(b["pronostico"], key=lambda p: p["bajo_minimos"]["p"])
        frases.append(f"Sin riesgo alto de bajo mínimos en 12 h (máximo {_pct(mx['bajo_minimos']['p'])} "
                      f"a {mx['horizonte_h']} h).")
    difs = [p for p in b["pronostico"] if p.get("taf") and p["taf"]["p_bajo_minimos"] is not None
            and abs(p["bajo_minimos"]["p"] - p["taf"]["p_bajo_minimos"]) >= 0.3]
    if b["calidad"]["taf"] and difs:
        p = difs[0]
        lado = "más" if p["bajo_minimos"]["p"] > p["taf"]["p_bajo_minimos"] else "menos"
        frases.append(f"Discrepancia con el TAF a las {_hora(p['hora_utc'])}: NOVOK ve {lado} riesgo "
                      f"({_pct(p['bajo_minimos']['p'])} frente a {_pct(p['taf']['p_bajo_minimos'])}).")
    v = b["viento_pista"]
    if v["previsto"]:
        mx = max(v["previsto"], key=lambda r: r["cruzado_kt"] or 0)
        frases.append(f"Viento previsto: cruzado máximo {mx['cruzado_kt']:.0f} kt a las {_hora(mx['hora_utc'])} "
                      f"(pista favorable {mx['pista_favorable']}).")
    if b["version_modelo"] == "v1":
        frases.append("Sin pronóstico numérico disponible: se usa NOVOK v1 (solo METAR), menos fiable a 6–12 h.")
    return frases


# ------------------------------------------------------------------ motor

def generar(fuentes, ahora: pd.Timestamp) -> dict:
    b = {"estacion": activa().icao, "aeropuerto": activa().a_dict(),
         "generado_utc": _iso(ahora), "version_software": __version__, "aviso": AVISO,
         "modo": "repeticion" if isinstance(fuentes, FuentesRepeticion) else "directo",
         "nota_repeticion": ("Repetición de una fecha pasada: los modelos finales se entrenaron con datos que "
                             "pueden incluir esa fecha; para medir el acierto real están las métricas del README.")
         if isinstance(fuentes, FuentesRepeticion) else None,
         "calidad": {"mensajes": [], "pronostico_emitido": False, "metar_ultimo_utc": None, "metar_edad_min": None,
                     "nwp": False, "taf": False}}
    cal = b["calidad"]

    try:
        textos = fuentes.metar()
    except Exception as e:  # noqa: BLE001
        cal["mensajes"].append(f"no se pudo obtener el METAR ({e.__class__.__name__})")
        b["resumen"] = _resumen(b)
        return b
    h, ultimo, ultimo_texto = _observado(textos, ahora)
    if h is None:
        cal["mensajes"].append("no hay METAR de las últimas 48 h")
        b["resumen"] = _resumen(b)
        return b
    i = h.attrs["i"]
    edad = (ahora - ultimo) / pd.Timedelta(minutes=1)
    cal.update(metar_ultimo_utc=_iso(ultimo), metar_edad_min=round(edad), hora_base_utc=_iso(i))
    horas_24 = int(h.loc[i - pd.Timedelta(hours=23): i, "hay_obs"].sum())
    if edad > EDAD_MAX_METAR_MIN:
        cal["mensajes"].append(f"el último METAR tiene {edad:.0f} min (máximo {EDAD_MAX_METAR_MIN})")
    if horas_24 < HORAS_MIN_ULTIMAS_24:
        cal["mensajes"].append(f"solo {horas_24} de las últimas 24 h tienen observación "
                               f"(mínimo {HORAS_MIN_ULTIMAS_24})")

    f = h.loc[i]
    vdir, vkt = _num(f["viento_dir"], 0), _num(f["viento_kt"], 0)
    b["ahora"] = {"hora_utc": _iso(i), "metar": ultimo_texto, "vis_m": _num(f["vis_m"], 0),
                  "rvr_min_m": _num(f["rvr_min_m"], 0), "techo_ft": _num(f["techo_ft"], 0),
                  "temp_c": _num(f["temp_c"]), "rocio_c": _num(f["rocio_c"]), "viento_dir": vdir, "viento_kt": vkt,
                  "racha_kt": _num(f["racha_kt"], 0),
                  "viento_texto": "en calma" if vkt == 0 else (f"{vdir:03.0f}° {vkt:.0f} kt" if vdir is not None
                                                               else f"variable {vkt:.0f} kt"),
                  "niebla": bool(f["obs_niebla"] == 1), "bajo_minimos": bool(f["obs_bajo_minimos_cat1"] == 1)}
    hist = h.loc[i - pd.Timedelta(hours=11): i]
    b["historial"] = [{"hora_utc": _iso(t), "vis_m": _num(r["vis_m"], 0), "techo_ft": _num(r["techo_ft"], 0),
                       "temp_c": _num(r["temp_c"]), "rocio_c": _num(r["rocio_c"]),
                       "niebla": bool(r["obs_niebla"] == 1), "bajo_minimos": bool(r["obs_bajo_minimos_cat1"] == 1)}
                      for t, r in hist.iterrows()]

    try:
        nwp = _nwp_util(fuentes.nwp(), i)
    except Exception as e:  # noqa: BLE001
        nwp = None
        cal["mensajes"].append(f"pronóstico numérico no disponible ({e.__class__.__name__})")
    cal["nwp"] = nwp is not None
    cargados = _cargar_modelos("v2") if nwp is not None else None
    version = "v2" if cargados else "v1"
    cargados = cargados or _cargar_modelos("v1")
    if cargados is None:
        cal["mensajes"].append("faltan los modelos entrenados en modelos/")
    b["version_modelo"] = version

    objetivos = [i + pd.Timedelta(hours=L + 1) for L in HORIZONTES_H]
    horas_taf = list(pd.date_range(i + pd.Timedelta(hours=1), i + pd.Timedelta(hours=max(HORIZONTES_H) + 1), freq="1h"))
    try:
        taf, taf_por_hora = _taf_vigente(fuentes.taf(), ahora, horas_taf)
    except Exception as e:  # noqa: BLE001
        taf, taf_por_hora = None, {}
        cal["mensajes"].append(f"TAF no disponible ({e.__class__.__name__})")
    cal["taf"] = taf is not None
    b["taf"] = None if taf is None else {"emision_utc": _iso(taf.emision), "enmienda": taf.enmienda,
                                         "valido_desde_utc": _iso(taf.inicio), "valido_hasta_utc": _iso(taf.fin),
                                         "texto": getattr(taf, "texto", None)}
    # TAF hora a hora para la línea de tiempo de la página
    b["taf_horas"] = [{"hora_utc": _iso(t), **taf_por_hora[t]} for t in horas_taf if t in taf_por_hora]

    base = variables_base(h)
    b["viento_pista"] = _viento(h, nwp, i)
    b["episodio"] = _disipacion(h, base, i)

    if cargados is None or edad > EDAD_MAX_METAR_MIN or horas_24 < HORAS_MIN_ULTIMAS_24:
        b["pronostico"] = []
        b["resumen"] = _resumen(b)
        return b

    modelos, umbrales = cargados
    pron = []
    for L, t in zip(HORIZONTES_H, objetivos, strict=True):
        p = {"horizonte_h": L, "hora_utc": _iso(t), "taf": taf_por_hora.get(t)}
        for s, clave in (("niebla", "niebla"), ("bajo_minimos_cat1", "bajo_minimos")):
            x, _ = matriz(h, base, s, L, nwp=nwp if version == "v2" else None)
            m = modelos[(s, L)]
            prob = float(m.predict(x.loc[[i]].reindex(columns=m.feature_name()))[0])
            u = umbrales[f"{version}_{s}_{L}h"]
            # alerta = supera el umbral de decisión del modelo (el que maximiza el CSI en validación)
            p[clave] = {"p": round(prob, 3), "umbral": u, "nivel": _nivel(prob), "alerta": prob >= u}
        pron.append(p)
    b["pronostico"] = pron
    cal["pronostico_emitido"] = True
    b["resumen"] = _resumen(b)
    return b


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repetir", metavar="FECHA_UTC", help="p. ej. 2024-06-12T05:40 (usa los datos locales)")
    p.add_argument("--estacion", help="código OACI (por defecto, el aeropuerto activo)")
    p.add_argument("--salida", help="por defecto web/datos/<OACI>.json")
    args = p.parse_args()
    usar(args.estacion)
    if args.repetir:
        ahora = pd.Timestamp(args.repetir, tz="UTC")
        fuentes = FuentesRepeticion(ahora)
    else:
        ahora = pd.Timestamp.now(tz="UTC")
        fuentes = FuentesEnDirecto(ahora)
    b = generar(fuentes, ahora)
    salida = Path(args.salida or RAIZ / "web" / "datos" / f"{activa().icao}.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(b, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n".join(b["resumen"]))
    print(f"\n-> {salida}")


if __name__ == "__main__":
    main()
