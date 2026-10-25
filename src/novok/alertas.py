"""F10 · Alertas: avisa por Telegram o por correo cuando un aeropuerto de la red entra en riesgo alto.

Nadie en un centro de operaciones tiene una web abierta todo el día: NOVOK tiene que avisar. Después de cada
briefing (cada 30 min) se revisa red.json y se manda un único mensaje con lo que haya cambiado:

- AVISO: un aeropuerto supera el umbral de bajo mínimos (por defecto 30 %) en las próximas 12 h.
- SUBE: un aviso ya enviado empeora claramente (+20 puntos).
- SIGUE: recordatorio si el aviso sigue activo pasadas unas horas.
- FIN: el riesgo baja del umbral de cierre y no hay bajo mínimos ahora.

Con --vuelos se añaden los vuelos en el aire hacia cada aeropuerto en aviso, su riesgo a la llegada y el coste
esperado de los desvíos (mismo cálculo que web/vuelos.js), y con "aerolinea" en la configuración, solo los de esa
aerolínea. Cada aviso se manda una vez: el estado se guarda en un fichero entre ejecuciones.

Configuración en config/alertas.json; las claves, en variables de entorno (nunca en el repositorio):
    NOVOK_TELEGRAM_TOKEN, NOVOK_TELEGRAM_CHAT                        (Telegram)
    NOVOK_SMTP_SERVIDOR, NOVOK_SMTP_PUERTO, NOVOK_SMTP_USUARIO,
    NOVOK_SMTP_CLAVE, NOVOK_CORREO_PARA                               (correo)
Sin ninguna de ellas, los mensajes solo se escriben en pantalla.

Uso:
    python -m novok.alertas                       # revisa web/datos/red.json y avisa si hace falta
    python -m novok.alertas --vuelos              # ... con los vuelos en el aire
    python -m novok.alertas --probar              # manda un mensaje de prueba por los canales configurados
    python -m novok.alertas --vigilar --vuelos    # en tu ordenador: briefing + alertas cada 30 min
"""

from __future__ import annotations

import argparse
import json
import math
import os
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from novok.config import RAIZ

FICH_CONFIG = RAIZ / "config" / "alertas.json"
FICH_ESTADO = RAIZ / "data" / "alertas_estado.json"
RED = RAIZ / "web" / "datos" / "red.json"
ZONA = ZoneInfo("Europe/Madrid")
CONFIG_DEFECTO = {
    "umbral": 0.30, "umbral_fin": 0.15, "subida": 0.20, "recordar_cada_h": 3,
    "aeropuertos": [], "aerolinea": None, "web": "https://angelvf3.github.io/novok/",
}
CABECERA = {"User-Agent": "novok (https://github.com/angelvf3/novok)"}


def config() -> dict:
    c = dict(CONFIG_DEFECTO)
    if FICH_CONFIG.exists():
        c.update({k: v for k, v in json.loads(FICH_CONFIG.read_text(encoding="utf-8")).items() if k != "comentario"})
    return c


def hora_local(iso: str | None) -> str:
    if not iso:
        return "—"
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(ZONA).strftime("%H:%M")


def pct(p: float) -> str:
    return f"{round(p * 100)} %"


def eur(x: float) -> str:
    v = round(x / 100) * 100 if x >= 1000 else round(x / 10) * 10
    return f"{v:,.0f} €".replace(",", ".")


# ------------------------------------------------------------------ vuelos (igual que web/vuelos.js)

def distancia_km(lat1, lon1, lat2, lon2) -> float:
    r = math.pi / 180
    a = (math.sin((lat2 - lat1) * r / 2) ** 2
         + math.cos(lat1 * r) * math.cos(lat2 * r) * math.sin((lon2 - lon1) * r / 2) ** 2)
    return 2 * 6371 * math.asin(min(1, math.sqrt(a)))


def rumbo(lat1, lon1, lat2, lon2) -> float:
    r = math.pi / 180
    y = math.sin((lon2 - lon1) * r) * math.cos(lat2 * r)
    x = math.cos(lat1 * r) * math.sin(lat2 * r) - math.sin(lat1 * r) * math.cos(lat2 * r) * math.cos((lon2 - lon1) * r)
    return (math.atan2(y, x) / r + 360) % 360


def riesgo_en_hora(aero: dict, t: float, ahora: float) -> float | None:
    """Probabilidad de bajo mínimos a la hora t (s), interpolando la serie de NOVOK y lo observado ahora."""
    puntos = sorted((datetime.fromisoformat(s["hora_utc"].replace("Z", "+00:00")).timestamp(), s["p_bajo_minimos"])
                    for s in aero.get("serie") or [])
    if not puntos:
        return None
    actual = 0.9 if aero.get("bajo_minimos") else 0.4 if aero.get("niebla") else None
    if actual is not None:
        puntos = sorted([(ahora, actual), *puntos])
    if t <= puntos[0][0]:
        return puntos[0][1]
    for (t0, p0), (t1, p1) in zip(puntos, puntos[1:], strict=False):
        if t <= t1:
            return p0 + (p1 - p0) * (t - t0) / (t1 - t0)
    return puntos[-1][1]


def posiciones(cfg_vuelos: dict) -> list[dict]:
    """Aviones del proveedor principal o, si falla, de los de respaldo (formato readsb)."""
    lat, lon = cfg_vuelos["centro"]
    # sin CORS en Python: se pide directamente al principal y, si falla, a adsb.fi (respaldo de pruebas)
    urls = [cfg_vuelos["posiciones"], "https://opendata.adsb.fi/api/v3/lat/{lat}/lon/{lon}/dist/{radio}"]
    for u in urls:
        try:
            r = requests.get(u.format(lat=lat, lon=lon, radio=min(cfg_vuelos["radio_nm"], 250)), headers=CABECERA,
                             timeout=10)
            if r.ok:
                datos = r.json()
                return [a for a in datos.get("ac") or datos.get("aircraft") or [] if a.get("lat") is not None]
        except (requests.RequestException, ValueError):
            continue
    return []


def rutas(cfg_vuelos: dict, aviones: list[dict]) -> dict:
    planes = [{"callsign": a["flight"].strip(), "lat": a["lat"], "lng": a["lon"]} for a in aviones
              if (a.get("flight") or "").strip() and a.get("alt_baro") != "ground"]
    salida = {}
    for k in range(0, len(planes), 100):
        try:
            r = requests.post(cfg_vuelos["rutas"], json={"planes": planes[k:k + 100]}, headers=CABECERA, timeout=10)
            for x in r.json() if r.ok else []:
                codigos = [a.get("icao") for a in x.get("_airports") or []] or (x.get("airport_codes") or "").split("-")
                codigos = [c for c in codigos if c and len(c) == 4]
                if len(codigos) >= 2:
                    salida[x["callsign"].strip()] = {"origen": codigos[0], "destino": codigos[-1]}
        except (requests.RequestException, ValueError):
            break
    return salida


def coste_desvio(tipo: str | None, dest: dict, alt: dict, c: dict) -> float:
    t = tipo if tipo in c["consumo_kg_h"] else "_otros"
    km = distancia_km(dest["lat"], dest["lon"], alt["lat"], alt["lon"])
    tramo_h = (km / c["velocidad_desvio_kmh"] * 60 + c["aproximacion_min"]) / 60
    horas = (c["espera_min"] / 60 + tramo_h) + tramo_h
    pasajeros = round(c["plazas"].get(t, c["plazas"]["_otros"]) * c["ocupacion"])
    return (horas * c["consumo_kg_h"][t] * c["combustible_eur_kg"] + horas * c["tripulacion_eur_h"]
            + c["handling_desvio_eur"] + c["tasas_alternativo_eur"]
            + pasajeros * (c["traslado_pasajero_eur"] + c["atencion_pasajero_eur"]))


def vuelos_hacia(red: dict, aviones: list[dict], rutas_: dict, costes: dict | None, ahora: float) -> dict:
    """Por aeropuerto: vuelos en el aire que van allí, con su riesgo a la llegada y el coste esperado."""
    aeros = {a["icao"]: a for a in red["aeropuertos"] if a.get("lat") is not None}
    por_aero: dict[str, list] = {}
    for av in aviones:
        if av.get("alt_baro") == "ground" or not av.get("gs") or av["gs"] < 60:
            continue
        indic = (av.get("flight") or "").strip()
        destino = rutas_.get(indic, {}).get("destino")
        if destino not in aeros:                         # sin ruta: bajo, descendiendo y apuntando a un aeropuerto
            destino = None
            alt_ft, vs, trk = av.get("alt_baro"), av.get("baro_rate") or 0, av.get("track")
            if isinstance(alt_ft, (int, float)) and alt_ft <= 12000 and vs <= -300 and trk is not None:
                cerca = [(distancia_km(av["lat"], av["lon"], a["lat"], a["lon"]), i) for i, a in aeros.items()]
                for d, i in sorted(cerca):
                    dif = abs((rumbo(av["lat"], av["lon"], aeros[i]["lat"], aeros[i]["lon"]) - trk + 540) % 360 - 180)
                    if d <= 90 and dif < 35:
                        destino = i
                        break
        if not destino:
            continue
        dest = aeros[destino]
        km = distancia_km(av["lat"], av["lon"], dest["lat"], dest["lon"])
        minutos = km / (av["gs"] * 1.852) * 60 + (8 if km > 25 else 2)
        if minutos > 16 * 60:
            continue
        llegada = ahora + minutos * 60
        p = riesgo_en_hora(dest, llegada, ahora)
        coste = None
        if p is not None and costes and p >= 0.10:
            alternos = []
            for i, a in aeros.items():
                d = distancia_km(dest["lat"], dest["lon"], a["lat"], a["lon"])
                if i != destino and a.get("serie") and d <= 300:
                    alternos.append(((riesgo_en_hora(a, llegada + 1800, ahora) or 1) + d / 3000, i))
            if alternos:
                coste = p * coste_desvio(av.get("t"), dest, aeros[min(alternos)[1]], costes)
        operador = indic[:3] if len(indic) > 3 and indic[:3].isalpha() and indic[3].isdigit() else None
        por_aero.setdefault(destino, []).append({"indicativo": indic or av.get("hex"), "operador": operador,
                                                 "llegada": llegada, "p": p, "coste": coste})
    return por_aero


# ------------------------------------------------------------------ decidir qué avisar

def eventos(red: dict, estado: dict, c: dict, ahora: float) -> list[dict]:
    """Compara red.json con lo ya avisado y devuelve los cambios que merecen mensaje. Actualiza el estado."""
    salida = []
    for a in red.get("aeropuertos", []):
        icao, r = a["icao"], a.get("riesgo")
        if (c["aeropuertos"] and icao not in c["aeropuertos"]) or not a.get("emitido") or not r:
            continue
        p, previo = r["p"], estado.get(icao)
        pico = max(a.get("serie") or [{"hora_utc": None, "p_bajo_minimos": p}], key=lambda s: s["p_bajo_minimos"])
        datos = {"icao": icao, "nombre": a["nombre"], "p": p, "hora": pico.get("hora_utc"),
                 "categoria": a.get("categoria"), "bajo_minimos": a.get("bajo_minimos"),
                 "despeja": a.get("despeja_utc")}
        if p >= c["umbral"] or (previo and a.get("bajo_minimos")):
            if not previo:
                salida.append({**datos, "tipo": "AVISO"})
            elif p >= previo["p"] + c["subida"]:
                salida.append({**datos, "tipo": "SUBE", "p_antes": previo["p"]})
            elif ahora - previo["enviado"] >= c["recordar_cada_h"] * 3600:
                salida.append({**datos, "tipo": "SIGUE"})
            else:
                continue                                  # nada nuevo: no se repite el mensaje
            estado[icao] = {"p": p, "enviado": ahora}
        elif previo and p < c["umbral_fin"] and not a.get("bajo_minimos"):
            salida.append({**datos, "tipo": "FIN"})
            del estado[icao]
    return salida


TITULOS = {"AVISO": "Aviso de bajo mínimos", "SUBE": "Empeora", "SIGUE": "Sigue el aviso", "FIN": "Fin del aviso"}


def mensaje(evs: list[dict], c: dict, vuelos: dict | None = None, nombres: dict | None = None) -> str:
    nombres = nombres or {}
    lineas = ["NOVOK · niebla y techo bajo"]
    for e in evs:
        if e["tipo"] == "FIN":
            lineas.append(f"\n✅ {TITULOS['FIN']} en {e['icao']} {e['nombre']}: riesgo {pct(e['p'])}.")
            continue
        sube = f" (antes {pct(e['p_antes'])})" if e["tipo"] == "SUBE" else ""
        lineas.append(f"\n⚠️ {TITULOS[e['tipo']]} en {e['icao']} {e['nombre']}: {pct(e['p'])}{sube} "
                      f"hacia las {hora_local(e['hora'])} (hora peninsular).")
        ahora = "bajo mínimos ahora" if e["bajo_minimos"] else f"ahora {e['categoria'] or 'sin dato'}"
        lineas.append(f"{ahora[0].upper()}{ahora[1:]}" + (f"; despeje previsto hacia las {hora_local(e['despeja'])}"
                                                           if e.get("despeja") else "") + ".")
        if vuelos is not None:
            lista = [v for v in vuelos.get(e["icao"], []) if not c.get("aerolinea") or v["operador"] == c["aerolinea"]]
            quien = f" de {nombres.get(c['aerolinea'], c['aerolinea'])}" if c.get("aerolinea") else ""
            if lista:
                total = sum(v["coste"] or 0 for v in lista)
                primeros = sorted(lista, key=lambda v: v["llegada"])[:6]
                detalle = ", ".join(f"{v['indicativo']} {datetime.fromtimestamp(v['llegada'], ZONA):%H:%M}"
                                    + (f" ({pct(v['p'])})" if v["p"] is not None else "") for v in primeros)
                lineas.append(f"Vuelos{quien} en el aire hacia {e['icao']}: {len(lista)} — {detalle}."
                              + (f" Coste esperado de desvíos: {eur(total)}." if total else ""))
            else:
                lineas.append(f"Ningún vuelo{quien} en el aire hacia {e['icao']} ahora.")
        lineas.append(f"{c['web'].rstrip('/')}/aeropuerto.html?a={e['icao']}")
    lineas.append("\nApoyo a la planificación; no sustituye a la información meteorológica oficial "
                  "(METAR/TAF de AEMET).")
    return "\n".join(lineas)


# ------------------------------------------------------------------ enviar

def canales() -> list[str]:
    salida = []
    if os.environ.get("NOVOK_TELEGRAM_TOKEN") and os.environ.get("NOVOK_TELEGRAM_CHAT"):
        salida.append("telegram")
    smtp = ("NOVOK_SMTP_SERVIDOR", "NOVOK_SMTP_USUARIO", "NOVOK_SMTP_CLAVE", "NOVOK_CORREO_PARA")
    if all(os.environ.get(k) for k in smtp):
        salida.append("correo")
    return salida


def enviar(texto: str, asunto: str = "NOVOK · aviso") -> list[str]:
    hechos = []
    for canal in canales():
        try:
            if canal == "telegram":
                r = requests.post(f"https://api.telegram.org/bot{os.environ['NOVOK_TELEGRAM_TOKEN']}/sendMessage",
                                  json={"chat_id": os.environ["NOVOK_TELEGRAM_CHAT"], "text": texto,
                                        "disable_web_page_preview": True}, timeout=20)
                r.raise_for_status()
            else:
                m = EmailMessage()
                m["Subject"], m["From"] = asunto, os.environ["NOVOK_SMTP_USUARIO"]
                m["To"] = os.environ["NOVOK_CORREO_PARA"]
                m.set_content(texto)
                with smtplib.SMTP(os.environ["NOVOK_SMTP_SERVIDOR"], int(os.environ.get("NOVOK_SMTP_PUERTO", 587)),
                                  timeout=30) as s:
                    s.starttls()
                    s.login(os.environ["NOVOK_SMTP_USUARIO"], os.environ["NOVOK_SMTP_CLAVE"])
                    s.send_message(m)
            hechos.append(canal)
        except Exception as e:  # noqa: BLE001 - un canal caído no impide usar el otro
            print(f"No se pudo enviar por {canal}: {e.__class__.__name__}: {e}")
    return hechos


def revisar(fich_red: Path, fich_estado: Path, con_vuelos: bool) -> list[dict]:
    c = config()
    red = json.loads(fich_red.read_text(encoding="utf-8"))
    if red.get("modo") == "repeticion":
        print("red.json es una repetición de una fecha pasada: no se avisa.")
        return []
    estado = json.loads(fich_estado.read_text(encoding="utf-8")) if fich_estado.exists() else {}
    ahora = time.time()
    evs = eventos(red, estado, c, ahora)
    if evs:
        vuelos, nombres = None, {}
        if con_vuelos:
            web = RAIZ / "web"
            cfg_v = json.loads((web / "vuelos.json").read_text(encoding="utf-8"))
            leer = lambda f: json.loads((web / f).read_text(encoding="utf-8")) if (web / f).exists() else None  # noqa: E731
            costes = leer("costes.json")
            nombres = (leer("aerolineas.json") or {}).get("aerolineas", {})
            aviones = posiciones(cfg_v)
            vuelos = vuelos_hacia(red, aviones, rutas(cfg_v, aviones) if aviones else {}, costes, ahora)
        texto = mensaje(evs, c, vuelos, nombres)
        print(texto)
        hechos = enviar(texto, "NOVOK · " + ", ".join(f"{e['tipo'].lower()} {e['icao']}" for e in evs))
        print(f"-> enviado por: {', '.join(hechos) or 'ningún canal (configura Telegram o correo)'}")
    else:
        print("Sin cambios que avisar.")
    fich_estado.parent.mkdir(parents=True, exist_ok=True)
    fich_estado.write_text(json.dumps(estado, indent=1), encoding="utf-8")
    return evs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--red", default=str(RED), help="red.json generado por python -m novok.red briefing")
    p.add_argument("--estado", default=str(FICH_ESTADO), help="dónde se guarda lo ya avisado")
    p.add_argument("--vuelos", action="store_true", help="añade los vuelos en el aire hacia cada aeropuerto en aviso")
    p.add_argument("--probar", action="store_true", help="manda un mensaje de prueba")
    p.add_argument("--vigilar", action="store_true", help="briefing y alertas cada 30 min (en tu ordenador)")
    args = p.parse_args()
    if args.probar:
        hechos = enviar("NOVOK · mensaje de prueba. Si lo lees, las alertas están bien configuradas.", "NOVOK · prueba")
        print(f"Enviado por: {', '.join(hechos) or 'ningún canal: revisa las variables de entorno'}")
        return
    while True:
        if args.vigilar:
            from novok.red import briefing_red
            briefing_red(None, Path(args.red).parent)
        revisar(Path(args.red), Path(args.estado), args.vuelos)
        if not args.vigilar:
            break
        print("Siguiente revisión en 30 min (Ctrl+C para parar)")
        time.sleep(1800)


if __name__ == "__main__":
    main()
