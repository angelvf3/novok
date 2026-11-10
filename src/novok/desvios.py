"""F7b · Desvíos reales por niebla en el histórico de vuelos, y si NOVOK los habría avisado.

Es la prueba que entiende una aerolínea: "el día X desviasteis el vuelo Y de Asturias a Santander; NOVOK lo
avisaba N horas antes". Se hace en tres pasos:

1. descargar: para cada día, el histórico de adsb.lol (github.com/adsblol/globe_history_AAAA, ODbL 1.0) trae
   un "heatmap" con la posición de todos los aviones cada pocos segundos. Va al principio del fichero del día
   (unos 2 GB), así que se lee en continuo solo esa parte (unos 400-500 MB), se recorta al norte de España
   y se guarda en data/desvios/pistas/AAAA-MM-DD.parquet (unos pocos MB). El resto no se descarga.
2. detectar: se reconstruye cada vuelo y se buscan dos tipos de desvío hacia un aeropuerto distinto del previsto:
   - aproximación frustrada: el avión baja a menos de 3.000 ft cerca de un aeropuerto de la red, no aterriza y
     acaba aterrizando en otro;
   - desvío antes de llegar: el vuelo (mismo indicativo) aterriza casi siempre en un aeropuerto de la red y ese
     día aterriza en otro. El destino habitual se calcula con todos los días descargados.
3. evaluar: para cada desvío, se mira qué decía NOVOK para esa hora con 1, 3, 6 y 12 h de antelación (las
   predicciones de la verificación independiente, sin datos de la temporada de prueba) y la persistencia.

Uso:
    python -m novok.desvios descargar --desde 2026-01-01 --hasta 2026-03-31 [--solo-niebla] [--control 10]
    python -m novok.desvios detectar
    python -m novok.desvios evaluar
    python -m novok.desvios todo --desde 2026-01-01        # los tres pasos
    python -m novok.desvios web                            # web/desvios.json (ficha de cada aeropuerto)

Limitaciones: la cobertura de adsb.lol depende de sus colaboradores; cerca del suelo puede faltar señal, por
eso un aterrizaje se da por bueno si el vuelo termina por debajo de 2.000 ft cerca de un aeropuerto. Los desvíos
por otras causas (avería, pasajero enfermo) también aparecen: se marcan los que coinciden con bajo mínimos
observados en el METAR del aeropuerto previsto.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import tarfile
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from novok.config import RAIZ

DIR = RAIZ / "data" / "desvios"
DIR_PISTAS = DIR / "pistas"
DIR_INFORME = RAIZ / "reports" / "desvios"
URL = "https://github.com/adsblol/globe_history_{anio}/releases/download/{tag}/{tag}.tar.{parte}"
INSTANCIAS = ("prod-0", "prod-1", "staging-0")
CAJA = (40.2, 45.2, -10.5, 0.5)          # lat_min, lat_max, lon_min, lon_max: norte de España y alrededores
MARCA_TIEMPO = 0x0E7F7C9D                 # entrada del heatmap que lleva la hora (readsb)

# Coordenadas de referencia (OurAirports). Red de NOVOK y alternativos habituales fuera de ella.
RED = {
    "LEAS": ("Asturias", 43.5636, -6.0346), "LEXJ": ("Santander", 43.4271, -3.8200),
    "LEBB": ("Bilbao", 43.3011, -2.9106), "LESO": ("San Sebastián", 43.3565, -1.7906),
    "LEVT": ("Vitoria", 42.8828, -2.7245), "LEPP": ("Pamplona", 42.7700, -1.6463),
    "LELO": ("Logroño", 42.4610, -2.3223), "LEBG": ("Burgos", 42.3576, -3.6206),
    "LELN": ("León", 42.5890, -5.6556), "LEVD": ("Valladolid", 41.7061, -4.8519),
    "LEST": ("Santiago", 42.8963, -8.4151), "LECO": ("A Coruña", 43.3021, -8.3773),
    "LEVX": ("Vigo", 42.2318, -8.6268),
}
OTROS = {
    "LEMD": ("Madrid", 40.4719, -3.5626), "LEZG": ("Zaragoza", 41.6662, -1.0416),
    "LPPR": ("Oporto", 41.2481, -8.6814), "LESA": ("Salamanca", 40.9521, -5.5020),
    "LFBZ": ("Biarritz", 43.4684, -1.5233), "LEHC": ("Huesca", 42.0761, -0.3160),
    "LFBP": ("Pau", 43.3800, -0.4186), "LPBG": ("Bragança", 41.8578, -6.7071),
}
AEROPUERTOS = {**RED, **OTROS}
ELEV_FT = {"LEAS": 416, "LEXJ": 16, "LEBB": 138, "LESO": 16, "LEVT": 1682, "LEPP": 1504, "LELO": 1161, "LEBG": 2945,
           "LELN": 3006, "LEVD": 2776, "LEST": 1213, "LECO": 326, "LEVX": 856, "LEMD": 1998, "LEZG": 863, "LPPR": 228,
           "LESA": 2595, "LFBZ": 245, "LEHC": 1768, "LFBP": 616, "LPBG": 2241}

# Umbrales de la detección
# alturas sobre el aeropuerto (AGL): la antena más cercana no siempre ve los últimos cientos de pies
RADIO_ATERRIZAJE_KM = 15
ALT_ATERRIZAJE_FT = 2000
RADIO_APROX_KM = 15
ALT_APROX_FT = 2500
# donde la antena no baja tanto (Bilbao, Santander: montañas), un vuelo que desaparece bajando cerca de un
# aeropuerto se da por aterrizado en él ("inferido")
RADIO_INFERIDO_KM = 40
ALT_INFERIDO_FT = 10000
HUECO_VUELO_MIN = 40      # sin señal más de esto separa dos vuelos
ESCALA_MIN = 12           # ...o más de esto si antes estaba bajo junto a un aeropuerto (aterrizó y volvió a salir)
MIN_DIAS_HABITUAL = 3     # días que debe repetirse un indicativo para fijar su destino habitual
PROP_HABITUAL = 0.75


# ---------------------------------------------------------------- 1. descarga

def leer_heatmap(datos: bytes, caja=CAJA) -> pd.DataFrame:
    """Descodifica un fichero heatmap de readsb (entradas de 16 bytes) y se queda con lo que cae en la caja.

    Cada entrada: int32 hex (24 bits de dirección + marcas), int32 lat·1e6, int32 lon·1e6, int16 alt/25, int16 gs·10.
    Entradas especiales: la marca de tiempo (hex = MARCA_TIEMPO; lat, lon = milisegundos en dos mitades) y las de
    indicativo (lat con el bit 30 puesto; los 8 bytes siguientes son el indicativo en ASCII).
    """
    if datos[:2] == b"\x1f\x8b":
        datos = gzip.decompress(datos)
    n = len(datos) // 16
    if n == 0:
        return pd.DataFrame()
    a = np.frombuffer(datos[: n * 16], dtype="<i4").reshape(-1, 4)
    hexr, lat, lon = a[:, 0], a[:, 1], a[:, 2]
    resto = np.frombuffer(datos[: n * 16], dtype="<i2").reshape(-1, 8)
    alt, gs = resto[:, 6], resto[:, 7]

    marca = hexr == MARCA_TIEMPO
    if not marca.any():
        return pd.DataFrame()
    t_marca = (lat[marca].astype(np.int64) << 32) | (lon[marca].astype(np.int64) & 0xFFFFFFFF)
    cual = np.cumsum(marca) - 1                       # a qué marca de tiempo pertenece cada entrada
    valida = (cual >= 0) & ~marca
    t = np.where(cual >= 0, t_marca[np.clip(cual, 0, None)], 0)

    es_indicativo = valida & ((lat & (1 << 30)) != 0)
    posicion = valida & ~es_indicativo
    direccion = hexr & 0xFFFFFF

    # indicativos: último conocido por avión
    indicativos = {}
    if es_indicativo.any():
        crudo = np.frombuffer(datos[: n * 16], dtype="S16")[es_indicativo]
        for d, c in zip(direccion[es_indicativo], crudo, strict=True):
            txt = c[8:16].decode("ascii", "ignore").strip()
            if txt:
                indicativos[int(d)] = txt

    la, lo = lat[posicion] / 1e6, lon[posicion] / 1e6
    dentro = (la >= caja[0]) & (la <= caja[1]) & (lo >= caja[2]) & (lo <= caja[3])
    df = pd.DataFrame({
        "t": t[posicion][dentro],
        "hex": direccion[posicion][dentro].astype(np.int32),
        "lat": la[dentro].astype(np.float32), "lon": lo[dentro].astype(np.float32),
        # -123 (×25 = -3075 ft) es la marca de "en tierra"
        "alt_ft": np.where(alt[posicion][dentro] == -123, 0, alt[posicion][dentro].astype(np.int32) * 25),
        "gs_kt": (gs[posicion][dentro] / 10).astype(np.float32),
    })
    df["indicativo"] = df["hex"].map(indicativos)
    return df


class _Partes(io.RawIOBase):
    """Lee las partes .aa, .ab... de un release como si fueran un único fichero, en continuo."""

    def __init__(self, sesion, urls):
        self.sesion, self.urls, self.resp, self.it = sesion, list(urls), None, None

    def readable(self):
        return True

    def _siguiente(self):
        while self.urls:
            r = self.sesion.get(self.urls.pop(0), stream=True, timeout=60)
            if r.status_code == 200:
                self.resp, self.it = r, r.iter_content(1 << 20)
                return True
            r.close()
            return False
        return False

    def readinto(self, b):
        while True:
            if self.it is None and not self._siguiente():
                return 0
            try:
                trozo = next(self.it)
            except StopIteration:
                self.resp.close()
                self.it = None
                continue
            if not trozo:
                continue
            self._pend = getattr(self, "_pend", b"") + trozo
            k = min(len(b), len(self._pend))
            b[:k], self._pend = self._pend[:k], self._pend[k:]
            return k

    def close(self):
        if self.resp is not None:
            self.resp.close()
        super().close()


def _sesion():
    import requests
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    s = requests.Session()
    s.headers["User-Agent"] = "novok-desvios (investigacion de desvios por niebla)"
    return s


def descargar_dia(dia: date, sesion=None, caja=CAJA) -> pd.DataFrame | None:
    """Lee en continuo el heatmap del día y se detiene al llegar a las trazas. Devuelve None si no hay release."""
    sesion = sesion or _sesion()
    for inst in INSTANCIAS:
        tag = f"v{dia:%Y.%m.%d}-planes-readsb-{inst}"
        urls = [URL.format(anio=dia.year, tag=tag, parte=p) for p in ("aa", "ab", "ac", "ad")]
        cabeza = sesion.head(urls[0], allow_redirects=True, timeout=30)
        if cabeza.status_code != 200:
            continue
        flujo = io.BufferedReader(_Partes(sesion, urls), buffer_size=1 << 20)
        trozos, visto_heatmap = [], False
        try:
            with tarfile.open(fileobj=flujo, mode="r|") as tar:
                for m in tar:
                    nombre = m.name.lstrip("./")
                    if nombre.startswith("heatmap/") and m.isfile():
                        visto_heatmap = True
                        trozos.append(leer_heatmap(tar.extractfile(m).read(), caja))
                    elif visto_heatmap and nombre.startswith("traces"):
                        break                      # el heatmap ya pasó: no hace falta bajar el resto
        finally:
            flujo.close()
        if not trozos:
            continue
        df = pd.concat([x for x in trozos if len(x)], ignore_index=True)
        df.attrs["fuente"] = tag
        return df.sort_values(["hex", "t"], kind="stable").reset_index(drop=True)
    return None


def dias_con_niebla(desde: date, hasta: date) -> list[date]:
    """Días con al menos 2 horas de bajo mínimos CAT I (06-22 UTC) en algún aeropuerto de la red (METAR)."""
    from novok.aeropuertos import usar
    from novok.verificacion.baselines import cargar_observado
    dias = set()
    for icao in RED:
        try:
            usar(icao)
            h = cargar_observado()
        except Exception:      # aeropuerto sin datos descargados
            continue
        fin = pd.Timestamp(hasta + timedelta(days=1), tz="UTC")
        h = h[(h.index >= pd.Timestamp(desde, tz="UTC")) & (h.index < fin)]
        y = h["obs_bajo_minimos_cat1"].where((h.index.hour >= 6) & (h.index.hour <= 22)).fillna(0)
        cuenta = y.groupby(h.index.date).sum()
        dias |= set(cuenta[cuenta >= 2].index)
    return sorted(dias)


def descargar(desde: date, hasta: date, solo_niebla: bool, control: int = 0, rehacer: bool = False) -> None:
    DIR_PISTAS.mkdir(parents=True, exist_ok=True)
    todos = [desde + timedelta(days=i) for i in range((hasta - desde).days + 1)]
    if solo_niebla:
        niebla = set(dias_con_niebla(desde, hasta))
        if not niebla:
            raise SystemExit("No hay días con niebla en los METAR de esas fechas (¿están actualizados?).")
        # días de control sin niebla: sirven para conocer el destino habitual de cada vuelo
        resto = [d for d in todos if d not in niebla]
        paso = max(1, len(resto) // control) if control else 0
        dias = sorted(niebla | set(resto[::paso][:control] if control else []))
        print(f"{len(niebla)} días con niebla y {len(dias) - len(niebla)} de control")
    else:
        dias = todos
    s = _sesion()
    for i, d in enumerate(dias, 1):
        destino = DIR_PISTAS / f"{d:%Y-%m-%d}.parquet"
        if destino.exists() and not rehacer:
            continue
        t0 = time.time()
        try:
            df = descargar_dia(d, s)
        except Exception as e:                       # corte de red: se sigue con el siguiente día
            print(f"[{i}/{len(dias)}] {d}: error ({e}); se reintentará en la próxima ejecución")
            continue
        if df is None:
            print(f"[{i}/{len(dias)}] {d}: no hay histórico publicado")
            continue
        df.to_parquet(destino, index=False)
        print(f"[{i}/{len(dias)}] {d}: {len(df):,} posiciones de {df['hex'].nunique()} aviones "
              f"({time.time() - t0:.0f} s, {df.attrs.get('fuente')})")


# ---------------------------------------------------------------- 2. detección

def _km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _mas_cercano(lat, lon):
    """Aeropuerto conocido más cercano a cada punto y su distancia (km)."""
    codigos = list(AEROPUERTOS)
    d = np.stack([_km(lat, lon, AEROPUERTOS[c][1], AEROPUERTOS[c][2]) for c in codigos], axis=1)
    k = d.argmin(axis=1)
    return np.array(codigos)[k], d[np.arange(len(k)), k]


def vuelos_del_dia(df: pd.DataFrame) -> pd.DataFrame:
    """Un registro por vuelo: indicativo, aterrizaje (si lo hay), aproximaciones a la red que no acabaron en tierra."""
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["hex", "t"], kind="stable")
    cerca, dist = _mas_cercano(df["lat"].to_numpy(), df["lon"].to_numpy())
    agl = df["alt_ft"].to_numpy() - pd.Series(cerca).map(ELEV_FT).fillna(0).to_numpy()
    agl = np.where(df["alt_ft"].to_numpy() == 0, 0, agl)          # 0 = en tierra
    df = df.assign(cerca=cerca, dist=dist, agl=agl)
    filas = []
    for hexa, g in df.groupby("hex", sort=False):
        t, alt, dist = g["t"].to_numpy(), g["agl"].to_numpy(), g["dist"].to_numpy()
        hueco = np.diff(t)
        # nuevo vuelo: sin señal mucho rato, o un rato menor tras quedarse bajo junto a un aeropuerto (aterrizó y
        # despegó otra vez: la antena no ve el suelo). Una aproximación frustrada pierde la señal solo unos minutos.
        en_tierra = (hueco > ESCALA_MIN * 60_000) & (alt[:-1] <= ALT_INFERIDO_FT) & (dist[:-1] <= RADIO_INFERIDO_KM)
        # o se le ve en el suelo y luego en el aire (la antena llega hasta la pista)
        despega = (alt[:-1] <= 0) & (alt[1:] > 0) & (g["alt_ft"].to_numpy()[:-1] == 0)
        corte = np.r_[0, np.where((hueco > HUECO_VUELO_MIN * 60_000) | en_tierra | despega)[0] + 1, len(g)]
        for a, b in zip(corte[:-1], corte[1:], strict=True):
            v = g.iloc[a:b]
            if len(v) < 10 or v["alt_ft"].max() < 5000:
                continue                                # trozos sueltos o vuelos locales a baja altura
            ultimo = v.iloc[-1]
            final_bajo = v.tail(6)
            aterriza, inferido = None, False
            if ultimo["agl"] <= ALT_ATERRIZAJE_FT and ultimo["dist"] <= RADIO_ATERRIZAJE_KM:
                aterriza = ultimo["cerca"]
            elif (final_bajo["agl"] <= ALT_ATERRIZAJE_FT).all() and final_bajo["dist"].min() <= RADIO_ATERRIZAJE_KM:
                aterriza = final_bajo.loc[final_bajo["dist"].idxmin(), "cerca"]
            else:
                hace5 = v[v["t"] >= ultimo["t"] - 5 * 60_000]
                bajando = hace5["alt_ft"].iat[0] - ultimo["alt_ft"] >= 800
                # desaparece casi encima de un aeropuerto, aún alto pero bajando: está en la maniobra de llegada
                if ultimo["dist"] <= 8 and ultimo["agl"] <= 10000 and hace5["alt_ft"].iat[0] > ultimo["alt_ft"]:
                    bajando, ultimo = True, ultimo.copy()
                    ultimo["agl"] = min(ultimo["agl"], ALT_INFERIDO_FT)
                if bajando and ultimo["agl"] <= ALT_INFERIDO_FT and ultimo["dist"] <= RADIO_INFERIDO_KM:
                    # se prolonga la trayectoria lo que le queda de descenso (senda de 3°: ~300 ft por milla)
                    previo = hace5.iloc[max(0, len(hace5) - 7)]
                    rumbo = np.arctan2((ultimo["lon"] - previo["lon"]) * np.cos(np.radians(ultimo["lat"])),
                                       ultimo["lat"] - previo["lat"])
                    km = min(ultimo["agl"] / 300 * 1.852, 70)
                    paso = np.linspace(0, km, 20)        # aeropuerto que más se acerca a esa prolongación
                    lat_f = ultimo["lat"] + paso / 111.32 * np.cos(rumbo)
                    lon_f = ultimo["lon"] + paso / (111.32 * np.cos(np.radians(ultimo["lat"]))) * np.sin(rumbo)
                    c, dd = _mas_cercano(lat_f, lon_f)
                    if dd.min() <= 15:
                        aterriza, inferido = c[dd.argmin()], True
            # aproximaciones a la red: bajo 3.000 ft a menos de 15 km, en un tramo que no es el del aterrizaje
            bajos = v[(v["agl"] <= ALT_APROX_FT) & (v["dist"] <= RADIO_APROX_KM) & v["cerca"].isin(list(RED))]
            aprox = []
            for icao, gg in bajos.groupby("cerca"):
                # solo cuenta si venía bajando (no es la subida tras despegar de ahí)
                t_bajo = gg["t"].min()
                antes = v[(v["t"] < t_bajo) & (v["t"] >= t_bajo - 6 * 60_000)]
                if antes.empty or antes["agl"].max() < gg["agl"].iat[0] + 500:
                    continue
                if icao == aterriza:
                    # ¿se fue y volvió? solo cuenta si después subió por encima de 4.000 ft y se alejó
                    despues = v[v["t"] > gg["t"].min()]
                    if not ((despues["agl"] > 4000) & (despues["dist"] > 25)).any():
                        continue
                aprox.append({"icao": icao, "t": int(t_bajo), "agl_min_ft": int(gg["agl"].min())})
            ind = v["indicativo"].dropna()
            filas.append({
                "hex": f"{int(hexa):06x}", "indicativo": ind.mode().iat[0] if len(ind) else None,
                "t_ini": int(v["t"].iat[0]), "t_fin": int(ultimo["t"]),
                "aterriza": aterriza, "inferido": inferido, "aproximaciones": aprox,
            })
    return pd.DataFrame(filas)


def detectar() -> pd.DataFrame:
    ficheros = sorted(DIR_PISTAS.glob("*.parquet"))
    if not ficheros:
        raise SystemExit("No hay días descargados: python -m novok.desvios descargar --desde AAAA-MM-DD")
    todos = []
    for f in ficheros:
        v = vuelos_del_dia(pd.read_parquet(f))
        if len(v):
            v["dia"] = f.stem
            todos.append(v)
        print(f"{f.stem}: {len(v)} vuelos")
    v = pd.concat(todos, ignore_index=True)
    v["operador"] = v["indicativo"].str.extract(r"^([A-Z]{3})\d", expand=False)

    # destino habitual de cada indicativo (comercial), con todos los días
    lleg = v.dropna(subset=["indicativo", "aterriza"])
    lleg = lleg[lleg["operador"].notna()]
    habitual = {}
    for ind, g in lleg.groupby("indicativo"):
        if g["dia"].nunique() < MIN_DIAS_HABITUAL:
            continue
        cuenta = g["aterriza"].value_counts()
        if cuenta.iat[0] / cuenta.sum() >= PROP_HABITUAL and cuenta.index[0] in RED:
            habitual[ind] = cuenta.index[0]

    desvios = []
    for r in v.itertuples():
        if pd.isna(r.aterriza):
            continue
        motivo, previsto, t_prev = None, None, None
        frustradas = [a for a in r.aproximaciones if a["icao"] != r.aterriza]
        if frustradas:
            motivo, previsto, t_prev = "aproximacion_frustrada", frustradas[-1]["icao"], frustradas[-1]["t"]
        elif r.indicativo in habitual and habitual[r.indicativo] != r.aterriza:
            motivo, previsto = "destino_habitual", habitual[r.indicativo]
            # hora en la que habría llegado: el final del vuelo menos el tramo extra, aproximado con el propio final
            t_prev = r.t_fin
        if motivo:
            desvios.append({"dia": r.dia, "indicativo": r.indicativo, "operador": r.operador, "hex": r.hex,
                            "previsto": previsto, "aterriza": r.aterriza, "aterrizaje_inferido": bool(r.inferido),
                            "motivo": motivo,
                            "hora_prevista_utc": pd.Timestamp(t_prev, unit="ms", tz="UTC").isoformat(),
                            "aterrizaje_utc": pd.Timestamp(r.t_fin, unit="ms", tz="UTC").isoformat()})
    d = pd.DataFrame(desvios)
    DIR_INFORME.mkdir(parents=True, exist_ok=True)
    d.to_csv(DIR_INFORME / "desvios_detectados.csv", index=False)
    (DIR_INFORME / "destinos_habituales.json").write_text(json.dumps(habitual, indent=1), encoding="utf-8")
    print(f"\n{len(v)} vuelos, {len(habitual)} indicativos con destino habitual, {len(d)} desvíos detectados")
    return d


# ---------------------------------------------------------------- 3. evaluación

def evaluar(corte: str = "2026-01-01") -> pd.DataFrame:
    """Para cada desvío a un aeropuerto de la red: ¿había bajo mínimos? ¿lo avisaba NOVOK con 1, 3, 6 y 12 h?"""
    from novok.aeropuertos import usar
    from novok.config import dir_informes
    from novok.verificacion.baselines import cargar_observado

    ruta = DIR_INFORME / "desvios_detectados.csv"
    if not ruta.exists():
        raise SystemExit("Primero: python -m novok.desvios detectar")
    d = pd.read_csv(ruta)
    filas = []
    for r in d.itertuples():
        t = pd.Timestamp(r.hora_prevista_utc).floor("h")
        fila = r._asdict()
        fila.pop("Index")
        try:
            usar(r.previsto)
            h = cargar_observado()
            horas = pd.date_range(t - pd.Timedelta(hours=1), t + pd.Timedelta(hours=1), freq="h")
            ventana = h["obs_bajo_minimos_cat1"].reindex(horas)
            fila["bajo_minimos_obs"] = bool((ventana == 1).any())
        except (Exception, SystemExit):            # aeropuerto sin METAR descargados
            fila["bajo_minimos_obs"] = None
        series = dir_informes(r.previsto) / "independiente_series.parquet"
        if series.exists():
            s = pd.read_parquet(series)
            s = s[(s["suceso"] == "bajo_minimos_cat1")]
            for L in (1, 3, 6, 12):
                x = s[(s["horizonte_h"] == L) & (s["t"] == t)]
                if len(x):
                    fila[f"p_{L}h"] = round(float(x["p_novok"].iat[0]), 3)
                    fila[f"aviso_{L}h"] = bool(x["p_novok"].iat[0] >= x["u_novok"].iat[0])
                    fila[f"persistencia_{L}h"] = bool(x["persistencia"].iat[0] == 1)
        fila["en_prueba"] = t >= pd.Timestamp(corte, tz="UTC")
        filas.append(fila)
    e = pd.DataFrame(filas)
    e.to_csv(DIR_INFORME / "desvios_evaluados.csv", index=False)
    resumen(e)
    return e


def resumen(e: pd.DataFrame) -> dict:
    niebla = e[(e["bajo_minimos_obs"] == True) & e["previsto"].isin(list(RED))]  # noqa: E712
    out = {"desvios_detectados": int(len(e)), "desvios_con_bajo_minimos": int(len(niebla)),
           "por_aeropuerto": niebla["previsto"].value_counts().to_dict(), "avisados": {}}
    for L in (1, 3, 6, 12):
        c = f"aviso_{L}h"
        if c in niebla and niebla[c].notna().any():
            x = niebla[niebla[c].notna()]
            out["avisados"][f"{L}h"] = {"novok": int(x[c].sum()), "persistencia": int(x[f"persistencia_{L}h"].sum()),
                                        "total": int(len(x))}
    (DIR_INFORME / "resumen.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def web(antelacion: int = 6) -> dict:
    """Resumen por aeropuerto para la web (web/desvios.json): casos con bajo mínimos, avisados y los últimos."""
    ruta = DIR_INFORME / "desvios_evaluados.csv"
    if not ruta.exists():
        raise SystemExit("Primero: python -m novok.desvios evaluar")
    e = pd.read_csv(ruta)
    n = e[(e["bajo_minimos_obs"] == True) & e["previsto"].isin(list(RED))].copy()  # noqa: E712
    if "en_prueba" in n:
        n = n[n["en_prueba"] == True]  # noqa: E712
    c_av, c_pe = f"aviso_{antelacion}h", f"persistencia_{antelacion}h"
    out = {"antelacion_h": antelacion, "aeropuertos": {}}
    if len(n):
        dias = pd.to_datetime(n["dia"])
        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
                 "noviembre", "diciembre"]
        out["periodo"] = f"{meses[dias.min().month - 1]} a {meses[dias.max().month - 1]} de {dias.max().year}"
    for icao, g in n.groupby("previsto"):
        ev = g[g[c_av].notna()] if c_av in g else g.iloc[0:0]
        g = g.sort_values("hora_prevista_utc", ascending=False)
        out["aeropuertos"][icao] = {
            "casos": int(len(g)), "evaluados": int(len(ev)),
            "avisados": int((ev[c_av] == True).sum()) if len(ev) else 0,  # noqa: E712
            "persistencia": int((ev[c_pe] == True).sum()) if len(ev) else 0,  # noqa: E712
            "ultimos": [{"fecha": pd.Timestamp(r.dia).strftime("%d/%m"), "indicativo": r.indicativo,
                         "aterriza": r.aterriza,
                         "avisado": bool(c_av in g and pd.notna(getattr(r, c_av)) and getattr(r, c_av))}
                        for r in g.head(4).itertuples()],
        }
    destino = RAIZ / "web" / "desvios.json"
    destino.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    casos = sum(x["casos"] for x in out["aeropuertos"].values())
    print(f"{destino}: {casos} casos en {len(out['aeropuertos'])} aeropuertos")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="orden", required=True)
    for nombre in ("descargar", "todo"):
        p = sub.add_parser(nombre)
        p.add_argument("--desde", required=True, type=date.fromisoformat)
        p.add_argument("--hasta", type=date.fromisoformat, default=date.today() - timedelta(days=2))
        p.add_argument("--solo-niebla", action="store_true", help="solo días con bajo mínimos en algún aeropuerto")
        p.add_argument("--control", type=int, default=10, help="días sin niebla para el destino habitual")
        p.add_argument("--rehacer", action="store_true")
    sub.add_parser("detectar")
    pe = sub.add_parser("evaluar")
    pe.add_argument("--corte", default="2026-01-01")
    pw = sub.add_parser("web", help="escribe web/desvios.json para la ficha de cada aeropuerto")
    pw.add_argument("--antelacion", type=int, default=6)
    a = ap.parse_args()
    if a.orden in ("descargar", "todo"):
        descargar(a.desde, a.hasta, a.solo_niebla, a.control, a.rehacer)
    if a.orden in ("detectar", "todo"):
        detectar()
    if a.orden in ("evaluar", "todo"):
        evaluar(getattr(a, "corte", "2026-01-01"))
    if a.orden in ("evaluar", "todo", "web"):
        web(getattr(a, "antelacion", 6))


if __name__ == "__main__":
    main()
