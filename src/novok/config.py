"""Configuración central del proyecto."""

from pathlib import Path

# Aeropuerto por defecto. El aeropuerto activo se elige en novok.aeropuertos (activa / usar / estacion).
ESTACION = "LEAS"

# Rutas
RAIZ = Path(__file__).resolve().parents[2]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"



def dir_proc(icao: str | None = None) -> Path:
    """Datos procesados de un aeropuerto (por defecto, el activo)."""
    from novok.aeropuertos import activa
    return DATA_PROCESSED / (icao or activa().icao)


def dir_modelos(icao: str | None = None) -> Path:
    """Modelos entrenados de un aeropuerto (se suben al repo: los usa el briefing en directo)."""
    from novok.aeropuertos import activa
    return RAIZ / "modelos" / (icao or activa().icao)


def dir_informes(icao: str | None = None) -> Path:
    """Informes y gráficos. LEAS se queda en reports/ (los enlaza el README); el resto, en reports/<OACI>/."""
    from novok.aeropuertos import activa
    icao = icao or activa().icao
    d = REPORTS if icao == "LEAS" else REPORTS / icao
    d.mkdir(parents=True, exist_ok=True)  # la carpeta de un aeropuerto nuevo aún no existe
    return d


# Fuente: archivo ASOS/METAR de Iowa Environmental Mesonet
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Conversiones
MILLA_A_METROS = 1609.344

# Primer año con METAR las 24 h (antes, sin observaciones de madrugada, justo
# cuando más niebla hay). Los años anteriores infravaloran la niebla.
ANIO_INICIO_FIABLE = 2017

# Umbrales de visibilidad (metros). Aproximaciones educativas:
# los mínimos reales de aproximación usan RVR y altura de decisión,
# y dependen del procedimiento publicado. NO usar para decisiones de vuelo.
UMBRALES_VIS_M = {
    "niebla": 1000,      # definición meteorológica de niebla
    "aprox_cat1": 550,   # orden de magnitud del RVR mínimo CAT I
    "aprox_cat2": 300,   # orden de magnitud del RVR mínimo CAT II
}

# Umbrales de techo de nubes (pies). Se considera bajo si techo <= umbral
# (el METAR notifica en pasos de 100 ft: VV001 = 100 ft).
UMBRALES_TECHO_FT = {
    "techo_bajo": 500,
    "aprox_cat1": 200,   # altura de decisión típica CAT I
    "aprox_cat2": 100,   # altura de decisión típica CAT II
}

# Coberturas que forman techo (BKN, OVC y cielo invisible VV)
COBERTURAS_TECHO = {"BKN", "OVC", "VV"}

# Las pistas de cada aeropuerto están en novok.aeropuertos (rumbo verdadero, de OurAirports).

# Umbrales de referencia de viento (nudos). Los límites reales dependen del avión
# y del operador; estos sirven para marcar condiciones exigentes, no prohibidas.
LIMITE_CRUZADO_KT = 20
LIMITE_COLA_KT = 10
