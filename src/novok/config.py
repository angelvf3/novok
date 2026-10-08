"""Configuración central del proyecto."""

from pathlib import Path

# Estación: Aeropuerto de Asturias (código OACI)
ESTACION = "LEAS"

# Rutas
RAIZ = Path(__file__).resolve().parents[2]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"

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

# Pista de LEAS (11/29) y rumbo aproximado en grados.
# El viento del METAR es geográfico y la pista se designa por rumbo magnético;
# en Asturias la declinación es ~1°, despreciable para este cálculo.
PISTAS = {"11": 110, "29": 290}

# Umbrales de referencia de viento (nudos). Los límites reales dependen del avión
# y del operador; estos sirven para marcar condiciones exigentes, no prohibidas.
LIMITE_CRUZADO_KT = 20
LIMITE_COLA_KT = 10
