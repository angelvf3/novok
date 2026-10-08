<p align="center">
  <img src="docs/novok-logo.png" alt="NOVOK" width="520">
</p>

<p align="center"><em>Will it be CAVOK? NOVOK tells you when it won't.</em></p>

# NOVOK

**NOwcasting Of Visibility & Overcast Kit** — predicción de niebla y techo de nubes con IA
para el Aeropuerto de Asturias (LEAS).

El nombre juega con **CAVOK** (*Ceiling And Visibility OK*), el código aeronáutico de
«techo y visibilidad bien». NOVOK predice justo lo contrario: cuándo **no** lo estarán.

Modelo de IA que predice, hora a hora, la probabilidad de niebla y techo de nubes bajo
en el Aeropuerto de Asturias, y se compara de forma honesta con el pronóstico oficial (TAF).

**Lo que busca hacer que el TAF no hace:** probabilidad por hora y por umbral operativo,
actualización con cada METAR, hora estimada de disipación, explicación del porqué
y un marcador público contra el TAF.

> ⚠️ Proyecto educativo. No apto para decisiones de vuelo.

## Estado

| Fase | Descripción | Estado |
|---|---|---|
| F0 | Repositorio, estructura, CI | ✅ |
| F1 | Datos: METAR histórico, limpieza y análisis exploratorio | ✅ |
| F2 | Baselines (persistencia, climatología, TAF) | 🔄 persistencia y climatología ✅ · TAF grabándose en directo |
| F3 | Modelo: niebla + techo de nubes | 🔄 v1 solo con METAR ✅ · falta añadir pronósticos numéricos |
| F4 | Hora de disipación y tipo de niebla | ⏳ |
| F5 | Aplicación web en vivo + briefing operativo (niebla, techo y viento en pista) | ⏳ |

## Primeros resultados (METAR 2017–2025)

![Frecuencia de niebla por mes y hora](reports/frecuencia_mes_hora.png)

| Indicador (media anual) | Valor |
|---|---|
| Días con alguna hora de visibilidad < 1000 m | ~36 |
| Días por debajo de mínimos tipo CAT I (vis < 550 m o techo ≤ 200 ft) | ~58 |
| Horas con visibilidad < 1000 m | ~116 (1,3 % del año) |
| Horas con techo ≤ 200 ft | ~235 |
| Episodios de niebla | ~46, duración mediana 2 h |
| Días con viento cruzado (con racha) > 20 kt | ~28 |

- **Es niebla de advección marina, no de radiación:** pico de abril a agosto (máximo en junio, de 02 a 09 h),
  con viento del NW/W desde el mar en el 60 % de los casos.
- **El techo bajo pesa más que la visibilidad:** hay el doble de horas con techo ≤ 200 ft que con visibilidad < 550 m.
- Antes de 2017 no hay METAR de madrugada (00–06 h), justo cuando más niebla hay; esos años se excluyen de la estadística.

## Baselines (F2)

Antes de entrenar nada hay que saber qué hay que batir. Para cada hora y antelación L,
el pronóstico se "emite" L horas antes y **solo usa lo que se sabía en ese momento**:

| Método | Qué predice |
|---|---|
| Persistencia | Lo observado en la última hora completa antes de emitir |
| Climatología | Frecuencia del suceso para ese mes y hora local, sin contar el propio año |
| TAF | El último TAF/enmienda emitido como muy tarde L h antes: condición predominante, peor caso (con TEMPO/PROB) y probabilidad |

Todos se comparan en la misma muestra de horas, contra la peor visibilidad/techo observados en cada hora.

**Persistencia (2017–2025), CSI:**

| Antelación | Niebla (< 1000 m) | Bajo mínimos tipo CAT I |
|---|---|---|
| 1 h | 0,26 | 0,34 |
| 3 h | 0,15 | 0,21 |
| 6 h | 0,08 | 0,12 |
| 12 h | 0,03 | 0,07 |

La niebla de LEAS es breve (episodios de ~2 h): mirar la hora anterior deja de servir enseguida.

**Fuente de TAF:** el histórico de Ogimet no devuelve datos, así que los TAF se graban en directo desde
[aviationweather.gov](https://aviationweather.gov/data/api/) cada 30 min con GitHub Actions
(`.github/workflows/grabar_taf.yml` → `data/vivo/taf_LEAS.txt`). La comparación con el TAF crece con cada día grabado.

**Conversión del TAF a probabilidad** (el TAF no da un número para todos los grupos; convención propia):
predominante bajo umbral = 1; PROB30 = 0,3; PROB40 = 0,4; TEMPO y transición BECMG = 0,5.

## Modelo NOVOK v1 (F3)

LightGBM por suceso y antelación, entrenado **solo con METAR**: estado actual, tendencias de 1–3 h,
historial reciente de niebla y calendario. Validación por años con ventana creciente: para evaluar el
año Y se entrena con los años anteriores y el umbral de decisión se elige en Y−1. Nunca ve el año evaluado.

![NOVOK frente a persistencia](reports/modelo_csi.png)

**Niebla (< 1000 m), 2020–2025, mismas horas para todos:**

| Antelación | CSI NOVOK | CSI persistencia | Brier skill NOVOK | Brier skill persistencia |
|---|---|---|---|---|
| 1 h | 0,28 | 0,29 | +0,26 | −0,13 |
| 3 h | 0,19 | 0,17 | +0,14 | −0,46 |
| 6 h | 0,11 | 0,09 | +0,06 | −0,71 |
| 12 h | 0,05 | 0,03 | +0,01 | −0,94 |

(Brier skill respecto a la climatología: > 0 mejora, < 0 empeora.)

**Lectura honesta:**
- Como **probabilidad**, NOVOK es claramente mejor que la persistencia y está bien calibrado
  ([fiabilidad](reports/modelo_fiabilidad.png)).
- Como **decisión sí/no**, a 1 h empata con la persistencia y a 3–12 h gana por poco. El umbral elegido en un
  solo año de validación es inestable (0,08–0,30).
- **El techo de lo que se puede hacer solo con METAR:** a 6–12 h el modelo depende sobre todo de la época del
  año. Para anticipar la niebla con horas de antelación hacen falta pronósticos numéricos (viento y humedad
  previstos, temperatura del mar). Es el siguiente paso.
- Lo que más pesa ([importancia SHAP](reports/modelo_importancia.png)): humedad, temperatura, época del año y la
  componente oeste del viento — el aire que llega del mar, coherente con la niebla de advección.

## NOVOK v2: + pronóstico numérico (F3, en curso)

Se añaden pronósticos de modelos meteorológicos (ECMWF IFS desde 2024, JMA GSM desde 2018) de la API
**Previous Runs** de Open-Meteo, usando **el pronóstico emitido ~24 h antes** de cada hora (`_previous_day1`).

¿Por qué no la API de pronósticos históricos? Encadena las primeras horas de cada pasada del modelo:
valores que se publicaron *después* del momento en que NOVOK habría emitido. Usarlos inflaría los
resultados artificialmente. El pronóstico de 24 h antes ya estaba publicado con 1–12 h de antelación.

Variables: humedad, diferencia temperatura–rocío, nubes bajas, visibilidad y viento previstos para la hora
objetivo; cambio previsto respecto a ahora; y error actual del modelo frente al METAR.

v1 y v2 se entrenan con las mismas filas y la misma validación por años; los resultados se separan para
2024–2025, los únicos años con ECMWF.

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 1. Descargar el histórico de METAR (un CSV por año en data/raw/)
python -m novok.data.descargar_metar --inicio 2010 --fin 2025

# 2. Limpiar y agregar a tabla horaria (data/processed/)
python -m novok.data.limpiar_metar

# 3. Primer análisis: horas de niebla por año y mapa mes x hora (reports/)
python -m novok.analysis.resumen_niebla

# 4. TAF: se graban solos en el repo (GitHub Actions). Para unirlos a la verificación:
git pull
python -m novok.data.descargar_taf --extraer

# 5. Verificación de baselines (reports/verificacion_baselines.csv y verificacion_csi.png)
python -m novok.verificacion.baselines

# 6. Modelo NOVOK: validación por años y modelos finales en models/ (~7 min)
python -m novok.modelo.entrenar

# 7. Pronóstico numérico (Open-Meteo): primero la prueba de 1 día, luego todo
python -m novok.data.descargar_nwp --prueba
python -m novok.data.descargar_nwp

# 8. NOVOK v2 (METAR + pronóstico) frente a v1 (~15 min)
python -m novok.modelo.entrenar_nwp

# Tests y lint
pytest -q
ruff check .
```

## Datos

- **METAR/SPECI de LEAS**: archivo de [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml).
  Visibilidad en metros, techo = capa más baja BKN/OVC/VV en pies.
- La tabla horaria toma la **peor** visibilidad y el **peor** techo de cada hora: esas son las etiquetas.

## Umbrales

| Variable | Umbral | Significado |
|---|---|---|
| Visibilidad | < 1000 m | Niebla (definición meteorológica) |
| Visibilidad | < 550 m | Orden de magnitud del mínimo CAT I |
| Visibilidad | < 300 m | Orden de magnitud del mínimo CAT II |
| Techo | < 500 ft | Techo bajo |
| Techo | < 200 ft | Altura de decisión típica CAT I |
| Techo | < 100 ft | Altura de decisión típica CAT II |

Son aproximaciones: los mínimos reales usan RVR y la altura de decisión del procedimiento publicado.

## Viento en pista

El viento no es un modelo de IA: es una capa de información. A partir del viento (observado o
pronosticado) se calculan las componentes cruzada y de cara/cola sobre la pista **11/29**, la
cabecera favorable y avisos de referencia:

| Aviso | Umbral |
|---|---|
| Viento cruzado (con racha) | > 20 kt |
| Viento de cola | > 10 kt |

Los límites reales dependen del avión y del operador.

## Estructura

```
src/novok/
  config.py                 # estación, rutas y umbrales
  data/descargar_metar.py   # descarga año a año desde IEM
  data/limpiar_metar.py     # unidades métricas, techo, agregación horaria
  data/viento.py            # viento cruzado y de cola sobre la pista 11/29
  data/grabar_taf_vivo.py   # TAF en directo desde aviationweather.gov (GitHub Actions)
  data/descargar_taf.py     # histórico de Ogimet (no disponible) y unión de TAF
  data/descargar_nwp.py     # pronósticos de 24 h antes (ECMWF, JMA) desde Open-Meteo
  taf/parser.py             # TAF -> pronóstico horario (predominante, peor caso, probabilidad)
  verificacion/             # baselines y métricas (POD, FAR, CSI, Brier)
  modelo/                   # variables desde METAR y entrenamiento LightGBM con validación temporal
  analysis/resumen_niebla.py
tests/
```

## Autor

Ángel Villabrille Fernández
