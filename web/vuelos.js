/* NOVOK · vuelos en directo y riesgo a la llegada.
 *
 * Lo comparten el mapa de la red y la página de cada aeropuerto. La fuente de datos se define en vuelos.json:
 * hoy un servicio comunitario gratuito (adsb.lol, datos ODbL); mañana una antena propia o un proveedor de pago,
 * sin tocar este código. Acepta el formato de readsb/tar1090 (el habitual de los receptores ADS-B): {ac: [...]}
 * o {aircraft: [...]}.
 *
 * Lo que calcula para cada vuelo que va a un aeropuerto de la red:
 *  - hora estimada de llegada (distancia / velocidad, más unos minutos de aproximación),
 *  - probabilidad de bajo mínimos CAT I en el destino a esa hora (interpolando la previsión de NOVOK),
 *  - el aeropuerto de la red con mejor previsión cerca del destino, como alternativo orientativo.
 * Es apoyo a la planificación en tierra; la decisión de desviar es de la tripulación y de la aerolínea.
 */
(function (global) {
  "use strict";

  const R_TIERRA_KM = 6371;
  const NM_KM = 1.852;
  const MIN_APROX = 8;            // minutos que se suman a la llegada por la maniobra de aproximación
  const NIVELES = [0.10, 0.30];   // bajo < 10 % <= moderado < 30 % <= alto (los mismos que el briefing)

  function distanciaKm(lat1, lon1, lat2, lon2) {
    const r = Math.PI / 180, dlat = (lat2 - lat1) * r, dlon = (lon2 - lon1) * r;
    const a = Math.sin(dlat / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dlon / 2) ** 2;
    return 2 * R_TIERRA_KM * Math.asin(Math.min(1, Math.sqrt(a)));
  }

  function rumbo(lat1, lon1, lat2, lon2) {
    const r = Math.PI / 180, y = Math.sin((lon2 - lon1) * r) * Math.cos(lat2 * r);
    const x = Math.cos(lat1 * r) * Math.sin(lat2 * r) - Math.sin(lat1 * r) * Math.cos(lat2 * r) * Math.cos((lon2 - lon1) * r);
    return (Math.atan2(y, x) / r + 360) % 360;
  }

  const nivel = p => (p >= NIVELES[1] ? "alto" : p >= NIVELES[0] ? "moderado" : "bajo");

  // Aviones normalizados desde el formato readsb/tar1090
  function normalizar(json, ahoraMs) {
    const lista = json.ac || json.aircraft || [];
    const t = json.now ? (json.now > 1e12 ? json.now : json.now * 1000) : ahoraMs;
    return lista.filter(a => a.lat != null && a.lon != null).map(a => ({
      hex: String(a.hex || "").replace("~", ""),
      indicativo: (a.flight || "").trim() || null,
      operador: /^[A-Z]{3}\d/.test((a.flight || "").trim()) ? a.flight.trim().slice(0, 3) : null,
      matricula: a.r || null,
      tipo: a.t || null,
      lat: a.lat, lon: a.lon,
      alt_ft: a.alt_baro === "ground" ? 0 : (a.alt_baro ?? a.alt_geom ?? null),
      en_tierra: a.alt_baro === "ground",
      gs_kt: a.gs ?? null,
      rumbo: a.track ?? a.true_heading ?? null,
      vs_fpm: a.baro_rate ?? a.geom_rate ?? null,
      squawk: a.squawk || null,
      t: t - (a.seen_pos ?? a.seen ?? 0) * 1000,
    }));
  }

  // Ruta (origen-destino) de la respuesta de routeset: tolera los dos formatos que publica el servicio
  function leerRuta(r) {
    if (!r) return null;
    let codigos = [];
    if (Array.isArray(r._airports) && r._airports.length) codigos = r._airports.map(a => a.icao || a.ICAO).filter(Boolean);
    else if (typeof r.airport_codes === "string") codigos = r.airport_codes.split("-").filter(c => /^[A-Z]{4}$/.test(c));
    if (codigos.length < 2) return null;
    return { origen: codigos[0], destino: codigos[codigos.length - 1], escalas: codigos.slice(1, -1), plausible: r.plausible !== false };
  }

  // Si no hay ruta: un avión bajo, descendiendo y apuntando a un aeropuerto cercano de la red probablemente va allí
  function destinoProbable(av, aeropuertos) {
    if (av.en_tierra || av.alt_ft == null || av.alt_ft > 12000 || (av.vs_fpm ?? 0) > -300 || av.rumbo == null) return null;
    let mejor = null;
    for (const a of aeropuertos) {
      const d = distanciaKm(av.lat, av.lon, a.lat, a.lon);
      if (d > 90) continue;
      const dif = Math.abs(((rumbo(av.lat, av.lon, a.lat, a.lon) - av.rumbo + 540) % 360) - 180);
      if (dif < 35 && (!mejor || d < mejor.d)) mejor = { icao: a.icao, d };
    }
    return mejor ? mejor.icao : null;
  }

  function llegadaEstimada(av, destino, ahoraMs) {
    if (!av.gs_kt || av.gs_kt < 60) return null;
    const km = distanciaKm(av.lat, av.lon, destino.lat, destino.lon);
    const min = km / (av.gs_kt * NM_KM) * 60 + (km > 25 ? MIN_APROX : 2);
    return { ms: ahoraMs + min * 60000, km, min };
  }

  // Probabilidad de bajo mínimos a una hora dada, interpolando la serie de NOVOK (y lo observado ahora)
  function riesgoEnHora(aeropuerto, ms, ahoraMs) {
    const serie = (aeropuerto.serie || []).map(s => [Date.parse(s.hora_utc), s.p_bajo_minimos]).filter(s => !isNaN(s[0]));
    if (!serie.length) return null;
    const actual = aeropuerto.bajo_minimos ? 0.9 : aeropuerto.niebla ? 0.4 : null;
    const puntos = (actual != null ? [[ahoraMs, actual]] : []).concat(serie).sort((a, b) => a[0] - b[0]);
    if (ms <= puntos[0][0]) return puntos[0][1];
    for (let k = 1; k < puntos.length; k++) {
      if (ms <= puntos[k][0]) {
        const [t0, p0] = puntos[k - 1], [t1, p1] = puntos[k];
        return p0 + (p1 - p0) * (ms - t0) / (t1 - t0);
      }
    }
    return puntos[puntos.length - 1][1];
  }

  // Alternativo orientativo: aeropuerto de la red cercano al destino con mejor previsión a la llegada
  function alternativo(destino, aeropuertos, ms, ahoraMs) {
    let mejor = null;
    for (const a of aeropuertos) {
      if (a.icao === destino.icao || !a.serie?.length) continue;
      const d = distanciaKm(destino.lat, destino.lon, a.lat, a.lon);
      if (d > 300) continue;
      const p = riesgoEnHora(a, ms + 30 * 60000, ahoraMs);
      if (p == null) continue;
      const nota = p + d / 3000;               // a igualdad de previsión, el más cercano
      if (!mejor || nota < mejor.nota) mejor = { icao: a.icao, nombre: a.nombre, p, km: d, nota };
    }
    return mejor;
  }

  // Coste de desviar un vuelo de su destino al alternativo, con los parámetros de costes.json (cada aerolínea los suyos).
  // Incluye el combustible y la tripulación del tramo extra, la asistencia en el alternativo, llevar a los pasajeros
  // a su destino por carretera, atenderlos mientras esperan y devolver el avión vacío. No incluye compensaciones:
  // la niebla es una circunstancia extraordinaria y no da derecho a ellas (sí a asistencia).
  function costeDesvio(av, destino, alt, c) {
    const tipo = av.tipo && c.consumo_kg_h[av.tipo] ? av.tipo : "_otros";
    const consumo = c.consumo_kg_h[tipo], plazas = c.plazas[tipo] ?? c.plazas._otros;
    const kmAlt = distanciaKm(destino.lat, destino.lon, alt.lat, alt.lon);
    const minTramo = kmAlt / c.velocidad_desvio_kmh * 60 + c.aproximacion_min;   // del destino al alternativo
    const minExtra = c.espera_min + minTramo;                                     // esperar en el destino y desviarse
    const pasajeros = Math.round(plazas * c.ocupacion);
    const horasVuelta = (kmAlt / c.velocidad_desvio_kmh * 60 + c.aproximacion_min) / 60;  // reposicionar el avión
    const d = {
      combustible: (minExtra / 60 + horasVuelta) * consumo * c.combustible_eur_kg,
      tripulacion: (minExtra / 60 + horasVuelta) * c.tripulacion_eur_h,
      handling: c.handling_desvio_eur + c.tasas_alternativo_eur,
      traslado: pasajeros * c.traslado_pasajero_eur,
      atencion: pasajeros * c.atencion_pasajero_eur,
    };
    return { total: Object.values(d).reduce((a, b) => a + b, 0), desglose: d, km_extra: kmAlt, min_extra: minExtra,
             pasajeros, tipo_supuesto: tipo === "_otros" };
  }

  // Vuelos hacia la red, con su llegada estimada y el riesgo en destino a esa hora
  function analizar(aviones, rutas, aeropuertos, ahoraMs, costes = null) {
    const porIcao = Object.fromEntries(aeropuertos.map(a => [a.icao, a]));
    return aviones.map(av => {
      const ruta = av.indicativo ? leerRuta(rutas[av.indicativo]) : null;
      const destinoIcao = ruta && porIcao[ruta.destino] ? ruta.destino : destinoProbable(av, aeropuertos);
      const r = { ...av, ruta, destino: destinoIcao, destino_inferido: !!destinoIcao && !(ruta && porIcao[ruta.destino]) };
      if (!destinoIcao || av.en_tierra) return r;
      const dest = porIcao[destinoIcao];
      const llegada = llegadaEstimada(av, dest, ahoraMs);
      if (!llegada || llegada.min > 16 * 60) return r;
      const p = riesgoEnHora(dest, llegada.ms, ahoraMs);
      const alt = p != null && p >= NIVELES[0] ? alternativo(dest, aeropuertos, llegada.ms, ahoraMs) : null;
      let coste = null;
      if (alt && costes) {
        const dv = costeDesvio(av, dest, porIcao[alt.icao], costes);
        coste = { ...dv, esperado: p * dv.total };      // lo que cuesta, de media, seguir con el plan previsto
      }
      return { ...r, llegada, riesgo: p == null ? null : { p, nivel: nivel(p) }, alternativo: alt, coste };
    });
  }

  // ------------------------------------------------------------------ red (solo navegador)

  async function config(url = "vuelos.json") {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error("sin vuelos.json");
    return r.json();
  }

  // El proveedor puede no aceptar peticiones directas desde otra web (CORS). Entonces se usa el intermediario
  // del propio servidor de NOVOK (python -m novok.servidor), definido en vuelos.json como "*_intermediario".
  let viaIntermediario = false;
  // En local el intermediario es el propio servidor (api/vuelos/...); en la web publicada, el Worker de Cloudflare
  const enLocal = () => typeof location === "undefined" || ["localhost", "127.0.0.1", ""].includes(location.hostname);
  const intermediario = (cfg, u) => (u && !enLocal() && cfg.intermediario_publico
    ? cfg.intermediario_publico.replace(/\/$/, "") + "/" + u.replace(/^\//, "") : u);
  // Un proveedor saturado puede tardar mucho en contestar: mejor fallar pronto y probar el siguiente
  const conPlazo = (o, ms = 12000) => (typeof AbortSignal !== "undefined" && AbortSignal.timeout ? { ...o, signal: AbortSignal.timeout(ms) } : o);
  async function pedir(directa, intermediaria, opciones) {
    if (!viaIntermediario && directa) {
      try {
        const r = await fetch(directa, conPlazo(opciones, 6000));
        if (r.ok) return r;
      } catch (e) {
        if (!intermediaria) throw e;
      }
      if (!intermediaria) throw new Error("proveedor no disponible");
      viaIntermediario = true;
    }
    const r = await fetch(intermediaria, conPlazo(opciones));
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r;
  }

  async function posicionesDe(cfg) {
    const rellenar = u => u && u.replace("{lat}", cfg.centro[0]).replace("{lon}", cfg.centro[1])
      .replace("{radio}", Math.min(cfg.radio_nm, cfg.radio_max_nm || Infinity));
    const r = await pedir(rellenar(cfg.posiciones), intermediario(cfg, rellenar(cfg.posiciones_intermediario)), { cache: "no-store" });
    const json = await r.json();          // un aviso en texto (p. ej. "down for maintenance") no es JSON y falla aquí
    return normalizar(json, Date.now());
  }

  // Proveedor principal y, si falla, los de respaldo de vuelos.json (si los hay), en orden. cfg.atribucion_actual dice cuál se usó.
  let principalCaidoHasta = 0;            // si el principal falla, se deja descansar 2 minutos
  async function posiciones(cfg) {
    try {
      if (Date.now() < principalCaidoHasta && cfg.respaldos?.length) throw new Error("principal en pausa");
      const av = await posicionesDe(cfg);
      cfg.atribucion_actual = cfg.atribucion;
      return av;
    } catch (e) {
      if (Date.now() >= principalCaidoHasta) principalCaidoHasta = Date.now() + 120000;
      let error = e;
      for (const r of cfg.respaldos || []) {          // se prueban en orden hasta que uno responda
        try {
          const av = await posicionesDe({ ...cfg, posiciones: null, ...r });
          cfg.atribucion_actual = r.atribucion;
          return av;
        } catch (e2) {
          error = e2;
        }
      }
      throw error;
    }
  }

  const cacheRutas = {};
  let rutasEnCurso = null;
  const rutasConocidas = () => cacheRutas;
  // Las rutas (origen y destino) se piden aparte y sin bloquear: los aviones se dibujan aunque tarden o fallen
  function rutas(cfg, aviones) {
    if (!rutasEnCurso) rutasEnCurso = pedirRutas(cfg, aviones).finally(() => { rutasEnCurso = null; });
    return rutasEnCurso;
  }
  async function pedirRutas(cfg, aviones) {
    if (!cfg.rutas) return cacheRutas;
    const ahora = Date.now();
    const faltan = aviones.filter(a => a.indicativo && !a.en_tierra && !(a.indicativo in cacheRutas && ahora - cacheRutas[a.indicativo]._t < 30 * 60000));
    for (let k = 0; k < faltan.length; k += 100) {
      const lote = faltan.slice(k, k + 100);
      try {
        const r = await pedir(cfg.rutas, intermediario(cfg, cfg.rutas_intermediario), { method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ planes: lote.map(a => ({ callsign: a.indicativo, lat: a.lat, lng: a.lon })) }) });
        const datos = await r.json();
        for (const x of Array.isArray(datos) ? datos : []) if (x?.callsign) cacheRutas[x.callsign.trim()] = { ...x, _t: ahora };
        for (const a of lote) if (!(a.indicativo in cacheRutas)) cacheRutas[a.indicativo] = { _t: ahora };   // sin ruta conocida
      } catch { break; }
    }
    return cacheRutas;
  }

  async function json(url) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      return r.ok ? r.json() : null;
    } catch {
      return null;
    }
  }

  async function costes(url = "costes.json") {
    try {
      const r = await fetch(url, { cache: "no-store" });
      return r.ok ? r.json() : null;
    } catch {
      return null;
    }
  }

  const api = { distanciaKm, rumbo, nivel, normalizar, leerRuta, destinoProbable, llegadaEstimada, riesgoEnHora,
                alternativo, costeDesvio, analizar, config, posiciones, rutas, rutasConocidas, costes, json };
  global.NovokVuelos = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
