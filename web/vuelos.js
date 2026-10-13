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

  // Vuelos hacia la red, con su llegada estimada y el riesgo en destino a esa hora
  function analizar(aviones, rutas, aeropuertos, ahoraMs) {
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
      return { ...r, llegada, riesgo: p == null ? null : { p, nivel: nivel(p) },
               alternativo: p != null && p >= NIVELES[0] ? alternativo(dest, aeropuertos, llegada.ms, ahoraMs) : null };
    });
  }

  // ------------------------------------------------------------------ red (solo navegador)

  async function config(url = "vuelos.json") {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error("sin vuelos.json");
    return r.json();
  }

  async function posiciones(cfg) {
    const url = cfg.posiciones.replace("{lat}", cfg.centro[0]).replace("{lon}", cfg.centro[1]).replace("{radio}", cfg.radio_nm);
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return normalizar(await r.json(), Date.now());
  }

  const cacheRutas = {};
  async function rutas(cfg, aviones) {
    if (!cfg.rutas) return cacheRutas;
    const ahora = Date.now();
    const faltan = aviones.filter(a => a.indicativo && !a.en_tierra && !(a.indicativo in cacheRutas && ahora - cacheRutas[a.indicativo]._t < 30 * 60000));
    for (let k = 0; k < faltan.length; k += 100) {
      const lote = faltan.slice(k, k + 100);
      try {
        const r = await fetch(cfg.rutas, { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ planes: lote.map(a => ({ callsign: a.indicativo, lat: a.lat, lng: a.lon })) }) });
        if (!r.ok) break;
        const datos = await r.json();
        for (const x of Array.isArray(datos) ? datos : []) if (x?.callsign) cacheRutas[x.callsign.trim()] = { ...x, _t: ahora };
        for (const a of lote) if (!(a.indicativo in cacheRutas)) cacheRutas[a.indicativo] = { _t: ahora };   // sin ruta conocida
      } catch { break; }
    }
    return cacheRutas;
  }

  const api = { distanciaKm, rumbo, nivel, normalizar, leerRuta, destinoProbable, llegadaEstimada, riesgoEnHora,
                alternativo, analizar, config, posiciones, rutas };
  global.NovokVuelos = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
