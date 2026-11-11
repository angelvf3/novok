// NOVOK · intermediario de vuelos para la web publicada (Cloudflare Workers, plan gratuito).
//
// La web publicada en GitHub Pages es solo de ficheros: no puede pedir los vuelos a los proveedores que no aceptan
// peticiones desde otra web (CORS). Este Worker hace lo mismo que python -m novok.servidor en local: reenvía
// /api/vuelos/<proveedor>/<ruta> solo a los proveedores y rutas permitidos (no es un proxy abierto), guarda las
// posiciones 10 s para no repetir peticiones y solo responde a las webs de ORIGENES.
//
// Los proveedores comunitarios limitan las peticiones (HTTP 429) y los Workers comparten direcciones IP con otros
// usuarios de Cloudflare, así que a veces el límite salta aunque NOVOK pida poco. Para que la web no se quede sin
// aviones: si el proveedor limita o falla, se devuelve la última respuesta buena (hasta 5 min) marcada con la
// cabecera X-Novok-Copia (segundos de antigüedad), y tras un 429 se deja descansar al proveedor 30 s.
//
// Despliegue: ver LEEME.md de esta carpeta.

const PROVEEDORES = {
  adsblol: ["https://api.adsb.lol/", ["v2/point/", "api/0/routeset"]],
  adsbfi: ["https://opendata.adsb.fi/api/", ["v3/lat/"]],   // respaldo de pruebas (uso no comercial)
};
const CACHE_S = 10;
const COPIA_MAX_S = 300;
const PAUSA_429_S = 30;
const copias = new Map();          // última respuesta buena por dirección (memoria del Worker)
const pausas = new Map();          // proveedor -> hasta cuándo no se le pide nada tras un 429
const AGENTE = "novok (https://github.com/angelvf3/novok)";

export function destino(ruta) {
  const m = ruta.match(/^\/api\/vuelos\/([a-z]+)\/(.+)$/);
  if (!m || !(m[1] in PROVEEDORES) || m[2].includes("..")) return null;
  const [base, permitidas] = PROVEEDORES[m[1]];
  return permitidas.some(p => m[2].startsWith(p)) ? base + m[2] : null;
}

export default {
  async fetch(peticion, env = {}, ctx = { waitUntil() {} }) {
    const origenes = (env.ORIGENES || "https://angelvf3.github.io,http://localhost:8010").split(",").map(s => s.trim());
    const origen = peticion.headers.get("Origin") || "";
    const cors = {
      "Access-Control-Allow-Origin": origenes.includes(origen) ? origen : origenes[0],
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Vary": "Origin",
    };
    if (peticion.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
    if (origen && !origenes.includes(origen)) return new Response('{"error":"origen no permitido"}', { status: 403, headers: cors });
    const url = new URL(peticion.url);
    const arriba = destino(url.pathname);
    if (!arriba || !["GET", "POST"].includes(peticion.method)) {
      return new Response('{"error":"ruta no permitida"}', { status: 404, headers: { ...cors, "Content-Type": "application/json" } });
    }
    const cache = typeof caches !== "undefined" ? caches.default : null;
    const clave = new Request(arriba);
    if (peticion.method === "GET" && cache) {
      const guardada = await cache.match(clave);
      if (guardada) return new Response(guardada.body, { status: guardada.status, headers: { ...cors, "Content-Type": "application/json" } });
    }
    const proveedor = new URL(arriba).host;
    const copia = () => {
      const c = peticion.method === "GET" ? copias.get(arriba) : null;
      if (!c || Date.now() - c.t > COPIA_MAX_S * 1000) return null;
      return new Response(c.cuerpo, { status: 200, headers: { ...cors, "Content-Type": "application/json", "Cache-Control": "no-store",
        "X-Novok-Copia": String(Math.round((Date.now() - c.t) / 1000)), "Access-Control-Expose-Headers": "X-Novok-Copia" } });
    };
    if ((pausas.get(proveedor) || 0) > Date.now()) {
      return copia() || new Response('{"error":"proveedor limitando peticiones"}', { status: 429, headers: { ...cors, "Content-Type": "application/json" } });
    }
    let respuesta;
    try {
      respuesta = await fetch(arriba, {
        method: peticion.method,
        headers: { "User-Agent": AGENTE, ...(peticion.method === "POST" ? { "Content-Type": "application/json" } : {}) },
        body: peticion.method === "POST" ? await peticion.text() : undefined,
        signal: AbortSignal.timeout(8000),
      });
    } catch (e) {
      return copia() || new Response(JSON.stringify({ error: e.name }), { status: 502, headers: { ...cors, "Content-Type": "application/json" } });
    }
    if (respuesta.status === 429) pausas.set(proveedor, Date.now() + PAUSA_429_S * 1000);
    if (!respuesta.ok) {
      const c = copia();
      if (c) return c;
    }
    const cuerpo = await respuesta.arrayBuffer();
    if (peticion.method === "GET" && respuesta.status === 200) copias.set(arriba, { cuerpo, t: Date.now() });
    if (peticion.method === "GET" && cache && respuesta.status === 200) {
      ctx.waitUntil(cache.put(clave, new Response(cuerpo, { headers: { "Cache-Control": `max-age=${CACHE_S}`, "Content-Type": "application/json" } })));
    }
    return new Response(cuerpo, { status: respuesta.status, headers: { ...cors, "Content-Type": "application/json", "Cache-Control": "no-store" } });
  },
};
