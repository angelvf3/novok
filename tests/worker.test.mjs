// Pruebas del intermediario de Cloudflare (infra/vuelos-worker/worker.js) con fetch simulado.
import assert from "assert";
import worker, { destino } from "../infra/vuelos-worker/worker.js";

assert.strictEqual(destino("/api/vuelos/adsblol/v2/point/43.5/-6/100"), "https://api.adsb.lol/v2/point/43.5/-6/100");
assert.strictEqual(destino("/api/vuelos/adsbfi/v3/lat/1/lon/2/dist/3"), "https://opendata.adsb.fi/api/v3/lat/1/lon/2/dist/3");
assert.strictEqual(destino("/api/vuelos/adsblol/v2/../../secreto"), null);
assert.strictEqual(destino("/api/vuelos/otro/v2/point/1/2/3"), null);

const pedidas = [];
globalThis.fetch = async (url, op) => { pedidas.push([url, op.method]); return new Response('{"ac":[]}', { status: 200 }); };
const env = { ORIGENES: "https://angelvf3.github.io" };
const ok = await worker.fetch(new Request("https://w.dev/api/vuelos/adsblol/v2/point/43.5/-6/100", { headers: { Origin: "https://angelvf3.github.io" } }), env);
assert.strictEqual(ok.status, 200);
assert.strictEqual(ok.headers.get("Access-Control-Allow-Origin"), "https://angelvf3.github.io");
assert.deepStrictEqual(await ok.json(), { ac: [] });
const fuera = await worker.fetch(new Request("https://w.dev/api/vuelos/adsblol/v2/point/1/2/3", { headers: { Origin: "https://otra.web" } }), env);
assert.strictEqual(fuera.status, 403);
const mala = await worker.fetch(new Request("https://w.dev/cualquier/cosa"), env);
assert.strictEqual(mala.status, 404);
const pre = await worker.fetch(new Request("https://w.dev/api/vuelos/adsblol/api/0/routeset", { method: "OPTIONS", headers: { Origin: "https://angelvf3.github.io" } }), env);
assert.strictEqual(pre.status, 204);
const post = await worker.fetch(new Request("https://w.dev/api/vuelos/adsblol/api/0/routeset", { method: "POST", body: '{"planes":[]}', headers: { Origin: "https://angelvf3.github.io" } }), env);
assert.strictEqual(post.status, 200);
assert.deepStrictEqual(pedidas.map(p => p[1]), ["GET", "POST"]);
console.log("worker.js: todo bien");
