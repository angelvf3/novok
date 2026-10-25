// Pruebas de la lógica de vuelos (web/vuelos.js). Las lanza tests/test_vuelos.py si hay Node instalado.
const assert = require("assert");
const V = require("../web/vuelos.js");

const ahora = Date.parse("2025-06-30T04:40:00Z");
const red = [
  { icao: "LEAS", nombre: "Asturias", lat: 43.5636, lon: -6.0346, bajo_minimos: true, niebla: true,
    serie: [{ hora_utc: "2025-06-30T06:00Z", p_bajo_minimos: 0.8 }, { hora_utc: "2025-06-30T08:00Z", p_bajo_minimos: 0.5 },
            { hora_utc: "2025-06-30T11:00Z", p_bajo_minimos: 0.3 }, { hora_utc: "2025-06-30T17:00Z", p_bajo_minimos: 0.05 }] },
  { icao: "LEXJ", nombre: "Santander", lat: 43.4271, lon: -3.8200, bajo_minimos: false, niebla: false,
    serie: [{ hora_utc: "2025-06-30T06:00Z", p_bajo_minimos: 0.05 }, { hora_utc: "2025-06-30T17:00Z", p_bajo_minimos: 0.02 }] },
  { icao: "LEBB", nombre: "Bilbao", lat: 43.3011, lon: -2.9106, serie: [{ hora_utc: "2025-06-30T06:00Z", p_bajo_minimos: 0.2 }] },
];

// formato readsb/tar1090
const aviones = V.normalizar({ now: ahora / 1000, ac: [
  { hex: "3443c1", flight: "IBE0532 ", t: "A320", lat: 42.0, lon: -4.5, alt_baro: 31000, gs: 450, track: 330, baro_rate: 0 },
  { hex: "abc123", flight: "VLG1234", lat: 43.40, lon: -5.70, alt_baro: 5000, gs: 200, track: 290, baro_rate: -800 },
  { hex: "ffffff", flight: "AEA9999", lat: 40.4, lon: -3.7, alt_baro: "ground", gs: 0 },
  { hex: "sinpos", flight: "X" },
] }, ahora);
assert.strictEqual(aviones.length, 3);
assert.strictEqual(aviones[0].indicativo, "IBE0532");
assert.ok(aviones[2].en_tierra);

// distancia y rumbo
assert.ok(Math.abs(V.distanciaKm(43.5636, -6.0346, 43.4271, -3.82) - 179) < 3);
assert.ok(Math.abs(V.rumbo(43, -6, 44, -6) - 0) < 0.5);

// rutas en sus dos formatos
assert.deepStrictEqual(V.leerRuta({ airport_codes: "LEMD-LEAS" }).destino, "LEAS");
assert.strictEqual(V.leerRuta({ _airports: [{ icao: "LEBL" }, { icao: "LEXJ" }] }).origen, "LEBL");
assert.strictEqual(V.leerRuta({ airport_codes: "unknown" }), null);

// riesgo interpolado: ahora observado 0.9, a las 07:00 entre 0.8 (06:00) y 0.5 (08:00)
assert.ok(Math.abs(V.riesgoEnHora(red[0], Date.parse("2025-06-30T07:00Z"), ahora) - 0.65) < 1e-9);
assert.strictEqual(V.riesgoEnHora(red[0], Date.parse("2025-07-01T00:00Z"), ahora), 0.05);
assert.ok(V.riesgoEnHora(red[0], ahora, ahora) === 0.9);

// análisis: el IBE con ruta a LEAS; el VLG sin ruta, bajo y descendiendo hacia LEAS
const res = V.analizar(aviones, { IBE0532: { airport_codes: "LEMD-LEAS" } }, red, ahora);
const ibe = res.find(a => a.indicativo === "IBE0532"), vlg = res.find(a => a.indicativo === "VLG1234");
assert.strictEqual(ibe.destino, "LEAS"); assert.ok(!ibe.destino_inferido);
assert.ok(ibe.llegada.min > 15 && ibe.llegada.min < 40, ibe.llegada.min);
assert.strictEqual(ibe.riesgo.nivel, "alto");
assert.strictEqual(ibe.alternativo.icao, "LEXJ");                      // mejor previsión y cerca
assert.strictEqual(vlg.destino, "LEAS"); assert.ok(vlg.destino_inferido);
assert.strictEqual(res.find(a => a.hex === "ffffff").llegada, undefined);   // en tierra: sin llegada
console.log("vuelos.js: todo bien");
