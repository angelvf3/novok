import time

from novok import alertas


def red(p, bajo=False):
    return {"modo": "directo", "aeropuertos": [{
        "icao": "LEAS", "nombre": "Asturias", "lat": 43.56, "lon": -6.03, "emitido": True, "categoria": "VFR",
        "bajo_minimos": bajo, "niebla": bajo, "riesgo": {"p": p, "nivel": "alto"},
        "serie": [{"hora_utc": "2026-10-26T05:00Z", "p_bajo_minimos": p},
                  {"hora_utc": "2026-10-26T08:00Z", "p_bajo_minimos": p / 2}]}]}


def test_ciclo_de_un_aviso():
    c, estado, t = dict(alertas.CONFIG_DEFECTO), {}, time.time()
    assert [e["tipo"] for e in alertas.eventos(red(0.45), estado, c, t)] == ["AVISO"]
    assert alertas.eventos(red(0.50), estado, c, t + 600) == []               # sin cambios: no se repite
    assert [e["tipo"] for e in alertas.eventos(red(0.70), estado, c, t + 1200)] == ["SUBE"]
    assert [e["tipo"] for e in alertas.eventos(red(0.70), estado, c, t + 1200 + 3 * 3600)] == ["SIGUE"]
    assert alertas.eventos(red(0.20), estado, c, t + 4 * 3600 + 1300) == []   # entre umbrales: sigue abierto
    assert [e["tipo"] for e in alertas.eventos(red(0.05), estado, c, t + 5 * 3600)] == ["FIN"]
    assert estado == {}


def test_bajo_minimos_ahora_no_cierra_el_aviso():
    c, estado, t = dict(alertas.CONFIG_DEFECTO), {}, time.time()
    alertas.eventos(red(0.45), estado, c, t)
    assert alertas.eventos(red(0.05, bajo=True), estado, c, t + 600) == []
    assert "LEAS" in estado


def test_mensaje_con_vuelos_de_una_aerolinea():
    c = {**alertas.CONFIG_DEFECTO, "aerolinea": "VOE"}
    ev = alertas.eventos(red(0.45), {}, c, time.time())
    t = time.time()
    vuelos = {"LEAS": [
        {"indicativo": "VOE1234", "operador": "VOE", "llegada": t + 3600, "p": 0.42, "coste": 4200.0},
        {"indicativo": "IBE0532", "operador": "IBE", "llegada": t + 1800, "p": 0.40, "coste": 4000.0}]}
    texto = alertas.mensaje(ev, c, vuelos, {"VOE": "Volotea"})
    assert "Aviso de bajo mínimos en LEAS Asturias: 45 %" in texto
    assert "Vuelos de Volotea en el aire hacia LEAS: 1" in texto and "IBE0532" not in texto
    assert "4.200 €" in texto and "aeropuerto.html?a=LEAS" in texto


def test_riesgo_en_hora_interpola():
    a = red(0.6)["aeropuertos"][0]
    t0 = alertas.datetime.fromisoformat("2026-10-26T05:00+00:00").timestamp()
    assert abs(alertas.riesgo_en_hora(a, t0 + 1.5 * 3600, t0 - 3600) - 0.45) < 1e-9


def test_coste_igual_que_en_la_web():
    import json
    c = json.loads((alertas.RAIZ / "web" / "costes.json").read_text(encoding="utf-8"))
    leas, lexj = {"lat": 43.5636, "lon": -6.0346}, {"lat": 43.4271, "lon": -3.82}
    assert 9000 < alertas.coste_desvio("A320", leas, lexj, c) < 12000
