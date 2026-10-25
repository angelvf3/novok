from novok.servidor import destino


def test_solo_reenvia_rutas_de_los_proveedores():
    assert destino("/api/vuelos/adsblol/v2/point/43.5/-6/50") == "https://api.adsb.lol/v2/point/43.5/-6/50"
    assert destino("/api/vuelos/adsblol/api/0/routeset") == "https://api.adsb.lol/api/0/routeset"
    assert destino("/api/vuelos/adsbfi/v3/lat/43.5/lon/-6/dist/50") == \
        "https://opendata.adsb.fi/api/v3/lat/43.5/lon/-6/dist/50"
    assert destino("/api/vuelos/adsbfi/api/0/routeset") is None
    assert destino("/api/vuelos/adsblol/v2/../../otra") is None
    assert destino("/api/vuelos/otro/v2/point/1/2/3") is None
    assert destino("/index.html") is None


def test_sirve_trozos_de_fichero(tmp_path):
    import functools
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from novok.servidor import Manejador

    (tmp_path / "mapa.pmtiles").write_bytes(bytes(range(256)))
    s = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Manejador, directory=str(tmp_path)))
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{s.server_address[1]}/mapa.pmtiles"
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=10-19"}))
        assert r.status == 206 and r.read() == bytes(range(10, 20))
        assert r.headers["Content-Range"] == "bytes 10-19/256"
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=-6"}))
        assert r.read() == bytes(range(250, 256))
    finally:
        s.shutdown()
