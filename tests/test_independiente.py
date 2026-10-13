import numpy as np
import pandas as pd

from novok.verificacion.independiente import ahorro_por_100, inicios, valor_relativo


def test_valor_perfecto_y_sin_prevision():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], bool)
    assert valor_relativo(y, y, 0.2) == 1.0
    # no actuar nunca (alfa > frecuencia) equivale a no tener previsión
    assert valor_relativo(y, np.zeros_like(y), 0.5) == 0.0
    # actuar siempre cuando alfa < frecuencia también es la estrategia sin previsión
    assert abs(valor_relativo(y, np.ones_like(y), 0.1)) < 1e-12


def test_ahorro_por_100_llegadas():
    y = np.array([1] + [0] * 9, bool)            # 1 de cada 10 llegadas con bajo mínimos
    # sin previsión: min(C=100, 0.1 * 3000 = 300) = 100 EUR por llegada
    assert ahorro_por_100(y, y, 100, 3000) == 100 * (100 - 10)
    assert ahorro_por_100(y, np.zeros_like(y), 100, 3000) == 100 * (100 - 300)


def test_inicios_de_episodio():
    idx = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    y = pd.Series([0, 1, 1, 0, np.nan, 1, 0, 1], index=idx, dtype=float)
    # el de las 05 sigue a un hueco: no se cuenta (no se sabe si empezó ahí)
    assert list(inicios(y)) == [idx[1], idx[7]]
