"""Compara ADF y Johansen contra los resultados de referencia de statsmodels."""
from pathlib import Path

import numpy as np
import pytest

from cointegration.stats import _adf_design, _ols, half_life, johansen
from cointegration.vendor.adfvalues import mackinnonp
from strategy.signals import RuleParams, directions

FIXTURES = Path(__file__).parent / "fixtures"
TRACE_REFERENCE = [241.985452556075, 166.4781461662553, 110.3298006342814, 70.79801574443575,
                   44.90887371527634, 27.22385073668511, 11.74205493173769, 3.295435325623445]
MAX_REFERENCE = [75.50730638981975, 56.14834553197396, 39.5317848898456, 25.8891420291594,
                 17.68502297859124, 15.48179580494741, 8.446619606114249, 3.295435325623445]


def test_johansen_coincide_con_referencia():
    data = np.genfromtxt(FIXTURES / "johansen_reference.csv")
    result = johansen(data, 1, 2)
    assert result.trace_stat == pytest.approx(TRACE_REFERENCE, abs=1e-8)
    assert result.max_stat == pytest.approx(MAX_REFERENCE, abs=1e-8)


def test_adf_coincide_con_referencia():
    gdp = np.loadtxt(FIXTURES / "realgdp.csv")
    y, design = _adf_design(gdp, 4, True)
    beta, _, se = _ols(y, design)
    stat = beta[0] / se[0]
    assert stat == pytest.approx(0.97505319, abs=1e-4)
    assert mackinnonp(stat, "c", 1) == pytest.approx(0.99399563, abs=1e-5)


def test_vida_media_de_ar1_conocido():
    rng = np.random.default_rng(42)
    phi, s = 0.95, np.zeros(20000)
    for t in range(1, len(s)):
        s[t] = phi * s[t - 1] + rng.normal()
    assert half_life(s) == pytest.approx(-np.log(2) / (phi - 1), rel=0.1)


def test_regla_no_reabre_tras_stop_hasta_volver_a_salida():
    rule = RuleParams(z_entry=2.0, z_exit=0.5, z_stop=4.0, h_max=100)
    z = np.array([0, 2.5, 3.0, 4.5, 3.0, 2.5, 0.2, 2.5])
    assert directions(z, rule, 0).tolist() == [0, -1, -1, 0, 0, 0, 0, -1]


def test_regla_respeta_tenencia_maxima():
    rule = RuleParams(z_entry=2.0, z_exit=0.5, z_stop=4.0, h_max=2)
    z = np.array([2.5, 2.5, 2.5, 2.5])
    assert directions(z, rule, 0).tolist() == [-1, -1, 0, 0]


def test_johansen_eigenvectores_coinciden_con_referencia():
    data = np.genfromtxt(FIXTURES / "johansen_reference.csv")
    reference = np.loadtxt(FIXTURES / "johansen_evec_reference.txt")
    vectors = johansen(data, 1, 2).eigenvectors
    for column in range(vectors.shape[1]):
        sign = np.sign(vectors[:, column] @ reference[:, column])
        assert vectors[:, column] == pytest.approx(sign * reference[:, column], abs=1e-8)
