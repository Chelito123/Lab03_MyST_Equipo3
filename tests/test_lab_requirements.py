"""Las cuatro pruebas obligatorias del Lab03 (sección 3.11)."""
import numpy as np
import pytest

import config
from backtest.engine import run_backtest
from cointegration.selection import normalize_vector
from kalman.filters import filter_mean, filter_weights, innovation_diagnostics
from strategy.model import fit_signals

ACF_LAG1_TOLERANCE = 0.05


@pytest.mark.parametrize("offset", [20, 300, 900])
def test_causalidad_zscore_no_cambia_al_truncar(market, offset):
    """1. z_t recalculado con datos solo hasta t es idéntico al de la muestra completa."""
    log = market["log"]
    position = log.index.get_loc(market["train_end"]) + offset
    full = fit_signals(log, market["beta"], market["k"], market["train_end"], market["params"])
    cut = fit_signals(log.iloc[:position + 1], market["beta"], market["k"],
                      market["train_end"], market["params"])
    assert full.f2.zscore[position] == pytest.approx(cut.f2.zscore[position], abs=1e-12)
    assert full.f1.spread[position] == pytest.approx(cut.f1.spread[position], abs=1e-12)


def test_normalizacion_invariante_a_escala_y_signo():
    """2a. Multiplicar el vector por cualquier c != 0 da el mismo β normalizado."""
    vector = np.array([0.3, -1.2, 0.5, 0.1])
    order, reference = normalize_vector(vector)
    for scale in (-3.0, -0.5, 7.0):
        new_order, beta = normalize_vector(vector * scale)
        assert new_order == order
        assert beta == pytest.approx(reference)
    assert reference[0] == 1.0


def test_normalizacion_estable_entre_reestimaciones(market):
    """2b. La pata líder sigue en +1 y β no salta entre días consecutivos."""
    signals = fit_signals(market["log"], market["beta"], market["k"], market["train_end"],
                          market["params"])
    betas = signals.f1.beta_post
    assert np.all(betas[:, 0] == 1.0)
    assert np.abs(np.diff(betas, axis=0)).max() < 0.05
    dominant = np.abs(market["beta"]) > 0.2
    assert np.all(np.sign(betas[:, dominant]) == np.sign(market["beta"][dominant]))


def _synthetic_book(seed=config.SEED):
    """Precios, pesos y direcciones sintéticas con entradas, salidas y rebalanceo."""
    rng = np.random.default_rng(seed)
    T, n = 300, 4
    prices = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, n)), axis=0))
    betas = np.tile([1.0, -0.6, 0.4, -0.2], (T, 1)) + rng.normal(0, 0.02, (T, n))
    betas[:, 0] = 1.0
    direction = np.zeros(T, dtype=int)
    direction[20:60], direction[100:170], direction[200:240] = 1, -1, 1
    return prices, betas, direction


def test_contabilidad_multipata():
    """3. Valor = efectivo + patas; costos = comisión por pata + préstamo en cortos."""
    prices, betas, direction = _synthetic_book()
    result = run_backtest(prices, betas, direction)
    assert result.equity == pytest.approx(result.cash + (result.shares * prices).sum(axis=1))
    expected_comm = np.zeros(len(prices))
    expected_comm[1:] = (np.abs(np.diff(result.shares, axis=0)) * prices[1:]).sum(axis=1) * config.COMMISSION
    assert result.commissions == pytest.approx(expected_comm)
    expected_borrow = np.zeros(len(prices))
    shorts = np.abs(np.minimum(result.shares[:-1], 0))
    expected_borrow[1:] = (shorts * prices[1:]).sum(axis=1) * config.BORROW_RATE / config.TRADING_DAYS
    assert result.borrow == pytest.approx(expected_borrow)
    pnl = result.gross_pnl.sum() - result.commissions.sum() - result.borrow.sum()
    assert result.equity[-1] - config.INITIAL_CAPITAL == pytest.approx(pnl)
    assert np.all(result.shares[-1] == 0)
    assert result.trades["comisiones"].sum() == pytest.approx(result.commissions.sum())
    assert result.trades["pnl_neto"].sum() == pytest.approx(pnl)


def test_exposicion_y_ejecucion_al_dia_siguiente():
    """La señal de t se ejecuta en t+1 y el nocional bruto es 80 % del capital."""
    prices, betas, direction = _synthetic_book()
    result = run_backtest(prices, betas, direction)
    assert np.all(result.shares[20] == 0)
    gross = np.abs(result.shares[21] * prices[21]).sum()
    assert gross == pytest.approx(config.POSITION_FRACTION * result.equity[21], rel=0.01)
    assert np.all(np.sign(result.shares[21]) == np.sign(betas[20]))


def test_filtro_1_innovaciones_blancas_con_datos_del_modelo():
    """4a. Con datos generados por el modelo, innovaciones de varianza 1 y sin ACF."""
    rng = np.random.default_rng(config.SEED)
    T, n = 4000, 3
    x = np.column_stack([np.zeros(T), np.cumsum(rng.normal(0, 0.05, (T, n - 1)), axis=0) + 3])
    q, r = 1e-6, 1e-3
    z = np.cumsum(rng.normal(0, np.sqrt(q), (T, n)), axis=0) + np.array([0.5, 0.4, 0.3])
    for t in range(T):
        x[t, 0] = z[t, 0] - x[t, 1:] @ z[t, 1:] + rng.normal(0, np.sqrt(r))
    out = filter_weights(x, np.array([0.5, 0.4, 0.3]), np.eye(n) * 1e-4, np.eye(n) * q, r)
    stats = innovation_diagnostics((out.spread / np.sqrt(out.innovation_var))[100:])
    assert stats["varianza"] == pytest.approx(1.0, abs=0.1)
    assert abs(stats["acf1"]) < ACF_LAG1_TOLERANCE


def test_filtro_2_innovaciones_blancas_con_datos_del_modelo():
    """4b. Igual para el modelo de nivel local del filtro 2."""
    rng = np.random.default_rng(config.SEED)
    T, q, r = 4000, 1e-5, 1e-3
    mu = np.cumsum(rng.normal(0, np.sqrt(q), T))
    spread = mu + rng.normal(0, np.sqrt(r), T)
    out = filter_mean(spread, 0.0, r, q, r)
    stats = innovation_diagnostics(out.zscore[100:])
    assert stats["varianza"] == pytest.approx(1.0, abs=0.1)
    assert abs(stats["acf1"]) < ACF_LAG1_TOLERANCE


def test_filtro_varianza_cercana_a_uno_en_datos_reales(market):
    """4c. En train, varianza de innovaciones normalizadas dentro de [0.5, 2].

    La autocorrelación real no se exige aquí: un spread que revierte lentamente
    está autocorrelacionado por construcción, y esa persistencia es la señal que
    se opera. Su valor se reporta en artifacts/diagnosticos.json.
    """
    signals = fit_signals(market["log"], market["beta"], market["k"], market["train_end"],
                          market["params"])
    n_train = market["log"].index.get_loc(market["train_end"]) + 1
    begin = config.BURN_IN
    normalized = signals.f1.spread[begin:n_train] / np.sqrt(signals.f1.innovation_var[begin:n_train])
    assert 0.5 <= innovation_diagnostics(normalized)["varianza"] <= 2.0
    assert 0.5 <= innovation_diagnostics(signals.f2.zscore[begin:n_train])["varianza"] <= 2.0
