"""Búsqueda en rejilla sobre train y walk-forward anclado dentro de train."""
from itertools import product

import numpy as np
import pandas as pd

import config
from backtest.engine import run_backtest
from performance.metrics import daily_returns, sharpe
from strategy.model import FilterParams, fit_signals
from strategy.signals import RuleParams, directions

MIN_TRADES = 10
MIN_TRADES_FOLD = 3
DEFAULT_RULE = dict(z_entry=2.0, z_exit=0.5, z_stop=4.0, hold=3.0)


def _rule(z_entry, z_exit, z_stop, hold, half_life_days):
    """Convierte el múltiplo de vida media en una tenencia máxima en días."""
    return RuleParams(z_entry, z_exit, z_stop, int(round(hold * half_life_days)))


def score(prices, signals, rule, start, end):
    """Sharpe y número de operaciones de una regla en [start, end)."""
    d = directions(signals.f2.zscore[start:end], rule, 0)
    result = run_backtest(prices[start:end], signals.f1.beta_post[start:end], d)
    return sharpe(daily_returns(result.equity)), len(result.trades)


def search_filters(log_prices, prices, beta, k, train_end):
    """Etapa 1: elige (Q1/R1, Q2/R2) por Sharpe de train con la regla base."""
    n_train = len(log_prices.loc[:train_end])
    rows = []
    for q1, q2 in product(config.Q_RATIO_F1_GRID, config.Q_RATIO_F2_GRID):
        params = FilterParams(q1, q2)
        signals = fit_signals(log_prices, beta, k, train_end, params)
        rule = _rule(**DEFAULT_RULE, half_life_days=signals.half_life)
        value, trades = score(prices, signals, rule, config.BURN_IN, n_train)
        rows.append({"q_ratio_f1": q1, "q_ratio_f2": q2, "sharpe_train": value, "operaciones": trades})
    table = pd.DataFrame(rows)
    valid = table[table["operaciones"] >= MIN_TRADES]
    best = (valid if len(valid) else table).sort_values("sharpe_train", ascending=False).iloc[0]
    return FilterParams(float(best["q_ratio_f1"]), float(best["q_ratio_f2"])), table


def _rule_grid(half_life_days):
    """Todas las combinaciones de umbrales de la rejilla."""
    grid = product(config.Z_ENTRY_GRID, config.Z_EXIT_GRID, config.Z_STOP_GRID,
                   config.HOLD_MULTIPLE_GRID)
    return [(e, x, s, h, _rule(e, x, s, h, half_life_days)) for e, x, s, h in grid]


def search_rule(prices, signals, n_train):
    """Etapa 2: elige los umbrales por Sharpe de train."""
    rows = []
    for entry, exit_, stop, hold, rule in _rule_grid(signals.half_life):
        value, trades = score(prices, signals, rule, config.BURN_IN, n_train)
        rows.append({"z_entrada": entry, "z_salida": exit_, "z_stop": stop,
                     "multiplo_vida_media": hold, "h_max": rule.h_max,
                     "sharpe_train": value, "operaciones": trades})
    table = pd.DataFrame(rows)
    valid = table[table["operaciones"] >= MIN_TRADES]
    best = (valid if len(valid) else table).sort_values("sharpe_train", ascending=False).iloc[0]
    rule = RuleParams(float(best["z_entrada"]), float(best["z_salida"]),
                      float(best["z_stop"]), int(best["h_max"]))
    return rule, table


def walk_forward(prices, signals, n_train, folds=3):
    """Anclado en train: elige umbrales con el pasado y evalúa en el bloque siguiente."""
    edges = np.linspace(config.BURN_IN, n_train, folds + 2, dtype=int)
    rows = []
    for fold in range(1, folds + 1):
        split, stop = edges[fold], edges[fold + 1]
        best_value, best_rule = -np.inf, None
        for *_, rule in _rule_grid(signals.half_life):
            value, trades = score(prices, signals, rule, config.BURN_IN, split)
            if trades >= MIN_TRADES_FOLD and value > best_value:
                best_value, best_rule = value, rule
        if best_rule is None:
            rows.append({"fold": fold, "sharpe_ajuste": float("nan"), "sharpe_fuera": float("nan"),
                         "operaciones_fuera": 0})
            continue
        value, trades = score(prices, signals, best_rule, split, stop)
        rows.append({"fold": fold, "sharpe_ajuste": best_value, "sharpe_fuera": value,
                     "operaciones_fuera": trades, "z_entrada": best_rule.z_entry,
                     "z_salida": best_rule.z_exit, "z_stop": best_rule.z_stop,
                     "h_max": best_rule.h_max})
    return pd.DataFrame(rows)
