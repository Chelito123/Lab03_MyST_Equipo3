"""Motor de backtest multi-pata con comisión por pata y préstamo diario."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass
class BacktestResult:
    """Series diarias, bitácora de operaciones y totales de costos."""

    equity: np.ndarray
    cash: np.ndarray
    shares: np.ndarray
    commissions: np.ndarray
    borrow: np.ndarray
    gross_pnl: np.ndarray
    trades: pd.DataFrame


def target_shares(direction, beta, value, prices):
    """q_i = d·sgn(β_i)·N_i/P_i con N_i = 0.8·V·|β_i|/Σ|β_j| (S14 p.20)."""
    weights = np.abs(beta) / np.abs(beta).sum()
    notional = config.POSITION_FRACTION * value * weights
    return direction * np.sign(beta) * notional / prices


def _apply_band(current, target, band):
    """Mantiene las patas cuyo desvío relativo contra el objetivo está dentro de la banda."""
    keep = np.abs(target - current) <= band * np.abs(target)
    return np.where(keep, current, target)


def run_backtest(prices, betas, direction, dates=None, band=None):
    """Simula al cierre de t+1 la decisión tomada al cierre de t.

    `band=None` rebalancea a diario; con `band` usa la política de no-rebalanceo.
    Toda posición se cierra en la última fecha.
    """
    p = np.asarray(prices, float)
    T, n = p.shape
    d = np.array(direction, dtype=int)
    d[-2:] = 0
    cash = np.full(T, config.INITIAL_CAPITAL)
    shares, equity = np.zeros((T, n)), np.full(T, config.INITIAL_CAPITAL)
    commissions, borrow, gross = np.zeros(T), np.zeros(T), np.zeros(T)
    trades, open_trade = [], None
    for t in range(1, T):
        previous = shares[t - 1]
        gross[t] = float(previous @ (p[t] - p[t - 1]))
        borrow[t] = float(np.abs(np.minimum(previous, 0)) @ p[t]) * config.BORROW_RATE / config.TRADING_DAYS
        value_before = cash[t - 1] + float(previous @ p[t]) - borrow[t]
        signal, last_signal = d[t - 1], d[t - 2] if t > 1 else 0
        target = target_shares(signal, betas[t - 1], value_before, p[t])
        if band is not None and signal != 0 and signal == last_signal:
            target = _apply_band(previous, target, band)
        traded = target - previous
        commissions[t] = float(np.abs(traded) @ p[t]) * config.COMMISSION
        cash[t] = cash[t - 1] - borrow[t] - float(traded @ p[t]) - commissions[t]
        shares[t] = target
        equity[t] = cash[t] + float(target @ p[t])
        if open_trade is not None:
            open_trade["gross"] += gross[t]
            open_trade["costs_comm"] += commissions[t]
            open_trade["costs_borrow"] += borrow[t]
        if signal != 0 and open_trade is None:
            open_trade = {"entry": t, "side": int(signal), "gross": 0.0,
                          "costs_comm": commissions[t], "costs_borrow": 0.0,
                          "entry_value": value_before}
        elif signal == 0 and open_trade is not None:
            trades.append({**open_trade, "exit": t})
            open_trade = None
    return BacktestResult(equity, cash, shares, commissions, borrow, gross,
                          _trade_log(trades, dates, T))


def _trade_log(trades, dates, T):
    """Convierte las operaciones en tabla con P&L neto y retorno."""
    columns = ["entrada", "salida", "direccion", "dias", "pnl_bruto", "comisiones",
               "prestamo", "pnl_neto", "retorno", "resultado"]
    if not trades:
        return pd.DataFrame(columns=columns)
    index = dates if dates is not None else pd.RangeIndex(T)
    rows = []
    for item in trades:
        net = item["gross"] - item["costs_comm"] - item["costs_borrow"]
        rows.append({
            "entrada": index[item["entry"]], "salida": index[item["exit"]],
            "direccion": "largo" if item["side"] > 0 else "corto",
            "dias": item["exit"] - item["entry"], "pnl_bruto": item["gross"],
            "comisiones": item["costs_comm"], "prestamo": item["costs_borrow"],
            "pnl_neto": net, "retorno": net / item["entry_value"],
            "resultado": "ganadora" if net > 0 else "perdedora",
        })
    return pd.DataFrame(rows, columns=columns)
