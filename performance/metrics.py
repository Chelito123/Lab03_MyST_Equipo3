"""Métricas de desempeño. La tasa libre de riesgo se toma como 0."""
import numpy as np
import pandas as pd

import config


def daily_returns(equity):
    """Retornos simples diarios de la curva de capital."""
    return pd.Series(equity).pct_change().dropna().values


def sharpe(returns):
    """Sharpe anualizado; 0 si no hay variación."""
    r = np.asarray(returns, float)
    return float(r.mean() / r.std() * np.sqrt(config.TRADING_DAYS)) if r.std() > 0 else 0.0


def sortino(returns):
    """Sortino anualizado con desviación a la baja respecto de cero."""
    r = np.asarray(returns, float)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    return float(r.mean() / downside * np.sqrt(config.TRADING_DAYS)) if downside > 0 else 0.0


def drawdown(equity):
    """Serie de drawdown (valores negativos o cero)."""
    e = np.asarray(equity, float)
    return e / np.maximum.accumulate(e) - 1


def annual_return(equity):
    """Retorno anualizado compuesto."""
    e = np.asarray(equity, float)
    return float((e[-1] / e[0]) ** (config.TRADING_DAYS / (len(e) - 1)) - 1)


def summarize(equity, trades):
    """Tarjeta de desempeño del tramo: Sharpe, Sortino, Calmar, MDD, Win Rate, PF."""
    returns = daily_returns(equity)
    max_dd = float(drawdown(equity).min())
    annual = annual_return(equity)
    wins = trades.loc[trades["pnl_neto"] > 0, "pnl_neto"].sum() if len(trades) else 0.0
    losses = -trades.loc[trades["pnl_neto"] <= 0, "pnl_neto"].sum() if len(trades) else 0.0
    return {
        "retorno_total": float(equity[-1] / equity[0] - 1),
        "retorno_anual": annual,
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "calmar": annual / abs(max_dd) if max_dd < 0 else 0.0,
        "max_drawdown": max_dd,
        "win_rate": float((trades["pnl_neto"] > 0).mean()) if len(trades) else 0.0,
        "profit_factor": float(wins / losses) if losses > 0 else float("nan"),
        "operaciones": int(len(trades)),
    }


def rolling_sharpe(equity, window=config.TRADING_DAYS):
    """Sharpe móvil de 252 días."""
    returns = pd.Series(equity).pct_change()
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return (mean / std * np.sqrt(config.TRADING_DAYS)).values
