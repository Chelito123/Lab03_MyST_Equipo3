"""Calibra el modelo con train y genera señales causales para toda la muestra."""
from dataclasses import dataclass

import numpy as np

import config
from cointegration.stats import half_life
from kalman.filters import filter_mean, filter_weights
from strategy.vecm import fit_vecm


@dataclass(frozen=True)
class FilterParams:
    """Razones Q/R de cada filtro."""

    q_ratio_f1: float
    q_ratio_f2: float


@dataclass
class Signals:
    """Salidas de ambos filtros y parámetros calibrados en train."""

    f1: object
    f2: object
    vecm: object
    r1: float
    r2: float
    half_life: float


def initial_state_f1(train, beta, constant, r1, q_ratio):
    """Regresa (z0, P0, Q): prior con la incertidumbre de una observación (C5)."""
    n = train.shape[1]
    regressors = np.concatenate(([1.0], train[:, 1:].mean(axis=0)))
    z0 = np.concatenate(([constant], beta[1:]))
    P0 = np.eye(n) * r1 / float(regressors @ regressors)
    return z0, P0, np.eye(n) * q_ratio * r1


def fit_signals(log_prices, beta, k_ar_diff, train_end, params):
    """Corre el pipeline de filtros usando solo train para calibrar.

    Recibe log-precios ya ordenados con la pata líder primero. Si se le pasan datos
    truncados en t > train_end, debe devolver el mismo z_t: de eso depende la
    prueba de causalidad.
    """
    x = log_prices.values
    train = log_prices.loc[:train_end].values
    vecm = fit_vecm(train, beta, k_ar_diff)
    static_spread = train @ beta - vecm.constant
    r1 = float(static_spread.var())
    z0, P0, Q = initial_state_f1(train, beta, vecm.constant, r1, params.q_ratio_f1)
    f1 = filter_weights(x, z0, P0, Q, r1)
    n_train = len(train)
    r2 = float(f1.spread[config.BURN_IN:n_train].var())
    f2 = filter_mean(f1.spread, 0.0, r2, params.q_ratio_f2 * r2, r2)
    return Signals(f1, f2, vecm, r1, r2, half_life(static_spread))
