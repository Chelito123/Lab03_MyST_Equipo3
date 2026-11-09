"""Hace importables los módulos del proyecto y comparte datos entre pruebas."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_prices, split_dates  # noqa: E402
from strategy.model import FilterParams  # noqa: E402

ASSETS = ["DUK", "AEP", "D", "XEL", "EXC", "WEC"]


@pytest.fixture(scope="session")
def market():
    """Log-precios del grupo, β normalizado de train y parámetros de filtros."""
    from cointegration.selection import normalize_vector
    from cointegration.stats import johansen

    prices = load_prices()
    train_end, _ = split_dates(prices.index)
    log_all = np.log(prices[ASSETS])
    result = johansen(log_all.loc[:train_end].values, 0, 9)
    order, beta = normalize_vector(result.eigenvectors[:, 0])
    assets = [ASSETS[i] for i in order]
    return {
        "log": log_all[assets], "beta": beta, "k": 9, "train_end": train_end,
        "params": FilterParams(1e-6, 1e-4),
    }
