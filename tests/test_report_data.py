"""Ejemplos numéricos y carga de datos para el reporte."""
import numpy as np
import pandas as pd
import pytest

import config
from kalman.filters import filter_mean, filter_weights
from performance.report_data import (
    fmt_profit_factor, load_context, worked_example_f1, worked_example_f2,
)
from strategy.model import fit_signals, initial_state_f1
from strategy.vecm import fit_vecm


def test_ejemplo_f2_coincide_con_filtro():
    spread = np.array([0.02, 0.06, 0.115, 0.13, 0.07, 0.015])
    table = worked_example_f2(spread, 0.0, 4e-4, 1e-5, 2.5e-3)
    out = filter_mean(spread, 0.0, 4e-4, 1e-5, 2.5e-3)
    assert table["z_previo"].values == pytest.approx(out.zscore)
    assert table.loc[3, "z_previo"] == pytest.approx(2.04, abs=0.01)
    assert table.loc[3, "z_posterior"] == pytest.approx(1.82, abs=0.01)


def test_ejemplo_f1_coincide_con_filtro(market):
    log = market["log"]
    train = log.loc[:market["train_end"]].values
    constant = fit_vecm(train, market["beta"], market["k"]).constant
    r1 = float((train @ market["beta"] - constant).var())
    z0, P0, Q = initial_state_f1(train, market["beta"], constant, r1, 1e-6)
    table = worked_example_f1(log.values, z0, P0, Q, r1)
    out = filter_weights(log.values, z0, P0, Q, r1)
    assert len(table) == 4
    assert table["innovacion"].values == pytest.approx(out.spread[:4])
    assert table["K_norma"].values == pytest.approx(out.gain_norm[:4])
    assert table["beta_2"].values == pytest.approx(out.beta_post[:4, 1])


def test_initial_state_no_cambia_fit_signals(market):
    signals = fit_signals(market["log"], market["beta"], market["k"], market["train_end"],
                          market["params"])
    saved = pd.read_csv(config.ARTIFACTS / "series.csv", index_col=0)["zscore"].values
    assert signals.f2.zscore[-1] == pytest.approx(saved[-1], abs=1e-9)


def test_profit_factor_nan():
    assert fmt_profit_factor(float("nan")) == "sin pérdidas"
    assert fmt_profit_factor(7.596) == "7.60"


def test_load_context_trae_todo():
    context = load_context(config.ARTIFACTS)
    for key in ("modelo", "metricas", "costos", "rebalanceo", "diagnosticos", "seleccion",
                "vectores", "walk_forward", "operaciones", "grid_filtros", "grid_reglas"):
        assert key in context
    assert len(context["seleccion"]) == context["modelo"]["grupos_evaluados"]
