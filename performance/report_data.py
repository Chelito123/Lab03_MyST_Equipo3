"""Datos y ejemplos numéricos que alimentan el reporte."""
import json

import numpy as np
import pandas as pd

from kalman.filters import filter_mean, filter_weights


def worked_example_f1(log_prices, z0, P0, Q, R, days=4):
    """Primeros días del filtro 1, calculados con el filtro real."""
    x = np.asarray(log_prices, float)[:days]
    out = filter_weights(x, z0, P0, Q, R)
    return pd.DataFrame({
        "t": np.arange(1, days + 1), "y": x[:, 0], "innovacion": out.spread,
        "S": out.innovation_var, "K_norma": out.gain_norm, "beta_2": out.beta_post[:, 1],
    })


def worked_example_f2(spread, mu0, p0, q, r, days=6):
    """Primeros días del filtro 2, con el z previo contra el posterior (S13 p.21)."""
    s = np.asarray(spread, float)[:days]
    out = filter_mean(s, mu0, p0, q, r)
    mu_posterior = out.mu_prior + out.gain * (s - out.mu_prior)
    return pd.DataFrame({
        "t": np.arange(1, days + 1), "s": s, "mu_previa": out.mu_prior,
        "sigma_previa": out.sigma_prior, "z_previo": out.zscore, "K": out.gain,
        "mu_posterior": mu_posterior, "z_posterior": (s - mu_posterior) / out.sigma_prior,
    })


def fmt_profit_factor(value):
    """Texto del profit factor; NaN significa que el tramo no tuvo pérdidas."""
    return "sin pérdidas" if value is None or np.isnan(value) else f"{value:.2f}"


def load_context(artifacts):
    """Lee las salidas del pipeline que usa el reporte."""
    def read_json(name):
        return json.loads((artifacts / f"{name}.json").read_text(encoding="utf-8"))

    def read_csv(name):
        return pd.read_csv(artifacts / f"{name}.csv")

    context = {key: read_json(key) for key in
               ("modelo", "metricas", "costos", "rebalanceo", "diagnosticos")}
    context.update({
        "seleccion": read_csv("seleccion_grupos"), "vectores": read_csv("seleccion_vectores"),
        "walk_forward": read_csv("walk_forward"), "operaciones": read_csv("operaciones"),
        "grid_filtros": read_csv("grid_filtros"), "grid_reglas": read_csv("grid_reglas"),
    })
    return context
