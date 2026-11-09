"""Pipeline completo del Lab03. Uso: python main.py"""
import json
import sys

import numpy as np
import pandas as pd

import config
from backtest.calibration import search_filters, search_rule, walk_forward
from backtest.engine import run_backtest
from cointegration.selection import NoGroupSurvived, choose_group
from cointegration.stats import adf_pvalue, half_life, johansen
from data.loader import DataUnavailable, load_prices, split_dates, validate_prices
from kalman.filters import innovation_diagnostics
from performance import figures
from performance.metrics import drawdown, rolling_sharpe, summarize
from strategy.model import fit_signals
from strategy.signals import directions


def segment_bounds(index, train_end, test_end):
    """Índices [inicio, fin) de cada tramo; comparten el cierre anterior como base."""
    n_train = index.get_loc(train_end) + 1
    n_test = index.get_loc(test_end) + 1
    return {"train": (0, n_train), "test": (n_train - 1, n_test),
            "validacion": (n_test - 1, len(index))}


def segment_trades(trades, index, start, stop):
    """Operaciones cuya entrada cae dentro del tramo."""
    return trades[(trades["entrada"] > index[start]) & (trades["entrada"] <= index[stop - 1])]


def carried_trades(trades, index, start, stop):
    """Operaciones abiertas antes del tramo que siguen vivas cuando este empieza."""
    return int(((trades["entrada"] <= index[start]) & (trades["salida"] > index[start])).sum())


def rolling_first_eigenvector(log_prices):
    """Primer eigenvector de Johansen móvil, normalizado en la pata líder (C10)."""
    x = log_prices.values
    window, step = config.ROLLING_JOHANSEN_WINDOW, config.ROLLING_JOHANSEN_STEP
    rows, dates = [], []
    for end in range(window, len(x) + 1, step):
        vector = johansen(x[end - window:end], config.JOHANSEN_DET_ORDER, 1).eigenvectors[:, 0]
        rows.append(vector / vector[0])
        dates.append(log_prices.index[end - 1])
    frame = pd.DataFrame(rows, index=dates, columns=log_prices.columns)
    return frame.reindex(log_prices.index).ffill()


def diagnostics_by_segment(signals, bounds):
    """Innovaciones normalizadas y estacionariedad del spread en cada tramo."""
    output = {}
    for name, (start, stop) in bounds.items():
        begin = max(start, config.BURN_IN)
        spread = signals.f1.spread[begin:stop]
        output[name] = {
            "filtro1": innovation_diagnostics(spread / np.sqrt(signals.f1.innovation_var[begin:stop])),
            "filtro2": innovation_diagnostics(signals.f2.zscore[begin:stop]),
            "adf_p_spread": adf_pvalue(spread),
            "vida_media_spread": half_life(spread),
        }
    return output


def save_json(name, payload):
    """Guarda un JSON en artifacts/."""
    (config.ARTIFACTS / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )


def run():
    """Ejecuta datos → selección → filtros → backtest → artefactos."""
    np.random.seed(config.SEED)
    config.FIGURES.mkdir(parents=True, exist_ok=True)

    prices = load_prices()
    quality = validate_prices(prices)
    train_end, test_end = split_dates(prices.index)
    print(f"[1/6] Datos: {quality['observaciones']} días; train hasta {train_end.date()}, "
          f"test hasta {test_end.date()}")

    log_all = np.log(prices)
    selection = choose_group(log_all.loc[:train_end])
    selection.table.to_csv(config.ARTIFACTS / "seleccion_grupos.csv", index=False)
    selection.vectors.to_csv(config.ARTIFACTS / "seleccion_vectores.csv", index=False)
    alive = int((selection.table["etapa_descarte"] == "").sum())
    print(f"[2/6] Selección: {alive} de {len(selection.table)} grupos sobreviven; "
          f"elegido {selection.assets}")

    assets, beta, k = selection.assets, selection.beta, selection.k_ar_diff
    log_prices, levels = log_all[assets], prices[assets]
    array = levels.values
    index = prices.index
    n_train = index.get_loc(train_end) + 1

    filter_params, filter_grid = search_filters(log_prices, array, beta, k, train_end)
    filter_grid.to_csv(config.ARTIFACTS / "grid_filtros.csv", index=False)
    signals = fit_signals(log_prices, beta, k, train_end, filter_params)
    rule, rule_grid = search_rule(array, signals, n_train)
    rule_grid.to_csv(config.ARTIFACTS / "grid_reglas.csv", index=False)
    wf = walk_forward(array, signals, n_train)
    wf.to_csv(config.ARTIFACTS / "walk_forward.csv", index=False)
    print(f"[3/6] Calibración: {filter_params} {rule}")

    direction = directions(signals.f2.zscore, rule, config.BURN_IN)
    result = run_backtest(array, signals.f1.beta_post, direction, dates=index)
    band = run_backtest(array, signals.f1.beta_post, direction, dates=index,
                        band=config.REBALANCE_BAND)
    print(f"[4/6] Backtest: {len(result.trades)} operaciones; capital final "
          f"${result.equity[-1]:,.0f}")

    bounds = segment_bounds(index, train_end, test_end)
    metrics = {
        name: summarize(result.equity[start:stop],
                        segment_trades(result.trades, index, start, stop))
        for name, (start, stop) in bounds.items()
    }
    for name, (start, stop) in bounds.items():
        metrics[name]["operaciones_arrastradas"] = carried_trades(result.trades, index, start, stop)
    metrics["total"] = summarize(result.equity, result.trades)
    costs = {
        "pnl_bruto": float(result.gross_pnl.sum()),
        "comisiones": float(result.commissions.sum()),
        "prestamo": float(result.borrow.sum()),
        "pnl_neto": float(result.equity[-1] - config.INITIAL_CAPITAL),
    }
    rebalance = {
        "diario": {"comisiones": float(result.commissions.sum()),
                   **summarize(result.equity, result.trades)},
        "banda": {"banda": config.REBALANCE_BAND, "comisiones": float(band.commissions.sum()),
                  **summarize(band.equity, band.trades)},
    }

    series = pd.DataFrame({
        "tramo": np.select([index <= train_end, index <= test_end],
                           ["train", "test"], "validacion"),
        "capital": result.equity, "drawdown": drawdown(result.equity),
        "spread": signals.f1.spread, "zscore": signals.f2.zscore,
        "mu": signals.f2.mu_prior, "sigma": signals.f2.sigma_prior,
        "direccion": direction,
        "ganancia_f1": signals.f1.gain_norm, "ganancia_f2": signals.f2.gain,
        "var_posterior_f1": signals.f1.posterior_trace,
        "var_posterior_f2": signals.f2.posterior_var,
        "sharpe_252": rolling_sharpe(result.equity),
    }, index=index)
    series.to_csv(config.ARTIFACTS / "series.csv")
    pd.DataFrame(signals.f1.beta_post, index=index, columns=assets).to_csv(
        config.ARTIFACTS / "pesos.csv")
    eigenvector = rolling_first_eigenvector(log_prices)
    eigenvector.to_csv(config.ARTIFACTS / "eigenvector_movil.csv")
    levels.to_csv(config.ARTIFACTS / "precios_grupo.csv")
    result.trades.to_csv(config.ARTIFACTS / "operaciones.csv", index=False)
    print("[5/6] Artefactos en artifacts/")

    diagnostics = diagnostics_by_segment(signals, bounds)
    save_json("metricas.json", metrics)
    save_json("costos.json", costs)
    save_json("rebalanceo.json", rebalance)
    save_json("diagnosticos.json", diagnostics)
    save_json("modelo.json", {
        "semilla": config.SEED, "calidad_datos": quality,
        "train_fin": str(train_end.date()), "test_fin": str(test_end.date()),
        "activos": assets, "beta": beta.tolist(), "k_ar_diff": k,
        "grupos_evaluados": len(selection.table), "grupos_sobrevivientes": alive,
        "q_ratio_f1": filter_params.q_ratio_f1, "q_ratio_f2": filter_params.q_ratio_f2,
        "r1": signals.r1, "r2": signals.r2, "vida_media": signals.half_life,
        "vecm": {"alpha": signals.vecm.alpha.tolist(), "c": signals.vecm.constant,
                 "beta_alpha": signals.vecm.beta_alpha,
                 "vida_media": signals.vecm.half_life, "corrige": signals.vecm.corrects},
        "regla": {"z_entrada": rule.z_entry, "z_salida": rule.z_exit,
                  "z_stop": rule.z_stop, "h_max": rule.h_max},
    })
    figures.save_all(series, levels, pd.DataFrame(signals.f1.beta_post, index=index, columns=assets),
                     eigenvector, result.trades, rule)
    print("[6/6] Figuras en artifacts/figures/")
    print(json.dumps(metrics, indent=2, default=float))


def main():
    """Corre el pipeline y convierte los fallos esperados en mensajes claros."""
    try:
        run()
    except DataUnavailable as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    except NoGroupSurvived as error:
        config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
        error.table.to_csv(config.ARTIFACTS / "seleccion_grupos.csv", index=False)
        print(f"ERROR: {error} Revisa artifacts/seleccion_grupos.csv.")
        sys.exit(1)


if __name__ == "__main__":
    main()
