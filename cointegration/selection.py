"""Embudo de selección del grupo cointegrado. Usa solo la partición de train."""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

import config
from cointegration.stats import (
    adf_pvalue, engle_granger_pvalue, half_life, johansen, select_k_ar_diff,
)


def rolling_mean_correlation(log_prices):
    """Correlación media entre pares, en ventana móvil, sobre retornos (C8)."""
    returns = log_prices.diff().dropna()
    n = returns.shape[1]
    corr = returns.rolling(config.CORR_WINDOW).corr().dropna()
    per_day = corr.groupby(level=0).apply(lambda m: (m.values.sum() - n) / (n * (n - 1)))
    return per_day


def engle_granger_pairs(log_prices):
    """Cuenta pares con p < 0.05 en ambos órdenes de regresión (S12 p.16)."""
    columns = list(log_prices.columns)
    passed, p_values = 0, []
    for a, b in combinations(columns, 2):
        p = max(
            engle_granger_pvalue(log_prices[a], log_prices[b]),
            engle_granger_pvalue(log_prices[b], log_prices[a]),
        )
        p_values.append(p)
        passed += p < config.PVALUE
    return passed, min(p_values)


def evaluate_vectors(log_prices, result, rank):
    """Construye el spread de cada vector y mide ADF y vida media."""
    rows = []
    for column in range(max(rank, 1)):
        spread = log_prices.values @ result.eigenvectors[:, column]
        rows.append({
            "vector": column,
            "eigenvalor": float(result.eigenvalues[column]),
            "adf_p": adf_pvalue(spread),
            "vida_media": half_life(spread),
        })
    return rows


def usable(row):
    """Spread estacionario y con vida media operable (S14 p.16)."""
    low, high = config.HALF_LIFE_RANGE
    return row["adf_p"] < config.PVALUE and low <= row["vida_media"] <= high


def stability(log_prices, k_ar_diff):
    """Fracción de subventanas móviles donde Johansen encuentra r >= 1."""
    values = log_prices.values
    starts = range(0, len(values) - config.STABILITY_WINDOW + 1, config.STABILITY_STEP)
    hits = [
        johansen(values[s:s + config.STABILITY_WINDOW], config.JOHANSEN_DET_ORDER,
                 k_ar_diff).rank("trace") >= 1
        for s in starts
    ]
    return float(np.mean(hits))


class NoGroupSurvived(RuntimeError):
    """Ningún grupo pasó el embudo; conserva la tabla para reportarla."""

    def __init__(self, table):
        super().__init__("Ningún grupo sobrevivió el embudo en train.")
        self.table = table


@dataclass
class Selection:
    """Grupo elegido y evidencia para reportarlo."""

    assets: list
    beta: np.ndarray  # normalizado, en el orden de `assets`
    k_ar_diff: int
    table: pd.DataFrame
    vectors: pd.DataFrame


def normalize_vector(vector):
    """Reordena para que la pata de mayor |β| sea la primera con β = +1 (C4)."""
    lead = int(np.argmax(np.abs(vector)))
    order = [lead] + [i for i in range(len(vector)) if i != lead]
    return order, vector[order] / vector[lead]


def evaluate_group(log_train, assets):
    """Corre el embudo completo para un grupo y regresa su fila de la tabla."""
    frame = log_train[assets]
    row = {"grupo": " ".join(assets), "etapa_descarte": ""}
    correlation = rolling_mean_correlation(frame)
    row["corr_media"] = float(correlation.mean())
    row["corr_ventanas_sobre_umbral"] = float((correlation > config.CORR_THRESHOLD).mean())
    row["eg_pares_p05"], row["eg_min_p"] = engle_granger_pairs(frame)
    k = select_k_ar_diff(frame.values)
    row["k_ar_diff"] = k
    result = johansen(frame.values, config.JOHANSEN_DET_ORDER, k)
    row["r_traza"] = result.rank("trace")
    row["r_max_eig"] = result.rank("max")
    row["r_traza_k1"] = johansen(frame.values, config.JOHANSEN_DET_ORDER, 1).rank("trace")
    row["traza_r0"] = float(result.trace_stat[0])
    row["traza_cv95"] = float(result.trace_cv95[0])
    row["max_eig_r0"] = float(result.max_stat[0])
    row["max_eig_cv95"] = float(result.max_cv95[0])
    vectors = evaluate_vectors(frame, result, row["r_traza"])
    for item in vectors:
        item["grupo"] = row["grupo"]
    candidates = [v for v in vectors if usable(v)]
    best = min(candidates, key=lambda v: v["adf_p"]) if candidates else None
    row["vector_elegido"] = best["vector"] if best else -1
    row["adf_p"] = best["adf_p"] if best else float("nan")
    row["vida_media"] = best["vida_media"] if best else float("nan")
    row["estabilidad"] = stability(frame, k)
    if row["corr_media"] <= config.CORR_THRESHOLD:
        row["etapa_descarte"] = "correlacion"
    elif row["r_traza"] < 1:
        row["etapa_descarte"] = "johansen"
    elif best is None:
        row["etapa_descarte"] = "validacion_spread"
    return row, vectors, result


def run_funnel(log_train):
    """Evalúa todos los grupos de GROUP_SIZE y regresa tablas y resultados."""
    rows, vectors, results = [], [], {}
    for combo in combinations(config.UNIVERSE, config.GROUP_SIZE):
        assets = list(combo)
        row, group_vectors, result = evaluate_group(log_train, assets)
        rows.append(row)
        vectors.extend(group_vectors)
        results[row["grupo"]] = (assets, result, row)
    return pd.DataFrame(rows), pd.DataFrame(vectors), results


def choose_group(log_train):
    """Elige entre sobrevivientes: más pares EG (C2), luego estabilidad y ADF."""
    table, vectors, results = run_funnel(log_train)
    alive = table[table["etapa_descarte"] == ""]
    if alive.empty:
        raise NoGroupSurvived(table)
    winner = alive.sort_values(["eg_pares_p05", "estabilidad", "adf_p"],
                               ascending=[False, False, True]).iloc[0]
    assets, result, row = results[winner["grupo"]]
    order, beta = normalize_vector(result.eigenvectors[:, row["vector_elegido"]])
    return Selection([assets[i] for i in order], beta, int(row["k_ar_diff"]), table, vectors)
