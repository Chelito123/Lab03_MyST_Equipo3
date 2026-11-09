"""VECM con el vector de Johansen como término de corrección de error (S14 p.14)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class VecmFit:
    """Velocidades de ajuste y diagnósticos del VECM estimado en train."""

    alpha: np.ndarray
    constant: float  # c del término de corrección: media del spread en train
    beta_alpha: float  # βᵀα: debe caer en (-2, 0)
    half_life: float
    corrects: bool


def fit_vecm(log_train, beta, k_ar_diff):
    """OLS de ΔP_t sobre ECT_{t-1} y k rezagos de ΔP. No re-estima β."""
    x = np.asarray(log_train, float)
    constant = float((x @ beta).mean())
    ect = x @ beta - constant
    dx = np.diff(x, axis=0)
    y = dx[k_ar_diff:]
    lags = [dx[k_ar_diff - j:-j] for j in range(1, k_ar_diff + 1)]
    design = np.column_stack([ect[k_ar_diff:-1]] + lags)
    coefs = np.linalg.lstsq(design, y, rcond=None)[0]
    alpha = coefs[0]
    beta_alpha = float(beta @ alpha)
    corrects = -2 < beta_alpha < 0
    implied = float(-np.log(2) / np.log(1 + beta_alpha)) if corrects else float("inf")
    return VecmFit(alpha, constant, beta_alpha, implied, corrects)
