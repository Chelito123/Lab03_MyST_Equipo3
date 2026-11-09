"""Los dos filtros de Kalman del laboratorio."""
from dataclasses import dataclass

import numpy as np


@dataclass
class WeightFilterOutput:
    """Salida del filtro 1, con T filas."""

    beta_post: np.ndarray  # vector de pesos completo tras actualizar en t
    spread: np.ndarray  # innovación con el estado previo: causal en t
    innovation_var: np.ndarray
    gain_norm: np.ndarray
    posterior_trace: np.ndarray


def filter_weights(log_prices, z0, P0, Q, R):
    """Filtro 1: estado z_t = (c_t, β2..βn) y observación y_t = log P1 (S14 p.18).

    El peso del activo 1 queda fijo en +1. El spread S_t = β̂_{t|t-1}ᵀ x_t - ĉ_{t|t-1}
    es exactamente la innovación, así que solo usa información previa a t.
    """
    x = np.asarray(log_prices, float)
    T, n = x.shape
    z, P = np.array(z0, float), np.array(P0, float)
    beta_post, spread = np.zeros((T, n)), np.zeros(T)
    innovation_var, gain_norm, posterior_trace = np.zeros(T), np.zeros(T), np.zeros(T)
    identity = np.eye(n)
    for t in range(T):
        P = P + Q
        C = np.concatenate(([1.0], -x[t, 1:]))
        innovation = x[t, 0] - C @ z
        S = float(C @ P @ C + R)
        K = P @ C / S
        z = z + K * innovation
        A = identity - np.outer(K, C)
        P = A @ P @ A.T + np.outer(K, K) * R  # forma de Joseph: numéricamente estable
        spread[t], innovation_var[t] = innovation, S
        gain_norm[t], posterior_trace[t] = np.linalg.norm(K), np.trace(P)
        beta_post[t] = np.concatenate(([1.0], z[1:]))
    return WeightFilterOutput(beta_post, spread, innovation_var, gain_norm, posterior_trace)


@dataclass
class MeanFilterOutput:
    """Salida del filtro 2, con T filas."""

    mu_prior: np.ndarray
    sigma_prior: np.ndarray
    zscore: np.ndarray
    gain: np.ndarray
    posterior_var: np.ndarray


def filter_mean(spread, mu0, p0, q, r):
    """Filtro 2: nivel local sobre el spread; z_t usa estadísticos previos (S13 p.20)."""
    s = np.asarray(spread, float)
    T = len(s)
    mu_prior, sigma_prior, zscore = np.zeros(T), np.zeros(T), np.zeros(T)
    gain, posterior_var = np.zeros(T), np.zeros(T)
    mu, p = float(mu0), float(p0)
    for t in range(T):
        p_prior = p + q
        S = p_prior + r
        mu_prior[t], sigma_prior[t] = mu, np.sqrt(S)
        zscore[t] = (s[t] - mu) / sigma_prior[t]
        K = p_prior / S
        mu = mu + K * (s[t] - mu)
        p = (1 - K) * p_prior
        gain[t], posterior_var[t] = K, p
    return MeanFilterOutput(mu_prior, sigma_prior, zscore, gain, posterior_var)


def innovation_diagnostics(normalized):
    """Varianza, autocorrelación de rezago 1 y banda de significancia al 95 %."""
    v = np.asarray(normalized, float)
    centered = v - v.mean()
    rho = float(centered[1:] @ centered[:-1] / (centered @ centered))
    return {"varianza": float(v.var()), "acf1": rho, "banda_95": 1.96 / np.sqrt(len(v))}
