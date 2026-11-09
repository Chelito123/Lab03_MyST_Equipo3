"""Pruebas de raíz unitaria y cointegración con numpy.

statsmodels no tiene binarios para Windows ARM64, así que ADF, Engle-Granger y
Johansen se implementan aquí siguiendo sus mismos algoritmos. Los valores
críticos y p-values salen de `cointegration/vendor`, copiados de statsmodels 0.14.6
(licencia BSD). `tests/test_stats.py` compara contra resultados de referencia.
"""
from dataclasses import dataclass

import numpy as np

from cointegration.vendor.adfvalues import mackinnonp
from cointegration.vendor.coint_tables import c_sja, c_sjt


def _ols(y, x):
    """Devuelve (coeficientes, suma de cuadrados de residuales, errores estándar)."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    ssr = float(resid @ resid)
    dof = len(y) - x.shape[1]
    cov = ssr / dof * np.linalg.pinv(x.T @ x)
    return beta, ssr, np.sqrt(np.diag(cov))


def _adf_design(x, lag, constant):
    """Arma la regresión ADF con `lag` diferencias rezagadas."""
    dx = np.diff(x)
    y = dx[lag:]
    cols = [x[lag:-1]]
    cols += [dx[lag - k:-k] for k in range(1, lag + 1)]
    if constant:
        cols.append(np.ones(len(y)))
    return y, np.column_stack(cols)


def _default_maxlag(n, constant):
    """Regla de Schwert, con el tope que usa statsmodels."""
    ntrend = int(constant)
    maxlag = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    return min(n // 2 - ntrend - 1, maxlag)


def adf_stat(x, constant=True, maxlag=None):
    """Estadístico ADF con rezagos elegidos por AIC. Regresa (t, rezago)."""
    x = np.asarray(x, dtype=float)
    if maxlag is None:
        maxlag = _default_maxlag(len(x), constant)
    best_aic, best_lag = np.inf, 0
    n_common = len(x) - 1 - maxlag
    for lag in range(0, maxlag + 1):
        y, design = _adf_design(x, lag, constant)
        y, design = y[-n_common:], design[-n_common:]
        _, ssr, _ = _ols(y, design)
        aic = n_common * np.log(ssr / n_common) + 2 * design.shape[1]
        if aic < best_aic:
            best_aic, best_lag = aic, lag
    y, design = _adf_design(x, best_lag, constant)
    beta, _, se = _ols(y, design)
    return float(beta[0] / se[0]), best_lag


def adf_pvalue(x):
    """p-value MacKinnon de ADF con constante (H0: raíz unitaria)."""
    stat, _ = adf_stat(x, constant=True)
    return float(mackinnonp(stat, regression="c", N=1))


def engle_granger_pvalue(y, x):
    """Engle-Granger: OLS y ~ c + x y ADF sin constante sobre los residuales."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    stat, _ = adf_stat(y - design @ beta, constant=False)
    return float(mackinnonp(stat, regression="c", N=2))


def half_life(spread):
    """Vida media Ornstein-Uhlenbeck: ds_t = a + λ s_{t-1}; HL = -ln2/λ (S12 p.17)."""
    s = np.asarray(spread, float)
    design = np.column_stack([s[:-1], np.ones(len(s) - 1)])
    beta, *_ = np.linalg.lstsq(design, np.diff(s), rcond=None)
    return float(-np.log(2) / beta[0]) if beta[0] < 0 else float("inf")


@dataclass
class JohansenResult:
    """Resultado de Johansen; `eigenvectors` tiene un vector por columna."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    trace_stat: np.ndarray
    max_stat: np.ndarray
    trace_cv95: np.ndarray
    max_cv95: np.ndarray

    def rank(self, statistic="trace"):
        """Primer r0 que no se rechaza al 95 % (S14 p.11)."""
        stats = self.trace_stat if statistic == "trace" else self.max_stat
        cvs = self.trace_cv95 if statistic == "trace" else self.max_cv95
        for r0, (value, cv) in enumerate(zip(stats, cvs)):
            if value <= cv:
                return r0
        return len(stats)


def _detrend(y, order):
    """Quita un polinomio de grado `order`; -1 no hace nada."""
    if order == -1:
        return y
    grid = np.vander(np.linspace(-1, 1, len(y)), order + 1)
    return y - grid @ (np.linalg.pinv(grid) @ y)


def _partial_out(y, x):
    """Residuales de y regresado sobre x."""
    return y if x.size == 0 else y - x @ (np.linalg.pinv(x) @ y)


def johansen(endog, det_order=0, k_ar_diff=1):
    """Prueba de Johansen (traza y máximo eigenvalor), igual que `coint_johansen`."""
    endog = _detrend(np.asarray(endog, float), det_order)
    neqs = endog.shape[1]
    f = 0 if det_order > -1 else det_order
    dx_full = np.diff(endog, axis=0)
    if k_ar_diff:
        z = np.column_stack(
            [dx_full[k_ar_diff - k:-k] for k in range(1, k_ar_diff + 1)]
        )
        z = _detrend(z, f)
    else:
        z = np.empty((len(dx_full), 0))
    r0t = _partial_out(_detrend(dx_full[k_ar_diff:], f), z)
    levels = endog[: endog.shape[0] - k_ar_diff][1:]
    rkt = _partial_out(_detrend(levels, f), z)
    nobs = rkt.shape[0]
    skk = rkt.T @ rkt / nobs
    sk0 = rkt.T @ r0t / nobs
    s00 = r0t.T @ r0t / nobs
    eigval, eigvec = np.linalg.eig(
        np.linalg.inv(skk) @ sk0 @ np.linalg.inv(s00) @ sk0.T
    )
    eigval, eigvec = eigval.real, eigvec.real
    scale = np.linalg.inv(np.linalg.cholesky(eigvec.T @ skk @ eigvec))
    vectors = eigvec @ scale
    order = np.argsort(eigval)[::-1]
    eigval, vectors = eigval[order], vectors[:, order]
    first = vectors.flat[np.flatnonzero(vectors.flat)[0]]
    vectors = vectors * np.sign(first)
    log_term = np.log(1 - eigval)
    trace = np.array([-nobs * log_term[i:].sum() for i in range(neqs)])
    maxim = -nobs * log_term
    trace_cv = np.array([c_sjt(neqs - i, det_order)[1] for i in range(neqs)])
    max_cv = np.array([c_sja(neqs - i, det_order)[1] for i in range(neqs)])
    return JohansenResult(eigval, vectors, trace, maxim, trace_cv, max_cv)


def select_k_ar_diff(endog, maxlags=10):
    """Rezagos del VECM por AIC del VAR en niveles menos 1, con mínimo 1 (C9)."""
    x = np.asarray(endog, float)
    k = x.shape[1]
    best_aic, best_p = np.inf, 1
    y = x[maxlags:]
    for p in range(1, maxlags + 1):
        lags = [x[maxlags - j:-j] for j in range(1, p + 1)]
        design = np.column_stack(lags + [np.ones(len(y))])
        resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        sigma = resid.T @ resid / len(y)
        aic = np.linalg.slogdet(sigma)[1] + 2 * p * k * k / len(y)
        if aic < best_aic:
            best_aic, best_p = aic, p
    return max(best_p - 1, 1)
