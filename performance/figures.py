"""Las 8 figuras obligatorias del reporte (Lab03, sección 3.10)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import config

SEGMENT_COLORS = {"train": "#1f77b4", "test": "#ff7f0e", "validacion": "#2ca02c"}


def _finish(ax, title, ylabel, name, legend=True):
    """Pone título, ejes y leyenda, y guarda la figura."""
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(config.FIGURES / name, dpi=130)
    plt.close(ax.figure)


def _new_axes():
    """Ejes con el tamaño común de las figuras."""
    return plt.subplots(figsize=(10, 4.5))[1]


def _entries_exits(series):
    """Índices de entrada y salida a partir de la dirección."""
    change = series["direccion"].diff().fillna(0)
    entries = series[(series["direccion"] != 0) & (change != 0)]
    exits = series[(series["direccion"] == 0) & (change != 0)]
    return entries, exits


def save_all(series, prices, betas, eigenvector, trades, rule):
    """Genera y guarda las figuras 1 a 8."""
    ax = _new_axes()
    (prices / prices.iloc[0]).plot(ax=ax, linewidth=0.9)
    _finish(ax, "1. Precios del grupo (base 1 al inicio)", "Precio normalizado", "01_precios.png")

    entries, exits = _entries_exits(series)
    ax = _new_axes()
    ax.plot(series.index, series["spread"], color="#333333", linewidth=0.7, label="Spread $S_t$")
    ax.scatter(entries.index, entries["spread"], color="green", s=20, label="Entrada", zorder=3)
    ax.scatter(exits.index, exits["spread"], color="red", s=20, label="Salida", zorder=3)
    _finish(ax, "2. Spread del portafolio con señales", "Spread (log-precios)", "02_spread.png")

    ax = _new_axes()
    betas.plot(ax=ax, linewidth=0.9)
    _finish(ax, r"3. Evolución del vector de pesos $\hat{\beta}_t$", "Peso", "03_pesos.png")

    ax = _new_axes()
    eigenvector.clip(-5, 5).plot(ax=ax, linewidth=0.8)
    _finish(ax, "4. Primer eigenvector de Johansen móvil (252 días, recortado a ±5)",
            "Peso normalizado", "04_eigenvector.png")

    ax = _new_axes()
    ax.plot(series.index, series["ganancia_f1"] / series["ganancia_f1"].iloc[-1],
            label="Filtro 1 (norma de K, relativa al último valor)")
    ax.plot(series.index, series["ganancia_f2"] / series["ganancia_f2"].iloc[-1],
            label="Filtro 2 (K, relativa al último valor)")
    ax.set_yscale("log")
    _finish(ax, "5. Ganancia de Kalman de ambos filtros", "Ganancia relativa (escala log)",
            "05_ganancia.png")

    ax = _new_axes()
    ax.plot(series.index, series["zscore"], color="#333333", linewidth=0.7, label="$z_t$")
    for value, label, color in [(rule.z_entry, "±z entrada", "red"),
                                (rule.z_exit, "±z salida", "blue"),
                                (rule.z_stop, "±z stop", "black")]:
        ax.axhline(value, color=color, linestyle="--", linewidth=0.8, label=label)
        ax.axhline(-value, color=color, linestyle="--", linewidth=0.8)
    _finish(ax, "6. Z-score con umbrales", "z", "06_zscore.png")

    ax = _new_axes()
    for name, color in SEGMENT_COLORS.items():
        part = series[series["tramo"] == name]
        ax.plot(part.index, part["capital"], color=color, label=name)
    _finish(ax, "7. Curva de capital por tramo", "Capital (USD)", "07_capital.png")

    ax = _new_axes()
    returns = trades["retorno"].values * 100
    if len(returns):
        ax.hist(returns, bins=max(5, len(returns) // 2), density=True, color="#8fb8de",
                edgecolor="white", label="Operaciones")
        grid = np.linspace(returns.min() - 1, returns.max() + 1, 200)
        ax.plot(grid, norm.pdf(grid, returns.mean(), returns.std()), color="red", label="Normal")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Sin operaciones", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("8. Distribución de retornos por operación")
    ax.set_xlabel("Retorno por operación (%)")
    ax.set_ylabel("Densidad")
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(config.FIGURES / "08_retornos.png", dpi=130)
    plt.close(ax.figure)
