"""Ingesta, limpieza y validación de precios diarios ajustados."""
import numpy as np
import pandas as pd

import config


class DataUnavailable(RuntimeError):
    """No hay precios locales y tampoco se pudieron descargar."""


def download_prices():
    """Descarga cierres ajustados de yfinance para el universo y la ventana."""
    import yfinance as yf

    end = pd.Timestamp(config.END) + pd.Timedelta(days=1)
    raw = yf.download(
        config.UNIVERSE, start=config.START, end=end.strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
    )["Close"]
    return raw[config.UNIVERSE]


def validate_prices(prices):
    """Revisa orden, huecos y positividad; regresa un resumen de calidad."""
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("El índice de fechas debe ser creciente y sin duplicados.")
    if prices.isna().any().any():
        raise ValueError("Hay precios faltantes en la ventana.")
    if (prices <= 0).any().any():
        raise ValueError("Hay precios no positivos.")
    returns = np.log(prices).diff().dropna()
    return {
        "observaciones": len(prices),
        "inicio": str(prices.index[0].date()),
        "fin": str(prices.index[-1].date()),
        "max_retorno_abs": float(returns.abs().max().max()),
        "dias_retorno_abs_mayor_15pct": int((returns.abs() > 0.15).any(axis=1).sum()),
    }


def load_prices(refresh=False):
    """Lee `data/prices.csv`, o lo descarga la primera vez.

    El CSV se versiona: yfinance reajusta la historia con cada dividendo nuevo,
    así que descargar de nuevo cambiaría los resultados.
    """
    if refresh or not config.PRICES_CSV.exists():
        try:
            prices = download_prices()
        except Exception as error:
            raise DataUnavailable(
                f"No se pudo descargar precios ({error}). Restaura data/prices.csv o revisa la conexión."
            ) from error
        if prices.empty:
            raise DataUnavailable("La descarga llegó vacía. Restaura data/prices.csv o revisa la conexión.")
        prices.to_csv(config.PRICES_CSV)
    else:
        prices = pd.read_csv(config.PRICES_CSV, index_col=0, parse_dates=True)
    prices.index.name = "Date"
    return prices


def split_dates(index):
    """Últimas fechas de train y de test para la partición 60/20/20."""
    n = len(index)
    train_end = index[int(n * config.TRAIN_FRACTION) - 1]
    test_end = index[int(n * (config.TRAIN_FRACTION + config.TEST_FRACTION)) - 1]
    return train_end, test_end
