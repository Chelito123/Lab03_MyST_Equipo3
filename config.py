"""Parámetros del proyecto. Todo lo que cambia un resultado vive aquí."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
PRICES_CSV = ROOT / "data" / "prices.csv"

SEED = 42

# Universo: utilities reguladas de EE.UU. (README, sección 5.2).
UNIVERSE = ["DUK", "SO", "AEP", "D", "XEL", "ED", "EXC", "WEC"]
GROUP_SIZE = 6  # 2 x 3 integrantes

# Ventana y partición 60/20/20 cronológica (README, sección 5.4).
START = "2012-01-03"
END = "2026-10-30"
TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.20

# Lab03, sección 3.1.
COMMISSION = 0.00125
BORROW_RATE = 0.0025
TRADING_DAYS = 252
INITIAL_CAPITAL = 1_000_000.0
POSITION_FRACTION = 0.80

# Embudo de selección (README, convenciones C2, C7, C8, C9).
CORR_WINDOW = 252
CORR_THRESHOLD = 0.70
PVALUE = 0.05
HALF_LIFE_RANGE = (5.0, 60.0)
JOHANSEN_DET_ORDER = 0
STABILITY_WINDOW = 504
STABILITY_STEP = 126
ROLLING_JOHANSEN_WINDOW = 252
ROLLING_JOHANSEN_STEP = 5

# Filtros de Kalman (README, convención C5).
BURN_IN = 60

# Rejillas buscadas solo en train (README, sección 3).
Q_RATIO_F1_GRID = [1e-7, 1e-6, 1e-5]
Q_RATIO_F2_GRID = [1e-4, 1e-3, 1e-2]
Z_ENTRY_GRID = [1.0, 1.5, 2.0, 2.5]
Z_EXIT_GRID = [0.25, 0.5]  # positivos: la regla sale con |z| < z_salida
Z_STOP_GRID = [3.0, 4.0]
HOLD_MULTIPLE_GRID = [2.0, 3.0]
REBALANCE_BAND = 0.10
