"""Regla de bandas sobre el z-score (política PFA, S12 p.20)."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuleParams:
    """θ = (z_entrada, z_salida, z_stop, h_max)."""

    z_entry: float
    z_exit: float
    z_stop: float
    h_max: int

    def __post_init__(self):
        if self.z_exit <= 0:
            raise ValueError("z_salida debe ser positivo: con |z| < 0 nunca habría salida.")


def directions(zscore, params, burn_in):
    """Dirección objetivo d_t en {-1, 0, +1} decidida al cierre de t.

    Tras un stop o tenencia máxima no se reabre hasta que |z| regrese a z_salida:
    si no, la misma divergencia volvería a abrir la posición al día siguiente.
    """
    z = np.asarray(zscore, float)
    d = np.zeros(len(z), dtype=int)
    current, held, locked = 0, 0, False
    for t in range(burn_in, len(z)):
        magnitude = abs(z[t])
        if locked and magnitude < params.z_exit + 1e-12:
            locked = False
        if current != 0:
            held += 1
            if magnitude < params.z_exit:
                current, held = 0, 0
            elif magnitude > params.z_stop or held >= params.h_max:
                current, held, locked = 0, 0, True
        elif not locked and params.z_entry < magnitude <= params.z_stop:
            current, held = (1 if z[t] < 0 else -1), 0
        d[t] = current
    return d
