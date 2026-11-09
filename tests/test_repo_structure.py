"""La estructura obligatoria del repositorio está completa."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUERIDOS = ["README.md", "requirements.txt", ".gitignore", "main.py", "dashboard/app.py",
              "project_management/charter.md", "project_management/postmortem.md",
              "project_management/sprint_01.md", "docs/reporte.pdf", "docs/dashboard.md"]
CARPETAS = ["data", "cointegration", "kalman", "strategy", "backtest", "performance", "artifacts", "tests"]


@pytest.mark.parametrize("ruta", REQUERIDOS + CARPETAS)
def test_existe(ruta):
    assert (ROOT / ruta).exists()


def test_requirements_fijados():
    lineas = (ROOT / "requirements.txt").read_text().split()
    assert lineas and all("==" in linea for linea in lineas)
