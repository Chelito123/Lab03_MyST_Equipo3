"""Casos límite: sin operaciones, sin grupos, sin datos y filtros vacíos."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config
import main as pipeline
from backtest.engine import run_backtest
from cointegration import selection
from cointegration.selection import NoGroupSurvived, choose_group
from data import loader
from data.loader import DataUnavailable
from performance import figures
from performance.metrics import summarize
from strategy.signals import RuleParams

ROOT = Path(__file__).resolve().parents[1]


def _figure_inputs():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    series = pd.DataFrame({
        "spread": np.linspace(-1, 1, 30), "zscore": np.linspace(-1, 1, 30),
        "direccion": np.zeros(30, dtype=int), "ganancia_f1": np.ones(30),
        "ganancia_f2": np.ones(30), "tramo": ["train"] * 30, "capital": np.ones(30),
    }, index=dates)
    frame = pd.DataFrame({"A": np.linspace(1, 2, 30), "B": np.linspace(2, 1, 30)}, index=dates)
    return series, frame


def test_figuras_sin_operaciones(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES", tmp_path)
    series, frame = _figure_inputs()
    figures.save_all(series, frame, frame, frame, pd.DataFrame({"retorno": []}),
                     RuleParams(2.0, 0.5, 4.0, 10))
    assert (tmp_path / "08_retornos.png").exists()


def test_metricas_sin_operaciones():
    prices = np.full((20, 2), 10.0)
    betas = np.tile([1.0, -0.5], (20, 1))
    result = run_backtest(prices, betas, np.zeros(20, dtype=int))
    card = summarize(result.equity, result.trades)
    assert card["operaciones"] == 0
    assert card["sharpe"] == 0.0


def test_ningun_grupo_sobrevive_conserva_tabla(monkeypatch):
    table = pd.DataFrame({"grupo": ["a", "b"], "etapa_descarte": ["johansen", "johansen"]})
    monkeypatch.setattr(selection, "run_funnel", lambda _: (table, pd.DataFrame(), {}))
    with pytest.raises(NoGroupSurvived) as err:
        choose_group(pd.DataFrame())
    assert len(err.value.table) == 2


def test_main_sin_grupos_guarda_tabla_y_sale(tmp_path, monkeypatch):
    table = pd.DataFrame({"grupo": ["a"], "etapa_descarte": ["johansen"]})

    def fail(_):
        raise NoGroupSurvived(table)

    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(config, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(pipeline, "choose_group", fail)
    with pytest.raises(SystemExit) as err:
        pipeline.main()
    assert err.value.code == 1
    assert (tmp_path / "seleccion_grupos.csv").exists()


def _no_network():
    raise ConnectionError("sin red")


def test_sin_csv_y_sin_red(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PRICES_CSV", tmp_path / "no.csv")
    monkeypatch.setattr(loader, "download_prices", _no_network)
    with pytest.raises(DataUnavailable, match="data/prices.csv"):
        loader.load_prices()


def test_descarga_vacia_no_crea_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PRICES_CSV", tmp_path / "no.csv")
    monkeypatch.setattr(loader, "download_prices", pd.DataFrame)
    with pytest.raises(DataUnavailable):
        loader.load_prices()
    assert not (tmp_path / "no.csv").exists()


def _captions(app):
    return " ".join(c.value.lower() for c in app.caption)


def test_dashboard_filtros_vacios():
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    app.multiselect[1].set_value([]).run()
    assert not app.exception
    assert "sin operaciones" in _captions(app)


def test_dashboard_sin_operaciones(tmp_path):
    (tmp_path / "dashboard").mkdir()
    shutil.copy(ROOT / "dashboard" / "app.py", tmp_path / "dashboard" / "app.py")
    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    header = (ROOT / "artifacts" / "operaciones.csv").read_text().splitlines()[0]
    (tmp_path / "artifacts" / "operaciones.csv").write_text(header + "\n")
    app = AppTest.from_file(str(tmp_path / "dashboard" / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert "sin operaciones" in _captions(app)
