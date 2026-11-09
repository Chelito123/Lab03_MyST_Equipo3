"""Defectos encontrados en la revisión final."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

import config
import main as pipeline
from cointegration import selection
from strategy.signals import RuleParams


def test_salida_cero_no_es_una_regla_valida():
    with pytest.raises(ValueError):
        RuleParams(2.0, 0.0, 4.0, 10)
    assert all(value > 0 for value in config.Z_EXIT_GRID)


def test_operacion_que_cruza_tramos_se_cuenta_como_arrastrada():
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    trades = pd.DataFrame({"entrada": [index[2], index[6]], "salida": [index[6], index[8]]})
    assert pipeline.carried_trades(trades, index, 4, 10) == 1
    assert pipeline.carried_trades(trades, index, 0, 5) == 0


def test_engle_granger_ordena_a_los_sobrevivientes(monkeypatch):
    table = pd.DataFrame({
        "grupo": ["a", "b"], "etapa_descarte": ["", ""], "eg_pares_p05": [1, 0],
        "estabilidad": [0.1, 0.9], "adf_p": [0.01, 0.01],
    })
    result = SimpleNamespace(eigenvectors=np.eye(2))
    row = {"vector_elegido": 0, "k_ar_diff": 1}
    results = {"a": (["X", "Y"], result, row), "b": (["Z", "W"], result, row)}
    monkeypatch.setattr(selection, "run_funnel", lambda _: (table, pd.DataFrame(), results))
    assert selection.choose_group(pd.DataFrame()).assets == ["X", "Y"]


@pytest.fixture(scope="module")
def report_text(tmp_path_factory):
    from docs.build_report import build

    pdf = build(tmp_path_factory.mktemp("r") / "r.pdf")
    return " ".join(" ".join(p.extract_text() for p in PdfReader(pdf).pages).split())


def test_reporte_aclara_atribucion_por_entrada(report_text):
    assert "por fecha de entrada" in report_text


def test_reporte_aclara_walk_forward(report_text):
    assert "no es estrictamente fuera de muestra" in report_text
    assert "Sharpe fuera" not in report_text
