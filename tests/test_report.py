"""El reporte PDF contiene lo exigido y sus cifras salen de artifacts/."""
import json

import pytest
from pypdf import PdfReader

import config
from docs.build_report import build


@pytest.fixture(scope="module")
def report_pdf(tmp_path_factory):
    return build(tmp_path_factory.mktemp("reporte") / "r.pdf")


@pytest.fixture(scope="module")
def report_text(report_pdf):
    return "\n".join(page.extract_text() for page in PdfReader(report_pdf).pages)


def test_reporte_contiene_cifras_de_artifacts(report_text):
    metricas = json.loads((config.ARTIFACTS / "metricas.json").read_text("utf-8"))
    assert f"{metricas['train']['sharpe']:.2f}" in report_text
    assert f"{metricas['test']['sharpe']:.2f}" in report_text
    assert "sin pérdidas" in report_text
    assert "nan" not in report_text.lower().split()


@pytest.mark.parametrize("titulo", ["Selección", "Filtro 1", "Filtro 2", "Sistema completo",
                                    "VECM", "Resultados", "Limitaciones", "Uso de IA"])
def test_secciones(report_text, titulo):
    assert titulo in report_text


def test_powell_sin_risk_parity(report_text):
    for name in ("Transición", "Observación", "creencia", "físico"):
        assert name in report_text
    assert "Risk Parity" not in report_text


def test_ocho_figuras(report_pdf):
    assert sum(len(page.images) for page in PdfReader(report_pdf).pages) == 8


def test_tabla_con_todos_los_grupos(report_text):
    for group in ("DUK SO AEP D XEL ED", "AEP D XEL ED EXC WEC"):
        assert group in report_text
