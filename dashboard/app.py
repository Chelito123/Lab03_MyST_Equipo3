"""Dashboard Streamlit del Lab03. Lee artifacts/ y nunca corre el backtest.

Uso: python -m streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
SEGMENT_COLORS = {"train": "#0071e3", "test": "#ff9f0a", "validacion": "#30d158"}
FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif'
STYLE = f"""
<style>
html, body, [class*="css"], .stMarkdown, .stRadio, .stMultiSelect {{ font-family: {FONT}; }}
.block-container {{ max-width: 1180px; padding-top: 3rem; }}
h1 {{ font-size: 3rem !important; font-weight: 600 !important; letter-spacing: -0.015em; }}
h2, h3 {{ font-weight: 600 !important; letter-spacing: -0.01em; }}
[data-testid="stCaptionContainer"] {{ opacity: 0.75; font-size: 0.95rem; }}
[data-testid="stMetric"] {{ background: rgba(128, 128, 128, 0.10); border-radius: 18px; padding: 18px 22px; }}
[data-testid="stMetricLabel"] {{ opacity: 0.75; }}
[data-testid="stMetricValue"] {{ font-weight: 600; letter-spacing: -0.02em; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 28px; border-bottom: 1px solid rgba(128, 128, 128, 0.3); }}
.stTabs [data-baseweb="tab"] {{ font-size: 1.05rem; }}
[data-testid="stPlotlyChart"] {{ background: rgba(128, 128, 128, 0.05); border-radius: 18px; padding: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08); }}
</style>
"""


@st.cache_data
def load(folder_key, stamp_key):
    """Carga las salidas del backtest; los argumentos solo invalidan la caché."""
    def read(name):
        return pd.read_csv(ARTIFACTS / name, index_col=0, parse_dates=True)

    def read_json(name):
        return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))

    trades = pd.read_csv(ARTIFACTS / "operaciones.csv", parse_dates=["entrada", "salida"])
    return {
        "series": read("series.csv"), "pesos": read("pesos.csv"),
        "eigen": read("eigenvector_movil.csv"), "trades": trades,
        "metricas": read_json("metricas.json"), "costos": read_json("costos.json"),
        "modelo": read_json("modelo.json"),
    }


def chart(figure, height=380):
    """Dibuja una figura interactiva con estilo común."""
    figure.update_layout(height=height, margin=dict(l=40, r=20, t=50, b=40), hovermode="x unified",
                         font=dict(family=FONT), title_font=dict(size=17), colorway=["#0071e3", "#ff9f0a", "#30d158",
                                                             "#bf5af2", "#ff375f", "#64d2ff"])
    st.plotly_chart(figure, width="stretch")


def executive(data):
    """Vista para portfolio manager o comité de inversión."""
    series, metricas, costos, trades = data["series"], data["metricas"], data["costos"], data["trades"]
    tramo = st.radio("Tarjeta de desempeño", ["total", "train", "test", "validacion"], horizontal=True)
    card = metricas[tramo]
    columns = st.columns(6)
    profit_factor = card["profit_factor"]
    for column, (label, value) in zip(columns, [
        ("Sharpe", f"{card['sharpe']:.2f}"), ("Sortino", f"{card['sortino']:.2f}"),
        ("Calmar", f"{card['calmar']:.2f}"), ("Max Drawdown", f"{card['max_drawdown']:.2%}"),
        ("Win Rate", f"{card['win_rate']:.0%}"),
        ("Profit Factor", "sin pérdidas" if profit_factor is None or pd.isna(profit_factor) else f"{profit_factor:.2f}"),
    ]):
        column.metric(label, value)
    st.caption(f"{card['operaciones']} operaciones por fecha de entrada. Con tan pocas operaciones, "
               "las métricas tienen mucha incertidumbre.")
    if card.get("operaciones_arrastradas"):
        st.caption(f"{card['operaciones_arrastradas']} operación(es) abierta(s) en el tramo anterior afectan el "
                   "retorno y el Sharpe de este tramo, pero no su win rate ni su profit factor.")

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                           subplot_titles=("Curva de capital (USD)", "Drawdown"))
    for name, color in SEGMENT_COLORS.items():
        part = series[series["tramo"] == name]
        figure.add_trace(go.Scatter(x=part.index, y=part["capital"], name=name, line=dict(color=color)), 1, 1)
    figure.add_trace(go.Scatter(x=series.index, y=series["drawdown"], name="Drawdown", fill="tozeroy",
                                line=dict(color="#c0392b")), 2, 1)
    figure.update_yaxes(tickformat=".1%", row=2, col=1)
    chart(figure, 520)

    left, right = st.columns(2)
    with left:
        figure = go.Figure(go.Waterfall(
            measure=["absolute", "relative", "relative", "total"],
            x=["P&L bruto", "Comisiones", "Préstamo", "P&L neto"],
            y=[costos["pnl_bruto"], -costos["comisiones"], -costos["prestamo"], 0],
        ))
        figure.update_layout(title="Atribución de costos (USD)")
        chart(figure)
    with right:
        returns = trades["retorno"].values * 100
        if len(returns):
            figure = go.Figure(go.Histogram(x=returns, histnorm="probability density", name="Operaciones"))
            grid = np.linspace(returns.min() - 1, returns.max() + 1, 200)
            figure.add_trace(go.Scatter(x=grid, y=norm.pdf(grid, returns.mean(), returns.std()), name="Normal"))
            figure.update_layout(title="Retorno por operación (%) contra normal")
            chart(figure)
        else:
            st.caption("Sin operaciones para mostrar la distribución.")

    st.subheader("Bitácora de operaciones")
    trades = trades.copy()
    trades["tramo"] = pd.cut(trades["entrada"],
                             bins=[pd.Timestamp.min, pd.Timestamp(data["modelo"]["train_fin"]),
                                   pd.Timestamp(data["modelo"]["test_fin"]), pd.Timestamp.max],
                             labels=["train", "test", "validacion"]).astype(str)
    columns = st.columns(3)
    period = columns[0].multiselect("Periodo", ["train", "test", "validacion"], ["train", "test", "validacion"])
    side = columns[1].multiselect("Dirección", ["largo", "corto"], ["largo", "corto"])
    outcome = columns[2].multiselect("Resultado", ["ganadora", "perdedora"], ["ganadora", "perdedora"])
    view = trades[trades["tramo"].isin(period) & trades["direccion"].isin(side) & trades["resultado"].isin(outcome)]
    if view.empty:
        st.caption("Sin operaciones para este filtro.")
        return
    shown = view[["tramo", "entrada", "salida", "direccion", "dias", "pnl_bruto", "comisiones", "prestamo", "pnl_neto", "resultado"]].copy()
    shown["entrada"], shown["salida"] = shown["entrada"].dt.date.astype(str), shown["salida"].dt.date.astype(str)
    for column in ["pnl_bruto", "comisiones", "prestamo", "pnl_neto"]:
        shown[column] = shown[column].map("{:,.0f}".format)
    figure = go.Figure(go.Table(header=dict(values=list(shown.columns), align="left"),
                                cells=dict(values=[shown[c] for c in shown.columns], align="left")))
    chart(figure, 420)


def technical(data):
    """Vista para investigador cuantitativo o gestor de riesgos."""
    series, regla = data["series"], data["modelo"]["regla"]
    change = series["direccion"].diff().fillna(0)
    entries, exits = series[(series["direccion"] != 0) & (change != 0)], series[(series["direccion"] == 0) & (change != 0)]

    figure = go.Figure(go.Scatter(x=series.index, y=series["spread"], name="Spread", line=dict(color="#444", width=1)))
    figure.add_trace(go.Scatter(x=entries.index, y=entries["spread"], mode="markers", name="Entrada", marker=dict(color="green", size=8)))
    figure.add_trace(go.Scatter(x=exits.index, y=exits["spread"], mode="markers", name="Salida", marker=dict(color="red", size=8)))
    figure.update_layout(title="Spread S_t con señales")
    chart(figure)

    figure = go.Figure(go.Scatter(x=series.index, y=series["zscore"], name="z", line=dict(color="#444", width=1)))
    for value, label, color in [(regla["z_entrada"], "entrada", "red"), (regla["z_salida"], "salida", "blue"), (regla["z_stop"], "stop", "black")]:
        for sign in (1, -1):
            figure.add_hline(y=sign * value, line_dash="dash", line_color=color, annotation_text=label if sign > 0 else None)
    figure.update_layout(title="Z-score con umbrales")
    chart(figure)

    left, right = st.columns(2)
    with left:
        figure = go.Figure([go.Scatter(x=data["pesos"].index, y=data["pesos"][c], name=c) for c in data["pesos"].columns])
        figure.update_layout(title="Vector de pesos β̂_t")
        chart(figure)
    with right:
        clipped = data["eigen"].clip(-5, 5)
        figure = go.Figure([go.Scatter(x=clipped.index, y=clipped[c], name=c) for c in clipped.columns])
        figure.update_layout(title="Primer eigenvector de Johansen móvil (±5)")
        chart(figure)

    left, right = st.columns(2)
    with left:
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(go.Scatter(x=series.index, y=series["ganancia_f1"], name="Filtro 1, ‖K‖"))
        figure.add_trace(go.Scatter(x=series.index, y=series["ganancia_f2"], name="Filtro 2, K"), secondary_y=True)
        figure.update_layout(title="Ganancia de Kalman")
        chart(figure)
    with right:
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(go.Scatter(x=series.index, y=series["var_posterior_f1"], name="Filtro 1, traza P"))
        figure.add_trace(go.Scatter(x=series.index, y=series["var_posterior_f2"], name="Filtro 2, p"), secondary_y=True)
        figure.update_layout(title="Varianza posterior (convergencia)")
        chart(figure)

    figure = go.Figure(go.Scatter(x=series.index, y=series["sharpe_252"], name="Sharpe 252d"))
    figure.update_layout(title="Sharpe móvil de 252 días")
    chart(figure)


def main():
    """Punto de entrada de la app."""
    st.set_page_config(page_title="Lab03 Equipo 3", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Lab03: arbitraje estadístico con Kalman")
    if not (ARTIFACTS / "metricas.json").exists():
        st.error("Faltan los artefactos. Corre primero: python main.py")
        return
    stamp = max(path.stat().st_mtime for path in ARTIFACTS.glob("*.*"))
    data = load(str(ARTIFACTS), stamp)
    st.caption("Grupo: " + ", ".join(data["modelo"]["activos"]) +
               " · Modo claro u oscuro: menú ⋮ arriba a la derecha → Light o Dark")
    tab_exec, tab_tech = st.tabs(["Vista ejecutiva", "Vista técnica"])
    with tab_exec:
        executive(data)
    with tab_tech:
        technical(data)


main()
