"""Genera docs/reporte.pdf a partir de artifacts/. Uso: python docs/build_report.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

import config  # noqa: E402
from performance.report_data import (  # noqa: E402
    fmt_profit_factor, load_context, worked_example_f1, worked_example_f2,
)
from strategy.model import initial_state_f1  # noqa: E402

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                              italic="DejaVu", boldItalic="DejaVu-Bold")

BODY = ParagraphStyle("body", fontName="DejaVu", fontSize=9, leading=12.5, spaceAfter=5)
H1 = ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=13, leading=16, spaceBefore=9, spaceAfter=5, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=10, leading=13, spaceBefore=5, spaceAfter=3, keepWithNext=1)
TITLE = ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=17, leading=21, spaceAfter=4)
CELL = ParagraphStyle("cell", fontName="DejaVu", fontSize=7.5, leading=9.5)

FIGURES = [
    ("01_precios.png", "Las seis utilities se mueven juntas, pero con tendencias que se separan por años."),
    ("02_spread.png", "Cada entrada ocurre en un extremo del spread; se ve su deriva fuera de train."),
    ("03_pesos.png", "Los pesos cambian poco: la Q elegida hace del filtro 1 un ajuste lento."),
    ("04_eigenvector.png", "En ventanas de un año el eigenvector es muy inestable: la relación no se reestima igual."),
    ("05_ganancia.png", "Ambas ganancias caen rápido al inicio y se estabilizan: los filtros convergen."),
    ("06_zscore.png", "El z-score cruza la entrada pocas veces, de ahí el número bajo de operaciones."),
    ("07_capital.png", "El capital crece sobre todo en train; test queda casi plano."),
    ("08_retornos.png", "Con tan pocas operaciones, la forma de la distribución no se puede evaluar."),
]


def fmt(value, decimals=3):
    """Número con decimales, o guion si no hay dato."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{decimals}f}"


def pct(value, decimals=1):
    """Fracción expresada en porcentaje."""
    return f"{value * 100:.{decimals}f} %"


def para(text):
    """Párrafo con el estilo del cuerpo."""
    return Paragraph(text, BODY)


def table(rows, widths=None, font_size=7.5):
    """Tabla con encabezado sombreado; las celdas de texto largo se ajustan."""
    style = ParagraphStyle("c", parent=CELL, fontSize=font_size, leading=font_size + 2)
    wrapped = [[Paragraph(str(cell), style) for cell in row] for row in rows]
    result = Table(wrapped, colWidths=widths, repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f1")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return result


def frame_rows(frame, formats):
    """Encabezado y filas de un DataFrame con un formato por columna."""
    rows = [list(formats)]
    for _, row in frame.iterrows():
        rows.append([f(row[column]) for column, f in formats.items()])
    return rows


def selected_row(context):
    """Fila del grupo elegido en la tabla de selección."""
    table_ = context["seleccion"]
    target = set(context["modelo"]["activos"])
    mask = table_["grupo"].map(lambda g: set(g.split()) == target)
    return table_[mask].iloc[0]


def powell_table(rows):
    """Tabla de dos columnas para las formulaciones de Powell."""
    return table([["Elemento", "Formulación"]] + rows, widths=[3.6 * cm, 13.4 * cm], font_size=8)


def examples(context, artifacts):
    """Ejemplos numéricos del filtro 1 y 2 con los parámetros calibrados."""
    model = context["modelo"]
    prices = pd.read_csv(artifacts / "precios_grupo.csv", index_col=0, parse_dates=True)[model["activos"]]
    log_prices = np.log(prices)
    train = log_prices.loc[:model["train_fin"]].values
    beta = np.array(model["beta"])
    z0, P0, Q = initial_state_f1(train, beta, model["vecm"]["c"], model["r1"], model["q_ratio_f1"])
    spread = pd.read_csv(artifacts / "series.csv", index_col=0)["spread"].values
    return (worked_example_f1(log_prices.values, z0, P0, Q, model["r1"]),
            worked_example_f2(spread, 0.0, model["r2"], model["q_ratio_f2"] * model["r2"], model["r2"]))


def section_summary(context):
    model, metrics, row = context["modelo"], context["metricas"], selected_row(context)
    diag = context["diagnosticos"]
    alive = model["grupos_sobrevivientes"]
    return [
        Paragraph("1. Resumen", H1),
        para(f"Se buscó un portafolio cointegrado de {len(model['activos'])} activos entre "
             f"{model['grupos_evaluados']} grupos de utilities reguladas de EE.UU. y se operó con dos filtros de "
             f"Kalman y un VECM. <b>{alive} de {model['grupos_evaluados']}</b> grupos pasó Johansen en train; al 5 % "
             f"se esperan {0.05 * model['grupos_evaluados']:.1f} por azar."),
        para(f"<b>Veredicto: la relación encontrada es más probablemente un artefacto de la búsqueda que una "
             f"relación estructural.</b> La traza apenas supera su valor crítico ({row['traza_r0']:.2f} contra "
             f"{row['traza_cv95']:.2f}), el máximo eigenvalor no rechaza ({row['max_eig_r0']:.2f} contra "
             f"{row['max_eig_cv95']:.2f}), solo {pct(row['estabilidad'], 0)} de las subventanas encuentra "
             f"cointegración y el spread deja de ser estacionario fuera de train (ADF p = "
             f"{diag['test']['adf_p_spread']:.3f} en test y {diag['validacion']['adf_p_spread']:.3f} en validación)."),
        para(f"La estrategia terminó con ganancia neta (Sharpe {metrics['train']['sharpe']:.2f} en train, "
             f"{metrics['test']['sharpe']:.2f} en test y {metrics['validacion']['sharpe']:.2f} en validación), pero "
             f"con solo {metrics['total']['operaciones']} operaciones no es estadísticamente concluyente."),
    ]


def section_data(context):
    model = context["modelo"]
    quality = model["calidad_datos"]
    return [
        Paragraph("2. Datos y universo", H1),
        para(f"Precios diarios ajustados de yfinance del {quality['inicio']} al {quality['fin']} "
             f"({quality['observaciones']} días). Partición cronológica 60/20/20: train hasta {model['train_fin']}, "
             f"test hasta {model['test_fin']} y validación hasta el final. Todo se calibra en train. El mayor "
             f"movimiento diario fue de {pct(quality['max_retorno_abs'])}."),
        para("<b>Universo:</b> DUK, SO, AEP, D, XEL, ED, EXC y WEC, rivales directos con negocio regulado y "
             "exposición común a las tasas de interés y a la regulación tarifaria. Esa justificación económica se "
             "fijó antes de probar y es la defensa frente a las pruebas múltiples. El grupo tiene 6 activos por "
             "2 × 3 integrantes."),
    ]


def section_selection(context):
    data, model, row = context["seleccion"].copy(), context["modelo"], selected_row(context)
    data["etapa"] = data["etapa_descarte"].fillna("sobrevive")
    formats = {
        "Grupo": lambda v: v, "Corr": lambda v: fmt(v, 2), "EG": lambda v: int(v), "k": lambda v: int(v),
        "r traza": lambda v: int(v), "r máx": lambda v: int(v), "r k=1": lambda v: int(v),
        "Traza": lambda v: fmt(v, 1), "Máx": lambda v: fmt(v, 1), "ADF p": lambda v: fmt(v, 3),
        "VM": lambda v: fmt(v, 1), "Estab.": lambda v: fmt(v, 2), "Etapa": lambda v: v,
    }
    columns = {"Grupo": "grupo", "Corr": "corr_media", "EG": "eg_pares_p05", "k": "k_ar_diff",
               "r traza": "r_traza", "r máx": "r_max_eig", "r k=1": "r_traza_k1", "Traza": "traza_r0",
               "Máx": "max_eig_r0", "ADF p": "adf_p", "VM": "vida_media", "Estab.": "estabilidad",
               "Etapa": "etapa"}
    shown = data[list(columns.values())].rename(columns={v: k for k, v in columns.items()})
    widths = [x * cm for x in (3.9, .85, .7, .6, .9, .8, .8, 1.05, .95, 1.05, .9, 1.05, 2.5)]
    vectors = context["vectores"]
    chosen = vectors[vectors["grupo"].map(lambda g: set(g.split()) == set(model["activos"]))]
    vector_rows = frame_rows(chosen.rename(columns={
        "vector": "Vector", "eigenvalor": "Eigenvalor", "adf_p": "ADF p", "vida_media": "Vida media"}), {
        "Vector": lambda v: int(v), "Eigenvalor": lambda v: fmt(v, 4),
        "ADF p": lambda v: fmt(v, 4), "Vida media": lambda v: fmt(v, 1)})
    return [
        Paragraph("3. Selección del grupo", H1),
        para(f"Se evaluaron <b>{len(data)} grupos en total</b>, solo con train. Embudo: correlación móvil de 252 días "
             "sobre retornos, Engle-Granger, Johansen, validación del spread y estabilidad en subventanas. "
             "Engle-Granger <b>prioriza y no elimina</b>, porque un grupo puede no tener pares estacionarios y serlo "
             "como portafolio. Esto se aparta de la lectura literal del laboratorio y es deliberado. "
             f"Si Engle-Granger eliminara, seguirían {int((data['eg_pares_p05'] >= 1).sum())} de {len(data)} grupos "
             f"y el elegido sobreviviría con {int(row['eg_pares_p05'])} par significativo."),
        table(frame_rows(shown, formats), widths=widths, font_size=6),
        para("EG: pares con p &lt; 0.05 de 15. k: rezagos del VECM. r k=1: rango con un solo rezago. "
             "Traza y Máx: estadísticos para r = 0. VM: vida media en días. Estab.: fracción de subventanas de "
             "2 años con r ≥ 1."),
        para(f"<b>Vector seleccionado.</b> Con rango {int(row['r_traza'])} hay un solo vector candidato: mayor "
             "eigenvalor significa corrección de error más fuerte."),
        table(vector_rows, widths=[2 * cm, 3 * cm, 3 * cm, 3 * cm], font_size=8),
        Spacer(1, 4),
        para("Grupo y pesos normalizados: " + ", ".join(
            f"{a} ({b:+.3f})" for a, b in zip(model["activos"], model["beta"])) + "."),
    ]


def gain_summary(artifacts):
    """Ganancias inicial y final de ambos filtros."""
    series = pd.read_csv(artifacts / "series.csv", index_col=0)
    return {c: (series[c].iloc[0], series[c].iloc[-1]) for c in ("ganancia_f1", "ganancia_f2")}


def section_filter1(context, table_f1, gains):
    model = context["modelo"]
    rows = frame_rows(table_f1, {
        "t": lambda v: int(v), "y": lambda v: fmt(v, 4), "innovacion": lambda v: fmt(v, 4),
        "S": lambda v: fmt(v, 5), "K_norma": lambda v: fmt(v, 4), "beta_2": lambda v: fmt(v, 5)})
    return [
        Paragraph("4. Filtro 1: pesos dinámicos (marco de Powell)", H1),
        para(f"<b>Normalización y signo.</b> El activo 1 es la pata de mayor |β| en train ({model['activos'][0]}) y su "
             "peso se fija en +1; una sola división fija escala y signo. El estado es un intercepto más los 5 pesos "
             "libres. <b>Inicialización:</b> z<sub>0</sub> es el vector de Johansen normalizado con c = media del "
             f"spread en train. <b>R</b> = {model['r1']:.2e}, varianza del spread estático en train. "
             f"<b>Q</b> = {model['q_ratio_f1']:.0e} · R · I, razón elegida por rejilla en train. "
             "P<sub>0</sub> equivale a la incertidumbre de una observación. <b>Burn-in:</b> "
             f"{config.BURN_IN} días sin operar."),
        Paragraph("Proceso de 6 pasos", H2),
        powell_table([
            ["1. Problema", "Estimar cada día los pesos del portafolio reversivo cuando la relación deriva."],
            ["2. Estado", "z<sub>t</sub> = (c<sub>t</sub>, β2…β6) y su covarianza P<sub>t</sub>."],
            ["3. Transición", "Caminata aleatoria: z<sub>t</sub> = z<sub>t-1</sub> + w<sub>t</sub>, w ~ N(0, Q)."],
            ["4. Observación", "y<sub>t</sub> = log P1<sub>t</sub> = C<sub>t</sub> z<sub>t</sub> + v<sub>t</sub>, con C<sub>t</sub> = (1, −log P2…−log P6) y v ~ N(0, R)."],
            ["5. Regla de decisión", "x<sub>t</sub> = K<sub>t</sub> ỹ<sub>t</sub>: corrección de la estimación. Política PFA con θ = (Q, R)."],
            ["6. Aprendizaje", "Predicción y actualización de Kalman; el spread es la innovación con el estado previo."],
        ]),
        Paragraph("5 elementos del modelo", H2),
        powell_table([
            ["Variables de estado", "B<sub>t</sub> = (ẑ<sub>t</sub>, P<sub>t</sub>), creencia; I<sub>t</sub> = precios observados."],
            ["Variables de decisión", "x<sub>t</sub> = K<sub>t</sub> ỹ<sub>t</sub>."],
            ["Información exógena", "W<sub>t+1</sub> = log-precios del día siguiente."],
            ["Función de transición", "Ecuaciones de predicción y actualización."],
            ["Función objetivo", "min E[ỹ<sub>t</sub>²], error de predicción cuadrático."],
        ]),
        para(f"<b>Ganancia.</b> K<sub>t</sub> = P<sub>t|t-1</sub> C<sub>t</sub>ᵀ / S<sub>t</sub>. Su norma pasa de "
             f"{gains['ganancia_f1'][0]:.4f} el primer día a {gains['ganancia_f1'][1]:.4f} al final: cuando baja la "
             "incertidumbre, cada precio nuevo mueve menos los pesos. No llega a cero porque Q se suma en cada paso."),
        Paragraph("Ejemplo numérico: primeros 4 días", H2),
        table(rows, widths=[1.2 * cm] + [2.8 * cm] * 5, font_size=8),
    ]


def section_filter2(context, table_f2, gains):
    model = context["modelo"]
    rows = frame_rows(table_f2, {
        "t": lambda v: int(v), "s": lambda v: fmt(v, 4), "mu_previa": lambda v: fmt(v, 4),
        "sigma_previa": lambda v: fmt(v, 4), "z_previo": lambda v: fmt(v, 3), "K": lambda v: fmt(v, 4),
        "mu_posterior": lambda v: fmt(v, 4), "z_posterior": lambda v: fmt(v, 3)})
    return [
        Paragraph("5. Filtro 2: generación de señal (marco de Powell)", H1),
        para(f"Modelo de nivel local sobre S<sub>t</sub>. <b>r</b> = {model['r2']:.2e}, varianza del spread de train "
             f"tras el burn-in; <b>q</b> = {model['q_ratio_f2']:.0e} · r. La varianza σ² es la de la innovación, "
             "p + q + r. El z-score usa los estadísticos <b>previos</b> a la actualización: con los posteriores, el "
             "dato del día se juzgaría a sí mismo y el z se encogería por (1 − K)."),
        Paragraph("Proceso de 6 pasos", H2),
        powell_table([
            ["1. Problema", "Saber si el spread de hoy está lejos de su nivel normal, sin look-ahead."],
            ["2. Estado", "μ<sub>t</sub> y su varianza p<sub>t</sub>."],
            ["3. Transición", "μ<sub>t</sub> = μ<sub>t-1</sub> + w<sub>t</sub>, w ~ N(0, q)."],
            ["4. Observación", "S<sub>t</sub> = μ<sub>t</sub> + v<sub>t</sub>, v ~ N(0, r)."],
            ["5. Regla de decisión", "x<sub>t</sub> = K<sub>t</sub> (S<sub>t</sub> − μ<sub>t|t-1</sub>). Política PFA con θ = (q, r)."],
            ["6. Aprendizaje", "Actualización recursiva de μ<sub>t</sub> y p<sub>t</sub>."],
        ]),
        Paragraph("5 elementos del modelo", H2),
        powell_table([
            ["Variables de estado", "B<sub>t</sub> = (μ<sub>t</sub>, p<sub>t</sub>), creencia; I<sub>t</sub> = S<sub>t</sub>."],
            ["Variables de decisión", "Corrección de la media, x<sub>t</sub>."],
            ["Información exógena", "W<sub>t+1</sub> = S<sub>t+1</sub>."],
            ["Función de transición", "Predicción y actualización del nivel local."],
            ["Función objetivo", "min E[ν<sub>t</sub>²], error de la innovación."],
        ]),
        para(f"<b>Ganancia.</b> K<sub>t</sub> = (p + q)/(p + q + r); pasa de {gains['ganancia_f2'][0]:.4f} a "
             f"{gains['ganancia_f2'][1]:.4f}. Un valor bajo significa que el nivel normal se mueve despacio, "
             "que es lo que permite ver desviaciones que duran semanas."),
        Paragraph("Ejemplo numérico: primeros 6 días (z previo contra posterior)", H2),
        table(rows, font_size=7.5),
    ]


def section_system():
    return [
        Paragraph("6. Sistema completo como proceso de decisión secuencial de Powell", H1),
        powell_table([
            ["Estado", "<b>Componente físico</b> R<sub>t</sub>: efectivo, acciones por pata y días en posición. <b>Componente de información</b> I<sub>t</sub>: precios y spread. <b>Componente de creencia</b> B<sub>t</sub>: (ẑ<sub>t</sub>, P<sub>t</sub>) del filtro 1 y (μ<sub>t</sub>, p<sub>t</sub>) del filtro 2. En S<sub>0</sub> quedan fijos β de Johansen, α del VECM, θ y los costos."],
            ["Decisión", "d<sub>t</sub> ∈ {−1, 0, +1} y acciones objetivo q<sub>i</sub> = d·sgn(β<sub>i</sub>)·N<sub>i</sub>/P<sub>i</sub>, con N<sub>i</sub> = 0.8·V·|β<sub>i</sub>|/Σ|β<sub>j</sub>|. Se decide al cierre de t y se ejecuta al cierre de t+1."],
            ["Información exógena", "W<sub>t+1</sub> = precios del día siguiente."],
            ["Transición", "Filtros, marcado a mercado, devengo de préstamo y comisión por pata."],
            ["Objetivo", "max E[Σ C(S<sub>t</sub>, X<sup>π</sup>(S<sub>t</sub>), W<sub>t+1</sub>)], P&amp;L neto; se calibra por Sharpe de train."],
            ["Política", "PFA con θ = (z entrada, z salida, z stop, h máx)."],
        ]),
        Spacer(1, 4),
        para("Los 6 pasos para el sistema: el <b>problema</b> es capturar la reversión del portafolio con costos; "
             "el <b>estado</b> es el de la tabla; la <b>transición</b> combina filtros y contabilidad; la "
             "<b>observación</b> son los precios de cierre; la <b>regla de decisión</b> son las bandas de z-score; "
             "y el <b>aprendizaje</b> son las actualizaciones recursivas de ambos filtros."),
    ]


def section_vecm(context):
    model, vecm, rule = context["modelo"], context["modelo"]["vecm"], context["modelo"]["regla"]
    alpha = ", ".join(f"{a} {b:+.4f}" for a, b in zip(model["activos"], vecm["alpha"]))
    top = context["grid_reglas"].sort_values("sharpe_train", ascending=False).head(6)
    rows = frame_rows(top.rename(columns={"z_entrada": "z entrada", "z_salida": "z salida", "z_stop": "z stop",
                                          "h_max": "h máx", "sharpe_train": "Sharpe train",
                                          "operaciones": "Operaciones"}), {
        "z entrada": lambda v: fmt(v, 2), "z salida": lambda v: fmt(v, 2), "z stop": lambda v: fmt(v, 1),
        "h máx": lambda v: int(v), "Sharpe train": lambda v: fmt(v, 3), "Operaciones": lambda v: int(v)})
    filters = frame_rows(context["grid_filtros"].rename(columns={
        "q_ratio_f1": "Q/R filtro 1", "q_ratio_f2": "q/r filtro 2", "sharpe_train": "Sharpe train",
        "operaciones": "Operaciones"}), {
        "Q/R filtro 1": lambda v: f"{v:.0e}", "q/r filtro 2": lambda v: f"{v:.0e}",
        "Sharpe train": lambda v: fmt(v, 3), "Operaciones": lambda v: int(v)})
    return [
        Paragraph("7. VECM y señales", H1),
        para("<b>Desviación respecto a la sección 3.6 del laboratorio.</b> El z-score se calcula sobre el término de corrección de error ECT<sub>t</sub> = βᵀ log P<sub>t</sub> − c, "
             "que es el spread del filtro 1. Los residuales ε<sub>t</sub> de la ecuación del VECM son seis choques "
             "impredecibles por construcción: no revierten y no dan una señal única."),
        para(f"<b>Velocidades de ajuste α:</b> {alpha}. βᵀα = {vecm['beta_alpha']:.4f}, dentro de (−2, 0): el "
             f"sistema corrige. Vida media implícita del VECM: {vecm['vida_media']:.1f} días; por Ornstein-Uhlenbeck "
             f"en train: {model['vida_media']:.1f} días."),
        para(f"<b>Búsqueda en rejilla, solo en train</b>, en dos etapas y exigiendo al menos 10 operaciones. "
             f"Elegidos: z entrada {rule['z_entrada']}, z salida {rule['z_salida']}, z stop {rule['z_stop']}, "
             f"tenencia máxima {rule['h_max']} días. Tras un stop o una tenencia máxima no se reabre hasta que |z| "
             "regrese a la salida."),
        Paragraph("Etapa 1: filtros", H2),
        table(filters, widths=[3 * cm] * 4, font_size=8),
        Paragraph("Etapa 2: mejores reglas", H2),
        table(rows, font_size=8),
    ]


def carried_note(metrics):
    """Advierte qué tramos heredan operaciones abiertas del tramo anterior."""
    parts = [f"{name}: {metrics[name]['operaciones_arrastradas']}"
             for name in ("train", "test", "validacion") if metrics[name]["operaciones_arrastradas"]]
    if not parts:
        return ""
    return ("Operaciones heredadas del tramo anterior, cuyo resultado no entra en esas razones (" + ", ".join(parts) +
            "). Por eso un tramo puede decir «sin pérdidas» aunque haya absorbido una pérdida.")


def section_results(context, artifacts):
    metrics, costs, rebalance = context["metricas"], context["costos"], context["rebalanceo"]
    rows = [["Tramo", "Retorno anual", "Sharpe", "Sortino", "Calmar", "Max DD", "Win rate", "Profit factor", "Oper."]]
    for name, label in (("train", "Train"), ("test", "Test"), ("validacion", "Validación"), ("total", "Total")):
        item = metrics[name]
        rows.append([label, pct(item["retorno_anual"], 2), fmt(item["sharpe"], 2), fmt(item["sortino"], 2),
                     fmt(item["calmar"], 2), pct(item["max_drawdown"], 2), pct(item["win_rate"], 0),
                     fmt_profit_factor(item["profit_factor"]), item["operaciones"]])
    daily, band = rebalance["diario"], rebalance["banda"]
    folds = context["walk_forward"].copy()
    folds.loc[folds["operaciones_fuera"] == 0, "sharpe_fuera"] = float("nan")
    wf = frame_rows(folds.rename(columns={
        "fold": "Bloque", "sharpe_ajuste": "Sharpe ajuste", "sharpe_fuera": "Sharpe bloque siguiente",
        "operaciones_fuera": "Oper. bloque"}), {
        "Bloque": lambda v: int(v), "Sharpe ajuste": lambda v: fmt(v, 3),
        "Sharpe bloque siguiente": lambda v: fmt(v, 3), "Oper. bloque": lambda v: int(v)})
    content = [
        Paragraph("8. Resultados", H1),
        table(rows, widths=[x * cm for x in (2, 1.9, 1.4, 1.5, 1.4, 1.8, 1.6, 2.6, 1.2)], font_size=8),
        Spacer(1, 5),
        para("Win rate, profit factor y número de operaciones se atribuyen por fecha de entrada; retorno, Sharpe y "
             "drawdown incluyen el resultado diario de cada tramo. " + carried_note(metrics)),
        para(f"<b>Atribución de costos.</b> P&amp;L bruto ${costs['pnl_bruto']:,.0f}, comisiones ${costs['comisiones']:,.0f}, "
             f"préstamo ${costs['prestamo']:,.0f}, P&amp;L neto ${costs['pnl_neto']:,.0f}. Las comisiones consumen "
             f"{pct(costs['comisiones'] / costs['pnl_bruto'], 0)} de la ganancia bruta."),
        para(f"<b>Rebalanceo.</b> A diario se pagan ${daily['comisiones']:,.0f} en comisiones y el Sharpe es "
             f"{daily['sharpe']:.2f}. Con una banda de no-rebalanceo de {pct(band['banda'], 0)} se pagan "
             f"${band['comisiones']:,.0f} y el Sharpe es {band['sharpe']:.2f}."),
        Paragraph("Walk-forward anclado dentro de train", H2),
        table(wf, widths=[2 * cm, 3 * cm, 4 * cm, 3 * cm], font_size=8),
        Spacer(1, 3),
        para("Este walk-forward no es estrictamente fuera de muestra: solo los umbrales usan el pasado de cada bloque; "
             "el grupo, β, R, Q y la vida media se calibraron con todo train. Un guion indica un bloque sin operaciones."),
        PageBreak(),
        Paragraph("Figuras", H2),
    ]
    for name, note in FIGURES:
        content += [Image(str(artifacts / "figures" / name), width=16 * cm, height=7.2 * cm), para(note)]
    return content


def section_limits(context):
    row, model, diag = selected_row(context), context["modelo"], context["diagnosticos"]
    return [
        Paragraph("9. Limitaciones", H1),
        para(f"<b>Pruebas múltiples.</b> {model['grupos_evaluados']} grupos probados; el único sobreviviente es "
             "compatible con el azar."),
        para(f"<b>Sensibilidad a rezagos.</b> El AIC eligió k = {int(row['k_ar_diff'])}, en el límite de la búsqueda. "
             f"Con k = 1 el rango del grupo elegido es {int(row['r_traza_k1'])}."),
        para(f"<b>Estacionariedad fuera de muestra.</b> ADF p del spread: {diag['train']['adf_p_spread']:.4f} en train, "
             f"{diag['test']['adf_p_spread']:.3f} en test y {diag['validacion']['adf_p_spread']:.3f} en validación."),
        para(f"<b>Muestra pequeña.</b> {context['metricas']['total']['operaciones']} operaciones en toda la muestra."),
        para(f"<b>Innovaciones.</b> En train, la varianza normalizada es {diag['train']['filtro1']['varianza']:.2f} y "
             f"{diag['train']['filtro2']['varianza']:.2f}, pero la autocorrelación de rezago 1 es "
             f"{diag['train']['filtro1']['acf1']:.2f} y {diag['train']['filtro2']['acf1']:.2f}. Un spread que revierte "
             "despacio es persistente por construcción; una ganancia alta la reduciría, pero blanquearía la señal. "
             "Con datos simulados del modelo, los filtros sí cumplen ambas condiciones. Con datos reales se "
             "cumple la varianza, no la autocorrelación: unas innovaciones tan persistentes indican que el filtro 1 "
             "no está bien especificado como modelo de estado. Es un costo aceptado para conservar la señal."),
        para(f"<b>Varianza fija en el filtro 2.</b> Con r fijo, σ converge a una constante y no se adapta a cambios "
             f"de volatilidad. La varianza normalizada del filtro 1 sube de {diag['train']['filtro1']['varianza']:.2f} "
             f"en train a {diag['validacion']['filtro1']['varianza']:.2f} en validación."),
        para("<b>Métricas de train dentro de muestra.</b> Filtros y umbrales se calibran con todo train, así que el "
             "z-score y el Sharpe de train no son fuera de muestra."),
        para("<b>Supuestos.</b> Ejecución al cierre del día siguiente sin deslizamiento, tasa libre de riesgo cero, "
             "préstamo devengado sobre 252 días y ruido gaussiano. El filtro no modela colas pesadas. Se permiten "
             "acciones fraccionarias y el 80 % del capital se calcula antes de descontar la comisión."),
        para("<b>Implementación.</b> ADF, Engle-Granger y Johansen se programaron con numpy porque statsmodels no "
             "tiene binarios para Windows ARM64; se validaron contra sus resultados de referencia."),
        Paragraph("10. Uso de IA", H1),
        para("Se usó Claude Code (Anthropic) para implementar el pipeline, las pruebas, el dashboard y generar este "
             "reporte. Cada integrante es responsable del código y debe poder explicarlo."),
    ]


def build(output, artifacts=config.ARTIFACTS):
    """Construye el PDF en `output` y regresa su ruta."""
    output = Path(output)
    context = load_context(artifacts)
    table_f1, table_f2 = examples(context, artifacts)
    gains = gain_summary(artifacts)
    story = [
        Paragraph("Arbitraje estadístico con cointegración y filtros de Kalman", TITLE),
        para("Laboratorio 03 · Microestructuras y Sistemas de Trading · ITESO · Equipo 3<br/>"
             "Adrián Marcelo Ballesteros Herrera · Erik del Castillo Román · Jesús Emmanuel Flores Cortés<br/>"
             f"Semilla fijada: {context['modelo']['semilla']}"),
    ]
    story += section_summary(context) + section_data(context) + section_selection(context)
    story += section_filter1(context, table_f1, gains) + section_filter2(context, table_f2, gains)
    story += section_system() + section_vecm(context)
    story += section_results(context, artifacts) + section_limits(context)
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="Lab03 Equipo 3", author="Equipo 3").build(story)
    return output


if __name__ == "__main__":
    print(build(ROOT / "docs" / "reporte.pdf"))
