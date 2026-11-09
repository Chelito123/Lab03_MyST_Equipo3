# Project charter — Lab03_MyST_Equipo3

## Alcance

**Sí hará:** seleccionar un grupo de 6 utilities cointegradas, estimar pesos dinámicos y señal con dos
filtros de Kalman, operar con VECM y bandas de z-score, y evaluar con costos en train, test y validación.

**No hará:** Risk Parity entre grupos (excluido por el profesor), datos intradía ni ejecución real.

## Hitos

| Hito | Fecha objetivo | Estado |
|---|---|---|
| Pipeline de datos completo | 2026-11-08 | Hecho |
| Prueba de cointegración completa | 2026-11-08 | Hecho |
| Filtro 1 validado | 2026-11-08 | Hecho |
| Backtest completo | 2026-11-08 | Hecho |
| Dashboard funcionando | 2026-11-08 | Hecho |
| Reporte `docs/reporte.pdf` | 2026-11-08 | Hecho |
| Exposición | 2026-11-09 | Pendiente |

## Registro de riesgos

| # | Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|---|
| 1 | Pérdida de cointegración fuera de muestra. Ya observada: ADF p = 0.086 en test y 0.119 en validación | Alta | Alto | Stop sobre \|z\|, tenencia máxima y reporte explícito |
| 2 | Falso positivo por pruebas múltiples: 1 de 28 grupos pasa y se esperan 1.4 por azar | Alta | Alto | Declarar los 28 grupos y discutirlo en la exposición |
| 3 | Sensibilidad al número de rezagos: r = 1 con k = 9 y r = 0 con k = 1 | Alta | Medio | Reportar la columna `r_traza_k1` de la tabla de selección |
| 4 | Divergencia de los filtros | Baja | Alto | Forma de Joseph, burn-in de 60 días y diagnóstico de innovaciones |
| 5 | Calidad de datos: reajustes de `yfinance` | Media | Medio | `data/prices.csv` versionado y validación en cada corrida |
| 6 | Eventos corporativos en el grupo: Exelon–Pepco, escisión de Constellation (2022), desinversiones de Dominion | Media | Medio | Revisar los quiebres de β̂_t alrededor de esas fechas |
| 7 | Muestra pequeña: 16 operaciones en 14.8 años | Alta | Alto | No concluir significancia estadística con estas métricas |

## Roles

| Integrante | Responsabilidad | Carpetas (contribución esperada en Git) |
|---|---|---|
| Adrián Marcelo Ballesteros Herrera | Datos, universo, embudo de selección y Johansen | `data/`, `cointegration/`, `config.py`, `main.py`, `README.md`, `tests/test_stats.py` |
| Erik del Castillo Román | Filtros de Kalman, VECM, señales y formulación de Powell | `kalman/`, `strategy/`, `project_management/`, `tests/test_lab_requirements.py`, `tests/test_report_data.py`, `tests/conftest.py` |
| Jesús Emmanuel Flores Cortés | Backtest, métricas, dashboard y reporte | `backtest/`, `performance/`, `dashboard/`, `docs/`, `artifacts/`, `tests/test_report.py`, `tests/test_robustness.py`, `tests/test_review_fixes.py`, `tests/test_repo_structure.py` |
