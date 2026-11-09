# Sprint 01

**Objetivo:** tener el pipeline completo corriendo con un solo comando, desde los datos hasta el dashboard.
Debe producir los artefactos y las figuras exigidas, y pasar las pruebas obligatorias.

| Tarea | Estado |
|---|---|
| Ingesta y validación de precios | Hecho |
| Embudo de selección sobre 28 grupos | Hecho |
| Filtros de Kalman 1 y 2 | Hecho |
| VECM y regla de señales | Hecho |
| Backtest con costos y walk-forward | Hecho |
| Dashboard en Streamlit | Hecho |
| Pruebas con pytest | Hecho |
| Reporte PDF | Hecho |

**Bloqueos y desviaciones:**
- `statsmodels` y `pyarrow` no tienen binarios para Windows ARM64. ADF y Johansen se implementaron con `numpy` y se validaron contra referencia.
- Solo 1 de 28 grupos pasó Johansen. Ver la sección 5.6 del README.
