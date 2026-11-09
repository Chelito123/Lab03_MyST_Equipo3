# Dashboard del Lab03

El código está en `dashboard/app.py`, como pide la estructura obligatoria. Lee `artifacts/` y no corre el backtest.

```bash
python main.py
python -m streamlit run dashboard/app.py
```

Se abre en http://localhost:8501 con dos pestañas:

- **Vista ejecutiva:** curva de capital con drawdown, tarjeta de desempeño por tramo, atribución de costos, distribución de retornos y bitácora filtrable.
- **Vista técnica:** spread con señales, z-score con umbrales, pesos, eigenvector móvil, ganancia de Kalman, varianza posterior y Sharpe móvil.

La guía completa de navegación está en la sección 12 del `README.md`.
