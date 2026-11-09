# Post-mortem — Lab03_MyST_Equipo3

> **Borrador.** Los puntos marcados con `[EQUIPO]` solo los puede responder el equipo. Hay que completarlos y borrar este aviso antes de entregar.

## 1. Hitos: qué se cumplió y qué se retrasó

Todos los entregables técnicos quedaron listos, pero no con un ritmo sostenido: pipeline, selección, filtros, backtest, dashboard y reporte se construyeron cerca de la fecha límite, y el repositorio se creó el mismo día de la entrega (2026-11-08). Por eso:

- El charter y el sprint se escribieron junto con el código, no antes. Sus fechas registran cuándo se terminó cada hito, no un plan previo.
- Hay un solo archivo de sprint, aunque el laboratorio pide uno por semana.
- El historial de commits refleja la subida final y no el desarrollo real.

`[EQUIPO]` ¿Qué impidió empezar antes? ¿Qué hitos se pensaba tener en semanas anteriores?

## 2. Cuellos de botella

**Técnicos**

- **Plataforma.** `statsmodels` y `pyarrow` no tienen binarios para Windows ARM64. ADF, Engle-Granger y Johansen se programaron con `numpy` y se validaron contra resultados de referencia. El costo es más código propio que defender.
- **Contradicciones en las fuentes.** `Lab03.pdf` y las sesiones S11–S14 difieren en varios puntos: los 6 pasos de Powell, si Engle-Granger elimina o prioriza, y si el z-score sale del spread o de los residuales del VECM. Hubo que fijar 14 convenciones (README, sección 4).
- **Tensión entre la prueba 4 y la estrategia.** Se pide que las innovaciones no estén autocorrelacionadas, pero un spread que revierte despacio es persistente: la autocorrelación de rezago 1 es 0.97 en train. Con datos simulados del modelo la prueba se cumple; con datos reales solo se exige la varianza.
- **Errores encontrados en la revisión final.** La rejilla incluía z_salida = 0, que nunca cerraba posiciones. Además, el tramo de test aparecía "sin pérdidas" aunque heredó −$7,950 de una operación abierta en train. Ambos se corrigieron o se declararon; las métricas pasaron pruebas antes de que alguien notara lo segundo.

**Organizacionales**

- La búsqueda del universo y las decisiones de diseño se concentraron al final, lo que dejó poco margen para explorar alternativas.
- El grupo solo se conoció después de construir el embudo, con muy poco tiempo para reaccionar a un resultado débil.
- `[EQUIPO]` ¿Cómo se repartió y coordinó el trabajo en la práctica? ¿Qué parte quedó concentrada en una persona? ¿Cómo aseguraron que los tres entienden todo el código, parte del cual se generó con IA?

## 3. Qué haríamos distinto en producción

- **No operar este grupo.** Una relación que pasa por muy poco la traza, no pasa el máximo eigenvalor y pierde estacionariedad fuera de muestra no justifica arriesgar capital.
- **Exigir robustez antes de operar:** cointegración con varios rezagos, en ambos estadísticos y en la mayoría de las subventanas, con corrección por pruebas múltiples.
- **Corregir el sesgo de supervivencia.** El universo se eligió con empresas que existen hoy. Haría falta un universo histórico con datos point-in-time.
- **Modelar la ejecución real:** deslizamiento, profundidad de libro y disponibilidad y costo real del préstamo por acción. Aquí se ejecuta al cierre sin deslizamiento y con préstamo fijo de 0.25 %.
- **Monitorear en vivo:** retesteo periódico de la cointegración, alertas sobre innovaciones y salida automática del grupo cuando la relación se rompa.
- **Separar quien calibra de quien valida.** El walk-forward de este proyecto no es estrictamente fuera de muestra, porque el grupo y los filtros se calibraron con todo train.
- `[EQUIPO]` ¿Qué cambiarían en la forma de trabajar como equipo?

## 4. Cómo se materializaron los riesgos del registro

| # | Riesgo | ¿Se materializó? | Evidencia |
|---|---|---|---|
| 1 | Pérdida de cointegración fuera de muestra | **Sí** | ADF p del spread: 0.0008 en train, 0.086 en test y 0.119 en validación |
| 2 | Falso positivo por pruebas múltiples | **Probable** | 1 de 28 grupos pasa y se esperan 1.4 por azar; traza de 96.87 contra 95.75 y máximo eigenvalor sin rechazo; solo 14 % de subventanas con cointegración |
| 3 | Sensibilidad a rezagos | **Sí** | r = 1 con k = 9 y k = 5; r = 0 con k = 1 y k = 2; el AIC eligió el límite de la búsqueda |
| 4 | Divergencia de los filtros | **No** | Las ganancias convergen. La varianza de innovaciones del filtro 1 sube de 0.87 en train a 1.55 en validación, señal de deterioro y no de divergencia |
| 5 | Calidad de datos | **No** | `data/prices.csv` versionado; sin faltantes. Queda sin cubrir una descarga parcial, que guardaría el CSV antes de validarlo |
| 6 | Eventos corporativos en el grupo | **No evaluado** | No se analizaron los quiebres de β alrededor de cada evento: es una omisión nuestra |
| 7 | Muestra pequeña | **Sí** | 16 operaciones en 14.8 años; las métricas por tramo (10, 3 y 3 operaciones) no son estadísticamente concluyentes |

**Riesgo que no anticipamos:** la conclusión del laboratorio depende menos de la estrategia que del proceso de selección. El resultado neto positivo (Sharpe de 0.60 en train, 0.08 en test y 0.51 en validación) no valida la relación de equilibrio.
