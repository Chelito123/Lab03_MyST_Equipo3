# Desviaciones respecto al Lab03 y su justificación

Este proyecto se aparta de la letra del laboratorio en tres puntos del modelo. Ninguno es una omisión: cada uno se decidió antes de ver los resultados de test y validación, se justifica con la teoría del curso (S12–S14) y se reporta con cifras de `artifacts/`. Las convenciones completas están en el `README.md`, sección 4.

| # | El laboratorio pide | Lo que hace el proyecto | Convención |
|---|---|---|---|
| 1 | Z-score derivado de los residuales del VECM (sección 3.6) | Z-score sobre el término de corrección de error del VECM | C1 |
| 2 | Media μ<sub>t</sub> y varianza σ<sub>t</sub>² como estado del filtro 2 (sección 3.4) | μ<sub>t</sub> es el estado; σ<sub>t</sub>² es la varianza de la innovación | C6 |
| 3 | Engle-Granger como pre-filtro con p < 0.05 (sección 3.2) | Engle-Granger ordena los grupos y no los elimina | C2 |

---

## 1. El z-score sale del término de corrección de error, no de ε<sub>t</sub>

**Qué hacemos.** El z-score se calcula sobre ECT<sub>t</sub> = β̂ᵀ log P<sub>t</sub> − c, la desviación del equilibrio. Es el spread que produce el filtro 1 y sobre el que trabaja el filtro 2.

**Por qué no usamos los residuales ε<sub>t</sub> de la ecuación del VECM.**
- ε<sub>t</sub> son los choques de ΔP<sub>t</sub> = α·ECT<sub>t−1</sub> + ΣΓ<sub>j</sub>ΔP<sub>t−j</sub> + ε<sub>t</sub>. Por construcción son impredecibles: no tienen memoria y no revierten a la media, así que no hay nada que operar.
- Son un vector de 6 dimensiones, una por activo, y la regla de bandas necesita una sola serie.
- Lo que revierte a la media es el término de corrección de error. Es la serie que la sesión S14 (p.14) define como ECT y la que se opera en arbitraje estadístico (Chan, 2013).

**Qué aporta el VECM.** Comprueba que el sistema corrige y da la velocidad de reversión:
- βᵀα = −0.0256, dentro de (−2, 0): el sistema corrige.
- Vida media implícita de 26.7 días, consistente con los 29.7 días por Ornstein-Uhlenbeck sobre el spread de train.

**Costo.** Las secciones 3.4 y 3.6 del laboratorio describen la misma serie solo si se acepta esta lectura. Si el profesor esperaba ε<sub>t</sub>, la señal y todos los resultados serían distintos.

## 2. σ<sub>t</sub>² es la varianza de la innovación, no un estado aparte

**Qué hacemos.** El filtro 2 es un modelo de nivel local: su estado es la media latente μ<sub>t</sub>. La varianza que normaliza el z-score es la de la innovación del propio filtro, σ̂²<sub>t|t−1</sub> = p<sub>t−1</sub> + q + r, calculada con los estadísticos **previos** a la actualización.

**Por qué.** Es la forma que da la sesión S13 (p.20) para un z-score causal. Un filtro de Kalman lineal no puede llevar la varianza como estado sin volverse no lineal: haría falta un filtro extendido, de partículas o un modelo de volatilidad estocástica.

**Costo, declarado como limitación.** Con r fijo, σ̂ converge a una constante y no se adapta a cambios de volatilidad. La varianza de las innovaciones normalizadas del filtro 1 sube de 0.87 en train a 1.36 en test y 1.55 en validación. La del filtro 2 pasa de 0.84 en train a 1.07 en test y 0.66 en validación.

## 3. Engle-Granger ordena los grupos en lugar de eliminarlos

**Qué hacemos.** Se corre Engle-Granger en los 15 pares de cada grupo y se cuenta cuántos tienen p < 0.05. Ese conteo ordena a los grupos que sobreviven, y Johansen es el criterio definitivo.

**Por qué.** Un grupo puede no tener ningún par estacionario y aun así ser estacionario como portafolio: es el caso que la sesión S14 (p.6) usa para justificar pasar de pares a grupos, y la p.16 dice explícitamente "prioritization, not elimination". Usar Engle-Granger para eliminar descartaría justo esos grupos.

**El resultado no cambia.** Con Engle-Granger como eliminador, como pide la letra del laboratorio, seguirían 15 de los 28 grupos, y el grupo elegido sobrevive con 1 par significativo. La selección final es la misma con cualquiera de las dos lecturas.

---

## Relacionado: la prueba 4 con datos reales

No es una desviación del modelo, pero se declara por la misma razón. Con datos reales, las innovaciones normalizadas cumplen la varianza cercana a uno, pero no la autocorrelación no significativa: la autocorrelación de rezago 1 es 0.97 en ambos filtros, contra una banda de ±0.042.

Un spread que revierte despacio es persistente por construcción. Bajar esa autocorrelación exigiría una ganancia alta, que blanquearía el spread y dejaría sin señal la estrategia. También indica que el filtro 1 no está bien especificado como modelo de estado.

Con datos simulados del propio modelo, los dos filtros sí cumplen ambas condiciones (`tests/test_lab_requirements.py`).
