# Lab03_MyST_Equipo3 — Arbitraje Estadístico con Cointegración y Filtros de Kalman

Microestructuras y Sistemas de Trading · ITESO · Otoño 2026 · Módulo 3 (Arbitraje Estadístico)

Las indicaciones oficiales están en `Lab03.pdf` y la teoría en `S11` a `S14`, en la carpeta `Teoria/`
del curso.

**Resultado en una línea:** de 28 grupos de utilities probados, solo 1 pasa Johansen en train, y la
evidencia apunta a que es más un artefacto de la búsqueda que una relación estructural (sección 5.6).

---

## 1. Integrantes

| Integrante | Rol |
|---|---|
| Adrián Marcelo Ballesteros Herrera | Datos, selección de activos y cointegración: `data/`, `cointegration/`, `config.py`, `main.py`. Secciones 1–3 del reporte |
| Erik del Castillo Román | Filtros de Kalman, VECM y señales: `kalman/`, `strategy/`. Secciones 4–7 del reporte (Powell) |
| Jesús Emmanuel Flores Cortés | Backtest, métricas, dashboard y reporte: `backtest/`, `performance/`, `dashboard/`, `docs/`. Secciones 8–10 del reporte |

Equipo de 3 integrantes, así que el grupo tiene **2 × 3 = 6 activos** (justificación exigida en 3.1).

## 2. Nivel de alcance

**Nivel base: un solo grupo cointegrado.**
Por indicación del profesor, **no** se implementa Risk Parity (sección 3.7 de `Lab03.pdf`). Eso también
excluye la figura 9 y la prueba de contribuciones al riesgo de nivel B.

## 3. Descripción del proyecto

Estrategia de arbitraje estadístico sobre un portafolio de 6 activos cointegrados:

1. **Selección del grupo** (solo con la partición de entrenamiento; ver la sección 5):
   universo con justificación económica → correlación móvil > 0.7 → Engle-Granger (ADF de residuales,
   p < 0.05; prioriza, no elimina) → Johansen (traza y máximo eigenvalor) → validación del spread
   (ADF y vida media).
   Se documentan **todos** los grupos probados y en qué etapa se descartó cada uno.
2. **Kalman 1, pesos dinámicos** (S14, p.18):
   - **Estado:** z_t = (c_t, β⁽²⁾_t, …, β⁽⁶⁾_t)ᵀ ∈ ℝ⁶, como caminata aleatoria z_t = z_{t−1} + w_t,
     con w_t ~ N(0, Q). El estado tiene 6 componentes, pero **no** son los 6 pesos: es un intercepto
     más 5 pesos libres.
   - **Normalización:** el peso del activo 1 está fijo en β⁽¹⁾ ≡ +1. El activo 1 es la pata con mayor
     |β| en el vector de Johansen de entrenamiento. Así queda fijada la escala y el signo.
   - **Observación:** y_t = log P⁽¹⁾_t = C_t z_t + v_t, con C_t = (1, −log P⁽²⁾_t, …, −log P⁽⁶⁾_t)
     y v_t ~ N(0, R).
   - **Salida:** vector de pesos β̂_t = (1, β̂⁽²⁾_t, …, β̂⁽⁶⁾_t) y spread
     S_t = β̂_{t|t−1}ᵀ log P_t − ĉ_{t|t−1}, que es la innovación del filtro 1 calculada con el estado
     previo, por lo que es causal.
3. **Kalman 2, señal:** modelo de nivel local sobre S_t que estima μ_t y σ_t².
   Se calcula z_t = (S_t − μ̂_{t|t−1}) / σ̂_{t|t−1} **con los estadísticos previos a la actualización**
   (sin look-ahead).
4. **VECM:** usa el vector de Johansen como término de corrección de error. El z-score de señales se calcula
   sobre ese término, normalizado con el filtro 2 (ver la convención C1 en la sección 4).
5. **Reglas de trading:**
   - Largo en el spread si z < −z_entrada, corto si z > +z_entrada, salida si |z| < z_salida.
   - Stop sobre |z| y tenencia máxima calibrada a la vida media.
   - Umbrales buscados con grid search en entrenamiento.
6. **Rebalanceo:** diario según β̂_t, comparado contra una banda de no-rebalanceo.
7. **Powell:** ambos filtros y el sistema completo se formulan como procesos de decisión secuencial.
   Se usan los 6 pasos y los 5 elementos, con el estado separado en físico, información y creencia.

### Parámetros fijos (sección 3.1)

| Parámetro | Valor |
|---|---|
| Historia | 10–15 años, precios diarios |
| División | 60 / 20 / 20 (train / test / validación), cronológica |
| Comisión | 0.125 % por operación, a la entrada y a la salida, por pata |
| Tasa de préstamo | 0.25 % anual sobre el nocional en corto, devengo diario |
| Capital inicial | $1,000,000 |
| Tamaño de posición | 80 % del capital, repartido proporcional a \|β̂_t\| |
| Semilla aleatoria | `SEED = 42`, fijada en `main.py` |

### Teoría de referencia (carpeta `Teoria/` del curso, no incluida en este repositorio)

| Archivo | Para qué sirve |
|---|---|
| `S11_Sequential_Decision_Analytics.pdf` | Marco de Powell: 5 elementos, clases de política |
| `S12_Pairs_Trading_as_a_Sequential_Decision_Problem.pdf` | ADF, Engle-Granger, vida media, bandas de z-score |
| `S13_State_Space_Models_and_the_Kalman_Filter.pdf` | Predict/update, ganancia K, Q/R, diagnóstico de innovaciones, z causal |
| `S14_Johansen_VECM_and_Multi_Asset_Kalman_Arbitrage.pdf` | Johansen, VECM, normalización de β, embudo de selección, sizing n-patas |
| `Arbitraje_Estadístico_Secuencial.pdf` | Documento completo del módulo |
| `Repaso_Examen_Pares_Johansen_Kalman_SDAM.html` | Repaso rápido para el quiz |

---

## 4. Convenciones del modelo

Las presentaciones S11–S14 y `Lab03.pdf` no siempre dicen lo mismo. Estas son las convenciones que
usa el proyecto. Si alguna cambia, hay que actualizar esta tabla y el código al mismo tiempo.

| # | Tema | Convención | Por qué / fuente |
|---|---|---|---|
| C1 | **¿A qué se le calcula el z-score?** | El z-score se calcula sobre el **término de corrección de error** ECT_t = β̂ᵀ log P_t − c, es decir, la desviación del equilibrio. Esa serie es el spread S_t del filtro 1 y sobre ella opera el filtro 2. **No** se usan los residuales ε_t de la ecuación del VECM. Del VECM se toman α (velocidades de ajuste), la condición −2 < βᵀα < 0 y la vida media implícita. | Es la práctica usual: z-score sobre el spread de cointegración o la innovación de Kalman (Chan, 2013). S14 p.14 define ese ECT. Los ε_t son 6 choques impredecibles por construcción, así que no revierten a la media ni dan una señal única. Así, el Lab03 3.4 y 3.6 describen la misma serie. |
| C2 | **Engle-Granger** | **Prioriza, no elimina.** Se aplica a todos los pares dentro de cada grupo y se registra cuántos tienen p < 0.05, para ordenar los grupos. Todos los grupos que pasan la correlación siguen a Johansen, que es el criterio definitivo. | Un grupo puede no tener ningún par estacionario y ser estacionario como portafolio (S14 p.6 y p.16). Si Engle-Granger eliminara, esos grupos se perderían. La tabla de selección reporta el resultado de esta etapa para cada grupo. |
| C3 | **Log-precios** | Johansen, Engle-Granger, el filtro 1 y el spread usan **log-precios**. El sizing usa precios en nivel. | S12 p.14 y S14 p.7: con log-precios, β son pesos en dólares, que es lo que pide el sizing N_i ∝ \|β_i\| (S14 p.20). |
| C4 | **Estado del filtro 1** | Es el descrito en la sección 3, punto 2: (c_t, β⁽²⁾…β⁽⁶⁾) y β⁽¹⁾ ≡ +1 sobre la pata de mayor \|β\|. | S14 p.13 y p.18. |
| C5 | **Inicialización y burn-in del filtro 1** | ẑ_0 = vector de Johansen normalizado (más c como media del spread en train), P_0 diagonal y R estimado con la varianza de los residuales en train. Q se calibra con la razón Q/R. Burn-in: 60 días sin operar. | S13 p.16: solo importa Q/R y se necesita un burn-in hasta que P_t se estabilice. |
| C6 | **σ̂ del filtro 2** | Varianza del modelo: σ̂²_{t\|t−1} = p_{t−1} + q + r, con el **prior**. La EWMA de innovaciones (λ) queda como alternativa documentada. | S13 p.20. El estado de Kalman es μ_t; σ_t² es la varianza de la innovación, no un estado aparte. |
| C7 | **Vida media** | Se usa la fórmula de Ornstein-Uhlenbeck, −ln2/λ, sobre el spread de train, para calibrar la tenencia máxima (h_max, en múltiplos de la vida media). La vida media implícita del VECM, −ln2/ln(1+βᵀα), se reporta como verificación. El rango utilizable es de 5 a 60 días. | S12 p.17 y S14 p.14 y p.16. |
| C8 | **Filtro de correlación** | Correlación móvil de 252 días sobre **retornos logarítmicos**, como promedio de los pares del grupo, con umbral > 0.7 y estable entre subventanas. Es solo una heurística. | S14 p.16. El Lab03 no dice si es sobre retornos o precios. |
| C9 | **Johansen** | Término determinístico constante (`det_order = 0`), rezagos `k_ar_diff` elegidos por AIC del VAR en niveles menos 1 y valores críticos al 95 %. | S12 p.10 y S14 p.12: los términos determinísticos cambian los valores críticos. |
| C10 | **Johansen móvil (figura 4)** | Ventana de 252 días. Cada eigenvector se normaliza sobre la **misma pata** y con el mismo signo que el vector de train. | Los eigenvectores solo están definidos hasta escala y signo (S14 p.13). Sin normalizar, la gráfica salta. |
| C11 | **Costos de rebalanceo** | Cada cambio de β̂_t que mueve acciones paga 0.125 % en esa pata. El préstamo se devenga a diario sobre todas las patas cortas. | S14 p.19. La prueba de contabilidad debe incluir estos costos. |
| C12 | **Partición y uso de cada tramo** | Orden cronológico train (60 %) → test (20 %) → validación (20 %). Se calibra **solo en train**, test sirve para comparar pocas variantes y validación es el holdout final, que no se toca. | El Lab03 pone la validación al final. Se declara para que no haya data snooping. |
| C13 | **Los 6 pasos de Powell** | En el reporte se usa la lista del **Lab03**: definir el problema, el estado, la transición, la observación, la regla de decisión y el aprendizaje. La lista de S11 (narrativa, elementos, modelo, incertidumbre, políticas, evaluación) solo se cita como referencia. | Las dos listas son distintas y la rúbrica es la del Lab03. |
| C14 | **Decisión en Powell para cada filtro** | La variable de decisión del filtro es la corrección x_t = K_t ỹ_t. La política es una PFA con θ = (Q, R). La regla de bandas también es una PFA, con θ = (z_in, z_out, z_stop, h_max). | S13 p.23 y S12 p.22. |

## 5. Búsqueda y selección del universo de activos

El criterio principal es la **justificación económica**: empresas del mismo giro o rivales directos que
comparten una tendencia estocástica común (el mismo factor de precio, de tasa o de demanda). Esta es la
defensa frente al problema de pruebas múltiples (S12 p.23 y S14 p.15).

### 5.1 Criterios para que un activo entre al universo

1. **Mismo giro o rival directo**, con exposición al mismo factor: precio del crudo, del oro, curva de
   tasas o demanda regulada.
2. **Mismo mercado y moneda:** NYSE o NASDAQ en USD (los ADR son válidos).
3. **Historia completa en la ventana** de 10–15 años y con precios ajustados por splits y dividendos.
4. **Sin fusiones, escisiones ni reestructuras transformacionales** dentro de la ventana. Si hay alguna,
   se documenta como riesgo de ruptura.
5. **Alta liquidez**, para que la comisión de 0.125 % por pata sea el costo real.

### 5.2 Candidatos revisados (verificados con `yfinance` el 2026-11-08)

La correlación es la media de los pares de **retornos logarítmicos diarios, 2010–2026, con toda la
muestra**. Es **solo preliminar**; la prueba oficial se repite únicamente sobre train (convención C8).

| Sector (rivales directos) | Tickers | Correlación media | Historia común | Grupos de 6 posibles | Riesgos de ruptura a documentar |
|---|---|---|---|---|---|
| **Bancos grandes EE.UU.** | JPM, BAC, C, WFC, GS, MS, USB, PNC | **0.78** | desde 2010 | C(8,6) = 28 | GS y MS son banca de inversión, con otro modelo de negocio. WFC tuvo un tope de activos (2018–2025). Crisis de bancos regionales en 2023 (USB, PNC). |
| **Utilities reguladas** | DUK, SO, AEP, D, XEL, ED, EXC, WEC | **0.77** | desde 2010 | 28 | Todas dependen de las tasas de interés. EXC escindió Constellation en 2022. D vendió negocios entre 2020 y 2024. |
| **Refinadoras** | VLO, MPC, PSX, DINO, PBF, DK | **0.76** | desde dic-2012 | 1 | MPC (2011) y PSX (2012) son escisiones y PBF salió a bolsa en 2012, así que la ventana arranca en 2013. DINO es la antigua HollyFrontier. |
| **Petroleras** | XOM, CVX, SHEL, BP, TTE, COP, EQNR, E | **0.75** | desde 2010 | 28 | COP es solo exploración y producción (no integrada). BP, SHEL y TTE tienen estrategias de transición distintas. Choque de 2020. Cambios de ticker: SHEL (antes RDS-A) y TTE (antes TOT). |
| **Mineras de oro (+ GLD)** | NEM, B, AEM, KGC, AU, GFI, GLD | **0.73** | desde 2010 | C(7,6) = 7 | Factor común: el precio del oro. B es Barrick (antes GOLD, cambió en 2025). KGC, AU y GFI tienen beta alta y riesgo país. NEM compró Newcrest en 2023. |
| **Aerolíneas EE.UU.** | DAL, UAL, AAL, LUV, ALK, JBLU | **0.72** | desde 2010 | 1 | Ruptura estructural por COVID en 2020. AAL nació de la fusión de 2013, y su historia previa en `yfinance` debe verificarse. ALK compró Hawaiian en 2024. |

**Descartados en el prefiltro de correlación (documentarlos en la tabla de grupos):**

| Sector | Tickers | Correlación media | Motivo |
|---|---|---|---|
| Ferrocarriles | UNP, CSX, NSC, CP, CNI | 0.72 | Solo 5 rivales y el equipo necesita 6 activos. Además, UNP–NSC anunciaron fusión en 2025. |
| Pagos | V, MA, AXP | 0.69 | Menos de 0.7 y solo 3 activos. |
| Consumo del hogar | PG, CL, KMB, CHD, CLX, UL | 0.56 | Menos de 0.7. |
| Telecom | T, VZ, TMUS | 0.46 | Menos de 0.7 y solo 3 activos. |

Otras opciones que no se revisaron: doble listado (por ejemplo, una emisora mexicana en la BMV y su ADR
en NYSE; agrega riesgo cambiario) y cadena de suministro (por ejemplo, productoras de crudo contra
refinadoras).

### 5.3 Conteo de pruebas múltiples

Se evaluaron **28 grupos** (utilities). Al 5 % se esperan unos **1.4 falsos positivos** aunque no haya
cointegración real. Si se probaran los 6 sectores serían 93 grupos y unos 5 falsos positivos. Por eso se
reporta el número total de grupos probados.

### 5.4 Ventana y partición

- **Ventana:** 2012-01-03 a 2026-10-30, 3,729 días (unos 14.8 años), dentro del rango de 10–15.
- **Partición:**
  - train: hasta 2020-11-19;
  - test: hasta 2023-11-08;
  - validación: hasta 2026-10-30.
- **Nota:** el COVID de 2020 cae al final de train. Hay que revisar si rompe la relación en el sector
  elegido.

### 5.5 Universo elegido

**Utilities reguladas de EE.UU.:** DUK, SO, AEP, D, XEL, ED, EXC, WEC, con 28 grupos de 6. Se eligió por
la justificación económica: negocio regulado y exposición común a las tasas de interés. El embudo
completo corre **solo sobre train** y la tabla con todos los grupos está en
`artifacts/seleccion_grupos.csv`. Si se cambia de universo, se edita `UNIVERSE` en `config.py`.

### 5.6 Resultado de la selección

| Etapa | Grupos | Comentario |
|---|---|---|
| Correlación > 0.7 | 28 de 28 pasan | Media de 0.73 a 0.83 |
| Engle-Granger (solo prioriza) | 15 de 28 con un par con p < 0.05 | Nunca más de 1 de 15 pares por grupo |
| Johansen, traza al 95 % | **1 de 28** | Se esperan 1.4 por azar al 5 % |
| Validación del spread | 1 de 1 | ADF p = 0.001, vida media 29.7 días |

**Grupo elegido:** DUK, AEP, D, XEL, EXC, WEC, con β = (1, −0.457, −0.219, −0.300, −0.096, 0.324).

**Por qué es probablemente un artefacto y no una relación estructural:**
- La traza apenas supera el valor crítico (96.87 contra 95.75) y el máximo eigenvalor **no** rechaza
  (38.17 contra 40.08).
- La conclusión depende de los rezagos: con k = 9 y k = 5 hay r = 1; con k = 1 y k = 2, r = 0. El AIC
  elige el tope de la búsqueda (10 rezagos del VAR).
- Solo 14 % de las subventanas de 2 años encuentran cointegración, y solo 4 de los 28 grupos tienen una cifra menor.
- Cada una de las tres señales (traza, máximo eigenvalor, k = 1) elige un grupo **distinto**.
- El spread deja de ser estacionario fuera de train: ADF p = 0.086 en test y 0.119 en validación.

La regla se declaró antes de ver los datos y no se ajustó para que pasaran más grupos.

---

## 6. Requisitos previos

- Python 3.12 y Git
- [Claude Code](https://claude.com/claude-code) (opcional, para trabajar con las skills de la sección 7)

## 7. Skills de Claude Code (instalar antes de programar)

Ejecuta estos comandos **dentro de una sesión de Claude Code**.

**a) Karpathy guidelines.** Cuatro reglas: pensar antes de codificar, simplicidad, cambios quirúrgicos
y ejecución orientada a metas.

```
/plugin marketplace add forrestchang/andrej-karpathy-skills
/plugin install andrej-karpathy-skills@karpathy-skills
```

(Repo: https://github.com/multica-ai/andrej-karpathy-skills. Su propio README instala desde `forrestchang`.)

**b) Superpowers.** Flujo brainstorming → plan → TDD → code review. Se activa solo.

```
/plugin install superpowers@claude-plugins-official
```

Alternativa:

```
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace
```

(Repo: https://github.com/obra/superpowers.)

**c) Claude Octopus.** Varios modelos de IA revisan la misma tarea, con comandos `/octo:*`.
Se instala desde la **terminal**, no dentro de la sesión:

```
claude plugin marketplace add https://github.com/nyldn/plugins.git
claude plugin install octo@nyldn-plugins
```

Después, dentro de Claude Code, ejecuta `/octo:setup`.

> ⚠️ **Windows:** Octopus solo funciona dentro de **WSL**. No es compatible con PowerShell ni con Git Bash
> nativos. Instala Claude Code y Octopus dentro de WSL.
> Repo: https://github.com/nyldn/claude-octopus.

Para verificar la instalación, escribe `/plugin` y revisa que aparezcan las tres.

## 8. Instalación del proyecto

```powershell
git clone https://github.com/Chelito123/Lab03_MyST_Equipo3.git
cd Lab03_MyST_Equipo3
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

> **Windows ARM64 con Smart App Control:** crea el venv con `--system-site-packages` y ejecuta las
> herramientas como módulo (`python -m pytest`, `python -m streamlit`). Así evitas que Windows bloquee
> los `.exe` y DLL recién instalados. En otros equipos funciona un venv normal.
>
> `pyarrow`, que Streamlit pide, no tiene versión para ARM64, y `pip install -r requirements.txt` falla
> ahí. En ese caso instala Streamlit sin dependencias y luego las demás:
> `python -m pip install --no-deps streamlit==1.67.0`,
> `python -m pip install numpy==2.5.3 pandas==3.0.5 scipy==1.18.1 matplotlib==3.11.1 plotly==7.1.0 yfinance==1.7.0 pytest==9.1.1 reportlab==5.0.1 pypdf==6.21.4 altair blinker cachetools click gitpython packaging pillow protobuf pydeck requests tenacity toml tornado typing-extensions watchdog starlette uvicorn anyio httpx itsdangerous python-multipart websockets`.
> El dashboard no usa `pyarrow`.

## 9. Reproducción: un solo comando

```bash
python main.py
```

Este comando valida los datos y luego corre la selección del grupo, los dos filtros, el VECM, el backtest
y las métricas. Tarda alrededor de un minuto. Todo lo que genera se guarda en `artifacts/`, que es lo que
lee el dashboard.

Los precios están en `data/prices.csv`, versionado: `yfinance` reajusta la historia con cada dividendo, así
que descargarlos de nuevo cambiaría los resultados. El pipeline es determinista, con `SEED = 42` fijada
en `config.py`.

ADF, Engle-Granger y Johansen están implementados en `cointegration/stats.py`, porque `statsmodels` no tiene
binarios para Windows ARM64. Las tablas de valores críticos vienen de `statsmodels` 0.14.6 (BSD) y
`tests/test_stats.py` valida contra sus resultados de referencia.

Pruebas:

```bash
python -m pytest tests/ -v
```

Reporte (se genera desde `artifacts/`):

```bash
python docs/build_report.py
```

Dashboard:

```bash
python -m streamlit run dashboard/app.py
```

## 10. Estructura del repositorio (obligatoria)

```
Lab03_MyST_Equipo3/
├── README.md
├── requirements.txt          # versiones fijadas
├── .gitignore
├── main.py                   # pipeline completo
├── config.py                 # parámetros y semilla
├── data/                     # ingesta, limpieza, validación y prices.csv
├── cointegration/            # screening, Engle-Granger, Johansen
├── kalman/                   # filtro 1 (c_t, β_t) y filtro 2 (μ_t, σ_t²)
├── strategy/                 # VECM, z-score, señales, sizing
├── backtest/                 # motor con costos y walk-forward
├── performance/              # métricas
├── dashboard/app.py          # Streamlit
├── project_management/       # charter.md, sprint_01.md…, postmortem.md
├── artifacts/                # salidas del backtest
├── tests/
└── docs/                     # reporte.pdf, build_report.py, dashboard.md
```

## 11. Pruebas obligatorias (`tests/`, pytest)

1. **Causalidad:** z_t no cambia si se recalcula el pipeline con datos solo hasta t (al menos 3 valores de t).
2. **Normalización:** β̂ conserva escala y signo entre reestimaciones consecutivas.
3. **Contabilidad multi-pata:**
   - valor del portafolio = efectivo + valor de todas las patas;
   - costo total = operaciones por pata × comisión + devengo del préstamo en cortos (incluye el
     rebalanceo, convención C11).
4. **Filtro:** las innovaciones normalizadas tienen varianza ≈ 1 y autocorrelación no significativa.
   - **Con datos generados por el modelo:** se exigen ambas, con varianza 1 ± 0.1 y |ACF(1)| < 0.05.
   - **Con datos reales:** se exige varianza en [0.5, 2] en train. La autocorrelación **no** se exige:
     con esta calibración vale 0.97, porque un spread que revierte despacio es persistente por construcción,
     y esa persistencia es la señal que se opera. Bajarla exigiría una ganancia alta, y entonces el spread
     quedaría blanqueado y no habría nada que operar. Los valores están en `artifacts/diagnosticos.json`.

También hay pruebas adicionales de la referencia estadística, la regla de señales y la ejecución al día
siguiente.

## 12. Guía de uso del dashboard

**Versión publicada:** https://chelito123-lab03-myst-equipo3-dashboardapp-azxhb5.streamlit.app/

**Abrir localmente:** primero corre `python main.py` y después `python -m streamlit run dashboard/app.py`.
Se abre en http://localhost:8501.

**Reglas:**
- Lee los archivos de `artifacts/` y no vuelve a correr el backtest.
- Todas las gráficas son interactivas (zoom, hover, filtros).

**Vista ejecutiva** (portfolio manager):
- Curva de capital en train, test y validación, con el drawdown superpuesto.
- Tarjeta de desempeño: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate y Profit Factor.
- Atribución de costos: P&L bruto → P&L neto, comisiones y costo de préstamo.
- Bitácora de operaciones, filtrable por periodo, dirección y resultado.
- Histograma de retornos por operación con una normal superpuesta.

**Vista técnica** (quant o riesgos):
- Spread S_t con las señales de entrada y salida.
- Evolución de β̂_t, con una línea por activo.
- Primer eigenvector de la prueba de Johansen móvil.
- Ganancia de Kalman de ambos filtros.
- z-score con los umbrales de entrada y salida.
- Varianza posterior de cada filtro (convergencia).
- Sharpe móvil de 252 días.

**Cómo navegar:**
1. Hay dos pestañas arriba: **Vista ejecutiva** y **Vista técnica**.
2. En la ejecutiva, el selector *Tarjeta de desempeño* cambia entre total, train, test y validación. Compara
   siempre train contra los otros dos: la diferencia indica sobreajuste.
3. Abajo, *Periodo*, *Dirección* y *Resultado* filtran la bitácora de operaciones.
4. En la técnica, revisa primero el z-score y el spread (¿revierte tras cada entrada?), luego β̂_t y el
   eigenvector móvil (¿son estables?) y al final la ganancia y la varianza posterior (¿convergieron los filtros?).
5. En cualquier gráfica: arrastra para hacer zoom, doble clic para reiniciar, y clic en la leyenda para
   ocultar una serie.

## 13. Figuras obligatorias (van en el reporte)

1. Precios individuales del grupo
2. Spread con las señales
3. Evolución de β̂_t
4. Primer eigenvector en el tiempo
5. Ganancia de Kalman de ambos filtros
6. z-score con los umbrales
7. Curvas de capital en train, test y validación
8. Distribución de retornos por operación

## 14. Estado de la entrega

- **Entrega:** 2026-11-08 a las 23:59. **Exposición:** 2026-11-09, sobre el dashboard.
- **Repositorio:** https://github.com/Chelito123/Lab03_MyST_Equipo3
- **Dashboard publicado:** https://chelito123-lab03-myst-equipo3-dashboardapp-azxhb5.streamlit.app/
- **Reporte:** `docs/reporte.pdf`.
- **Universo:** se mantuvo utilities a pesar del resultado de la sección 5.6, que se reporta como hallazgo:
  la relación encontrada es probablemente un artefacto de la búsqueda.

## 15. Gestión del proyecto

En `project_management/`:
- `charter.md` (1 página): alcance, hitos con fechas, al menos 5 riesgos (probabilidad, impacto y
  mitigación) y roles. Los riesgos de ruptura de la tabla 5.2 sirven como punto de partida.
- `sprint_XX.md` (uno por semana): objetivo, tareas con estado y bloqueos.
- `postmortem.md` (1–2 páginas).

## 16. Uso de herramientas de IA

Se usó Claude Code (Anthropic) para implementar el pipeline, las pruebas y el dashboard, generar el reporte y
revisar la consistencia entre `Lab03.pdf` y las sesiones S11–S14. Las decisiones de universo, herramienta y
ventana las tomó el equipo. Cada integrante es responsable de todo el código y debe poder explicar cualquier línea.

## 17. Entrega y penalizaciones

- **Entregables:**
  - repo público `Lab03_MyST_Equipo3`;
  - `docs/reporte.pdf`;
  - exposición: 10 min + 5 min de preguntas, sin leer y con todos los integrantes exponiendo.
- **Fecha límite:** 23:59 del día anterior a la exposición. Se evalúa el último commit antes de esa hora.
- **Penalizaciones:**
  - entrega tardía: −20;
  - si el repo no corre con `python main.py`: −10;
  - si falta el reporte o el dashboard en `docs/`: −5;
  - integrante sin commits: 0 en la parte técnica.
- **Rúbrica:** 40 % técnica, 40 % exposición y 20 % quiz.
