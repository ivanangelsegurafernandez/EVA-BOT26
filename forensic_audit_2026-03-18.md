# Auditoría forense IA trading — 2026-03-18

## Contexto y alcance
- Ruta solicitada por el usuario: `/mnt/data/...`.
- Evidencia real disponible en este entorno: archivos homónimos en `/workspace/EVA-BOT26`.
- No se detectó `/mnt/data` en el contenedor.

## Snapshoting
### SNAPSHOT_A (activo probable)
Criterio: artefactos de entrenamiento con `mtime` 2026-03-18 13:42 UTC.

| archivo | tamaño (bytes) | mtime (UTC) | sha256 |
|---|---:|---|---|
| model_meta_v2.json | 3527 | 2026-03-18T13:42:10.627906Z | 2604397c359fb804a4f6b2b9fb4de4ed0e9ed8ebf482a7055f72203d72e1769e |
| dataset_incremental.csv | 106919 | 2026-03-18T13:42:10.627906Z | 53ba73f32b17c271f30e2cc8603f8eeedcb9b8a7ef6ad3c276b0da7e5fd03e7e |
| dataset_incremental_v3.csv | 107847 | 2026-03-18T13:42:10.627906Z | 1c93f79339f59310c33e59c7696aab87c6d1ce6de0da3138bad5e94b4303e99e |
| feature_names_v2.pkl | 80 | 2026-03-18T13:42:10.627906Z | f2828bff26c69f98597ce19448b1e192ddbe1bac893fa5487248072aad1945b7 |
| scaler_v2.pkl | 1039 | 2026-03-18T13:42:10.631906Z | 573b248661a837b9fadd15605febe30659c3cc69c38a959efe15732a66595a1e |
| modelo_xgb_v2.pkl | 518737 | 2026-03-18T13:42:10.631906Z | 1f7373505953ac173c538b3847063ad8585ca850bfcca45d84f7b670554b36e8 |

### SNAPSHOT_B (histórico/auxiliar)
Criterio: archivos de logging/reporte y backup con `mtime` ~2026-03-18 02:32 UTC.

| archivo | tamaño (bytes) | mtime (UTC) | sha256 |
|---|---:|---|---|
| ia_signals_log.csv | 30 | 2026-03-18T02:32:00.188316Z | 4d4a0d0344d1938458c7fa5048f5dbe5377becad3e0c14cbd153dc6fd9b7c0ea |
| ia_temporal_degradation_report.json | 2472 | 2026-03-18T02:32:00.188316Z | c55fb3aa86d378f48274b8639bd1bb39ef87bd7f58f0ec984172d35473ab4973 |
| 5R6M-1-2-4-8-16.py | 637751 | 2026-03-18T02:32:00.184316Z | c2dfb4ee180404010fea5bc6c2049b187bf54c7dd4acf809c0c991c797d5c00c |
| botttt45-1-2-4-8-16-32.py | 120899 | 2026-03-18T02:32:00.184316Z | 85cf0630f3fc8ab8cc7eb87815dd378a188786c4a7223eea8f6f76c3582ba9d8 |
| registro_enriquecido_fulll45.csv | 133142 | 2026-03-18T02:32:00.192316Z | 0009d07c33fe39194f598877c34a6e878d4e0d551c16025bdfa83d77b488c473 |
| dataset_incremental.csv.bak_1773789678 | 533 | 2026-03-18T02:32:00.188316Z | a2b4584c472a749224bb841a2956dead9ef6e5dda7c344b2f693d769928cdca5 |

### Nota de consistencia de snapshots
- `model_meta_v2.json` existe en una sola versión física (no hay segundo `model_meta_v2*.json`).
- Sí existen evidencias históricas incompatibles: `dataset_incremental.csv.bak_1773789678` (solo columnas `racha_actual,result_bin`) y reporte temporal de ventanas con deduplicación extrema.

## Dataset audit
### dataset_incremental.csv (v2)
- rows_total: 493
- rows_useful(label binario): 493
- exact_duplicates: 36
- duplicates por features+label: 36
- columnas: 14
- pos/neg: 251 / 242 (base_rate=0.5091)

### dataset_incremental_v3.csv
- rows_total: 457
- rows_useful(label binario): 457
- exact_duplicates: 0
- duplicates por features+label: 0
- columnas: 15 (`+ts_ingest`)
- pos/neg: 228 / 229 (base_rate=0.4989)

### Diferencia exacta v2 vs v3
- Diferencia de columnas: solo `ts_ingest` está en v3.
- Diferencia de filas sobre columnas comunes: `only_in_v2=36`, `only_in_v3=0`.
- Conclusión forense: v3 = v2 deduplicado + columna temporal (`ts_ingest`), no dataset "nuevo" con muestras nuevas.

## Modelo / meta / artefactos
### model_meta_v2.json (snapshot único)
- rows_total/n_samples: 457
- split: train=286, calib=80, test=91
- auc=0.2934, f1=0.3210, brier=0.2660
- cv_auc=0.4877
- threshold=0.515
- reliable=false
- test_precision_at_thr=0.3095 con n=42
- feature_names(meta): `['racha_actual','ret_5m','ret_3m','micro_trend_persist','ret_1m']`
- trained_on_incremental: `dataset_incremental_v2`

### Compatibilidad de artefactos
- `feature_names_v2.pkl` carga correctamente y coincide con `meta.feature_names` (5 features).
- `scaler_v2.pkl` contiene clase `sklearn.preprocessing._data.StandardScaler` y atributo `feature_names_in_` (inspección pickle).
- `modelo_xgb_v2.pkl` referencia `xgboost.sklearn.XGBClassifier` + calibrador `sklearn.linear_model._logistic` + `n_features_in_` (inspección pickle).
- Inconsistencia semántica: meta declara `trained_on_incremental=dataset_incremental_v2`, pero `n_samples=457` coincide exacto con v3 deduplicado (457) y no con v2 (493).

### Limitación de ejecución de scoring directo
- En este contenedor no están disponibles módulos `numpy/sklearn/xgboost/pandas` y no hay acceso de red a pip para instalarlos.
- Por lo tanto no se pudo ejecutar `predict_proba` real del campeón para recomputar AUC/Brier/F1 fuera de métricas guardadas en `model_meta_v2.json`.

## Auditoría de 13 core features (sobre v3)
Formato: `feature | nunique | dominance | default_ratio(0 o vacío) | corr(label) | auc_univariado`

- racha_actual | 17 | 0.2473 | 0.0000 | -0.0381 | 0.4794
- puntaje_estrategia | 431 | 0.0066 | 0.0000 | +0.0194 | 0.5236
- payout | 4 | 0.5974 | 0.0000 | -0.0098 | 0.4957
- ret_1m | 401 | 0.0088 | 0.0000 | -0.1039 | 0.4424
- ret_3m | 401 | 0.0088 | 0.0000 | -0.0946 | 0.4454
- ret_5m | 401 | 0.0088 | 0.0000 | -0.1342 | 0.4182
- slope_5m | 401 | 0.0088 | 0.0000 | -0.1048 | 0.4448
- rv_20 | 432 | 0.0066 | 0.0000 | +0.0673 | 0.5299
- range_norm | 433 | 0.0066 | 0.0000 | +0.0362 | 0.5138
- bb_z | 432 | 0.0066 | 0.0000 | -0.0662 | 0.4602
- body_ratio | 376 | 0.0656 | 0.0000 | -0.0005 | 0.5287
- wick_imbalance | 374 | 0.0700 | 0.0700 | -0.0229 | 0.4887
- micro_trend_persist | 22 | 0.2188 | 0.0000 | -0.1211 | 0.4464

## Drift temporal robusto (reconstruido)
Criterio de robustez usado: 5 bloques temporales por `ts_ingest`, tamaño ~91 filas cada uno (no 1-2 filas).

### Ventanas (v3)
- W1 n=91, win_rate=0.5714
- W2 n=91, win_rate=0.5275
- W3 n=91, win_rate=0.4176
- W4 n=91, win_rate=0.5495
- W5 n=93, win_rate=0.4301

### AUC univariado promedio del set campeón por ventana (proxy)
(Features: `racha_actual, ret_5m, ret_3m, micro_trend_persist, ret_1m`)
- W1: 0.3425
- W2: 0.3784
- W3: 0.5107
- W4: 0.4232
- W5: 0.5693

### Estabilidad temporal por signo de correlación (v3)
- Las 13 features muestran inversión de signo entre ventanas (corr<0 en algunas y corr>0 en otras).

### PSI (primera vs última ventana, deciles sobre W1)
- racha_actual: 0.239
- puntaje_estrategia: 0.370
- payout: 0.092
- ret_1m: 0.573
- ret_3m: 2.811
- ret_5m: 3.106
- slope_5m: 2.694
- rv_20: 5.249
- range_norm: 4.244
- bb_z: 0.629
- body_ratio: 0.428
- wick_imbalance: 0.481
- micro_trend_persist: 1.531

## Reporte temporal previo (invalidez)
`ia_temporal_degradation_report.json` muestra deduplicación por ventana que deja 2,1,1,1 filas post-dedup.
Eso invalida su diagnóstico de degradación para inferencia estadística robusta.

## Colapso de features (causa mecánica)
Parámetros relevantes en `5R6M-1-2-4-8-16.py`:
- `FEATURE_MAX_PROD=6`
- `FEATURE_DYNAMIC_SELECTION=True`
- `FEATURE_FREEZE_CORE_DURING_WARMUP=True`
- `FEATURE_MIN_AUC_DELTA=0.015`
- `FEATURE_MAX_DOMINANCE_GATE=0.965`
- `TRAIN_PROMOTE_MIN_AUC=0.50`
- `TRAIN_PROMOTE_MIN_FEATURES=5`

Efecto conjunto observado:
1. Selección dinámica puede dejar hasta 6 features máximas.
2. Promoción del campeón exige mínimo 5 features; por debajo se bloquea promoción.
3. Meta actual tiene 5 features (cumple mínimo pero es set angosto).
4. Dos features de interacción reportadas rotas (`nunique=1`) y descartadas.
5. Con señales univariadas cercanas a 0.5 y cambios de signo, la política quality-first + cap a 6 favorece compactación agresiva.

## ia_signals_log y compuerta REAL
- `ia_signals_log.csv` contiene solo header (`ts,bot,epoch,prob,thr,modo,y`), sin filas cerradas.
- Esto impide auditoría cerrada de calibración real (sin n cerrado, sin WR observado, sin ECE/Brier real de producción).
- GO/NO-GO en código exige: `n_samples>=180`, `closed>=50`, `reliable=true`, `auc>=0.53`, `hard_guard!=RED`.
- Con evidencia actual: `reliable=false`, `auc=0.293`, `closed=0`; por diseño la compuerta debe quedar en NO-GO/SHADOW.

## Matriz forense (0-100)
- ausencia de señal: 65
- drift: 75
- sobreajuste/leakage: 80
- colapso del selector: 70
- artefact drift / snapshots inconsistentes: 85
- bloqueo correcto de REAL: 90
- readiness para REAL manual: 20
- readiness para REAL automático: 5

## Causas raíz priorizadas
1. Inconsistencia de artefactos/snapshots y trazabilidad rota (meta vs dataset declarado + logs vacíos).
2. Rendimiento holdout/meta muy débil (`auc=0.293`) con `reliable=false`, incompatible con promoción operativa REAL.
3. Selector dinámico bajo régimen inestable (drift + señales débiles) compacta demasiado el set productivo.

## Acciones mínimas recomendadas
1. Cerrar auditoría de señales reales: poblar `ia_signals_log` con cierres válidos antes de cualquier decisión REAL.
2. Congelar snapshot consistente (dataset + meta + modelo + scaler + features) con checksum y etiqueta única de versión.
3. Mantener SHADOW/DEMO y revalidar con ventana temporal robusta; no habilitar REAL auto hasta cumplir `closed>=50`, `reliable=true`, `auc>=0.53`.
