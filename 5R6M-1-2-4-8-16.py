# -*- coding: utf-8 -*-

# === BLOQUE 0 — OBJETIVOS DEL PROGRAMA 5R6M-1-2-4-8-16 ===
#
# Este script coordina:
# - Lectura de CSV enriquecidos de los bots fulll45–fulll50
# - Control de Martingala 1-2-4-8-16
# - Gestión de tokens DEMO/REAL
# - IA (XGBoost) para probabilidades de éxito
# - HUD visual con Prob IA, % éxito, saldo, meta y eventos
#
# ÍNDICE DE BLOQUES:
#   BLOQUE 1 — IMPORTS Y ENTORNO BÁSICO
#   BLOQUE 2 — CONFIGURACIÓN GLOBAL (MARTINGALA, HUD, AUDIO, IA)
#   BLOQUE 3 — CONFIGURACIÓN DE REENTRENAMIENTO Y MODOS IA
#   BLOQUE 4 — AUDIO (INIT Y REPRODUCCIÓN)
#   BLOQUE 5 — TOKENS, BOT_NAMES Y ESTADO GLOBAL
#   BLOQUE 6 — LOCKS, FIRMAS Y UTILIDADES CSV
#   BLOQUE 7 — ORDEN DE REAL Y CONTROL DE TOKEN
#   BLOQUE 8 — NORMALIZACIÓN Y PUNTAJE DE ESTRATEGIA
#   BLOQUE 9 — DETECCIÓN DE MARTINGALA Y REINICIOS
#   BLOQUE 10 — IA: DATASET, MODELO Y PREDICCIÓN
#   BLOQUE 11 — HUD Y PANEL VISUAL
#   BLOQUE 12 — CONTROL MANUAL REAL Y CONDICIONES SEGURAS
#   BLOQUE 13 — LOOP PRINCIPAL, WEBSOCKET Y TECLADO
#   BLOQUE 99 — RESUMEN FINAL DE LO QUE SE LOGRA
#
# Nota:
#   Esta organización NO cambia la lógica del programa.
#   Solo añade estructura para facilitar futuras modificaciones.
#
# === FIN BLOQUE 0 ===

# === BLOQUE 1 — IMPORTS Y ENTORNO BÁSICO ===
import os, csv, time, random, asyncio, json, re
from collections import deque
from unicodedata import normalize
import threading
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import sys
import shutil
import joblib
import numpy as np
import pandas as pd
import importlib
import traceback

import math
import hashlib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)
warnings.filterwarnings("ignore", message="Columns .* have mixed types.*")
try:
    from pandas.errors import DtypeWarning
    warnings.filterwarnings("ignore", category=DtypeWarning)
except Exception:
    pass


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

def _load_optional_module(name: str):
    try:
        if str(name) == "pygame":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="pkg_resources is deprecated as an API.*",
                    category=UserWarning,
                )
                return importlib.import_module(name)
        return importlib.import_module(name)
    except Exception:
        return None


def _safe_mean_np(values, default=None):
    """Media robusta: evita RuntimeWarning en slices vacíos y NaN-only."""
    try:
        arr = np.asarray(values)
        if arr.size <= 0:
            return default
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.nanmean(arr.astype(float))
        if not np.isfinite(m):
            return default
        return float(m)
    except Exception:
        return default


websockets = _load_optional_module("websockets")
WEBSOCKETS_OK = websockets is not None


colorama = _load_optional_module("colorama")
if colorama is not None:
    Fore = colorama.Fore
    Style = colorama.Style
    init = colorama.init
else:
    class _NoColor:
        def __getattr__(self, _name):
            return ""
    Fore = _NoColor()
    Style = _NoColor()
    def init(*args, **kwargs):
        return None

pygame = _load_optional_module("pygame")
PYGAME_OK = pygame is not None
if not PYGAME_OK:
    class _DummyMixer:
        def get_init(self):
            return False
        def pre_init(self, *args, **kwargs):
            return None
        def init(self, *args, **kwargs):
            return None
        def quit(self):
            return None
        def Sound(self, *args, **kwargs):
            return None

    class _DummyPygame:
        mixer = _DummyMixer()

    pygame = _DummyPygame()

winsound = _load_optional_module("winsound")

# ============================================================
# XGBoost (robusto): permite correr aunque xgboost no esté
# ============================================================
try:
    import xgboost as xgb  # opcional (por compatibilidad)
    from xgboost import XGBClassifier
    _XGBOOST_OK = True
except Exception:
    xgb = None
    XGBClassifier = None
    _XGBOOST_OK = False

# --- Teclado Windows (seguro y único) ---
try:
    import msvcrt as _msvcrt
    class _MSWrap:
        def __bool__(self): return True
        def kbhit(self):
            try: return _msvcrt.kbhit()
            except Exception: return False
        def getch(self):
            try: return _msvcrt.getch()
            except Exception: return b''
    msvcrt = _MSWrap()
    HAVE_MSVCRT = True
except Exception:
    class _DummyMS:
        def __bool__(self): return False
        def kbhit(self): return False
        def getch(self): return b''
    msvcrt = _DummyMS()
    HAVE_MSVCRT = False

# Forzar la ruta fija al directorio del script
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Directorio de trabajo fijado a: {script_dir}")
except Exception as e:
    print(f"⚠️ No se pudo cambiar al directorio del script: {e}. Usando cwd actual.")

init(autoreset=True)
# === FIN BLOQUE 1 ===

# === BLOQUE 2 — CONFIGURACIÓN GLOBAL (MARTINGALA, HUD, AUDIO, IA) ===
# === CONFIGURACIÓN DE MARTINGALA ===
MARTI_ESCALADO = [1, 2, 4, 8, 16]  # Escalado oficial de 5 pasos
MONTO_TOL = 0.01  # Tolerancia para redondeos
SONAR_TAMBIEN_EN_DEMO = False  # Activar sonidos para victorias en DEMO
SONAR_SOLO_EN_GATEWIN = True   # Solo sonar dentro de la ventana GateWIN
SONAR_FUERA_DE_GATEWIN = False # Permitir sonidos fuera de GateWIN si se se habilita
AUDIO_TIMEOUT_S = 0  # 0 significa sin timeout

# === CTT FASE (Consenso Temporal de Trades cerrados) ===
# 3 relojes: ola (WAVE), rezago (LAG) y expiración de permiso (TTL)
CTT_WAVE_WINDOW_S = 180            # W_wave: ventana de ola (120-180s recomendado)
CTT_WAVE_TTL_S = 150               # TTL_wave: permiso útil de la ola (<= W_wave)
CTT_THR_GREEN = 0.85               # Verde fuerte de régimen
CTT_THR_GREEN_OPERABLE = 0.85      # Verde operable: habilita solo con rezago válido + ola viva
CTT_THR_GREEN_WEAK = 0.75          # Verde diagnóstica mínima
CTT_THR_RED = 0.15                 # Rojo fuerte de régimen
CTT_THR_RED_WEAK = 0.25            # Rojo débil (endurece, no siempre veto total)
CTT_LAG_MIN_S = 45                 # rezago mínimo válido
CTT_LAG_MAX_S = 120                # rezago máximo válido (evita "arqueología")
CTT_DENSITY_MIN_CPM = 1.6          # densidad mínima de cierres/min para verde operable
CTT_RED_WEAK_SCORE_PENALTY = 0.02  # castigo suave en rojo débil
CTT_GREEN_OPERABLE_SCORE_BONUS = 0.01  # premio leve en verde operable
CTT_REQUIRE_SAME_ASSET = True      # no mezclar activos en consenso
CTT_ACTIVO_UNICO = "1HZ50V"         # opción 1: todos los bots operan el mismo sintético
CTT_NEUTRAL_POLICY = "normal"      # normal | block
CTT_CIERRE_LOOKBACK_MAX = 600       # higiene memoria eventos
CTT_ENABLE_GREEN_IN_MARTI_ADVANCED = False  # C2..C{MAX_CICLOS}: CTT actúa como freno más que como habilitador

def _ctt_min_confirmadores() -> int:
    n = int(len(BOT_NAMES))
    if n >= 10:
        return 7
    if n >= 6:
        return 4
    return max(1, int(math.ceil(0.7 * n)))


# === REMATE (modo cierre solo con WIN) ===
MODO_REMATE = True           # Continuar hasta WIN o fin de Martingala
REMATE_SIN_TOPE = False      # Limitado por MAX_CICLOS

# === HUD / Layout ===
HUD_LAYOUT = "bottom_center"  # Fijado en centro inferior
HUD_VISIBLE = True       # Para ocultarlo con tecla
# Visual/UI (no afecta lógica funcional): reducir ruido en consola
HUD_COMPACT_MODE = True
HUD_SHOW_TOP3_GATES = False
HUD_SHOW_RACHA_BLOQUES = False
HUD_MINIMAL_MODE = True
HUD_SHOW_DEBUG_BLOCKS = False
HUD_SHOW_VERBOSE_TOP = False
HUD_SHOW_VERBOSE_EVENTS = False
HUD_EVENTS_MAX = 3
HUD_SHOW_IA_LONG_TEXT = False
HUD_MERGE_SIDE_PANELS = True
HUD_ROUND_LIVE_COMPACT = True
HUD_LIVE_ACK_COL_WIDTH = 84
HUD_TABLE_COMPACT_WIDTH = True
HUD_SIDE_PANEL_INLINE = True
HUD_SHOW_SALDO_DEBUG = False
HUD_EVENT_MAX_CHARS = 150
HUD_SHOW_CONTROL_PANEL = False
HUD_MARTI_CLEAN_LAYOUT = True
HUD_BOX_WIDTH = 92
HUD_TABLE_WIDTH = 132
ROUND_LIVE_INVEST_WINDOW_S = 45
MANUAL_CONFIRM_TIMEOUT_S = 20
MANUAL_REAL_DECISION_WINDOW_S = 35
REAL_ORDER_TTL_S = 45
REAL_CLOSE_MAX_AGE_S = 45
KEYBOARD_ENABLE = True
KEYBOARD_DEBUG = False


def _keyboard_can_start():
    try:
        return bool(KEYBOARD_ENABLE and HAVE_MSVCRT and os.name == "nt")
    except Exception:
        return False


def _windows_console_input_safe_mode():
    """
    Best-effort para consola Windows:
    desactiva QuickEdit para evitar congelamientos por selección accidental.
    """
    try:
        if os.name != "nt":
            return
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h_stdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        if h_stdin in (None, 0):
            return
        mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
            return
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080
        new_mode = (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE
        kernel32.SetConsoleMode(h_stdin, new_mode)
    except Exception:
        pass

# --- Objetivos / umbrales globales de IA ---
IA_OBJETIVO_REAL_THR = 0.75   # objetivo de calidad REAL (meta: 75% aprox)
IA_ACTIVACION_REAL_THR = 0.60 # perfil moderado: habilitar REAL desde 60% con candados activos
IA_ACTIVACION_REAL_THR_POST_N15 = 0.58  # post-n15: bajar piso operativo para destrabar REAL moderado
# En modo unreliable (reliable=false), permitir piso post-n15 más realista para no congelar entradas.
IA_ACTIVACION_REAL_THR_POST_N15_UNREL = 0.56
IA_ACTIVACION_REAL_THR_POST_N15_UNREL_MIN_SAMPLES = 300
IA_ACTIVACION_REAL_MIN_N_POR_BOT = 5   # condición: todos los bots deben tener al menos n=5

# --- Oráculo visual ---
ORACULO_THR_MIN   = IA_ACTIVACION_REAL_THR
ORACULO_N_MIN     = 40
ORACULO_DELTA_PRE = 0.05

# Umbral visual/alerta: alineado al mínimo operativo REAL
IA_VERDE_THR = IA_ACTIVACION_REAL_THR
IA_SUCESO_LOOKBACK = 16
IA_SUCESO_DELTA_MIN = 0.035
IA_SUCESO_EVENTO_MIN = 0.20
IA_SUCESO_EVENTO_Q = 0.85
IA_SUCESO_EVENTO_HIST = 120
IA_SENSOR_DOM_HOT = 0.95
IA_SENSOR_MIN_HOT_FEATS = 3
IA_SENSOR_MIN_SAMPLE = 30
IA_REDUNDANCY_SCORE_PENALTY = 0.03
IA_SENSOR_PLANO_SCORE_PENALTY = 0.04
IA_SUCESO_SCORE_WEIGHT = 0.08
IA_OBSERVE_THR = 0.70
AUTO_REAL_THR = IA_OBJETIVO_REAL_THR      # techo dinámico objetivo (70%)
AUTO_REAL_BASE_FLOOR = 0.60                 # piso base dinámico para evitar bloqueo permanente en MODELO experimental
AUTO_REAL_THR_MIN = max(float(IA_ACTIVACION_REAL_THR), float(AUTO_REAL_BASE_FLOOR))
AUTO_REAL_TOP_Q = 0.80    # cuantíl de probs históricas para calibrar el gate REAL
AUTO_REAL_MARGIN = 0.01   # pequeño margen para evitar quedar fuera por décimas
AUTO_REAL_LOG_MAX_ROWS = 300  # máximo de señales históricas usadas en la calibración
AUTO_REAL_LIVE_MIN_BOTS = 3   # mínimos bots con prob viva para calibración por tick

# Umbral "operativo/UI" (señales actuales, semáforo, etc.)
IA_METRIC_THRESHOLD = AUTO_REAL_THR_MIN
# Modo clásico: activación REAL con umbral operativo vigente (hoy 65%, con techo dinámico base 70%).
# Mantiene lock de un solo bot en REAL y ciclo martingala global en HUD.
REAL_CLASSIC_GATE = True
MODO_PURIFICACION_REAL = True  # Llave maestra: bypassea toda promoción/activación REAL sin apagar IA/HUD.
LXV_SYNC_REAL_ROUTE_ENABLE = True
LXV_SYNC_REAL_SOURCE = "LXV_SYNC"
LXV_5V1X_ENABLE = True
LXV_5V1X_ONLY_ENABLE = False  # 5V1X en reposo para emisión REAL (compatibilidad conservada)
LXV_5V1X_REAL_SOURCE = "LXV_5V1X"
LXV_5V1X_REQUIRE_DATA_QUALITY_OK = True
LXV_5V1X_REQUIRE_ROUND_COMPLETE = True
LXV_4V2X_ENABLE = True
LXV_4V2X_REAL_SOURCE = "LXV_4V2X"
LXV_4V2X_REQUIRE_DATA_QUALITY_OK = True
LXV_4V2X_REQUIRE_ROUND_COMPLETE = True
LXV_GREEN_EXHAUSTION_GATE_ENABLE = True
LXV_GREEN_EXHAUSTION_BLOCK_5V1X = True
LXV_GREEN_EXHAUSTION_BLOCK_4V2X = True
LXV_GREEN_EXHAUSTION_LOOKBACK = 12
LXV_GREEN_EXHAUSTION_STREAK80_BLOCK = 2
LXV_GREEN_EXHAUSTION_STREAK90_BLOCK = 1
LXV_GREEN_EXHAUSTION_CURRENT90_BLOCK = True
LXV_GREEN_EXHAUSTION_PREV90_BLOCK = True
LXV_GREEN_EXHAUSTION_LOG_COOLDOWN_S = 20.0
LXV_GREEN_EXHAUSTION_PREV_FULL_GREEN_MIN = 3
LXV_GREEN_EXHAUSTION_ONLY_FULL_GREEN = True
LXV_GREEN_EXHAUSTION_FAIL_OPEN = False
LXV_BOTX_MAX_AGE_S = 90
MANUAL_REAL_ROUTE_ENABLE = True
MANUAL_REAL_REQUIRE_CONFIRM_RISK = False
MANUAL_REAL_FORCE_BYPASS_FASE_ZV = True
LXV_FASE_ZONA_VERDE_ENABLE = True
LXV_FASE_MIN_COLUMNS = 3
LXV_FASE_MAX_STREAK_VERDE_TEMPRANO = 3
LXV_FASE_STREAK_VERDE_MADURO = 4
LXV_FASE_LOG_COOLDOWN_S = 20.0

# ✅ Umbral SOLO para auditoría/calibración (señales CERRADAS en ia_signals_log)
# Esto es lo que querías: contar cierres desde 60% sin afectar la operativa.
IA_CALIB_THRESHOLD = 0.60
IA_CALIB_GOAL_THRESHOLD = IA_OBJETIVO_REAL_THR  # objetivo real: medir cierres fuertes cerca de 70%
IA_CALIB_MIN_CLOSED = 200  # mínimo recomendado para considerar estable la auditoría
REAL_GO_N_MIN = 180
REAL_GO_CLOSED_MIN = 50

# Recomendaciones operativas conservadoras (anti-sobreconfianza)
IA_TEMP_THR_HIGH = 0.80              # umbral temporal sugerido cuando la muestra fuerte es baja
IA_MIN_CLOSED_70_FOR_STRUCT = 200    # mínimo de cierres IA>=70% para cambios estructurales
IA_SHRINK_ALPHA = 0.60               # p_ajustada = alpha*p + (1-alpha)*tasa_base
IA_SHRINK_ALPHA_MIN = 0.45           # piso de mezcla (más conservador en descalibración fuerte)
IA_SHRINK_ALPHA_MAX = 0.85           # techo de mezcla (más sensible cuando la calibración mejora)
IA_BASE_RATE_WINDOW = 300            # cierres recientes para tasa base rolling
# Guardrail explícito de sobreconfianza en bucket alto (fase 1, bajo riesgo).
IA_OVERCONF_BUCKET_MIN_PROB = 0.90
IA_OVERCONF_MIN_N = 20
IA_OVERCONF_GAP_MAX_PP = 0.15
IA_OVERCONF_DYNAMIC_CAP = 0.90
IA_CHECKPOINT_CLOSED_STEP = 20
# Guardrail duro de salud IA (global+por bot): evita sobreconfianza con muestra inmadura.
IA_HARD_GUARD_ENABLE = True
IA_HARD_GUARD_RED_MIN_CLOSED = 0
IA_HARD_GUARD_AMBER_MIN_CLOSED = 80
IA_HARD_GUARD_RED_MIN_AUC = 0.48
IA_HARD_GUARD_GREEN_MIN_AUC = 0.55
IA_HARD_GUARD_MIN_FEATURES_RED = 3
IA_HARD_GUARD_MIN_FEATURES_GREEN = 6
IA_HARD_GUARD_RED_CAP = 0.66
IA_HARD_GUARD_AMBER_CAP = 0.66
IA_HARD_GUARD_RED_REQUIRE_MODEL_READY = True  # evita RED duro por AUC=0 cuando aún no existe modelo válido
IA_HARD_GUARD_SEVERE_GAP_MIN_N = 10
IA_HARD_GUARD_SEVERE_OVERCONF_GAP_PP = 0.25
IA_HARD_GUARD_AMBER_OVERCONF_GAP_PP = 0.15
IA_HARD_GUARD_GREEN_MAX_GAP_PP = 0.10
IA_HARD_GUARD_HYSTERESIS_S = 180.0
IA_HARD_GUARD_LOG_COOLDOWN_S = 45.0
IA_HARD_GUARD_BOT_MIN_N = 10
IA_HARD_GUARD_BOT_GAP_PP = 0.18
# Impulso por racha reciente (micro-ajuste dinámico para evitar Prob IA plana).
IA_RACHA_BOOST_ENABLE = True
IA_RACHA_BOOST_WINDOW = 8
IA_RACHA_BOOST_MAX_UP = 0.10
IA_RACHA_BOOST_MAX_DN = 0.05
IA_RACHA_BOOST_MIN_WINS = 5
IA_RACHA_BOOST_LOG_COOLDOWN_S = 25.0
# Cap conservador de probabilidad durante warmup para evitar inflado (ej. 99-100%).
IA_WARMUP_PROB_CAP_MIN = 0.70
IA_WARMUP_PROB_CAP_MAX = 0.85
IA_WARMUP_CAP_RAMP_ROWS = 120         # rampa de cap en warmup: permite tocar 75% antes sin abrir 90%
IA_WARMUP_LOW_EVIDENCE_CAP_BASE = 0.80
IA_WARMUP_LOW_EVIDENCE_CAP_POST_N15 = 0.85

AUTO_REAL_ALLOW_UNRELIABLE_POST_N15 = True
AUTO_REAL_UNRELIABLE_MIN_N = 0
AUTO_REAL_UNRELIABLE_MIN_PROB = 0.54  # modo unreliable conservador: exige mejor discriminación antes de REAL
AUTO_REAL_UNRELIABLE_MIN_AUC = 0.52   # unreliable conservador: evita activaciones con AUC marginal débil
AUTO_REAL_BLOCK_WHEN_WARMUP = False   # no bloquear REAL por warmup (perfil prueba protegida)
# Ajuste mínimo anti-congelamiento lateral: permite bajar el umbral UNREL
# solo cuando hay evidencia operativa consistente por bot.
AUTO_REAL_UNREL_LATERAL_ADAPT_ENABLE = True
AUTO_REAL_UNREL_LATERAL_MIN_N = 50
AUTO_REAL_UNREL_LATERAL_MIN_WR = 0.50
AUTO_REAL_UNREL_LATERAL_MIN_PROB = 0.50
AUTO_REAL_UNRELIABLE_FLOOR = 0.51      # piso REAL temporal cuando reliable=false y ya hay n mínimo por bot
# Micro-relajación gradual del umbral UNREL basada en cierres auditados reales.
# Solo aplica cuando ya hay muestra suficiente y rendimiento sostenido.
AUTO_REAL_UNREL_MICRO_RELAX_ENABLE = True
AUTO_REAL_UNREL_MICRO_RELAX_MIN_CLOSED = 20
AUTO_REAL_UNREL_MICRO_RELAX_MIN_WINRATE = 0.70
AUTO_REAL_UNREL_MICRO_RELAX_MAX_DELTA = 0.02
AUTO_REAL_UNREL_MICRO_RELAX_LOG_COOLDOWN_S = 45.0
# Bypass controlado: si la compuerta REAL ya está sólida en vivo, permitir AUTO
# aunque el modelo siga en warmup/reliable=false.
AUTO_REAL_UNRELIABLE_ALLOW_STRONG_GATE = True
AUTO_REAL_UNRELIABLE_GATE_MIN_PROB = IA_ACTIVACION_REAL_THR_POST_N15
AUTO_REAL_MICRO_EARLY_CONFIRM_ENABLE = True
AUTO_REAL_MICRO_EARLY_CONFIRM_MARGIN = 0.02
AUTO_REAL_MICRO_EARLY_CONFIRM_DEFICIT_MAX = 1

# Guardas por bot para reducir desalineación Prob IA vs % Éxito observado en HUD.
IA_PROMO_MIN_WR_POR_BOT = 0.45         # no promover bots con WR rolling claramente negativo
IA_PROMO_MAX_OVERCONF_GAP = 0.18       # si p_real supera WR por >18pp con evidencia, bloquear promoción

# Gate de calidad operativo (objetivo: mejorar precisión real, no volumen)
GATE_RACHA_NEG_BLOQUEO = -2.0        # bloquear señales con racha <= -2
GATE_PERMITE_REBOTE_EN_NEG = True    # permitir excepción si hay rebote confirmado
GATE_ACTIVO_MIN_MUESTRA = 40         # mínimo de cierres por activo para evaluar régimen
GATE_ACTIVO_MIN_WR = 0.48            # si WR reciente por activo cae debajo, bloquear temporalmente
GATE_ACTIVO_LOOKBACK = 180           # cierres recientes por bot para estimar régimen
ASSET_PROTECT_ENABLE = True          # protección dinámica por activo basada en degradación real
ASSET_PROTECT_LOOKBACK = 80
ASSET_COOLDOWN_S = 900
ASSET_MAX_CONSEC_LOSS = 4
ASSET_MAX_DRAWDOWN = -4.0
ASSET_MIN_WR = 0.42
ASSET_MAX_DEEP_CYCLE_RATIO = 0.55
ASSET_ALERT_COOLDOWN_S = 60.0
# Gate por segmentos (payout/vol/hora): prioriza zonas con señal estable de racha_actual
GATE_SEGMENTO_ENABLED = True  # gate segmento operativo para filtrar contexto débil
GATE_SEGMENTO_MIN_MUESTRA = 35
GATE_SEGMENTO_MIN_WR = 0.50
GATE_SEGMENTO_LOOKBACK = 240

# Candados inteligentes: evita bloqueos rígidos en empates/planicies cuando ya
# hay evidencia real robusta de un bot claramente apto.
SMART_LOCKS_ENABLE = True
SMART_CLONE_OVERRIDE_MIN_N = 20
SMART_CLONE_OVERRIDE_MIN_LB = 0.53
SMART_CLONE_OVERRIDE_MIN_PROB = 0.62
SMART_CLONE_OVERRIDE_MIN_GAP = 0.002

# Embudo IA en 2 capas: A=régimen (tradeable), B=prob fina (modelo)
REGIME_GATE_MIN_SCORE = 0.52          # mínimo score de régimen para considerar señal
REGIME_GATE_WEIGHT_PROB = 0.70        # peso de la prob del modelo en ranking final
REGIME_GATE_WEIGHT_REGIME = 0.20      # peso de la calidad de régimen
REGIME_GATE_WEIGHT_EVIDENCE = 0.10    # peso de evidencia histórica real (N + WR)
EVIDENCE_MIN_N_HARD = 60              # si hay >=N evidencia fuerte, exigir WR mínimo
EVIDENCE_MIN_WR_HARD = 0.70           # objetivo de calidad real por bot para habilitar auto-REAL
EVIDENCE_MIN_LB_HARD = 0.65           # candado conservador: límite inferior mínimo con evidencia fuerte
EVIDENCE_MIN_N_SOFT = 20              # evidencia mínima blanda para validar LB intermedio
EVIDENCE_MIN_LB_SOFT = 0.55           # LB mínimo cuando N aún es intermedio
EVIDENCE_LOW_N_EXTRA_MARGIN = 0.05    # margen extra de p_real si aún no hay N mínimo blando
POSTERIOR_EVIDENCE_K = 80             # inercia: más alto = más peso al histórico para p_real
POSTERIOR_REGIME_BLEND = 0.35         # mezcla del score de régimen dentro de p_real
EVIDENCE_CACHE_TTL_S = 20.0

# Guardas de honestidad operacional (alineadas al diagnóstico)
DIAG_PATH = "diagnostico_pipeline_ia.json"
ORIENTATION_RECHECK_S = 90.0
ORIENTATION_FLIP_MIN_DELTA = 0.03
ORIENTATION_MIN_CLOSED = 80
ORIENTATION_REQUIRE_RELIABLE_MODEL = True  # evita invertir p->1-p durante warmup/experimental
HARD_GATE_MAX_GAP_HIGH_BINS = 0.10
HARD_GATE_MIN_N_FOR_HIGH_THR = 200
INCREMENTAL_DUP_SCAN_LINES = 6000

# Umbral del aviso de audio (archivo ia_scifi_02_ia53_dry.wav)
AUDIO_IA53_THR = IA_ACTIVACION_REAL_THR

# Anti-spam + rearme
AUDIO_IA53_COOLDOWN_S = 20     # no repetir más de 1 vez cada X segundos por bot
AUDIO_IA53_RESET_HYST = 0.03   # se rearma cuando cae por debajo de (thr - hyst)

# === Caché de sonidos ===
SOUND_CACHE = {}
SOUND_LOAD_ERRORS = set()
SOUND_PATHS = {
    "ganancia_real": "ganabot.wav",
    "ganancia_demo": "ganabot.wav",
    "perdida_real": "perdida.wav",
    "perdida_demo": "perdida.wav",
    "meta_15": "meta15%.wav",
    "racha_detectada": "detectaracha.wav",
    "test": "test.wav",
    "ia_53": "ia_scifi_08_53porciento_dry.wav",

}
AUDIO_AVAILABLE = False
META_ACEPTADA = False
MODAL_ACTIVO = False
sonido_disparado = False
# === FIN BLOQUE 2 ===

# === BLOQUE 2.5 — PLAN OPERATIVO PATRÓN V1 (RESUMEN EJECUTIVO) ===
# IMPORTANTE:
# - Este bloque NO reemplaza todavía la lógica de entrada actual.
# - Sirve para dejar la integración preparada y revisable.
# - Los candados existentes (hard_guard/confirm/trigger/roof) se mantienen.
PATTERN_V1_ENABLE = True
PATTERN_V1_SCORE_THR = 6.0
PATTERN_V1_BONUS_DUAL = 1.0
PATTERN_V1_PENAL_TARDIA = 2.0
PATTERN_V1_REQUIRE_CONFIRM_FULL = True   # confirm=2/2
PATTERN_V1_REQUIRE_TRIGGER_OK = True     # trigger_ok=sí
PATTERN_V1_USE_HYBRID_RANKING = True    # ranking híbrido operativo (prob + pattern + evidencia)
PATTERN_V1_LOG_COOLDOWN_S = 25.0
PATTERN_V1_HYBRID_PTS_TO_PROB = 0.03  # 1 punto pattern = 3pp sobre score probabilístico
PATTERN_COL_WINDOW = 40
PATTERN_COL80_THRESHOLD = 0.80
PATTERN_COL90_THRESHOLD = 0.90
PATTERN_REBOTE_LOOKBACK = 12
PATTERN_REBOTE_MIN = 0.65
PATTERN_REBOTE_MIN_SAMPLES = 3
PATTERN_STRONG_STREAK_BLOCK = 2
PATTERN_ENABLE = True
PATTERN_COL_BONUS_CONTINUIDAD = 0.60
PATTERN_COL_BONUS_REBOTE = 0.80
PATTERN_COL_PENAL_SATURACION = 1.20
PATTERN_COL_PENAL_LATE_CHASE = 1.00
MRV_5V1X_ENABLE = True
MRV_5V1X_MIN_HISTORY_COLUMNS = 6
MRV_5V1X_REQUIRE_REBOTE_OR_CONTINUIDAD = True
MRV_5V1X_BLOCK_LATE_CHASE = True
MRV_5V1X_BLOCK_SATURACION = True
MRV_5V1X_PREV90_BLOCK = True
MRV_5V1X_LOG_COOLDOWN_S = 20.0
PATTERN_COL_LAST_STATE = {
    "green_ratio_col_actual": None,
    "total_verdes_col_actual": 0,
    "total_rojos_col_actual": 0,
    "rebote_rate_hist": None,
    "rebote_samples_hist": 0,
    "total_x_hist": 0,
    "total_x_rebote_hist": 0,
    "pattern_state": "BLOQUEADO",
    "strong_streak_80": 0,
    "strong_streak_90": 0,
    "late_chase": False,
    "pattern_delta": 0.0,
    "pattern_bonus_penalty": 0.0,
}
PATTERN_V1_Q3_PROXY = {
    "rsi_9": 64.0,
    "rsi_reversion": 0.060,
    "es_rebote": 0.090,
    "puntaje_estrategia": 0.28,
    "cruce_sma": 0.62,
    "breakout": 0.20,
    "payout": 0.9525,
    "racha_actual": 2.0,
}
PATTERN_V1_Q2_PROXY = {
    "volatilidad": 0.049,
}
PATTERN_V1_LAST_LOG_TS = {}
# Fase operativa REAL por madurez: SHADOW -> MICRO -> NORMAL
REAL_PILOT_MODE_ENABLE = True
REAL_MICRO_REQUIRE_PATTERN = False
REAL_MICRO_PATTERN_MIN_TOTAL = 4.0
REAL_MICRO_REQUIRE_DUAL = False
REAL_MICRO_REQUIRE_STRUCTURE = False
REAL_MICRO_MIN_WR = 0.50
REAL_MICRO_MIN_TRADES = 40
REAL_MICRO_TOP_K = 1
REAL_MICRO_ALLOW_SOFT_HIGH_PROB = True
REAL_MICRO_SOFT_MIN_PROB = 0.58
REAL_MICRO_SOFT_MIN_SUCESO = 18.0
REAL_MICRO_SOFT_MIN_WR = 0.47
REAL_SHADOW_MICRO_ENABLE = True
REAL_SHADOW_MICRO_MIN_PROB = 0.56
REAL_SHADOW_MICRO_MAX_ENTRIES = 6
REAL_SHADOW_MICRO_WINDOW_S = 300
REAL_SHADOW_MICRO_TOP_K = 1
REAL_SHADOW_MICRO_LOG_COOLDOWN_S = 20.0
_REAL_SHADOW_MICRO_OPEN_TS = deque(maxlen=64)
_REAL_SHADOW_MICRO_LAST_LOG_TS = 0.0
REAL_MICRO_STRONG_GATE_FALLBACK_ENABLE = True
REAL_MICRO_STRONG_GATE_MIN_PROB = 0.60
EMBUDO_FINAL_BLOCK_HARD = "BLOCK_HARD"
EMBUDO_FINAL_WAIT_SOFT = "WAIT_SOFT"
EMBUDO_FINAL_REAL_MICRO = "REAL_MICRO"
EMBUDO_FINAL_REAL_NORMAL = "REAL_NORMAL"
EMBUDO_FINAL_SHADOW_OK = "SHADOW_OK"
IA_PROB_POLARIZE_ENABLE = True
IA_PROB_POLARIZE_FACTOR_RELIABLE = 1.25
IA_PROB_POLARIZE_FACTOR_UNRELIABLE = 2.05
IA_PROB_POLARIZE_CENTER = 0.50


def _validar_pattern_v1_config() -> None:
    """Sanitiza parámetros para evitar valores inválidos en runtime."""
    global PATTERN_V1_SCORE_THR, PATTERN_V1_BONUS_DUAL, PATTERN_V1_PENAL_TARDIA, PATTERN_V1_LOG_COOLDOWN_S, PATTERN_V1_HYBRID_PTS_TO_PROB
    global PATTERN_COL_WINDOW, PATTERN_COL80_THRESHOLD, PATTERN_COL90_THRESHOLD, PATTERN_REBOTE_LOOKBACK
    global PATTERN_REBOTE_MIN, PATTERN_REBOTE_MIN_SAMPLES, PATTERN_STRONG_STREAK_BLOCK
    PATTERN_V1_SCORE_THR = max(0.0, float(PATTERN_V1_SCORE_THR))
    PATTERN_V1_BONUS_DUAL = max(0.0, float(PATTERN_V1_BONUS_DUAL))
    PATTERN_V1_PENAL_TARDIA = max(0.0, float(PATTERN_V1_PENAL_TARDIA))
    PATTERN_V1_LOG_COOLDOWN_S = max(5.0, float(PATTERN_V1_LOG_COOLDOWN_S))
    PATTERN_V1_HYBRID_PTS_TO_PROB = min(0.10, max(0.0, float(PATTERN_V1_HYBRID_PTS_TO_PROB)))
    PATTERN_COL_WINDOW = max(5, int(PATTERN_COL_WINDOW))
    PATTERN_COL80_THRESHOLD = min(0.99, max(0.50, float(PATTERN_COL80_THRESHOLD)))
    PATTERN_COL90_THRESHOLD = min(1.0, max(float(PATTERN_COL80_THRESHOLD), float(PATTERN_COL90_THRESHOLD)))
    PATTERN_REBOTE_LOOKBACK = max(2, int(PATTERN_REBOTE_LOOKBACK))
    PATTERN_REBOTE_MIN = min(1.0, max(0.0, float(PATTERN_REBOTE_MIN)))
    PATTERN_REBOTE_MIN_SAMPLES = max(1, int(PATTERN_REBOTE_MIN_SAMPLES))
    PATTERN_STRONG_STREAK_BLOCK = max(1, int(PATTERN_STRONG_STREAK_BLOCK))


def resumen_plan_cambios_5r6m() -> list[str]:
    """Resumen corto de cambios planificados en 5R6M-1-2-4-8-16.py."""
    return [
        "1) Añadir Pattern Score compuesto (señales duales + estructura técnica).",
        "2) Añadir veto tardío para evitar perseguir rachas verdes iniciadas.",
        "3) Separar detección de oportunidad vs permiso final de entrada.",
        "4) Mantener candados existentes (hard_guard, confirm, trigger, roof).",
        "5) Usar ranking híbrido: prob_ia_oper + bonus_patron - penal_tardia - crowding.",
        "6) Medir drift por ventanas y degradar score cuando no hay persistencia.",
    ]


def pattern_score_operativo_v1(features: dict, q3: dict, q2: dict) -> tuple[float, float, float, float]:
    """Score proxy para integración gradual (sin reemplazar la decisión vigente).

    Retorna: (score, bonus_dual, penal_tardia, score_final)
    """
    score = 0.0
    if features.get("rsi_9", 0.0) >= q3.get("rsi_9", 1e9):
        score += 2.0
    if features.get("rsi_reversion", 0.0) >= q3.get("rsi_reversion", 1e9):
        score += 2.0
    if features.get("es_rebote", 0.0) >= q3.get("es_rebote", 1e9):
        score += 2.0
    if features.get("puntaje_estrategia", 0.0) >= q3.get("puntaje_estrategia", 1e9):
        score += 1.0
    if features.get("cruce_sma", 0.0) >= q3.get("cruce_sma", 1e9):
        score += 1.0
    if features.get("breakout", 0.0) >= q3.get("breakout", 1e9):
        score += 1.0
    if features.get("payout", 0.0) >= q3.get("payout", 1e9):
        score += 1.0
    if features.get("volatilidad", 1e9) <= q2.get("volatilidad", -1e9):
        score += 1.0

    dual = (
        features.get("rsi_reversion", 0.0) >= q3.get("rsi_reversion", 1e9)
        or features.get("es_rebote", 0.0) >= q3.get("es_rebote", 1e9)
    )
    bonus_dual = (
        PATTERN_V1_BONUS_DUAL
        if dual and features.get("rsi_9", 0.0) >= q3.get("rsi_9", 1e9)
        else 0.0
    )
    penal_tardia = 0.0
    if features.get("racha_actual", 0.0) >= q3.get("racha_actual", 1e9) and not dual:
        penal_tardia = PATTERN_V1_PENAL_TARDIA

    score_final = score + bonus_dual - penal_tardia
    return score, bonus_dual, penal_tardia, score_final


def _pattern_v1_thresholds_proxy() -> tuple[dict, dict]:
    """Umbrales proxy (Q3/Q2) para operar Pattern V1 sin dependencia externa."""
    return dict(PATTERN_V1_Q3_PROXY), dict(PATTERN_V1_Q2_PROXY)


def _pattern_v1_log_bot(bot: str, pattern_score: float, bonus_dual: float, penal_tardia: float, score_hibrido: float) -> None:
    """Log por bot con cooldown para auditar impacto del Pattern V1."""
    try:
        ahora = time.time()
        last = float(PATTERN_V1_LAST_LOG_TS.get(bot, 0.0) or 0.0)
        if (ahora - last) < float(PATTERN_V1_LOG_COOLDOWN_S):
            return
        PATTERN_V1_LAST_LOG_TS[bot] = float(ahora)
        agregar_evento(
            f"🧠 PatternV1 {bot}: score={pattern_score:.1f} bonus={bonus_dual:.1f} "
            f"penal={penal_tardia:.1f} score_hibrido={score_hibrido*100:.1f}%"
        )
    except Exception:
        pass


def _purificacion_real_activa() -> bool:
    """Llave maestra centralizada para apagar capa REAL sin romper flujo IA/HUD."""
    try:
        route_src = str(globals().get("_REAL_ROUTE_SOURCE", "") or "").strip().upper()
        allow_sync = str(globals().get("LXV_SYNC_REAL_SOURCE", "LXV_SYNC")).upper()
        allow_5v1x = str(globals().get("LXV_5V1X_REAL_SOURCE", "LXV_5V1X")).upper()
        allow_4v2x = str(globals().get("LXV_4V2X_REAL_SOURCE", "LXV_4V2X")).upper()
        allowed_sources = set()
        if bool(globals().get("LXV_SYNC_REAL_ROUTE_ENABLE", False)):
            allowed_sources.add(allow_sync)
        if bool(globals().get("LXV_5V1X_ENABLE", False)):
            allowed_sources.add(allow_5v1x)
        if bool(globals().get("LXV_4V2X_ENABLE", False)):
            allowed_sources.add(allow_4v2x)
        if bool(globals().get("MANUAL_REAL_ROUTE_ENABLE", False)):
            allowed_sources.add("MANUAL")
        if route_src in allowed_sources:
            try:
                _lxv_5v1x_event_cooldown(
                    key=f"purif:allow:{route_src}",
                    msg=f"🛡️ Purificación bypass: source REAL permitida={route_src}",
                    cooldown_s=15.0,
                )
            except Exception:
                pass
            return False
        purif_on = bool(globals().get("MODO_PURIFICACION_REAL", False))
        if purif_on and route_src:
            try:
                _lxv_5v1x_event_cooldown(
                    key=f"purif:block:{route_src}",
                    msg=f"🧪 Purificación activa: source REAL bloqueada={route_src}",
                    cooldown_s=15.0,
                )
            except Exception:
                pass
        return purif_on
    except Exception:
        return False


def _emitir_marca_purificacion_real() -> None:
    """Marca visual/evento con cooldown para confirmar bypass REAL activo."""
    global _LAST_PURIFICACION_REAL_EVENT_TS
    try:
        if not _purificacion_real_activa():
            return
        now = float(time.time())
        if (now - float(_LAST_PURIFICACION_REAL_EVENT_TS or 0.0)) < float(PURIFICACION_REAL_EVENT_COOLDOWN_S):
            return
        _LAST_PURIFICACION_REAL_EVENT_TS = now
        agregar_evento("🧪 MODO PURIFICACION REAL ACTIVO: promoción/orden REAL desactivada (bypass).")
    except Exception:
        pass


def _guardar_real_owner_state(bot: str, ciclo: int, source: str, round_id: int | None = None, token_state: str | None = None) -> None:
    """Telemetría mínima del owner REAL asignado por el maestro."""
    try:
        payload = {
            "owner_bot": str(bot),
            "assigned_ts": float(time.time()),
            "ciclo": int(ciclo),
            "source": str(source or "UNKNOWN"),
            "round_id": int(round_id) if isinstance(round_id, (int, float)) else None,
            "token_state": str(token_state) if token_state is not None else None,
        }
        _atomic_write(REAL_OWNER_STATE_FILE, json.dumps(payload, ensure_ascii=False, indent=2))
        globals()["LAST_REAL_OWNER_STATE"] = payload
    except Exception:
        pass


def _registrar_real_close_trace(data: dict) -> None:
    """Append conservador de cierre REAL confirmado (sin tocar cálculo de saldo)."""
    try:
        payload = dict(data or {})
        payload.setdefault("ts", float(time.time()))
        _append_line_safe(REAL_CLOSE_TRACE_FILE, json.dumps(payload, ensure_ascii=False) + "\n")
        globals()["LAST_REAL_CLOSE_TRACE"] = payload
    except Exception:
        pass


def _resolver_estado_real(meta_live: dict | None = None) -> str:
    """Estado operativo REAL: SHADOW, MICRO, NORMAL."""
    try:
        if _purificacion_real_activa():
            return "SHADOW"
        if not bool(REAL_PILOT_MODE_ENABLE):
            return "NORMAL"
        meta = meta_live if isinstance(meta_live, dict) else (_ORACLE_CACHE.get("meta") or leer_model_meta() or {})
        n = int(meta.get("n_samples", meta.get("n", 0)) or 0)
        auc = float(meta.get("auc", 0.0) or 0.0)
        warmup = bool(meta.get("warmup_mode", n < int(TRAIN_WARMUP_MIN_ROWS)))
        reliable = bool(meta.get("reliable", False)) and (not warmup)
        hg = _estado_guardrail_ia_fuerte(force=False)
        if reliable and (auc >= 0.53) and (not bool(hg.get("hard_block", False))):
            return "NORMAL"
        if n >= int(MIN_FIT_ROWS_PROD):
            return "MICRO"
        return "SHADOW"
    except Exception:
        return "SHADOW"


def _micro_pattern_gate_ok(bot: str, ctx: dict | None = None) -> tuple[bool, str]:
    """Filtro principal en MICRO: patrón dual + estructura + recencia mínima."""
    try:
        if not bool(REAL_MICRO_REQUIRE_PATTERN):
            return True, "off"
        st = estado_bots.get(bot, {}) if isinstance(estado_bots, dict) else {}
        c = ctx if isinstance(ctx, dict) else _ultimo_contexto_operativo_bot(bot)
        q3, q2 = _pattern_v1_thresholds_proxy()
        p_score, p_bonus, p_penal, p_total = pattern_score_operativo_v1(c or {}, q3, q2)
        strict_ok = True
        why = f"pat={p_total:.1f}"
        if float(p_total) < float(REAL_MICRO_PATTERN_MIN_TOTAL):
            strict_ok = False
            why = f"pat<{REAL_MICRO_PATTERN_MIN_TOTAL:.1f}"
        if strict_ok and bool(REAL_MICRO_REQUIRE_DUAL) and float(p_bonus) <= 0.0:
            strict_ok = False
            why = "dual=no"
        if strict_ok and bool(REAL_MICRO_REQUIRE_STRUCTURE):
            breakout_ok = bool(float((c or {}).get("breakout", 0.0) or 0.0) >= float(q3.get("breakout", 1e9)))
            cruce_ok = bool(float((c or {}).get("cruce_sma", 0.0) or 0.0) >= float(q3.get("cruce_sma", 1e9)))
            if not (breakout_ok or cruce_ok):
                strict_ok = False
                why = "struct=no"

        g = int(st.get("ganancias", 0) or 0)
        d = int(st.get("perdidas", 0) or 0)
        n = int(max(0, g + d))
        wr = float((g + 1.0) / (n + 2.0))
        if strict_ok and n >= int(REAL_MICRO_MIN_TRADES) and wr < float(REAL_MICRO_MIN_WR):
            strict_ok = False
            why = f"wr<{REAL_MICRO_MIN_WR*100:.0f}%"
        if strict_ok and float(p_penal) > 0.0:
            strict_ok = False
            why = "late=veto"
        if strict_ok:
            return True, why

        # Fallback suave: permite fluir entradas cuando la calidad viva es alta
        # aunque el patrón dual/estructura no complete en ese tick.
        if bool(REAL_MICRO_ALLOW_SOFT_HIGH_PROB):
            p_oper = float(st.get("prob_ia_oper", st.get("prob_ia", 0.0)) or 0.0)
            suceso = float(st.get("ia_suceso_idx", 0.0) or 0.0)
            if (
                p_oper >= float(REAL_MICRO_SOFT_MIN_PROB)
                and suceso >= float(REAL_MICRO_SOFT_MIN_SUCESO)
                and wr >= float(REAL_MICRO_SOFT_MIN_WR)
                and float(p_penal) <= float(PATTERN_V1_PENAL_TARDIA)
            ):
                return True, f"soft:p={p_oper*100:.1f}%/s={suceso:.1f}"

        return False, why
    except Exception:
        return False, "pattern_err"


def _shadow_micro_quota_status(now_ts: float | None = None) -> tuple[int, int, float]:
    """Estado de cuota para micro-REAL temporal en SHADOW."""
    try:
        now = float(time.time() if now_ts is None else now_ts)
        window_s = max(60.0, float(REAL_SHADOW_MICRO_WINDOW_S))
        while _REAL_SHADOW_MICRO_OPEN_TS and (now - float(_REAL_SHADOW_MICRO_OPEN_TS[0])) > window_s:
            _REAL_SHADOW_MICRO_OPEN_TS.popleft()
        used = int(len(_REAL_SHADOW_MICRO_OPEN_TS))
        max_entries = max(1, int(REAL_SHADOW_MICRO_MAX_ENTRIES))
        left = max(0, max_entries - used)
        return left, used, window_s
    except Exception:
        return 0, 0, max(60.0, float(REAL_SHADOW_MICRO_WINDOW_S))


def _shadow_micro_gate_ok(candidatos: list, dyn_gate: dict | None = None) -> tuple[bool, str]:
    """Bypass seguro para permitir micro-REAL temporal aun en SHADOW."""
    global _REAL_SHADOW_MICRO_LAST_LOG_TS
    try:
        if not bool(REAL_SHADOW_MICRO_ENABLE):
            return False, "off"
        if not candidatos:
            return False, "sin_candidatos"
        if not bool(_todos_bots_con_n_minimo_real()):
            return False, "n_min_real"

        dgate = dyn_gate if isinstance(dyn_gate, dict) else {}
        confirm_need = int(dgate.get("confirm_need", DYN_ROOF_CONFIRM_TICKS) or DYN_ROOF_CONFIRM_TICKS)
        confirm_ok = int(dgate.get("confirm_streak", 0) or 0) >= confirm_need
        trigger_ok = bool(dgate.get("trigger_ok", False))
        allow_gate = bool(dgate.get("allow_real", False))

        best = candidatos[0]
        best_bot = str(best[1])
        p_best = float(best[2] or 0.0)
        if best_bot != str(dgate.get("best_bot", best_bot)):
            return False, "best_mismatch"
        if p_best < float(REAL_SHADOW_MICRO_MIN_PROB):
            return False, f"p<{REAL_SHADOW_MICRO_MIN_PROB*100:.0f}%"
        if not (confirm_ok and trigger_ok and allow_gate):
            return False, "gate_debil"

        hg = _estado_guardrail_ia_fuerte(force=False)
        if bool(hg.get("hard_block", False)):
            return False, "hard_guard"

        left, used, window_s = _shadow_micro_quota_status()
        if left <= 0:
            return False, f"quota:{used}/{max(1, int(REAL_SHADOW_MICRO_MAX_ENTRIES))}/{int(window_s//60)}m"

        now = time.time()
        if (now - float(_REAL_SHADOW_MICRO_LAST_LOG_TS or 0.0)) >= float(REAL_SHADOW_MICRO_LOG_COOLDOWN_S):
            _REAL_SHADOW_MICRO_LAST_LOG_TS = float(now)
            agregar_evento(
                f"🟢 REAL=SHADOW→MICRO temporal: {best_bot} p={p_best*100:.1f}% "
                f"confirm={int(dgate.get('confirm_streak', 0))}/{confirm_need} trigger_ok=sí "
                f"quota={used}/{max(1, int(REAL_SHADOW_MICRO_MAX_ENTRIES))}"
            )
        return True, "ok"
    except Exception:
        return False, "shadow_micro_err"


def _micro_strong_gate_fallback_ok(candidatos: list, dyn_gate: dict | None = None) -> tuple[bool, str]:
    """Fallback moderado para MICRO cuando el patrón no completa pero la compuerta está sólida."""
    try:
        if not bool(REAL_MICRO_STRONG_GATE_FALLBACK_ENABLE):
            return False, "off"
        if not candidatos:
            return False, "sin_candidatos"
        dgate = dyn_gate if isinstance(dyn_gate, dict) else {}
        top = candidatos[0]
        best_bot = str(top[1])
        p_best = float(top[2] or 0.0)
        confirm_need = int(dgate.get("confirm_need", DYN_ROOF_CONFIRM_TICKS) or DYN_ROOF_CONFIRM_TICKS)
        confirm_ok = int(dgate.get("confirm_streak", 0) or 0) >= confirm_need
        trigger_ok = bool(dgate.get("trigger_ok", False))
        allow_gate = bool(dgate.get("allow_real", False))
        if best_bot != str(dgate.get("best_bot", best_bot)):
            return False, "best_mismatch"
        if p_best < float(REAL_MICRO_STRONG_GATE_MIN_PROB):
            return False, f"p<{REAL_MICRO_STRONG_GATE_MIN_PROB*100:.0f}%"
        if not (confirm_ok and trigger_ok and allow_gate):
            return False, "gate_debil"
        hg = _estado_guardrail_ia_fuerte(force=False)
        if bool(hg.get("hard_block", False)):
            return False, "hard_guard"
        return True, f"p={p_best*100:.1f}%"
    except Exception:
        return False, "micro_fallback_err"


_validar_pattern_v1_config()


# === BLOQUE 3 — CONFIGURACIÓN DE REENTRENAMIENTO Y MODOS IA ===
# === CONFIGURACIÓN DE REENTRENAMIENTO ===
RETRAIN_INTERVAL_ROWS = 100     # por volumen
RETRAIN_INTERVAL_MIN  = 15      # por tiempo
MIN_NEW_ROWS_FOR_TIME = 20      # al menos 20 filas nuevas para reentrenar por tiempo
MAX_DATASET_ROWS = 10000
last_retrain_count = 0
last_retrain_ts    = time.time()  # Inicializado al boot para arranque en frío
AUTO_RETRAIN_TICK_S = 20.0  # reintento periódico para no quedarse sin modelo tras warmup
IA_NO_MODEL_LOG_COOLDOWN_S = 30.0  # evita spam cuando aún no hay modelo
_entrenando_lock = threading.Lock()  # Lock para antireentradas en maybe_retrain

# === MODO ENTRENAMIENTO CON POCA DATA (no toca la lógica de IA) ===
LOW_DATA_MODE = True           # True = permite entrenar con muy pocas filas
MIN_FIT_ROWS_PROD = 100        # umbral “confiable” para producción (lo que ya usabas)
MIN_FIT_ROWS_LOW  = 4          # umbral mínimo para permitir fit “experimental”
RELIABLE_POS_MIN  = 20         # mínimos para considerar fiable (calibración/umbral estable)
RELIABLE_NEG_MIN  = 20

# Modo manual desactivado: priorizamos automatización completa por Prob IA.
# Si luego quieres volver al modo manual, ponlo en True.
MODO_REAL_MANUAL = False

# Martingala global
marti_paso = 0
marti_activa = False

# Contador global de ciclos de martingala (HUD + orquestación automática)
# 0 = sin pérdidas consecutivas en REAL; 1..MAX_CICLOS = racha de pérdidas vigente.
marti_ciclos_perdidos = 0

# Anti-repetición de bot en REAL:
# - Si el HUD está en C1, se puede repetir bot.
# - Si el HUD está en C2..C{MAX_CICLOS}, se prioriza no repetir; puede haber fallback controlado.
ultimo_bot_real = None

# Rotación por corrida de martingala REAL (C1..C{MAX_CICLOS})
# Guarda el orden de bots usados en la corrida activa para evitar repeticiones.
bots_usados_en_esta_marti = []
# Continuidad inteligente C2..C{MAX_CICLOS}: si no hay bot nuevo elegible, permitir repetir
# el mejor candidato SOLO bajo umbral mínimo de probabilidad operativa.
MARTI_CYCLE_ALLOW_REPEAT_FALLBACK = True
MARTI_CYCLE_REPEAT_MIN_PROB = 0.68
MARTI_CYCLE_REPEAT_MIN_PROB_UNRELIABLE_CAP = 0.66


def _marti_repeat_min_prob_live(meta_live=None):
    """Umbral vivo para fallback C2..C{MAX_CICLOS}, con ajuste conservador en modo no confiable."""
    base = float(MARTI_CYCLE_REPEAT_MIN_PROB)
    try:
        if not isinstance(meta_live, dict):
            meta_live = resolver_canary_estado(leer_model_meta() or {})
        n_samples = int(meta_live.get("n_samples", meta_live.get("n", 0)) or 0)
        warmup = bool(meta_live.get("warmup_mode", n_samples < int(TRAIN_WARMUP_MIN_ROWS)))
        reliable = bool(meta_live.get("reliable", False)) and (not warmup)
        if not reliable:
            base = min(base, float(MARTI_CYCLE_REPEAT_MIN_PROB_UNRELIABLE_CAP))
    except Exception:
        pass
    return float(max(0.0, min(1.0, base)))

# Auditoría de secuencia martingala (C1..C{MAX_CICLOS}) para traza explícita.
marti_audit_run_id = 1
marti_audit_historial = deque(maxlen=80)
marti_audit_desviaciones = 0
marti_audit_ultimo_ciclo_ordenado = None

# Nueva: Umbrales mínimos para historial IA
MIN_IA_SENIALES_CONF = 10  # Mínimo señales cerradas para confiar en prob_hist
MIN_AUC_CONF = 0.65        # AUC mínimo para audios/colores verdes
MAX_CLASS_IMBALANCE = 0.8  # Máx proporción pos/neg para entrenar (evita 99% wins)
AUC_DROP_TOL = 0.05        # Tolerancia para no machacar modelo si AUC baja
TRAIN_REFRESH_STALE_MIN = 45 * 60   # forzar revisión de refresh si el campeón lleva mucho sin actualizar (s)
TRAIN_REFRESH_MIN_GROWTH = 0.20     # crecimiento mínimo relativo de dataset para considerar stale override
TRAIN_REFRESH_MIN_ABS_ROWS = 60      # crecimiento mínimo absoluto de filas para stale override
TRAIN_REFRESH_MIN_ABS_ROWS_LOWN = 20 # override para modelos pequeños: refresco más temprano
TRAIN_REFRESH_LOWN_CUTOFF = 180      # n por debajo de esto usa umbral absoluto reducido
TRAIN_CANARY_FORCE_UNRELIABLE = True # canary: refresca probs pero bloquea REAL hasta validar en operación cerrada
CANARY_MIN_CLOSED_SIGNALS = 20      # cierres mínimos para decidir salida de canary
CANARY_MIN_HITRATE = 0.50           # hit-rate mínimo de cierres durante canary para promover
CANARY_RETRY_BATCH = 10             # si canary falla, ampliar ventana en este tamaño
CANARY_EVAL_COOLDOWN_S = 10.0       # evaluar progreso canary como máximo cada N segundos
# Escape controlado: evita deadlock cuando CANARY no acumula cierres pero la compuerta REAL ya está sólida.
CANARY_ALLOW_STRONG_GATE_REAL = True
CANARY_STRONG_GATE_MIN_PROB = IA_ACTIVACION_REAL_THR_POST_N15
CANARY_STRONG_GATE_MIN_CONFIRM = 2
TRAIN_ROWS_DROP_GUARD_RATIO = 0.35  # no reemplazar modelo si la muestra cae demasiado vs meta anterior
TRAIN_ROWS_DROP_GUARD_MIN_PREV = 120  # activar guard solo si el modelo previo ya tenía muestra razonable
FEATURE_MAX_DOMINANCE = 0.90  # Si una feature repite >90%, se considera casi constante
FEATURE_DQ_MIN_OK = 5         # mínimo de features sanas para no bloquear warmup por 1 columna ruidosa
TRAIN_WARMUP_MIN_ROWS = 250          # evita declarar modo confiable sin muestra mínima
INPUT_DUP_DIAG_COOLDOWN_S = 25.0     # anti-spam de diagnóstico por inputs duplicados
CLONED_PROB_TICKS_ALERT = 3          # ticks consecutivos de probs clonadas para alertar
INPUT_DUP_FINGERPRINT_DECIMALS = 6   # precisión estable para huella de inputs IA


# Semáforo de calibración (lectura rápida PredMedia/Real/Inflación/n)
SEM_CAL_N_ROJO = 30
SEM_CAL_N_AMARILLO = 100
SEM_CAL_INFL_OK_PP = 5.0
SEM_CAL_INFL_WARN_PP = 15.0

# ============================================================
# Defaults IA (centralizados, sin duplicados)
# ============================================================
MIN_TRAIN_ROWS  = 250
TEST_SIZE_FRAC  = 0.20
MIN_TEST_ROWS   = 40
THR_DEFAULT = 0.50
MIN_TRAIN_ROWS_ADAPTIVE = 40  # evita entrenar con train ridículo cuando el dataset aún es chico
MIN_TRAIN_SHARE_ADAPTIVE = 0.60

# Split honesto: TRAIN_BASE (pasado) / CALIB (más reciente) / TEST (último)
CALIB_SIZE_FRAC = 0.15
MIN_CALIB_ROWS = 80

# Feature list canónica (si tu reentreno define otra, ahí la cambias UNA vez)
# ============================================================
# Feature set CORE (13) — estable y sin mutaciones
# ============================================================
FEATURE_NAMES_CORE_13 = [
    # CORE13_v2 (scalping 1-min): mantener aportantes + reemplazo de no-aportantes.
    "racha_actual", "puntaje_estrategia", "payout",
    "ret_1m", "ret_3m", "ret_5m", "slope_5m", "rv_20",
    "range_norm", "bb_z", "body_ratio", "wick_imbalance", "micro_trend_persist",
]

# Por defecto entrenamos SOLO con las 13 core (modo estable)
FEATURE_NAMES_INTERACCIONES = [
    "racha_x_rebote",
    "rev_x_breakout",
]

# Gobernanza calidad>cantidad: entrenar solo con features que realmente aporten.
FEATURE_ALWAYS_KEEP = ["racha_actual"]
FEATURE_MAX_PROD = 6
FEATURE_SET_PROD_WARMUP = ["racha_actual", "puntaje_estrategia", "ret_1m", "slope_5m", "rv_20", "bb_z"]
FEATURE_SET_CORE_EXT = ["racha_actual", "puntaje_estrategia", "ret_1m", "slope_5m", "rv_20", "bb_z"]
FEATURE_SET_CORE_EXT_MIN_ROWS = 500
FEATURE_MIN_AUC_DELTA = 0.015      # aporte mínimo (|AUC_uni - 0.5|)
FEATURE_MAX_DOMINANCE_GATE = 0.965 # evita casi-constantes
FEATURE_DYNAMIC_SELECTION = False
# Durante warmup evitamos selección agresiva para no colapsar a 2-4 features.
FEATURE_FREEZE_CORE_DURING_WARMUP = True
FEATURE_FREEZE_CORE_MIN_ROWS = TRAIN_WARMUP_MIN_ROWS
# Si el modelo anterior colapsó a muy pocas features, permitimos reemplazarlo
# aunque la AUC temporal baje levemente en un reentreno puntual.
FEATURE_MIN_ACCEPTED_COUNT = 6

# Meta objetivo (calidad real en señales fuertes)
IA_TARGET_PRECISION = 0.70
IA_TARGET_PRECISION_FLOOR = 0.65   # piso mínimo para declarar confiable
IA_TARGET_MIN_SIGNALS = 30         # mínimo de señales en zona alta para validar

# Guardas de promoción de campeón: evitar reemplazar por modelos débiles/colapsados.
TRAIN_PROMOTE_MIN_AUC = 0.50
TRAIN_PROMOTE_MIN_FEATURES = 5

FEATURE_NAMES_PROD = list(FEATURE_SET_PROD_WARMUP)
FEATURE_NAMES_SHADOW = [f for f in FEATURE_NAMES_CORE_13 if f not in FEATURE_NAMES_PROD]
FEATURE_NAMES_DEFAULT = list(FEATURE_NAMES_CORE_13)
PROXY_FEATURES_BLOCK_TRAIN = [
    "ret_1m", "ret_3m", "ret_5m", "slope_5m", "rv_20",
    "range_norm", "bb_z", "body_ratio", "wick_imbalance", "micro_trend_persist",
]

class ModeloXGBCalibrado:
    """
    Wrapper picklable para calibrar probabilidades con un holdout temporal (CALIB),
    sin re-entrenar el modelo base. El modelo espera X ya escalado.
    calib_kind: "sigmoid" (Platt con LogisticRegression sobre logit(p)) o "isotonic".
    """
    def __init__(self, modelo_base, calib_kind: str, calib_obj):
        self.modelo_base = modelo_base
        self.calib_kind = str(calib_kind)
        self.calib_obj = calib_obj

    def _calibrar_p(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)

        if self.calib_kind == "sigmoid":
            z = np.log(p / (1.0 - p)).reshape(-1, 1)
            p_cal = self.calib_obj.predict_proba(z)[:, 1]
            return np.clip(p_cal, 1e-6, 1.0 - 1e-6)

        # isotonic
        p_cal = self.calib_obj.transform(p)
        return np.clip(np.asarray(p_cal, dtype=float), 1e-6, 1.0 - 1e-6)

    def predict_proba(self, X):
        p_base = self.modelo_base.predict_proba(X)[:, 1]
        p_cal = self._calibrar_p(p_base)
        return np.vstack([1.0 - p_cal, p_cal]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

# === FIN BLOQUE 3 ===

# === BLOQUE 4 — AUDIO (INIT Y REPRODUCCIÓN) ===
# Inicialización de audio
def init_audio():
    global AUDIO_AVAILABLE, SOUND_CACHE

    # No asumimos nada: recalculamos disponibilidad cada vez
    AUDIO_AVAILABLE = False

    # 1) Asegurar mixer (si no está listo)
    if pygame.mixer.get_init():
        AUDIO_AVAILABLE = True
    else:
        drivers = ['directsound', 'winmm', 'wasapi', None]
        configs = [
            (44100, -16, 2, 1024),
            (22050, -16, 2, 512),
            (44100, -16, 1, 1024),
        ]
        for driver in drivers:
            for freq, size, channels, buffer in configs:
                try:
                    if driver:
                        os.environ["SDL_AUDIODRIVER"] = driver
                    pygame.mixer.pre_init(frequency=freq, size=size, channels=channels, buffer=buffer)
                    pygame.mixer.init()
                    AUDIO_AVAILABLE = True
                    break
                except Exception:
                    pass
            if AUDIO_AVAILABLE:
                break

    # 2) Fallback winsound (aunque no tengamos pygame)
    if not AUDIO_AVAILABLE and winsound:
        AUDIO_AVAILABLE = True

    # 3) Cargar sonidos SOLO si mixer está operativo
    if pygame.mixer.get_init():
        base_dir = os.path.dirname(__file__)
        for event, filename in SOUND_PATHS.items():
            if event in SOUND_LOAD_ERRORS:
                continue
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                try:
                    SOUND_CACHE[event] = pygame.mixer.Sound(path)
                except Exception:
                    SOUND_LOAD_ERRORS.add(event)

def reproducir_evento(evento, es_demo=False, dentro_gatewin=True):
    global sonido_disparado

    if not AUDIO_AVAILABLE:
        return

    # Reglas de GateWIN/DEMO (mismas que tenías)
    if evento != "ia_53":
        if SONAR_SOLO_EN_GATEWIN and (not dentro_gatewin) and (not SONAR_FUERA_DE_GATEWIN):
            return
        if es_demo and not SONAR_TAMBIEN_EN_DEMO:
            return

    # 1) Preferir pygame si está cargado
    try:
        if evento in SOUND_CACHE:
            SOUND_CACHE[evento].play()
            sonido_disparado = True
            return
    except Exception:
        pass

    # 2) Fallback winsound (si pygame no está usable o no cargó el sonido)
    if winsound:
        try:
            filename = SOUND_PATHS.get(evento)
            if not filename:
                return
            base_dir = os.path.dirname(__file__)
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                sonido_disparado = True
        except Exception:
            pass
# === FIN BLOQUE 4 ===

# === BLOQUE 5 — TOKENS, BOT_NAMES Y ESTADO GLOBAL ===
# Leer tokens del usuario
def leer_tokens_usuario():
    if not os.path.exists("tokens_usuario.txt"):
        return None, None
    try:
        with open("tokens_usuario.txt", "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
            if len(lines) < 2:
                return None, None
            token_demo, token_real = lines[0], lines[1]
            if not token_demo or not token_real:
                return None, None
            return token_demo, token_real
    except Exception:
        return None, None

# Escritura atómica de token
def write_token_atomic(path, content):
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        return True
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return False


# Orden operativo recomendado (calidad real primero):
# 1) fulll47: mejor hit-rate y menor inflación del set comparado.
# 2) fulll50/fulll45: rendimiento similar pero con muestra algo mayor.
# 3) fulll48: intermedio, baja muestra.
# 4) fulll49/fulll46: sobreconfianza alta y peor hit-rate reciente.
BOT_NAMES = ["fulll47", "fulll50", "fulll45", "fulll48", "fulll49", "fulll46"]
IA53_TRIGGERED = {bot: False for bot in BOT_NAMES}
IA53_LAST_TS = {bot: 0.0 for bot in BOT_NAMES}
TOKEN_FILE = "token_actual.txt"
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
saldo_real = "--"
SALDO_INICIAL = None
META = None
meta_mostrada = False
eventos_recentes = deque(maxlen=8)
reinicio_forzado = asyncio.Event()

salir = False
pausado = False
reinicio_manual = False
MAESTRO_PAUSE_FILE = os.path.abspath(os.path.expanduser(os.getenv("MAESTRO_PAUSE_STATE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "maestro_pause_state.json"))))
maestro_pause_active = False
maestro_pause_reason = ""
maestro_pause_resume_ts = 0.0
maestro_pause_started_ts = 0.0
maestro_pause_last_read_ts = 0.0
maestro_pause_last_log_ts = 0.0
maestro_pause_ref_balance = None
maestro_pause_trigger_balance = None
maestro_pause_source = ""
maestro_pause_last_state = False

LIMPIEZA_PANEL_HASTA = 0
ULTIMA_ACT_SALDO = 0
REFRESCO_SALDO = 12
MAX_CICLOS = len(MARTI_ESCALADO)
huellas_usadas = {bot: set() for bot in BOT_NAMES}
SNAPSHOT_FILAS = {bot: 0 for bot in BOT_NAMES}
REAL_ENTRY_BASELINE = {bot: 0 for bot in BOT_NAMES}  # filas al entrar/reafirmar REAL
OCULTAR_HASTA_NUEVO = {bot: False for bot in BOT_NAMES}
t_inicio_indef = {bot: None for bot in BOT_NAMES}
last_update_time = {bot: time.time() for bot in BOT_NAMES}
LAST_REAL_CLOSE_SIG = {bot: None for bot in BOT_NAMES}  # evita procesar el mismo cierre REAL varias veces
REAL_CLOSE_PENDING = {bot: None for bot in BOT_NAMES}
REAL_CLOSE_PENDING_TTL_S = 240
REAL_MANUAL_ALERT = {
    "active": False,
    "bot": None,
    "ciclo": None,
    "source": None,
    "ts": 0.0,
    "msg": "",
}

def _marcar_real_close_pending(bot, ciclo, source="UNKNOWN", round_id=None):
    try:
        REAL_CLOSE_PENDING[bot] = {
            "active": True,
            "bot": str(bot),
            "ciclo": int(ciclo),
            "baseline": int(REAL_ENTRY_BASELINE.get(bot, 0) or 0),
            "ts": float(time.time()),
            "source": str(source or "UNKNOWN"),
            "round_id": int(round_id) if round_id is not None else None,
        }
        agregar_evento(
            f"🧷 REAL_CLOSE_PENDING armado: bot={bot} C{int(ciclo)} "
            f"baseline={REAL_CLOSE_PENDING[bot]['baseline']} round={round_id}"
        )
    except Exception as e:
        agregar_evento(f"⚠️ Error armando REAL_CLOSE_PENDING: {bot} {e}")

def _hay_real_close_pending_activo():
    try:
        now = time.time()
        for bot, p in REAL_CLOSE_PENDING.items():
            if isinstance(p, dict) and p.get("active"):
                age = now - float(p.get("ts", 0) or 0)
                if age <= float(REAL_CLOSE_PENDING_TTL_S):
                    return True, bot, p
                REAL_CLOSE_PENDING[bot] = None
                agregar_evento(f"⚠️ REAL_CLOSE_PENDING expirado sin cierre: {bot} C{p.get('ciclo')}")
        return False, None, None
    except Exception:
        return False, None, None

def _real_close_sig(bot, res, monto, ciclo, payout_total, baseline=None):
    return (
        str(bot),
        str(res),
        round(float(monto or 0.0), 2),
        int(ciclo or 0),
        round(float(payout_total or 0.0), 4),
        int(baseline or REAL_ENTRY_BASELINE.get(bot, 0) or 0),
    )
CTT_CLOSE_EVENTS = deque(maxlen=6000)
CTT_CLOSE_SEEN = set()
CTT_STATE = {
    "status": "NEUTRAL",
    "regime": "NEUTRAL",
    "gate": "NEUTRAL",
    "asset": None,
    "t_front": 0.0,
    "wave_start": 0.0,
    "wave_age_s": None,
    "wave_ttl_ok": False,
    "wave_ratio": 0.0,
    "wave_total": 0,
    "confirmadores": 0,
    "density_cpm": 0.0,
    "diversity_ratio": 0.0,
    "redundancy_high": False,
    "green_mode": "none",
    "rezagados_validos": [],
    "no_participantes": [],
    "sample": 0,
    "roof_policy": "normal",
    "roof_delta": 0.0,
    "reason": "init",
}
REAL_OWNER_LOCK = None  # owner REAL en memoria (evita carreras de lectura de archivo)
REAL_LOCK_MISMATCH_SINCE = 0.0
REAL_LOCK_RECONCILE_S = 6.0
REAL_UI_RECON_LOG_TS = 0.0
REAL_OWNER_STATE_FILE = "real_owner_state.json"
REAL_CLOSE_TRACE_FILE = "real_close_trace.jsonl"
LAST_REAL_CLOSE_TRACE = {}

EMBUDO_DECISION_STATE = {
    "decision_final": EMBUDO_FINAL_WAIT_SOFT,
    "decision_reason": "init",
    "gate_quality": "weak",
    "risk_mode": "WAIT_SOFT",
    "hard_block_reason": "",
    "soft_wait_reason": "init",
    "top1_bot": None,
    "top2_bot": None,
    "gap_value": 0.0,
    "top1_prob": 0.0,
    "top2_prob": 0.0,
    "degrade_from": "none",
}

try:
    last_sig_por_bot
except NameError:
    last_sig_por_bot = {b: None for b in BOT_NAMES}

estado_bots = {
    bot: {
        "resultados": [], 
        "token": "DEMO", 
        "trigger_real": False,
        "ganancias": 0, 
        "perdidas": 0, 
        "porcentaje_exito": None,
        "tamano_muestra": 0,
        "prob_ia": None,              # guardará prob REAL (0..1). OJO: ya NO la forzamos a 0 por “no señal”
        "ia_ready": False,           # True solo si logramos armar features + predecir sin error
        "ia_last_err": None,         # texto corto del motivo si no se pudo predecir
        "ia_last_prob_ts": 0.0,      # timestamp de la última prob calculada
        "ciclo_actual": 1,
        "modo_real_anunciado": False, 
        "ultimo_resultado": None,
        "reintentar_ciclo": False,
        "remate_active": False,
        "remate_start": None,
        "remate_reason": "",
        "fuente": None,  
        "real_activado_en": 0.0,  
        "ignore_cierres_hasta": 0.0,
        "real_timeout_first_warn": 0.0,
        "modo_ia": "low_data",  # Arranca visible en warmup para evitar confusión de OFF al inicio
        "ia_seniales": 0,  # contadores para medir IA
        "ia_aciertos": 0,
        "ia_fallos": 0,
        "ia_senal_pendiente": False,  # Flag para operación recomendada por IA
        "ia_prob_senal": None,        # prob IA en el momento de la señal
        "ia_regime_score": 0.0,       # capa A (régimen)
        "ia_evidence_n": 0,           # soporte histórico en umbral objetivo
        "ia_evidence_wr": 0.0,        # win-rate real en umbral objetivo
        # Telemetría Pattern V1 (existente)
        "ia_pattern_bonus": 0.0,
        "ia_pattern_penal": 0.0,
        # Telemetría patrón por columnas (separada, evita colisión con Pattern V1)
        "ia_pattern_col_state": "BLOQUEADO",
        "ia_pattern_col_bonus": 0.0,
        "ia_pattern_col_penal": 0.0,
        "ia_pattern_col_delta": 0.0,
        "sync_wait": False,
        "last_sync_round": 1,
        "last_seen_ts": 0.0,
        "estado_visual": "ACTIVO",
    }
    for bot in BOT_NAMES
}
IA90_stats = {bot: {"n": 0, "ok": 0, "pct": 0.0, "pct_raw": 0.0, "pct_smooth": 50.0} for bot in BOT_NAMES}
# Ventana corta para diagnosticar el bloqueo dominante del embudo en HUD.
HUD_BLOQUEO_WINDOW = 120
HUD_BLOQUEOS_RECIENTES = deque(maxlen=HUD_BLOQUEO_WINDOW)
HUD_BOT_GATE_DIAG_EVERY_S = 6.0
_LAST_HUD_BOT_GATE_DIAG_TS = 0.0
PURIFICACION_REAL_EVENT_COOLDOWN_S = 90.0
_LAST_PURIFICACION_REAL_EVENT_TS = 0.0

EVENTO_MAX_CHARS = 220

def _normalizar_evento_texto(msg: str, max_chars: int = EVENTO_MAX_CHARS) -> str:
    try:
        txt = str(msg if msg is not None else "")
    except Exception:
        txt = ""
    for ch in ("\r", "\n", "\t"):
        txt = txt.replace(ch, " ")
    txt = " ".join(txt.split())
    if len(txt) > int(max_chars):
        txt = txt[: max(0, int(max_chars) - 1)] + "…"
    return txt

# === FIN BLOQUE 5 ===

# === BLOQUE 6 — LOCKS, FIRMAS Y UTILIDADES CSV ===
def _firma_registro(feature_names, row_vals, label):
    """
    Firma estable anti-duplicados:
    - Formato fijo para floats (evita variaciones 0.1 vs 0.10000000002)
    """
    parts = []
    for v in row_vals:
        try:
            parts.append(f"{float(v):.6f}")
        except Exception:
            parts.append(str(v))
    try:
        parts.append(str(int(label)))
    except Exception:
        parts.append(str(label))
    return "|".join(parts)

# Contar filas en CSV (sin header)
def contar_filas_csv(bot_name: str) -> int:
    ruta = f"registro_enriquecido_{bot_name}.csv"
    if not os.path.exists(ruta):
        return 0
    for encoding in ["utf-8", "latin-1", "windows-1252"]:
        try:
            with open(ruta, "r", newline="", encoding=encoding, errors="replace") as f:
                n = sum(1 for _ in f) - 1
                return max(0, n)
        except Exception:
            continue
    return 0

INCREMENTAL_LOCK_FILE = "incremental.lock"

# Contar filas en dataset_incremental.csv (sin contar header)
def contar_filas_incremental() -> int:
    """
    Devuelve número de filas (sin header) de dataset_incremental.csv.

    Optimizado:
    - Cachea (pos, size, rows) para evitar re-escaneo completo en cada llamada.
    - Si el archivo crece, cuenta solo las líneas nuevas.
    """
    try:
        path = "dataset_incremental.csv"

        if not os.path.exists(path):
            contar_filas_incremental._cache = {"pos": 0, "rows": 0, "size": 0}
            return 0

        cache = getattr(contar_filas_incremental, "_cache", None)
        size = os.path.getsize(path)

        # Recuento completo en binario (robusto a encoding)
        def _count_full_rows() -> int:
            total = 0
            last_byte = b""
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    total += chunk.count(b"\n")
                    last_byte = chunk[-1:]
            # Si el archivo NO termina en \n, hay una línea final sin salto
            if size > 0 and last_byte != b"\n":
                total += 1
            # Quita header si existe
            return max(0, total - 1)

        # Sin cache o el archivo se redujo/truncó: recuenta todo
        if (not cache) or (size < int(cache.get("size", 0) or 0)) or (int(cache.get("pos", 0) or 0) > size):
            rows = _count_full_rows()
            contar_filas_incremental._cache = {"pos": size, "rows": rows, "size": size}
            return rows

        # Si no cambió, devuelve cache
        if size == int(cache.get("size", 0) or 0):
            return int(cache.get("rows", 0) or 0)

        # Creció: cuenta solo líneas nuevas desde la última posición
        pos = int(cache.get("pos", 0) or 0)
        new_lines = 0
        with open(path, "rb") as f:
            f.seek(pos)
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                new_lines += chunk.count(b"\n")

        rows = int(cache.get("rows", 0) or 0) + new_lines
        contar_filas_incremental._cache = {"pos": size, "rows": rows, "size": size}
        return rows

    except Exception:
        return 0

# Lock de archivo
@contextmanager
def file_lock(path="real.lock", timeout=5.0, stale_after=30.0):
    """
    Lock por archivo (cross-platform) con protección anti-colisión:

    - NO borra el lock de otro proceso activo.
    - Solo intenta limpiar locks *stale* (viejos) si supera stale_after segundos.
    - Si no logra adquirir lock, continúa SIN exclusión (como ya venías haciendo),
      pero sin destruir el lock ajeno.
    """
    start_time = time.time()
    fd = None
    acquired = False

    try:
        # 1) Intento normal por timeout
        while (time.time() - start_time) < float(timeout):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.10)
            except Exception:
                time.sleep(0.10)

        # 2) Si no se pudo, evaluar si el lock parece "stale"
        if not acquired:
            age = None
            try:
                age = time.time() - os.path.getmtime(path)
            except Exception:
                age = None

            if age is not None and age > float(stale_after):
                # Solo si es viejo de verdad, intentamos limpiar
                try:
                    os.remove(path)
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    acquired = True
                except Exception as e:
                    try:
                        print(f"⚠️ Lock stale no se pudo limpiar ({path}): {e}. Continúo sin exclusión.")
                    except Exception:
                        pass
            else:
                # Lock reciente: NO tocarlo
                try:
                    print(f"⚠️ No se adquirió lock ({path}) en {timeout}s (lock reciente). Continúo sin exclusión.")
                except Exception:
                    pass

        # 3) Ejecutar la sección crítica (con o sin lock adquirido)
        yield

    finally:
        if acquired and fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                os.remove(path)
            except Exception:
                pass
# ============================================================
# DATASET INCREMENTAL — Reparación de esquema "mutante"
# (header viejo / columnas extra / filas con campos de más)
# Objetivo: mantener SIEMPRE un CSV estable para pandas/IA.
# ============================================================

# Reusar el set core (13) para que incremental y entrenamiento nunca diverjan
try:
    INCREMENTAL_FEATURES_V2 = list(FEATURE_NAMES_CORE_13)
except Exception:
    INCREMENTAL_FEATURES_V2 = [
        "racha_actual", "puntaje_estrategia", "payout",
        "ret_1m", "ret_3m", "ret_5m", "slope_5m", "rv_20",
        "range_norm", "bb_z", "body_ratio", "wick_imbalance", "micro_trend_persist",
    ]
INCREMENTAL_CLOSE_COLS = [f"close_{i}" for i in range(20)]
INCREMENTAL_META_FLAGS = ["row_has_proxy_features", "row_train_eligible"]
for _c in INCREMENTAL_CLOSE_COLS:
    if _c not in INCREMENTAL_FEATURES_V2:
        INCREMENTAL_FEATURES_V2.append(_c)
# === LOCK ESTRICTO (solo para escrituras sensibles como incremental.csv) ===
@contextmanager
def file_lock_required(path: str, timeout: float = 6.0, stale_after: float = 30.0):
    """
    Igual que file_lock, pero:
    - Si NO adquiere lock, NO ejecuta la sección crítica (yield False).
    - Para escrituras que NO toleran concurrencia (append CSV).
    """
    start_time = time.time()
    fd = None
    acquired = False

    try:
        while (time.time() - start_time) < float(timeout):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.10)
            except Exception:
                time.sleep(0.10)

        if not acquired:
            age = None
            try:
                age = time.time() - os.path.getmtime(path)
            except Exception:
                age = None

            if age is not None and age > float(stale_after):
                try:
                    os.remove(path)
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    acquired = True
                except Exception:
                    acquired = False

        yield acquired

    finally:
        if acquired and fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                os.remove(path)
            except Exception:
                pass
# === /LOCK ESTRICTO ===

def _canonical_incremental_cols(feature_names: list | None = None) -> list:
    fn = feature_names if feature_names else INCREMENTAL_FEATURES_V2
    out = list(fn)
    for mc in INCREMENTAL_META_FLAGS:
        if mc not in out:
            out.append(mc)
    return out + ["result_bin"]

_INCREMENTAL_INGEST_STATS = {
    "filas_incremental_aceptadas": 0,
    "filas_incremental_saneadas_close": 0,
    "filas_incremental_proxy_no_train": 0,
    "filas_incremental_descartadas_total": 0,
    "filas_incremental_close_reales_validas": 0,
    "last_log_ts": 0.0,
}

def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None

def _safe_int01(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        v = int(float(x))
        if v not in (0, 1):
            return None
        return v
    except Exception:
        return None

def reparar_dataset_incremental_mutante(ruta: str = "dataset_incremental.csv", cols: list | None = None) -> bool:
    """
    Repara dataset_incremental.csv cuando quedó 'mutante' por:
    - header corrupto (ej: racha...ia) o incompleto
    - filas con más/menos columnas (ej: bot_id, activo_id metidos)
    - mezcla de esquemas (Expected X fields, saw Y)

    Estrategia:
    - Reescribe un CSV limpio con columnas canónicas (cols).
    - Si el archivo actual tiene columnas canónicas presentes, mapea por header.
    - Si el header no es usable, intenta rescate por POSICIÓN:
        * len>=16: [0..12] + [15] (drop bot_id/activo_id)
        * len==15: [0..12] + [14]
        * len==14: [0..12] + [13]  (13 feats + label)
        * len>len(cols): toma primeras (len(cols)-1) + última como label
    - Crea backup del archivo original con sufijo .bak_<epoch>.
    """
    cols = cols or _canonical_incremental_cols()
    if not os.path.exists(ruta):
        return False

    # Leer header y detectar mutación
    header_list = None
    enc_usado = None
    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            with open(ruta, "r", newline="", encoding=enc, errors="replace") as f:
                first = f.readline()
            header_list = [h.strip() for h in first.strip().split(",")] if first else []
            enc_usado = enc
            break
        except Exception:
            continue

    if header_list is None:
        return False

    # Si el header ya es canónico, igual escanear rápido por longitudes
    header_ok = (header_list == cols)
    header_has_canonical = set(cols).issubset(set(header_list))

    needs_repair = not header_ok

    # Escaneo rápido de longitudes (si hay mezcla de campos, se marca mutante)
    try:
        with open(ruta, "r", newline="", encoding=enc_usado or "utf-8", errors="replace") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # header
            for j, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) != len(header_list):
                    needs_repair = True
                    break
                if j >= 3000:
                    break
    except Exception:
        needs_repair = True

    if not needs_repair:
        return False

    # Armar filas limpias
    cleaned_rows = []
    seen_rows = set()
    header_index = {name: i for i, name in enumerate(header_list)} if header_has_canonical else {}

    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            with open(ruta, "r", newline="", encoding=enc, errors="replace") as f:
                reader = csv.reader(f)
                _ = next(reader, None)  # header
                for row in reader:
                    if not row:
                        continue

                    new_row = None

                    # 1) Si el header contiene columnas canónicas, mapear por nombre
                    if header_has_canonical:
                        try:
                            new_row = [row[header_index[c]] if header_index[c] < len(row) else "" for c in cols]
                        except Exception:
                            new_row = None

                    # 2) Si el header no es canónico, intentar conversión por nombre (legacy->v2)
                    if new_row is None and header_list and len(header_list) == len(row):
                        try:
                            row_map = {str(header_list[i]).strip(): row[i] for i in range(len(row))}
                            row_map = _enriquecer_scalping_features_row(row_map)
                            lb = row_map.get("result_bin", row_map.get("label", row_map.get("y", None)))
                            new_row = [row_map.get(c, "") for c in cols[:-1]] + [lb]
                        except Exception:
                            new_row = None

                    # 3) Rescate por posición (último recurso legacy; evitar para v2 salvo emergencia)
                    if new_row is None:
                        ncols = len(cols)
                        rlen = len(row)
                        if rlen >= ncols:
                            new_row = list(row[:ncols - 1]) + [row[-1]]
                        else:
                            continue


                    # Validación y saneo defensivo (clip + contrato activo)
                    try:
                        row_map_clean = {cols[i]: new_row[i] for i in range(len(cols))}
                        feat_validate = [c for c in cols[:-1] if c not in INCREMENTAL_META_FLAGS and c != "ts_ingest"]
                        row_map_clean = clip_feature_values(row_map_clean, feat_validate)
                        for mc in INCREMENTAL_META_FLAGS:
                            if mc not in row_map_clean or row_map_clean.get(mc, "") in ("", None):
                                row_map_clean[mc] = 0 if mc == "row_has_proxy_features" else 1
                        if "ts_ingest" in cols and (row_map_clean.get("ts_ingest", "") in ("", None)):
                            row_map_clean["ts_ingest"] = float(time.time())
                        ok_row, _reason = validar_fila_incremental(row_map_clean, feat_validate)
                        if not ok_row:
                            continue
                        lab = _safe_int01(row_map_clean.get("result_bin", new_row[-1]))
                        if lab is None:
                            continue
                        row_clean = []
                        for c in cols[:-1]:
                            if c in INCREMENTAL_META_FLAGS:
                                row_clean.append(int(float(row_map_clean.get(c, 0 if c == "row_has_proxy_features" else 1) or 0)))
                            else:
                                row_clean.append(float(row_map_clean.get(c, 0.0) or 0.0))
                        row_clean = row_clean + [lab]
                        # Deduplicar durante repair para no inflar entrenamiento por filas repetidas.
                        sig = tuple(round(float(v), 10) for v in row_clean[:-1]) + (int(row_clean[-1]),)
                        if sig in seen_rows:
                            continue
                        seen_rows.add(sig)
                        cleaned_rows.append(row_clean)
                    except Exception:
                        continue
            break
        except Exception:
            continue

    # Reescritura atómica con backup
    tmp = ruta + ".tmp_repair"
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in cleaned_rows:
                w.writerow(r)
            f.flush()
            os.fsync(f.fileno())

        backup = f"{ruta}.bak_{int(time.time())}"
        backed_up = False

        # 1) Intento preferido: renombrar (rápido y atómico)
        try:
            os.replace(ruta, backup)
            backed_up = True
        except Exception:
            backed_up = False

        # 2) Fallback: copiar (cuando rename falla por permisos/locks)
        if not backed_up:
            try:
                shutil.copy2(ruta, backup)
                backed_up = True
            except Exception:
                backed_up = False

        # 3) Si NO hay backup, NO pisamos el original
        if not backed_up:
            raise RuntimeError("No se pudo crear backup del incremental; se aborta reparación para no perder datos.")

        os.replace(tmp, ruta)
        return True

    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        fn_evt = globals().get("agregar_evento", None)
        try:
            if callable(fn_evt):
                fn_evt(f"⚠️ Incremental: reparación falló: {e}")
            else:
                print(f"⚠️ Incremental: reparación falló: {e}")
        except Exception:
            print(f"⚠️ Incremental: reparación falló: {e}")
        return False

# Firma persistente anti-duplicados
_SIG_DIR = ".sigcache"
os.makedirs(_SIG_DIR, exist_ok=True)

def _sig_path(bot): 
    safe = str(bot).replace("/", "_").replace("\\", "_")
    return os.path.join(_SIG_DIR, f"{safe}.sig")

def _load_recent_sigs(bot: str, max_keep: int = 50) -> list:
    """
    Devuelve lista de firmas recientes (últimas N) desde disco.
    Compatible con formato viejo (1 sola firma).
    """
    try:
        p = _sig_path(bot)
        if not os.path.exists(p):
            return []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        if not lines:
            return []
        return lines[-int(max_keep):]
    except Exception:
        return []

def _sig_in_cache(bot: str, sig: str, max_keep: int = 50) -> bool:
    try:
        return sig in set(_load_recent_sigs(bot, max_keep=max_keep))
    except Exception:
        return False

def _append_sig_cache(bot: str, sig: str, max_keep: int = 50):
    """
    Guarda firma al final, manteniendo solo últimas N (y sin duplicados internos).
    """
    try:
        max_keep = int(max_keep)
        if max_keep < 5:
            max_keep = 5

        lst = _load_recent_sigs(bot, max_keep=max_keep)
        # mover al final si existe
        lst = [x for x in lst if x != sig] + [sig]
        lst = lst[-max_keep:]

        with open(_sig_path(bot), "w", encoding="utf-8") as f:
            f.write("\n".join(lst))
    except Exception:
        pass
# === COMPAT: helpers legacy (evita NameError y mantiene tu lógica actual) ===
def _load_last_sig(bot: str) -> str | None:
    """
    Compatibilidad: versiones antiguas esperaban una sola firma.
    Hoy guardamos varias en _sigcache; devolvemos la última.
    """
    try:
        lst = _load_recent_sigs(bot, max_keep=50)
        if not lst:
            return None
        return lst[-1]
    except Exception:
        return None

def _save_last_sig(bot: str, sig: str):
    """
    Compatibilidad: guarda como “última firma”, manteniendo historial.
    """
    try:
        _append_sig_cache(bot, sig, max_keep=50)
    except Exception:
        pass
# === /COMPAT ===

def _make_sig(row_dict):
    """Firma estable para comparar filas entre reinicios (sin timestamp si no existe)."""
    try:
        # Orden determinista
        data = {k: row_dict.get(k) for k in sorted(row_dict.keys())}
        s = json.dumps(data, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    except:
        return None

_INCREMENTAL_SIG_CACHE = {"mtime": 0.0, "sigs": set()}

def _load_incremental_signatures(ruta: str, feats: list, max_rows: int = INCREMENTAL_DUP_SCAN_LINES) -> set:
    """Carga firmas recientes del incremental para bloquear duplicados exactos aunque reinicie el bot."""
    try:
        if not os.path.exists(ruta):
            return set()
        mtime = float(os.path.getmtime(ruta) or 0.0)
        cache = _INCREMENTAL_SIG_CACHE
        if cache.get("mtime") == mtime and cache.get("sigs"):
            return set(cache.get("sigs") or set())

        sigs = set()
        with open(ruta, "r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.DictReader(f))
        if max_rows > 0:
            rows = rows[-int(max_rows):]
        for r in rows:
            try:
                vals = [float(r.get(k, 0.0) or 0.0) for k in feats]
                lab = int(float(r.get("result_bin", 0) or 0))
                sigs.add(_firma_registro(feats, vals, lab))
            except Exception:
                continue
        _INCREMENTAL_SIG_CACHE["mtime"] = mtime
        _INCREMENTAL_SIG_CACHE["sigs"] = set(sigs)
        return sigs
    except Exception:
        return set()

def _incremental_signature_exists(ruta: str, sig: str, feats: list) -> bool:
    try:
        return sig in _load_incremental_signatures(ruta, feats, max_rows=INCREMENTAL_DUP_SCAN_LINES)
    except Exception:
        return False

# Core scalping mínimo válido para no cuarentenar filas útiles por close_* incompleto
def _core_scalping_ready_from_row(row: dict) -> bool:
    try:
        keys = ("ret_1m", "slope_5m", "rv_20", "bb_z")
        for k in keys:
            v = row.get(k, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                return False
            vf = float(v)
            if not np.isfinite(vf):
                return False
        return True
    except Exception:
        return False

def _close_snapshot_issue_from_row(row: dict, required_closes: int = 20) -> bool:
    try:
        need = int(required_closes)
        valid = 0
        for i in range(need):
            v = row.get(f"close_{i}", None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                continue
            vf = float(v)
            if np.isfinite(vf) and vf > 0.0:
                valid += 1
        return bool(valid < need)
    except Exception:
        return True

# Nueva: Validar fila para incremental (blindaje contra basura)
def validar_fila_incremental(fila_dict, feature_names):
    close_sanitized = False
    close_valid_count = 0
    # Asegura numericidad real
    for k in feature_names:
        v = fila_dict.get(k, None)
        if str(k).startswith("close_"):
            try:
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    raise ValueError("close_missing")
                vf = float(v)
                if (not np.isfinite(vf)) or vf <= 0.0:
                    raise ValueError("close_invalid")
                fila_dict[k] = float(vf)
                close_valid_count += 1
                continue
            except Exception:
                fila_dict[k] = 0.0
                close_sanitized = True
                continue
        try:
            v = float(v)
            if not np.isfinite(v):
                return False, f"{k}=NaN/inf"
        except Exception:
            return False, f"{k}=no numérico"
        fila_dict[k] = v  # normaliza en sitio

    # Rangos lógicos por contrato activo (v2 + compat legacy opcional)
    ranges = {
        "rsi_9": (0, 100), "rsi_14": (0, 100),
        "payout": (0, 1.5), "volatilidad": (0, 1), "es_rebote": (0, 1), "hora_bucket": (0, 1),
        "ret_1m": (-1, 1), "ret_3m": (-1, 1), "ret_5m": (-1, 1),
        "slope_5m": (-1, 1), "rv_20": (0, 1), "range_norm": (0, 1),
        "bb_z": (-3, 3), "body_ratio": (0, 1), "wick_imbalance": (-1, 1), "micro_trend_persist": (-1, 1),
    }
    for k in feature_names:
        if k in ranges:
            lo, hi = ranges[k]
            v = float(fila_dict.get(k, 0.0) or 0.0)
            if not (lo <= v <= hi):
                return False, f"{k} fuera de rango [{lo},{hi}]"

    # Cuarentena conservadora: filas sospechosas que contaminan entrenamiento
    try:
        vals = [float(fila_dict.get(k, 0.0) or 0.0) for k in feature_names]
        nz = sum(1 for v in vals if abs(v) > 1e-12)
        if len(vals) >= 8 and nz <= 2:
            return False, "fila_sospechosa: casi_todo_cero"
        if sum(1 for v in vals if not np.isfinite(v)) > 0:
            return False, "fila_sospechosa: no_finito"
    except Exception:
        return False, "fila_sospechosa: parse"

    close_snapshot_issue = bool(close_sanitized or close_valid_count < 20)
    core_scalping_ready = _core_scalping_ready_from_row(fila_dict)
    if close_snapshot_issue and (not core_scalping_ready):
        fila_dict["row_has_proxy_features"] = 1
        fila_dict["row_train_eligible"] = 0

    return True, ""
        
def _anexar_incremental_desde_bot_CANON(bot: str, fila_dict_or_full: dict, label: int | None = None, feature_names: list | None = None) -> bool:
    """
    Anexa 1 fila al dataset_incremental.csv de forma estable:
    - Header canónico (anti "mutante")
    - Lock dedicado (incremental.lock) para evitar choques
    - Repair del CSV SOLO bajo lock (evita corrupción por concurrencia)
    - Retry ante PermissionError (Excel/OneDrive/AV)
    - Anti-duplicado por firma persistente (_sigcache por bot)
    """
    try:
        def _ingest_bump(key: str, delta: int = 1):
            try:
                stats = globals().get("_INCREMENTAL_INGEST_STATS", {})
                stats[key] = int(stats.get(key, 0) or 0) + int(delta)
                now_ts = time.time()
                if (now_ts - float(stats.get("last_log_ts", 0.0) or 0.0)) >= 15.0:
                    txt = (
                        "🧾 incremental-ingest: filas_incremental_aceptadas={a} "
                        "filas_incremental_saneadas_close={s} filas_incremental_proxy_no_train={p} "
                        "filas_incremental_descartadas_total={d} filas_incremental_close_reales_validas={r}"
                    ).format(
                        a=int(stats.get("filas_incremental_aceptadas", 0) or 0),
                        s=int(stats.get("filas_incremental_saneadas_close", 0) or 0),
                        p=int(stats.get("filas_incremental_proxy_no_train", 0) or 0),
                        d=int(stats.get("filas_incremental_descartadas_total", 0) or 0),
                        r=int(stats.get("filas_incremental_close_reales_validas", 0) or 0),
                    )
                    try:
                        agregar_evento(txt)
                    except Exception:
                        print(txt)
                    stats["last_log_ts"] = now_ts
            except Exception:
                pass

        ruta = "dataset_incremental.csv"
        feats = feature_names or INCREMENTAL_FEATURES_V2
        cols = _canonical_incremental_cols(feats)
        if "ts_ingest" not in cols:
            cols = list(cols[:-1]) + ["ts_ingest", cols[-1]]

        if not isinstance(fila_dict_or_full, dict) or not fila_dict_or_full:
            _ingest_bump("filas_incremental_descartadas_total", 1)
            return False

        # Normalizar/enriquecer fila para contrato CORE13_v2 (con fallback legacy).
        fila_dict_or_full = _enriquecer_scalping_features_row(fila_dict_or_full)

        # Label: aceptar parámetro o leer del dict
        if label is None:
            lb = fila_dict_or_full.get("result_bin", None)
            try:
                label = int(float(lb))
            except Exception:
                _ingest_bump("filas_incremental_descartadas_total", 1)
                return False

        try:
            label = int(label)
        except Exception:
            _ingest_bump("filas_incremental_descartadas_total", 1)
            return False
        if label not in (0, 1):
            _ingest_bump("filas_incremental_descartadas_total", 1)
            return False

        # Dict solo con features canónicas + metadatos de elegibilidad
        fila_dict = {k: fila_dict_or_full.get(k, None) for k in feats}
        try:
            row_has_proxy = int(float(fila_dict_or_full.get("row_has_proxy_features", 0) or 0))
        except Exception:
            row_has_proxy = 0
        try:
            row_train_eligible = int(float(fila_dict_or_full.get("row_train_eligible", 1) or 1))
        except Exception:
            row_train_eligible = 1
        if row_has_proxy == 1 and (not _core_scalping_ready_from_row(fila_dict_or_full)) and _close_snapshot_issue_from_row(fila_dict_or_full):
            row_train_eligible = 0
        ts_ing = fila_dict_or_full.get("ts_ingest", None)
        if ts_ing is None:
            try:
                ts_ing = float(time.time())
            except Exception:
                ts_ing = ""

        # Validación fuerte
        ok, why = validar_fila_incremental(fila_dict, feats)
        if not ok:
            fn_evt = globals().get("agregar_evento", None)
            try:
                if callable(fn_evt):
                    fn_evt(f"⚠️ Incremental: fila descartada {bot}: {why}")
            except Exception:
                pass
            _ingest_bump("filas_incremental_descartadas_total", 1)
            return False
        try:
            row_has_proxy = int(max(row_has_proxy, int(float(fila_dict.get("row_has_proxy_features", 0) or 0))))
        except Exception:
            pass
        try:
            row_train_eligible = int(min(row_train_eligible, int(float(fila_dict.get("row_train_eligible", 1) or 1))))
        except Exception:
            pass
        if row_has_proxy == 1 and (not _core_scalping_ready_from_row(fila_dict)) and _close_snapshot_issue_from_row(fila_dict):
            row_train_eligible = 0

        row_vals = [float(fila_dict[k]) for k in feats]
        row_all = list(row_vals) + [int(row_has_proxy), int(row_train_eligible)]
        sig = _firma_registro(feats, row_vals, label)

        # Anti-duplicado persistente (cache local + escaneo incremental reciente)
        if _sig_in_cache(bot, sig, max_keep=50):
            return False
        if _incremental_signature_exists(ruta, sig, feats):
            return False

        attempts = 8
        base_sleep = 0.08

        with file_lock_required("incremental.lock", timeout=6.0, stale_after=30.0) as got:
            if not got:
                fn_evt = globals().get("agregar_evento", None)
                try:
                    if callable(fn_evt):
                        fn_evt("⚠️ Incremental: no se pudo adquirir lock (incremental.lock). Fila omitida para evitar corrupción.")
                except Exception:
                    pass
                return False

            # ✅ Bajo lock: asegurar existencia + header estable + repair si hace falta
            if os.path.exists(ruta):
                try:
                    with open(ruta, "r", encoding="utf-8", errors="replace", newline="") as f:
                        first = f.readline().strip()
                    header_now = [h.strip() for h in first.split(",")] if first else []
                    if header_now != cols:
                        reparar_dataset_incremental_mutante(ruta=ruta, cols=cols)
                except Exception:
                    try:
                        reparar_dataset_incremental_mutante(ruta=ruta, cols=cols)
                    except Exception:
                        pass
            else:
                try:
                    with open(ruta, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(cols)
                        f.flush()
                        os.fsync(f.fileno())
                except Exception:
                    _ingest_bump("filas_incremental_descartadas_total", 1)
                    return False

            # Append con retry
            for n in range(attempts):
                try:
                    with open(ruta, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(row_all + [ts_ing, label])
                        f.flush()
                        os.fsync(f.fileno())

                    _save_last_sig(bot, sig)
                    try:
                        _INCREMENTAL_SIG_CACHE.setdefault("sigs", set()).add(sig)
                        _INCREMENTAL_SIG_CACHE["mtime"] = float(os.path.getmtime(ruta) or 0.0)
                    except Exception:
                        pass
                    _ingest_bump("filas_incremental_aceptadas", 1)
                    if int(row_has_proxy) == 1 or int(row_train_eligible) == 0:
                        _ingest_bump("filas_incremental_proxy_no_train", 1)
                    try:
                        close_real_valid = all(float(fila_dict.get(f"close_{i}", 0.0) or 0.0) > 0.0 for i in range(20))
                    except Exception:
                        close_real_valid = False
                    if close_real_valid:
                        _ingest_bump("filas_incremental_close_reales_validas", 1)
                    else:
                        _ingest_bump("filas_incremental_saneadas_close", 1)
                    return True

                except PermissionError:
                    time.sleep(base_sleep * (n + 1) + random.uniform(0, 0.07))
                    continue
                except Exception:
                    break

        _ingest_bump("filas_incremental_descartadas_total", 1)
        return False

    except Exception:
        try:
            globals().get("_INCREMENTAL_INGEST_STATS", {})["filas_incremental_descartadas_total"] = int(
                globals().get("_INCREMENTAL_INGEST_STATS", {}).get("filas_incremental_descartadas_total", 0) or 0
            ) + 1
        except Exception:
            pass
        return False
        
# === Canonización: aunque existan duplicados en el archivo, esta es la versión oficial ===
anexar_incremental_desde_bot = _anexar_incremental_desde_bot_CANON
       
# === FIN BLOQUE 6 ===

# === BLOQUE 7 — ORDEN DE REAL Y CONTROL DE TOKEN ===
# === ORDEN DE REAL (handshake maestro→bot) ===
ORDEN_DIR = "orden_real"

def _ensure_dir(p):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Falló creación de dir {p}: {e}")

def _atomic_write(path: str, text: str):
    _ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def path_orden(bot: str) -> str:
    _ensure_dir(ORDEN_DIR)
    return os.path.join(ORDEN_DIR, f"{bot}.json")


def leer_pause_state_maestro() -> dict:
    base = {
        "paused": False,
        "reason": "",
        "started_ts": 0.0,
        "resume_ts": 0.0,
        "duration_sec": 0,
        "source": "",
        "reference_balance": None,
        "trigger_balance": None,
    }
    path = MAESTRO_PAUSE_FILE
    if (not path) or (not os.path.exists(path)):
        return base
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read().strip()
        if not raw:
            return base
        data = json.loads(raw)
        if not isinstance(data, dict):
            return base
        out = dict(base)
        out.update(data)
        return out
    except Exception:
        return base


def tiempo_restante_pausa_maestro() -> int:
    try:
        if not maestro_pause_active:
            return 0
        return max(0, int(round(float(maestro_pause_resume_ts or 0.0) - time.time())))
    except Exception:
        return 0


def motivo_pausa_maestro() -> str:
    try:
        reason = str(maestro_pause_reason or "drawdown_20_monitor").strip()
        if reason == "drawdown_20_monitor":
            return "Protección por drawdown 20% activada desde monitor"
        if reason == "manual_resume":
            return "Pausa liberada manualmente desde monitor"
        return reason
    except Exception:
        return "Pausa externa"


def maestro_en_pausa() -> bool:
    try:
        return bool(maestro_pause_active)
    except Exception:
        return False


def _maybe_log_pause_state(force: bool = False):
    global maestro_pause_last_log_ts
    now = time.time()
    if (not force) and ((now - float(maestro_pause_last_log_ts or 0.0)) < 7.0):
        return
    maestro_pause_last_log_ts = now
    if maestro_pause_active:
        remain = tiempo_restante_pausa_maestro()
        mm, ss = divmod(max(0, int(remain)), 60)
        ref_txt = f"{float(maestro_pause_ref_balance):,.2f}" if isinstance(maestro_pause_ref_balance, (int, float)) else "--"
        trg_txt = f"{float(maestro_pause_trigger_balance):,.2f}" if isinstance(maestro_pause_trigger_balance, (int, float)) else "--"
        print(
            f"⛔ MAESTRO EN PAUSA | {mm:02d}:{ss:02d} | reason={maestro_pause_reason or 'drawdown_20_monitor'} "
            f"| ref={ref_txt} | trigger={trg_txt}"
        )


def actualizar_pause_state_maestro():
    global maestro_pause_active, maestro_pause_reason, maestro_pause_resume_ts, maestro_pause_started_ts
    global maestro_pause_last_read_ts, maestro_pause_ref_balance, maestro_pause_trigger_balance, maestro_pause_source
    global maestro_pause_last_state
    now = time.time()
    if (now - float(maestro_pause_last_read_ts or 0.0)) < 0.35:
        return
    maestro_pause_last_read_ts = now

    def _as_float(v):
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    data = leer_pause_state_maestro()
    paused_flag = bool(data.get("paused", False))
    resume_ts = float(data.get("resume_ts") or 0.0)
    started_ts = float(data.get("started_ts") or 0.0)
    reason = str(data.get("reason") or "")
    source = str(data.get("source") or "")
    ref_bal = _as_float(data.get("reference_balance"))
    trg_bal = _as_float(data.get("trigger_balance"))

    active_now = bool(paused_flag and resume_ts > now)
    just_changed = (active_now != bool(maestro_pause_last_state))

    if active_now:
        maestro_pause_active = True
        maestro_pause_resume_ts = resume_ts
        maestro_pause_started_ts = started_ts
        maestro_pause_reason = reason or "drawdown_20_monitor"
        maestro_pause_source = source
        maestro_pause_ref_balance = ref_bal
        maestro_pause_trigger_balance = trg_bal
        if just_changed:
            agregar_evento("⛔ MAESTRO EN PAUSA · Protección por drawdown 20% activada desde monitor.")
            _maybe_log_pause_state(force=True)
    else:
        maestro_pause_active = False
        maestro_pause_resume_ts = resume_ts
        maestro_pause_started_ts = started_ts
        if just_changed:
            if reason == "manual_resume":
                agregar_evento("✅ Pausa liberada manualmente desde monitor.")
                print("✅ Pausa liberada manualmente desde monitor")
            else:
                agregar_evento("✅ Pausa finalizada por tiempo.")
                print("✅ Pausa finalizada por tiempo")
        maestro_pause_reason = reason or ""
        maestro_pause_source = source
        maestro_pause_ref_balance = ref_bal
        maestro_pause_trigger_balance = trg_bal

    maestro_pause_last_state = bool(maestro_pause_active)

# === SALDO LIVE FEED (maestro -> monitor_saldo_pro) ===
SALDO_LIVE_FILE = "saldo_real_live.json"
SALDO_LIVE_HISTORY_FILE = "saldo_real_live_history.jsonl"
SALDO_SERIES_CSV_FILE = "saldo_real_series.csv"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SALDO_LIVE_PATH = os.path.join(os.path.expanduser("~"), SALDO_LIVE_FILE)
SALDO_LIVE_SHARED_PATH = os.path.abspath(
    os.path.expanduser(os.getenv("SALDO_LIVE_SHARED_PATH", _DEFAULT_SALDO_LIVE_PATH))
)
_DEFAULT_SALDO_HISTORY_PATH = os.path.join(os.path.dirname(SALDO_LIVE_SHARED_PATH), SALDO_LIVE_HISTORY_FILE)
SALDO_LIVE_HISTORY_SHARED_PATH = os.path.abspath(
    os.path.expanduser(os.getenv("SALDO_LIVE_HISTORY_SHARED_PATH", _DEFAULT_SALDO_HISTORY_PATH))
)
SALDO_SERIES_CSV_PATH = os.path.abspath(
    os.path.expanduser(os.getenv("SALDO_SERIES_CSV_PATH", os.path.join(SCRIPT_DIR, SALDO_SERIES_CSV_FILE)))
)

def _saldo_feed_targets() -> dict:
    return {
        "live": [SALDO_LIVE_SHARED_PATH],
        "history": [SALDO_LIVE_HISTORY_SHARED_PATH],
        "series": [SALDO_SERIES_CSV_PATH],
    }

def _append_line_safe(path: str, line: str):
    try:
        _ensure_dir(os.path.dirname(path) or ".")
    except Exception:
        pass
    tmp_lock = f"{os.path.basename(path)}.lock"
    with file_lock_required(tmp_lock, timeout=2.0, stale_after=20.0) as got:
        if not got:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

def _read_last_nonempty_line(path: str) -> str:
    try:
        if (not os.path.exists(path)) or os.path.getsize(path) <= 0:
            return ""
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            start = max(0, size - 8192)
            fh.seek(start, os.SEEK_SET)
            chunk = fh.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""

def _ensure_series_header_if_needed(path: str):
    try:
        _ensure_dir(os.path.dirname(path) or ".")
        if (not os.path.exists(path)) or os.path.getsize(path) <= 0:
            _append_line_safe(path, "timestamp,equity,source\n")
            return
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            first = (fh.readline() or "").strip().lower()
        if ("timestamp" not in first) or ("equity" not in first):
            bak = f"{path}.bak"
            try:
                os.replace(path, bak)
            except Exception:
                pass
            _append_line_safe(path, "timestamp,equity,source\n")
            if os.path.exists(bak):
                try:
                    with open(bak, "r", encoding="utf-8", errors="ignore") as bf:
                        for raw in bf:
                            row = raw.strip()
                            if not row:
                                continue
                            cols = [c.strip() for c in row.split(",")]
                            if len(cols) < 2 or cols[0].lower() in ("timestamp", "ts_utc"):
                                continue
                            _append_line_safe(path, f"{cols[0]},{cols[1]},{(cols[2] if len(cols) >= 3 else 'MAESTRO_5R6M')}\n")
                except Exception:
                    pass
    except Exception:
        pass

def _append_series_csv_if_new(path: str, ts_iso: str, val: float, source: str):
    try:
        _ensure_series_header_if_needed(path)
        line = f"{ts_iso},{float(val):.2f},{source}\n"
        last = _read_last_nonempty_line(path)
        if last and (last == line.strip()):
            return
        _append_line_safe(path, line)
    except Exception:
        pass

def _update_saldo_monitor_feed(valor_saldo: float):
    try:
        val = float(valor_saldo)
        now = float(time.time())
        ts_iso = datetime.now(timezone.utc).isoformat()
        payload_live = {
            "saldo_real": val,
            "equity": val,
            "balance": val,
            "timestamp": ts_iso,
            "ts": now,
            "source": "MAESTRO_5R6M",
        }
        payload_hist = {
            "timestamp": ts_iso,
            "equity": val,
            "saldo_real": val,
            "balance": val,
            "source": "MAESTRO_5R6M",
        }
        for p in dict.fromkeys(_saldo_feed_targets()["live"]):
            _atomic_write(p, json.dumps(payload_live, ensure_ascii=False))
        for p in dict.fromkeys(_saldo_feed_targets()["history"]):
            _append_line_safe(p, json.dumps(payload_hist, ensure_ascii=False) + "\n")
        for p in dict.fromkeys(_saldo_feed_targets()["series"]):
            _append_series_csv_if_new(p, ts_iso, val, "MAESTRO_5R6M")
        if bool(globals().get("HUD_SHOW_SALDO_DEBUG", False)):
            print(f"[SALDO LIVE] destino: {SALDO_LIVE_SHARED_PATH}")
            print(f"[SALDO HIST] destino: {SALDO_LIVE_HISTORY_SHARED_PATH}")
            print(f"[SALDO CSV] destino: {SALDO_SERIES_CSV_PATH}")
            print(f"[SALDO FEED][OK] saldo={val:.2f} ts={ts_iso}")
        return True
    except Exception as e:
        if bool(globals().get("HUD_SHOW_SALDO_DEBUG", False)):
            print(f"[SALDO FEED][ERROR] {e}")
        return False
# === /SALDO LIVE FEED ===

# === LXV_SYNC_COLUMN: sincronización de ronda/columna maestro↔bots ===
SYNC_ROUND_DIR = "sync_round"
SYNC_ROUND_STATE_PATH = os.path.join(SYNC_ROUND_DIR, "state.json")
TTL_ACK_SYNC_ROUND_S = 300.0
ACK_SYNC_ROUND_FUTURE_DRIFT_S = 20.0
LXV_SYNC_ROUND_FAILSAFE_ENABLE = True
LXV_SYNC_ROUND_MAX_WAIT_S = 240.0
LXV_SYNC_BOT_STALE_S = 120.0
LXV_SYNC_PENDING_MAX_WAIT_S = 240.0
LXV_SYNC_MIN_CLOSED_FOR_EVAL = 4
ACK_LIVE_HUD_ENABLE = True
ACK_LIVE_MAX_AGE_WARN_S = 10.0
ACK_LIVE_MAX_AGE_STALE_S = 120.0
ACK_LIVE_SHOW_MISSING = True
ACK_LIVE_COMPACT = True
HUD_MARTINGALA_LIVE_ENABLE = True
HUD_MARTINGALA_ALERT_ON_LOSS = True
HUD_MARTINGALA_SHOW_NEXT = True
HUD_MARTINGALA_SHOW_AMOUNT = True
HUD_MARTINGALA_ALERT_TTL_S = 90.0
ACK_TAPE_ENABLE = True
ACK_TAPE_WIDTH = 80
ACK_TAPE_MAX_SEEN = 2000
ACK_TAPE_USE_COLOR = True
ACK_TAPE_FILL_CHAR = "·"
ACK_LIVE_TAPE = {}
ACK_LIVE_TAPE_SEEN = deque(maxlen=2000)
_SYNC_ROUND_LAST_ANNOUNCED = None
_SYNC_ROUND_LAST_CLOSED_COUNT = {}
_LXV_LAST_EMITTED_ROUND = 0
LXV_REAL_EMITIDOS_POR_RONDA = set()
LXV_REAL_EMITIDOS_MAX_KEEP = 300
_SYNC_PENDING_WARN_TS = {}
_SYNC_STALE_WARN_TS = {}
_LXV_5V1X_EVENT_TS = {}
_LXV_HEADER_WARN_TS = {}
_LXV_HEADER_WARN_COOLDOWN_S = 180.0
_MATRIZ_SKIP_WARN_TS = 0.0
_MARTI_HUD_DEMO_IGNORED_TS = {}
_MATRIZ_STRICT_MODE_ANNOUNCED = False
_FOLLOWUP_5V1X_EVENT_TS = {}
_FOLLOWUP_5V1X_LAST_APPLIED = {}
_LXV_FASE_ZV_EVENT_TS = {}
_LXV_FASE_ZV_LAST_INFO = {"fase": "INSUFICIENTE", "allow_real": False, "g0": 0.0, "g1": 0.0, "g2": 0.0, "verdes0": 0, "verdes1": 0, "verdes2": 0, "streak_verde": 0, "motivo": "init"}
LXV_FASE_COLUMNS_CACHE = deque(maxlen=80)
LXV_ZONA_MIN_COLUMNS = 3
LXV_ZONA_GREEN_MIN = 4
LXV_ZONA_GREEN_STRONG = 5
LXV_ZONA_RED_STRONG = 4
LXV_ZONA_FULL_GREEN_MIN = 3
try:
    ZONA_SI_INVERTIR
except NameError:
    ZONA_SI_INVERTIR = "SI_INVERTIR"
try:
    ZONA_NO_INVERTIR
except NameError:
    ZONA_NO_INVERTIR = "NO_INVERTIR"
LXV_REAL_AUDIT = {
    "patrones_5v1x": 0,
    "patrones_4v2x": 0,
    "fase_ok": 0,
    "fase_bloq": 0,
    "real_emitidos": 0,
    "ultimo_bloqueo": "",
}
MATRIZ_COLUMNAS_LXV_CSV = "matriz_columnas_lxv.csv"
MATRIZ_CELDAS_LXV_CSV = "matriz_celdas_lxv.csv"
MATRIZ_FOLLOWUP_5V1X_CSV = "matriz_followup_5v1x.csv"
LEGACY_MATRIX_EXPORT_ENABLE = False
OFFICIAL_MATRIX_EXPORT_ENABLE = True
LXV_MATRIX_EXPORT_ENABLE = True
LXV_MATRIX_DIR = script_dir
LXV_MATRIX_EXPORT_LOCK = "lxv_matrix_export.lock"
LXV_MATRIX_EXPORT_LOG_EVERY_S = 10.0
_LXV_MATRIX_LAST_LOG_TS = 0.0
_LXV_MATRIX_HEADERS = {
    "matrix": [
        "round_id", "ts_round",
        "fulll47", "fulll50", "fulll45", "fulll48", "fulll49", "fulll46",
        "n_verdes", "n_rojos", "n_indef", "n_vacios",
        "ratio_verdes", "ratio_rojos",
        "patron_lxv", "bot_x1", "bot_x2", "bot_x_fuerte",
        "round_complete", "missing_bots", "data_quality",
        "source",
        "marti_bot", "marti_ciclo_actual", "marti_monto_actual", "marti_ultimo_resultado",
        "marti_ciclo_siguiente", "marti_monto_siguiente", "marti_estado", "marti_fuente",
    ],
    "long": [
        "round_id", "ts_round", "bot", "bot_order",
        "resultado_symbol", "resultado_texto", "result_bin",
        "activo", "direccion", "ciclo", "monto",
        "payout_total", "payout_multiplier",
        "token", "prob_ia", "modo_ia", "ia_gate_real",
        "trade_status", "epoch", "ts_trade",
        "ia_decision_id", "puntaje_estrategia",
        "marti_ciclo_bot", "marti_monto_bot",
        "round_complete", "missing_bots", "data_quality",
    ],
    "features": [
        "round_id", "ts_round", "secuencia_columna",
        "n_verdes", "n_rojos", "n_indef",
        "ratio_verdes", "ratio_rojos",
        "x_unica", "x_doble",
        "bots_rojos", "bots_verdes",
        "bot_x1", "bot_x2", "bot_x_fuerte",
        "avg_prob_verdes", "avg_prob_rojos", "max_prob_rojos", "min_prob_rojos", "std_prob_columna",
        "avg_score_verdes", "avg_score_rojos",
        "patron_lxv", "patron_simple", "patron_hash",
        "origin_marti_ciclo", "origin_marti_monto",
        "round_complete", "missing_bots", "data_quality",
    ],
    "followup": [
        "origin_round", "origin_ts_utc", "x_bot", "bot_objetivo", "regimen_fase", "origin_pattern",
        "resultado_origen", "ciclo_origen", "origin_marti_ciclo", "origin_marti_monto",
        "followup_c1", "followup_c2", "followup_c3", "followup_c4", "followup_c5",
        "future_sequence", "hit", "hit_step", "outcome_final", "resolved_round", "resolved_ts_utc",
    ],
}

def _lxv_matrix_paths() -> dict:
    base = os.path.abspath(os.path.expanduser(LXV_MATRIX_DIR or script_dir))
    return {
        # CSV oficial: una fila por columna cerrada
        "matrix": os.path.join(base, MATRIZ_COLUMNAS_LXV_CSV),
        # CSV oficial: una fila por bot por columna
        "long": os.path.join(base, MATRIZ_CELDAS_LXV_CSV),
        # CSV oficial: seguimiento 5V1X
        "followup": os.path.join(base, MATRIZ_FOLLOWUP_5V1X_CSV),
        # legacy desactivado / reemplazado por CSV oficial
        "features": os.path.join(base, "features_columnas_lxv.csv"),
        # legacy desactivado / reemplazado por CSV oficial
        "xlsx": os.path.join(base, "matriz_lxv.xlsx"),
    }

def _lxv_result_to_symbol(resultado: str | None) -> str:
    r = normalizar_resultado(resultado)
    if r == "GANANCIA":
        return "✓"
    if r == "PÉRDIDA":
        return "X"
    if r == "INDEFINIDO":
        return "·"
    return "-"

def _lxv_safe_float(v, default=None):
    try:
        if v is None or str(v).strip() == "":
            return default
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default

def _lxv_safe_int(v, default=None):
    try:
        if v is None or str(v).strip() == "":
            return default
        return int(float(v))
    except Exception:
        return default

def _lxv_csv_read_rows(path: str, max_lines: int = 1800) -> list[dict]:
    return _tail_rows_dict(path, max_lines=max_lines)

def _lxv_read_existing_header(path: str) -> list[str]:
    try:
        if (not os.path.exists(path)) or os.path.getsize(path) <= 0:
            return []
        with open(path, "r", encoding="utf-8", newline="") as f:
            row = next(csv.reader(f), [])
        return [str(x).strip() for x in list(row or []) if str(x).strip()]
    except Exception:
        return []

def _followup_5v1x_is_origin_valid(row_or_pack):
    d = dict(row_or_pack or {})
    pattern = str(d.get("pattern", d.get("patron_lxv", d.get("origin_pattern", ""))) or "").upper()
    try:
        v_count = int(d.get("v_count", d.get("n_verdes", 0)) or 0)
        x_count = int(d.get("x_count", d.get("n_rojos", 0)) or 0)
    except Exception:
        return False
    complete = bool(d.get("round_complete", d.get("complete", False)))
    quality = str(d.get("data_quality", d.get("quality", "")) or "").lower()
    x_bot = str(d.get("x_bot", d.get("bot_objetivo", "")) or "")
    return bool(pattern == "5V1X" and v_count == 5 and x_count == 1 and complete and quality == "ok" and x_bot in BOT_NAMES)

def _followup_5v1x_load_rows():
    path = _lxv_matrix_paths().get("followup")
    official = list(_LXV_MATRIX_HEADERS.get("followup", []))
    if not path or (not os.path.exists(path)):
        return [], official
    rows = []
    fieldnames = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for r in reader:
                try:
                    rr = dict(r or {})
                    if (not str(rr.get("origin_round", "")).strip()) or (not str(rr.get("x_bot", rr.get("bot_objetivo", "")) or "").strip()):
                        continue
                    rows.append(rr)
                except Exception:
                    continue
    except Exception:
        return [], official
    return rows, (fieldnames if fieldnames else official)

def _followup_5v1x_write_rows_atomic(rows, fieldnames):
    path = _lxv_matrix_paths().get("followup")
    if not path:
        return False
    try:
        _ensure_dir(os.path.dirname(path) or ".")
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames or []), extrasaction="ignore")
            w.writeheader()
            for r in list(rows or []):
                payload = {k: (r.get(k, "") if isinstance(r, dict) else "") for k in list(fieldnames or [])}
                w.writerow(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception as e:
        agregar_evento(f"⚠ followup write error: {str(e)[:70]}")
        return False

def _followup_5v1x_update_pending_with_column(column_pack):
    global _FOLLOWUP_5V1X_EVENT_TS, _FOLLOWUP_5V1X_LAST_APPLIED
    d = dict(column_pack or {})
    complete = bool(d.get("complete", d.get("round_complete", False)))
    quality = str(d.get("quality", d.get("data_quality", "")) or "").lower()
    results = dict(d.get("results", {}) or {})
    if (not complete) or (quality != "ok") or (len(results) != len(BOT_NAMES)) or any(str(results.get(b, "")) not in ("✓", "X") for b in BOT_NAMES):
        return
    current_round = int(d.get("round_id", 0) or 0)
    if current_round <= 0:
        return
    rows, fieldnames = _followup_5v1x_load_rows()
    fset = set(fieldnames or [])
    needed = {"followup_c1", "followup_c2", "followup_c3", "followup_c4", "followup_c5", "outcome_final"}
    if not needed.issubset(fset):
        now = time.time()
        last = float(_FOLLOWUP_5V1X_EVENT_TS.get("old_header", 0.0) or 0.0)
        if (now - last) >= 180.0:
            _FOLLOWUP_5V1X_EVENT_TS["old_header"] = now
            agregar_evento("⚠ followup header antiguo: no se puede completar C1..C5 hasta crear archivo nuevo")
        return

    changed = False
    for r in rows:
        try:
            x_bot = str(r.get("x_bot", r.get("bot_objetivo", "")) or "")
            origin_round = int(float(r.get("origin_round", 0) or 0))
        except Exception:
            continue
        if x_bot not in BOT_NAMES or origin_round <= 0 or current_round <= origin_round:
            continue
        if r.get("resolved_round", "") not in ("", None, "--"):
            continue
        if not _followup_5v1x_is_origin_valid(r):
            continue
        key = f"{origin_round}:{x_bot}"
        if int(_FOLLOWUP_5V1X_LAST_APPLIED.get(key, 0) or 0) >= current_round:
            continue
        outcome = str(r.get("outcome_final", "") or "").upper().strip()
        if outcome not in ("", "PENDING", "--"):
            continue
        if x_bot not in results:
            continue
        res = str(results.get(x_bot, "") or "")
        if res not in ("✓", "X"):
            continue
        slot = None
        for i in range(1, 6):
            c = f"followup_c{i}"
            if str(r.get(c, "") or "").strip() == "":
                slot = i
                break
        if slot is None:
            continue
        r[f"followup_c{slot}"] = res
        seq_vals = [str(r.get(f"followup_c{i}", "") or "").strip() for i in range(1, 6) if str(r.get(f"followup_c{i}", "") or "").strip() in ("✓", "X")]
        r["future_sequence"] = ",".join(seq_vals)
        r["outcome_final"] = "PENDING"
        if res == "✓":
            r["hit"] = "1"
            r["hit_step"] = str(slot)
            r["outcome_final"] = f"HIT_C{slot}"
            r["resolved_round"] = str(current_round)
            r["resolved_ts_utc"] = datetime.now(timezone.utc).isoformat()
        elif slot == 5:
            r["hit"] = "0"
            r["hit_step"] = ""
            r["outcome_final"] = "MISS_C5"
            r["resolved_round"] = str(current_round)
            r["resolved_ts_utc"] = datetime.now(timezone.utc).isoformat()
        _FOLLOWUP_5V1X_LAST_APPLIED[key] = int(current_round)
        changed = True

        now = time.time()
        ev_key = f"upd:{origin_round}:{x_bot}:{slot}"
        if (now - float(_FOLLOWUP_5V1X_EVENT_TS.get(ev_key, 0.0) or 0.0)) >= 15.0:
            _FOLLOWUP_5V1X_EVENT_TS[ev_key] = now
            agregar_evento(f"🧾 followup 5V1X update: origin={origin_round} {x_bot} C{slot}={res}")
        if str(r.get("outcome_final", "")).startswith("HIT_C"):
            ev2 = f"hit:{origin_round}:{x_bot}"
            if (now - float(_FOLLOWUP_5V1X_EVENT_TS.get(ev2, 0.0) or 0.0)) >= 30.0:
                _FOLLOWUP_5V1X_EVENT_TS[ev2] = now
                agregar_evento(f"✅ followup 5V1X {r.get('outcome_final')}: origin={origin_round} {x_bot}")
        elif str(r.get("outcome_final", "")) == "MISS_C5":
            ev2 = f"miss:{origin_round}:{x_bot}"
            if (now - float(_FOLLOWUP_5V1X_EVENT_TS.get(ev2, 0.0) or 0.0)) >= 30.0:
                _FOLLOWUP_5V1X_EVENT_TS[ev2] = now
                agregar_evento(f"❌ followup 5V1X MISS_C5: origin={origin_round} {x_bot}")

    if changed:
        _followup_5v1x_write_rows_atomic(rows, fieldnames)

def _followup_5v1x_create_origin_if_needed(column_pack):
    global _FOLLOWUP_5V1X_EVENT_TS
    d = dict(column_pack or {})
    if not _followup_5v1x_is_origin_valid(d):
        return
    round_id = int(d.get("round_id", 0) or 0)
    x_bot = str(d.get("x_bot", d.get("bot_objetivo", "")) or "")
    if round_id <= 0 or x_bot not in BOT_NAMES:
        return
    rows, fieldnames = _followup_5v1x_load_rows()
    exists = any(
        (str(r.get("origin_round", "")).strip() == str(round_id) and str(r.get("x_bot", r.get("bot_objetivo", "")) or "").strip() == x_bot)
        for r in rows
    )
    if exists:
        return
    base = {k: "" for k in list(fieldnames or _LXV_MATRIX_HEADERS.get("followup", []))}
    base["origin_round"] = str(round_id)
    base["origin_ts_utc"] = str(d.get("ts_utc", "") or datetime.now(timezone.utc).isoformat())
    base["x_bot"] = x_bot
    base["bot_objetivo"] = x_bot
    base["regimen_fase"] = str(d.get("regimen_fase", "") or "")
    base["origin_pattern"] = "5V1X"
    base["resultado_origen"] = "X"
    ciclo_map = dict(d.get("ciclo_by_bot", {}) or {})
    base["ciclo_origen"] = str(ciclo_map.get(x_bot, d.get("ciclo_origen", "")) or "")
    base["origin_marti_ciclo"] = str(d.get("origin_marti_ciclo", d.get("marti_ciclo_actual", "")) or "")
    base["origin_marti_monto"] = str(d.get("origin_marti_monto", d.get("marti_monto_actual", "")) or "")
    for i in range(1, 6):
        c = f"followup_c{i}"
        if c in base:
            base[c] = ""
    if "future_sequence" in base:
        base["future_sequence"] = ""
    if "hit" in base:
        base["hit"] = ""
    if "hit_step" in base:
        base["hit_step"] = ""
    if "outcome_final" in base:
        base["outcome_final"] = "PENDING"
    if "resolved_round" in base:
        base["resolved_round"] = ""
    if "resolved_ts_utc" in base:
        base["resolved_ts_utc"] = ""
    rows.append(base)
    if _followup_5v1x_write_rows_atomic(rows, fieldnames):
        now = time.time()
        k = f"new:{round_id}:{x_bot}"
        if (now - float(_FOLLOWUP_5V1X_EVENT_TS.get(k, 0.0) or 0.0)) >= 10.0:
            _FOLLOWUP_5V1X_EVENT_TS[k] = now
            agregar_evento(f"🧾 followup 5V1X creado: round={round_id} x_bot={x_bot}")

def _lxv_get_last_closed_row_for_bot(bot: str, ack_close: dict, round_id: int) -> dict | None:
    ruta = f"registro_enriquecido_{bot}.csv"
    rows = _lxv_csv_read_rows(ruta, max_lines=2500)
    if not rows:
        return None
    ack_res = normalizar_resultado((ack_close or {}).get("resultado"))
    ack_ts = _lxv_safe_float((ack_close or {}).get("ts"), default=0.0) or 0.0
    cands = []
    for r in rows:
        res = normalizar_resultado(r.get("resultado"))
        if res not in ("GANANCIA", "PÉRDIDA"):
            continue
        if ack_res in ("GANANCIA", "PÉRDIDA") and res != ack_res:
            continue
        ts_norm = normalizar_trade_status(r.get("trade_status_norm", None) or r.get("trade_status", None))
        if ts_norm and ts_norm != "CERRADO":
            continue
        ts_trade = _lxv_safe_float(r.get("ts"), default=0.0) or 0.0
        epoch = _lxv_safe_int(r.get("epoch"), default=0) or 0
        cycle = _lxv_safe_int(r.get("ciclo_martingala"), default=_lxv_safe_int(r.get("ciclo"), default=0) or 0) or 0
        cands.append({
            "_row": r,
            "_ts_trade": ts_trade,
            "_epoch": epoch,
            "_cycle": cycle,
            "_dt_ack": abs(ts_trade - ack_ts) if (ack_ts > 0 and ts_trade > 0) else 999999.0,
        })
    if not cands:
        return None
    cands.sort(key=lambda x: (x["_dt_ack"], -(x["_epoch"]), -(x["_ts_trade"])))
    if len(cands) >= 2:
        a, b = cands[0], cands[1]
        if abs(float(a["_dt_ack"]) - float(b["_dt_ack"])) <= 0.05 and a["_epoch"] == b["_epoch"]:
            return None
    if ack_ts > 0 and cands[0]["_dt_ack"] > 1200:
        return None
    out = dict(cands[0]["_row"])
    out["__round_id"] = int(round_id)
    out["__ack_resultado"] = ack_res
    out["__ack_ts"] = ack_ts
    out["__bot"] = bot
    return out

def _lxv_round_already_exported(round_id: int, bot: str | None = None) -> bool:
    p = _lxv_matrix_paths()
    if bot is None:
        path = p["matrix"]
        if not os.path.exists(path):
            return False
        rows = _lxv_csv_read_rows(path, max_lines=3000)
        rid = str(int(round_id))
        return any(str(r.get("round_id", "")).strip() == rid for r in rows)
    path = p["long"]
    if not os.path.exists(path):
        return False
    rows = _lxv_csv_read_rows(path, max_lines=6000)
    rid = str(int(round_id))
    for r in rows:
        if str(r.get("round_id", "")).strip() == rid and str(r.get("bot", "")).strip() == str(bot):
            return True
    return False

def _lxv_append_rows_csv(path: str, rows: list[dict], headers: list[str], unique_keys: list[str]) -> int:
    global _LXV_HEADER_WARN_TS
    if not rows:
        return 0
    _ensure_dir(os.path.dirname(path) or ".")
    wrote = 0
    active_headers = list(headers or [])
    existing_header = _lxv_read_existing_header(path)
    if existing_header:
        active_headers = list(existing_header)
        missing = [h for h in list(headers or []) if h not in set(existing_header)]
        missing_marti = [h for h in missing if h.startswith("marti_") or h.startswith("origin_marti_")]
        if missing_marti:
            now = time.time()
            ts_last = float(_LXV_HEADER_WARN_TS.get(path, 0.0) or 0.0)
            if (now - ts_last) >= float(_LXV_HEADER_WARN_COOLDOWN_S):
                _LXV_HEADER_WARN_TS[path] = now
                agregar_evento(f"⚠ matriz header antiguo: columnas nuevas omitidas en {os.path.basename(path)}")
    with file_lock_required(LXV_MATRIX_EXPORT_LOCK, timeout=3.0, stale_after=30.0) as got:
        if not got:
            return 0
        existing = set()
        if os.path.exists(path):
            old = _lxv_csv_read_rows(path, max_lines=20000)
            for r in old:
                k = tuple(str(r.get(c, "")).strip() for c in unique_keys)
                existing.add(k)
        need_header = (not os.path.exists(path)) or os.path.getsize(path) <= 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=active_headers, extrasaction="ignore")
            if need_header:
                w.writeheader()
            for row in rows:
                k = tuple(str(row.get(c, "")).strip() for c in unique_keys)
                if k in existing:
                    continue
                payload = {h: row.get(h, "") for h in active_headers}
                w.writerow(payload)
                existing.add(k)
                wrote += 1
            f.flush()
            os.fsync(f.fileno())
    return wrote

def _lxv_build_round_row(round_id: int, ts_round: float, rows_long: list[dict], missing_bots: list[str], round_complete: bool, data_quality: str, marti_snapshot: dict | None = None) -> dict:
    by_bot = {str(r.get("bot")): r for r in rows_long}
    symbols = []
    n_verdes = n_rojos = n_indef = n_vacios = 0
    rojos = []
    for b in BOT_NAMES:
        sym = str((by_bot.get(b, {}) or {}).get("resultado_symbol", "-") or "-")
        if sym not in ("✓", "X", "·", "-"):
            sym = "-"
        symbols.append(sym)
        if sym == "✓":
            n_verdes += 1
        elif sym == "X":
            n_rojos += 1
            rojos.append(b)
        elif sym == "·":
            n_indef += 1
        else:
            n_vacios += 1
    bot_x1 = rojos[0] if len(rojos) >= 1 else ""
    bot_x2 = rojos[1] if len(rojos) >= 2 else ""
    if n_verdes == 5 and n_rojos == 1:
        patron = "5V1X"
    elif n_verdes == 4 and n_rojos == 2:
        patron = "4V2X"
    else:
        patron = "OTHER"
    marti = dict(marti_snapshot or {})
    row = {
        "round_id": int(round_id),
        "ts_round": float(ts_round),
        "n_verdes": int(n_verdes),
        "n_rojos": int(n_rojos),
        "n_indef": int(n_indef),
        "n_vacios": int(n_vacios),
        "ratio_verdes": round(float(n_verdes) / float(len(BOT_NAMES)), 6),
        "ratio_rojos": round(float(n_rojos) / float(len(BOT_NAMES)), 6),
        "patron_lxv": patron,
        "bot_x1": bot_x1,
        "bot_x2": bot_x2,
        "bot_x_fuerte": "",
        "round_complete": bool(round_complete),
        "missing_bots": "|".join([b for b in BOT_NAMES if b in set(missing_bots or [])]),
        "data_quality": str(data_quality),
        "source": "sync_round",
        "marti_bot": str(marti.get("bot", "") or ""),
        "marti_ciclo_actual": marti.get("ciclo_actual", ""),
        "marti_monto_actual": marti.get("monto_actual", ""),
        "marti_ultimo_resultado": str(marti.get("ultimo_resultado", "") or ""),
        "marti_ciclo_siguiente": marti.get("ciclo_siguiente", ""),
        "marti_monto_siguiente": marti.get("monto_siguiente", ""),
        "marti_estado": str(marti.get("estado", "") or ""),
        "marti_fuente": str(marti.get("fuente", "") or ""),
    }
    for idx, b in enumerate(BOT_NAMES):
        row[b] = symbols[idx]
    return row

def _lxv_pick_bot_x_fuerte(rojos_rows: list[dict]) -> str:
    if len(rojos_rows) == 1:
        return str(rojos_rows[0].get("bot", ""))
    if len(rojos_rows) < 2:
        return ""
    stats_wr = {}
    for b in BOT_NAMES:
        stats_wr[b] = _lxv_safe_float(estado_bots.get(b, {}).get("porcentaje_exito"), default=-1e9)
    def rank(r):
        b = str(r.get("bot", ""))
        order = BOT_NAMES.index(b) if b in BOT_NAMES else 999
        return (
            _lxv_safe_float(r.get("prob_ia"), default=-1e9),
            _lxv_safe_float(r.get("puntaje_estrategia"), default=-1e9),
            _lxv_safe_float(r.get("payout_total"), default=-1e9),
            _lxv_safe_float(stats_wr.get(b), default=-1e9),
            -float(order),
        )
    ranked = sorted(rojos_rows, key=rank, reverse=True)
    return str(ranked[0].get("bot", "")) if ranked else ""

def _lxv_build_features_row(round_row: dict, rows_long: list[dict], marti_snapshot: dict | None = None) -> dict:
    vals_prob = []
    verdes_prob = []
    rojos_prob = []
    verdes_score = []
    rojos_score = []
    bots_rojos = []
    bots_verdes = []
    secuencia = []
    rojos_rows = []
    by_bot = {str(r.get("bot")): r for r in rows_long}
    for b in BOT_NAMES:
        r = by_bot.get(b, {})
        sym = str(r.get("resultado_symbol", "-") or "-")
        secuencia.append(sym)
        p = _lxv_safe_float(r.get("prob_ia"), default=None)
        s = _lxv_safe_float(r.get("puntaje_estrategia"), default=None)
        if p is not None:
            vals_prob.append(p)
        if sym == "✓":
            bots_verdes.append(b)
            if p is not None:
                verdes_prob.append(p)
            if s is not None:
                verdes_score.append(s)
        elif sym == "X":
            bots_rojos.append(b)
            rojos_rows.append(r)
            if p is not None:
                rojos_prob.append(p)
            if s is not None:
                rojos_score.append(s)
    bot_x_fuerte = _lxv_pick_bot_x_fuerte(rojos_rows)
    patron_simple = "".join(secuencia)
    marti = dict(marti_snapshot or {})
    out = {
        "round_id": round_row.get("round_id"),
        "ts_round": round_row.get("ts_round"),
        "secuencia_columna": "|".join(secuencia),
        "n_verdes": round_row.get("n_verdes", 0),
        "n_rojos": round_row.get("n_rojos", 0),
        "n_indef": round_row.get("n_indef", 0),
        "ratio_verdes": round_row.get("ratio_verdes", 0.0),
        "ratio_rojos": round_row.get("ratio_rojos", 0.0),
        "x_unica": bool(int(round_row.get("n_rojos", 0) or 0) == 1),
        "x_doble": bool(int(round_row.get("n_rojos", 0) or 0) == 2),
        "bots_rojos": "|".join(bots_rojos),
        "bots_verdes": "|".join(bots_verdes),
        "bot_x1": round_row.get("bot_x1", ""),
        "bot_x2": round_row.get("bot_x2", ""),
        "bot_x_fuerte": bot_x_fuerte,
        "avg_prob_verdes": float(np.mean(verdes_prob)) if verdes_prob else "",
        "avg_prob_rojos": float(np.mean(rojos_prob)) if rojos_prob else "",
        "max_prob_rojos": float(np.max(rojos_prob)) if rojos_prob else "",
        "min_prob_rojos": float(np.min(rojos_prob)) if rojos_prob else "",
        "std_prob_columna": float(np.std(vals_prob)) if vals_prob else "",
        "avg_score_verdes": float(np.mean(verdes_score)) if verdes_score else "",
        "avg_score_rojos": float(np.mean(rojos_score)) if rojos_score else "",
        "patron_lxv": round_row.get("patron_lxv", "OTHER"),
        "patron_simple": patron_simple,
        "patron_hash": hashlib.md5(patron_simple.encode("utf-8")).hexdigest()[:16],
        "origin_marti_ciclo": marti.get("ciclo_actual", ""),
        "origin_marti_monto": marti.get("monto_actual", ""),
        "round_complete": round_row.get("round_complete", False),
        "missing_bots": round_row.get("missing_bots", ""),
        "data_quality": round_row.get("data_quality", "partial"),
    }
    return out

def _lxv_export_excel_optional(paths: dict) -> None:
    # legacy desactivado / reemplazado por CSV oficial
    return

def _lxv_export_round_snapshot(round_id: int, ts_round: float, closed: dict, expected: list[str], stale_ignored: list[str], released_reason: str) -> None:
    global _LXV_MATRIX_LAST_LOG_TS, _MATRIZ_SKIP_WARN_TS, _MATRIZ_STRICT_MODE_ANNOUNCED
    if not bool(LXV_MATRIX_EXPORT_ENABLE):
        return
    if not bool(OFFICIAL_MATRIX_EXPORT_ENABLE):
        return
    if not bool(_MATRIZ_STRICT_MODE_ANNOUNCED):
        _MATRIZ_STRICT_MODE_ANNOUNCED = True
        agregar_evento("🧾 matrices: guardado estricto activado, solo columnas completas OK")
    if _lxv_round_already_exported(round_id):
        return
    expected = [b for b in list(expected or []) if b in BOT_NAMES]
    if not expected:
        expected = list(BOT_NAMES)
    missing = [b for b in expected if b not in closed]
    round_complete = len(missing) == 0
    data_quality = "ok" if round_complete else "partial"
    marti_snapshot = _marti_hud_snapshot() if "_marti_hud_snapshot" in globals() else {}
    rows_long = []
    for idx, bot in enumerate(BOT_NAMES, start=1):
        ack = closed.get(bot, {}) if isinstance(closed.get(bot, {}), dict) else {}
        csv_row = _lxv_get_last_closed_row_for_bot(bot, ack, round_id) if ack else None
        rtxt = normalizar_resultado(ack.get("resultado")) if ack else "INDEFINIDO"
        rsym = _lxv_result_to_symbol(rtxt) if ack else "-"
        if bot in missing:
            rsym = "-"
            rtxt = ""
        row = {
            "round_id": int(round_id),
            "ts_round": float(ts_round),
            "bot": bot,
            "bot_order": int(idx),
            "resultado_symbol": rsym,
            "resultado_texto": rtxt if ack else "",
            "result_bin": 1 if rsym == "✓" else (0 if rsym == "X" else ""),
            "activo": (csv_row or {}).get("activo", ""),
            "direccion": (csv_row or {}).get("direccion", (csv_row or {}).get("direction", "")),
            "ciclo": (csv_row or {}).get("ciclo_martingala", (ack or {}).get("ciclo", "")),
            "monto": (csv_row or {}).get("monto", ""),
            "payout_total": (csv_row or {}).get("payout_total", ""),
            "payout_multiplier": (csv_row or {}).get("payout_multiplier", ""),
            "token": (csv_row or {}).get("token", estado_bots.get(bot, {}).get("token", "")),
            "prob_ia": (csv_row or {}).get("ia_prob_en_juego", estado_bots.get(bot, {}).get("prob_ia", "")),
            "modo_ia": (csv_row or {}).get("ia_modo_ack", estado_bots.get(bot, {}).get("modo_ia", "")),
            "ia_gate_real": (csv_row or {}).get("ia_gate_real", ""),
            "trade_status": (csv_row or {}).get("trade_status", "CERRADO" if ack else ""),
            "epoch": (csv_row or {}).get("epoch", ""),
            "ts_trade": (csv_row or {}).get("ts", (ack or {}).get("ts", "")),
            "ia_decision_id": (csv_row or {}).get("ia_decision_id", estado_bots.get(bot, {}).get("ia_decision_id", "")),
            "puntaje_estrategia": (csv_row or {}).get("puntaje_estrategia", ""),
            "marti_ciclo_bot": marti_snapshot.get("ciclo_actual", ""),
            "marti_monto_bot": marti_snapshot.get("monto_actual", ""),
            "round_complete": bool(round_complete),
            "missing_bots": "|".join(missing),
            "data_quality": data_quality,
        }
        rows_long.append(row)
    # Guardado estricto: solo columnas cerradas completas OK
    bots_ok = len(rows_long) == len(BOT_NAMES)
    res_ok = all(str(r.get("resultado_symbol", "")) in ("✓", "X") for r in rows_long)
    no_missing = len(list(missing or [])) == 0
    strict_ok = bool(round_complete and str(data_quality).lower() == "ok" and bots_ok and res_ok and no_missing)
    if not strict_ok:
        now = time.time()
        if (now - float(_MATRIZ_SKIP_WARN_TS or 0.0)) >= 90.0:
            _MATRIZ_SKIP_WARN_TS = now
            agregar_evento(f"🧾 matriz skip: columna parcial o incompleta round={int(round_id)}")
        return

    if len(rows_long) != len(BOT_NAMES):
        now = time.time()
        if (now - float(_MATRIZ_SKIP_WARN_TS or 0.0)) >= 90.0:
            _MATRIZ_SKIP_WARN_TS = now
            agregar_evento(f"🧾 matriz skip: celdas inválidas round={int(round_id)} filas={len(rows_long)}")
        return

    rows_long = sorted(rows_long, key=lambda r: int(r.get("bot_order", 999)))
    round_row = _lxv_build_round_row(round_id, ts_round, rows_long, missing, round_complete, data_quality, marti_snapshot=marti_snapshot)
    feat_row = _lxv_build_features_row(round_row, rows_long, marti_snapshot=marti_snapshot)
    if str(data_quality) != "ok":
        feat_row["avg_prob_verdes"] = ""
        feat_row["avg_prob_rojos"] = ""
        feat_row["max_prob_rojos"] = ""
        feat_row["min_prob_rojos"] = ""
        feat_row["std_prob_columna"] = ""
        feat_row["avg_score_verdes"] = ""
        feat_row["avg_score_rojos"] = ""
    round_row["bot_x_fuerte"] = feat_row.get("bot_x_fuerte", "")
    v_count = int(round_row.get("n_verdes", 0) or 0)
    x_count = int(round_row.get("n_rojos", 0) or 0)
    pattern = str(round_row.get("patron_lxv", "") or "")
    is_5v1x = int(v_count == 5 and x_count == 1 and pattern == "5V1X")
    bot_obj = str(feat_row.get("bot_x_fuerte", "") or round_row.get("bot_x1", "") or "")
    row_obj = next((r for r in rows_long if str(r.get("bot", "")) == bot_obj), {}) if bot_obj else {}
    results_map = {str(r.get("bot", "")): str(r.get("resultado_symbol", "") or "") for r in rows_long}
    ciclo_map = {str(r.get("bot", "")): r.get("ciclo", "") for r in rows_long}
    column_pack = {
        "round_id": int(round_id),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "complete": bool(round_complete),
        "round_complete": bool(round_complete),
        "quality": str(data_quality),
        "data_quality": str(data_quality),
        "results": results_map,
        "pattern": pattern,
        "patron_lxv": pattern,
        "v_count": int(v_count),
        "x_count": int(x_count),
        "n_verdes": int(v_count),
        "n_rojos": int(x_count),
        "x_bot": bot_obj,
        "bot_objetivo": bot_obj,
        "resultado_origen": str(row_obj.get("resultado_symbol", "") or ""),
        "ciclo_origen": row_obj.get("ciclo", ""),
        "ciclo_by_bot": ciclo_map,
        "origin_marti_ciclo": marti_snapshot.get("ciclo_actual", ""),
        "origin_marti_monto": marti_snapshot.get("monto_actual", ""),
        "regimen_fase": "",
        "is_5v1x": int(is_5v1x),
    }
    paths = _lxv_matrix_paths()
    wrote_matrix = _lxv_append_rows_csv(paths["matrix"], [round_row], _LXV_MATRIX_HEADERS["matrix"], ["round_id"])
    _lxv_append_rows_csv(paths["long"], rows_long, _LXV_MATRIX_HEADERS["long"], ["round_id", "bot"])
    _followup_5v1x_update_pending_with_column(column_pack)
    _followup_5v1x_create_origin_if_needed(column_pack)
    _lxv_export_excel_optional(paths)
    now_ts = time.time()
    if wrote_matrix > 0 and (now_ts - float(_LXV_MATRIX_LAST_LOG_TS or 0.0)) >= float(LXV_MATRIX_EXPORT_LOG_EVERY_S):
        _LXV_MATRIX_LAST_LOG_TS = now_ts
        agregar_evento(f"📊 Export ronda LXV #{int(round_id)} -> columnas/celdas/followup OK ({data_quality}; reason={released_reason}).")

def _sync_round_ack_path(bot: str) -> str:
    _ensure_dir(SYNC_ROUND_DIR)
    return os.path.join(SYNC_ROUND_DIR, f"{bot}.json")

def _sync_round_safe_read_json(path: str):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def _ack_live_symbol(resultado):
    if resultado is None:
        return "-"
    try:
        txt = str(resultado).strip()
    except Exception:
        return "·"
    if txt == "":
        return "·"
    up = txt.upper()
    up = up.replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
    if up in ("GANANCIA", "WIN", "CHECK", "✓"):
        return "✓"
    if up in ("PÉRDIDA", "PERDIDA", "LOSS", "X", "✗"):
        return "X"
    if up in ("INDEFINIDO", "PENDING", "", "NONE", "NULL", "-"):
        return "·"
    return "·"


def _ack_live_fmt_age(age_s):
    try:
        if age_s is None:
            return "--"
        age = float(age_s)
        if not math.isfinite(age):
            return "--"
        if age < 0:
            return "FUTURE"
        if age < 10.0:
            return f"{age:.1f}s"
        if age < 60.0:
            return f"{int(age)}s"
        return f"{(age / 60.0):.1f}m"
    except Exception:
        return "--"


def _ack_live_snapshot():
    out = {
        "ok": True,
        "msg": "",
        "rows": [],
        "round_id_actual": 1,
        "released_round": 1,
        "closed_bots_state": {},
        "bots_missing_state": [],
        "status_state": "",
        "reason_state": "",
        "expected_count": 0,
        "closed_count": 0,
        "missing_bots": [],
        "max_lag_s": None,
        "avg_lag_s": None,
        "stale_bots": [],
        "warn_bots": [],
    }
    try:
        bots = list(BOT_NAMES)
    except Exception:
        out["ok"] = False
        out["msg"] = "📡 ACK LIVE: BOT_NAMES no disponible"
        return out
    if not os.path.exists(SYNC_ROUND_STATE_PATH):
        out["ok"] = False
        out["msg"] = "📡 ACK LIVE: esperando state.json"
        return out

    state = _sync_round_safe_read_json(SYNC_ROUND_STATE_PATH) or {}
    round_id_actual = int(state.get("round_id", state.get("released_round", 1)) or 1)
    released_round = int(state.get("released_round", round_id_actual) or round_id_actual)
    closed_bots_state = state.get("closed_bots", {})
    bots_missing_state = state.get("bots_missing", [])
    status_state = state.get("status", "")
    reason_state = state.get("reason", "")

    out["round_id_actual"] = round_id_actual
    out["released_round"] = released_round
    out["closed_bots_state"] = closed_bots_state
    out["bots_missing_state"] = bots_missing_state
    out["status_state"] = status_state
    out["reason_state"] = reason_state

    lag_samples = []
    closed_count = 0
    missing_bots = []
    stale_bots = []
    warn_bots = []
    rows = []
    now = time.time()

    for bot in bots:
        ack_path = _sync_round_ack_path(bot)
        ack = _sync_round_safe_read_json(ack_path)
        raw_exists = os.path.exists(ack_path)
        ack_err = False
        if raw_exists and ack is None:
            try:
                with open(ack_path, "r", encoding="utf-8") as f:
                    json.load(f)
            except Exception:
                ack_err = True
        if not isinstance(ack, dict):
            ack = {}

        ack_round_id = int(ack.get("round_id", 0) or 0)
        resultado = ack.get("resultado", "")
        ts = ack.get("ts")
        age_s = None
        future = False
        if ts is not None:
            try:
                tsf = float(ts)
                age_s = now - tsf
                if age_s < 0:
                    future = True
            except Exception:
                age_s = None
        asset = ack.get("asset", "--")
        ciclo = ack.get("ciclo", "--")
        ack_status = ack.get("status", "")
        sync_wait = ack.get("sync_wait", None)
        waiting_release_round = ack.get("waiting_release_round", None)
        last_seen_ts = ack.get("last_seen_ts", None)

        stale = bool(age_s is not None and age_s > float(ACK_LIVE_MAX_AGE_STALE_S))
        warn = bool(age_s is not None and age_s > float(ACK_LIVE_MAX_AGE_WARN_S))

        symbol = "-"
        row_status = "missing"
        valid_closed = False

        if ack_err:
            symbol = "ERR"
            row_status = "error"
            missing_bots.append(bot)
        elif not ack:
            symbol = "-"
            row_status = "missing"
            missing_bots.append(bot)
        elif ack_round_id != round_id_actual:
            symbol = "-"
            row_status = "waiting"
            missing_bots.append(bot)
        else:
            symbol_eval = _ack_live_symbol(resultado)
            if str(ack_status).lower() == "closed" and symbol_eval in ("✓", "X"):
                symbol = symbol_eval
                row_status = "closed"
                valid_closed = True
                closed_count += 1
                if (age_s is not None) and (not future):
                    lag_samples.append(float(age_s))
            else:
                symbol = "·"
                row_status = "nonclose"
                missing_bots.append(bot)

        if stale:
            stale_bots.append(bot)
        if warn:
            warn_bots.append(bot)

        rows.append({
            "bot": bot,
            "symbol": symbol,
            "round": ack_round_id,
            "age_s": age_s,
            "age_txt": _ack_live_fmt_age(age_s),
            "ciclo": ciclo if ciclo not in (None, "") else "--",
            "asset": asset if asset not in (None, "") else "--",
            "status": row_status,
            "ack_status": ack_status,
            "sync_wait": sync_wait,
            "waiting_release_round": waiting_release_round,
            "last_seen_ts": last_seen_ts,
            "stale": stale,
            "warn": warn,
            "future": future,
            "last_round": ack_round_id,
            "valid_closed": valid_closed,
        })

    out["rows"] = rows
    out["expected_count"] = len(bots)
    out["closed_count"] = closed_count
    out["missing_bots"] = missing_bots
    out["max_lag_s"] = max(lag_samples) if lag_samples else None
    out["avg_lag_s"] = (sum(lag_samples) / len(lag_samples)) if lag_samples else None
    out["stale_bots"] = stale_bots
    out["warn_bots"] = warn_bots
    return out


def _ack_live_norm_resultado(value):
    try:
        s = str(value or "").strip().upper()
        s_ascii = normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    except Exception:
        return "OTHER"

    if s in ("✓", "CHECK") or s_ascii in ("GANANCIA", "WIN", "CHECK"):
        return "WIN"

    if s in ("X", "✗") or s_ascii in ("PERDIDA", "LOSS", "X"):
        return "LOSS"

    return "OTHER"


def _ack_tape_init():
    try:
        if "BOT_NAMES" not in globals():
            return
        if not isinstance(globals().get("ACK_LIVE_TAPE"), dict):
            globals()["ACK_LIVE_TAPE"] = {}
        width = int(globals().get("ACK_TAPE_WIDTH", 80) or 80)
        for bot in BOT_NAMES:
            if bot not in ACK_LIVE_TAPE or not isinstance(ACK_LIVE_TAPE.get(bot), deque):
                ACK_LIVE_TAPE[bot] = deque(maxlen=width)
        if not isinstance(globals().get("ACK_LIVE_TAPE_SEEN"), deque):
            max_seen = int(globals().get("ACK_TAPE_MAX_SEEN", 2000) or 2000)
            globals()["ACK_LIVE_TAPE_SEEN"] = deque(maxlen=max_seen)
    except Exception:
        pass


def _ack_tape_seen_contains(key):
    try:
        if key in (None, ""):
            return False
        return str(key) in ACK_LIVE_TAPE_SEEN
    except Exception:
        return False


def _ack_tape_seen_add(key):
    try:
        if key in (None, ""):
            return
        k = str(key)
        if not _ack_tape_seen_contains(k):
            ACK_LIVE_TAPE_SEEN.append(k)
    except Exception:
        pass


def _ack_tape_symbol_plain(value):
    try:
        norm_fn = globals().get("_ack_live_norm_resultado", None)
        if callable(norm_fn):
            norm = str(norm_fn(value) or "").upper()
            if norm == "WIN":
                return "✓"
            if norm == "LOSS":
                return "X"
    except Exception:
        pass
    try:
        s = str(value or "").strip().upper()
        s_ascii = normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    except Exception:
        return "·"
    if s in ("✓", "CHECK") or s_ascii in ("GANANCIA", "WIN", "CHECK"):
        return "✓"
    if s in ("X", "✗") or s_ascii in ("PERDIDA", "LOSS", "X"):
        return "X"
    return "·"


def _ack_tape_update_from_ack_live():
    try:
        if not bool(globals().get("ACK_TAPE_ENABLE", True)):
            return
        _ack_tape_init()
        if "BOT_NAMES" not in globals():
            return
        for bot in BOT_NAMES:
            try:
                path = _sync_round_ack_path(bot)
                ack = _sync_round_safe_read_json(path)
                if not isinstance(ack, dict):
                    continue
                status = str(ack.get("status", "") or "").strip().lower()
                if status != "closed":
                    continue
                round_id = int(ack.get("round_id", 0) or 0)
                if round_id <= 0:
                    continue
                resultado = ack.get("resultado", "")
                symbol = _ack_tape_symbol_plain(resultado)
                if symbol not in ("✓", "X"):
                    continue
                result_norm = "WIN" if symbol == "✓" else "LOSS"
                contract_id = str(ack.get("contract_id", "") or "").strip()
                if contract_id:
                    key = f"{bot}:{round_id}:{contract_id}:{result_norm}"
                else:
                    key = f"{bot}:{round_id}:{result_norm}:{ack.get('ts', '')}"
                if _ack_tape_seen_contains(key):
                    continue
                if bot not in ACK_LIVE_TAPE or not isinstance(ACK_LIVE_TAPE.get(bot), deque):
                    ACK_LIVE_TAPE[bot] = deque(maxlen=int(globals().get("ACK_TAPE_WIDTH", 80) or 80))
                ACK_LIVE_TAPE[bot].append(symbol)
                _ack_tape_seen_add(key)
            except Exception:
                continue
    except Exception:
        pass


def _ack_tape_color_symbol(symbol):
    base = str(symbol or "")
    try:
        if not bool(globals().get("ACK_TAPE_USE_COLOR", True)):
            return base
        if base == "✓":
            return f"{Fore.GREEN}✓{Style.RESET_ALL}"
        if base == "X":
            return f"{Fore.RED}X{Style.RESET_ALL}"
        if base == "·":
            return f"{Fore.LIGHTBLACK_EX}·{Style.RESET_ALL}"
        return base
    except Exception:
        return base


def _ack_tape_strip_ansi(text):
    try:
        return re.sub(r"\x1b\[[0-9;]*m", "", str(text or ""))
    except Exception:
        return str(text or "")


def _ack_tape_pad_visible(text, width):
    try:
        w = int(width or 0)
    except Exception:
        w = 0
    txt = str(text or "")
    if w <= 0:
        return txt
    try:
        visible_len = len(_ack_tape_strip_ansi(txt))
    except Exception:
        visible_len = len(txt)
    if visible_len < w:
        return txt + (" " * (w - visible_len))
    return txt


def hud_strip_ansi(text):
    return _ack_tape_strip_ansi(text)


def hud_visible_len(text):
    return len(hud_strip_ansi(text))


def hud_pad(text, width):
    return _ack_tape_pad_visible(text, width)


def hud_border_top(width=HUD_BOX_WIDTH):
    inner = max(1, int(width or HUD_BOX_WIDTH))
    return "╔" + ("═" * inner) + "╗"


def hud_border_mid(width=HUD_BOX_WIDTH):
    inner = max(1, int(width or HUD_BOX_WIDTH))
    return "╠" + ("═" * inner) + "╣"


def hud_border_bottom(width=HUD_BOX_WIDTH):
    inner = max(1, int(width or HUD_BOX_WIDTH))
    return "╚" + ("═" * inner) + "╝"


def hud_box_line(text="", width=HUD_BOX_WIDTH, color=None):
    inner = max(1, int(width or HUD_BOX_WIDTH))
    payload = hud_pad(str(text or ""), inner)
    line = f"║{payload}║"
    if color:
        try:
            return f"{color}{line}{Style.RESET_ALL}"
        except Exception:
            return f"{color}{line}"
    return line


def _ack_tape_render_bot(bot, fallback_resultados=None, width=None):
    _ack_tape_init()
    try:
        w = int(width if width is not None else globals().get("ACK_TAPE_WIDTH", 80))
    except Exception:
        w = 80
    if w <= 0:
        w = 80
    fill_char = str(globals().get("ACK_TAPE_FILL_CHAR", "·") or "·")
    tape = list(ACK_LIVE_TAPE.get(bot, []) or [])
    symbols = []
    if tape:
        symbols = [_ack_tape_symbol_plain(x) for x in tape][-w:]
    else:
        fb = list(fallback_resultados or [])
        symbols = [_ack_tape_symbol_plain(x) for x in fb][-w:]
    while len(symbols) < w:
        symbols.insert(0, fill_char)
    colored = "".join(_ack_tape_color_symbol(_ack_tape_symbol_plain(s)) for s in symbols[-w:])
    return _ack_tape_pad_visible(colored, w)

def _hud_get_gp_stats(bot):
    st = estado_bots.get(bot, {}) if isinstance(estado_bots.get(bot, {}), dict) else {}
    g_state = int(st.get("ganancias", 0) or 0)
    p_state = int(st.get("perdidas", 0) or 0)
    total_state = max(0, g_state + p_state, int(st.get("tamano_muestra", 0) or 0))

    tape = list((ACK_LIVE_TAPE.get(bot, []) if isinstance(globals().get("ACK_LIVE_TAPE"), dict) else []) or [])
    g_ack = 0
    p_ack = 0
    for item in tape:
        sym = _ack_tape_symbol_plain(item)
        if sym == "✓":
            g_ack += 1
        elif sym == "X":
            p_ack += 1
    total_ack = g_ack + p_ack

    g_fb = 0
    p_fb = 0
    for item in list(st.get("resultados", []) or []):
        tok = str(item or "").strip().upper()
        if tok in ("GANANCIA", "✓", "WIN", "CHECK"):
            g_fb += 1
        elif tok in ("PÉRDIDA", "PERDIDA", "X", "✗", "LOSS"):
            p_fb += 1
    total_fb = g_fb + p_fb

    if total_ack > total_state:
        g, p, total = g_ack, p_ack, total_ack
    elif total_state > 0:
        g, p, total = g_state, p_state, total_state
    else:
        g, p, total = g_fb, p_fb, total_fb

    porc = (float(g) / float(total) * 100.0) if total > 0 else None
    return int(g), int(p), porc, int(total)


def _ack_live_build_rows():
    state = _sync_round_safe_read_json(SYNC_ROUND_STATE_PATH) or {}
    obj_round = int(state.get("round_id", state.get("released_round", 1)) or 1)
    released_round = int(state.get("released_round", obj_round) or obj_round)
    rows = []
    now = time.time()

    for bot in BOT_NAMES:
        ack = _sync_round_safe_read_json(_sync_round_ack_path(bot))

        if not isinstance(ack, dict):
            ack_round = 0
            gap = None
            res = "-"
            estado = "missing"
            age_s = None
            age_txt = "--"
            ciclo = "--"
            asset = "--"
            sync_wait = False
            is_current = False
            is_closed_result = False
            stale = False
            future = False
            warn = False
        else:
            ack_round = int(ack.get("round_id", 0) or 0)
            resultado = ack.get("resultado", "")
            ts = ack.get("ts", None)
            asset = ack.get("asset", "--")
            ciclo = ack.get("ciclo", "--")
            status_ack = str(ack.get("status", "") or "")
            sync_wait = bool(ack.get("sync_wait", False))

            gap = (obj_round - ack_round) if ack_round > 0 else None
            is_current = ack_round == obj_round
            norm = _ack_live_norm_resultado(resultado)

            age_s = None
            future = False
            try:
                if ts is not None:
                    age_try = now - float(ts)
                    if age_try < 0:
                        future = True
                        age_s = None
                    else:
                        age_s = age_try
            except Exception:
                age_s = None

            if is_current and norm == "WIN" and status_ack == "closed":
                res = "✓"
            elif is_current and norm == "LOSS" and status_ack == "closed":
                res = "X"
            elif ack_round != obj_round:
                res = "-"
            else:
                res = "·"

            is_closed_result = bool(is_current and status_ack == "closed" and norm in ("WIN", "LOSS"))
            stale = bool(age_s is not None and age_s > ACK_LIVE_MAX_AGE_STALE_S)
            warn = bool(age_s is not None and age_s > ACK_LIVE_MAX_AGE_WARN_S)

            if ack_round > 0 and ack_round < obj_round:
                estado = "waiting"
            elif ack_round > obj_round or future:
                estado = "future"
            elif ack_round <= 0:
                estado = "missing"
            elif is_current and status_ack != "closed":
                estado = "open"
            elif is_current and status_ack == "closed" and norm in ("WIN", "LOSS"):
                estado = "closed"
            else:
                estado = "open"

            if stale:
                estado_visual = f"{estado}‼"
            elif warn:
                estado_visual = f"{estado}⚠"
            else:
                estado_visual = estado

            age_txt = _ack_live_fmt_age(age_s) if age_s is not None else "--"
            ciclo = "--" if ciclo in (None, "") else ciclo
            asset = "--" if asset in (None, "") else asset
        if not isinstance(ack, dict):
            estado_visual = estado

        rows.append({
            "bot": bot,
            "obj_round": obj_round,
            "ack_round": ack_round,
            "gap": gap,
            "res": res,
            "age_s": age_s,
            "age_txt": age_txt,
            "ciclo": ciclo,
            "asset": asset,
            "estado": estado,
            "estado_visual": estado_visual,
            "sync_wait": sync_wait,
            "is_current": is_current,
            "is_closed_result": is_closed_result,
            "stale": stale,
            "future": future,
            "warn": warn,
        })

    return {
        "obj_round": obj_round,
        "released_round": released_round,
        "rows": rows,
    }


def _ack_live_calc_summary(rows_pack):
    rows = list((rows_pack or {}).get("rows", []) or [])
    st = _sync_round_safe_read_json(SYNC_ROUND_STATE_PATH) or {}
    obj_round = int((rows_pack or {}).get("obj_round", 1) or 1)
    released_round = int((rows_pack or {}).get("released_round", obj_round) or obj_round)

    valid_current = [r for r in rows if bool(r.get("is_current")) and r.get("res") in ("✓", "X")]
    verdes_count = sum(1 for r in valid_current if r.get("res") == "✓")
    rojas_count = sum(1 for r in valid_current if r.get("res") == "X")
    closed_count = verdes_count + rojas_count
    expected_count = len(BOT_NAMES)
    faltan_count = max(0, expected_count - closed_count)

    partial_pattern = f"{verdes_count}V{rojas_count}X"
    bot_x_actual = ""
    if rojas_count == 1:
        for row in valid_current:
            if row.get("res") == "X":
                bot_x_actual = str(row.get("bot") or "")
                break

    complete = closed_count == expected_count
    if complete:
        data_quality = "ok"
    elif closed_count > 0:
        data_quality = "partial"
    else:
        data_quality = "missing"

    lag_values = []
    for row in valid_current:
        if row.get("future"):
            continue
        age_s = row.get("age_s")
        if age_s is None:
            continue
        lag_values.append(float(age_s))

    max_lag_s = max(lag_values) if lag_values else None
    avg_lag_s = (sum(lag_values) / len(lag_values)) if lag_values else None

    waiting_bots = [str(r.get("bot")) for r in rows if str(r.get("estado")) in ("waiting", "missing")]
    stale_bots = [str(r.get("bot")) for r in rows if bool(r.get("stale"))]
    all_prev_waiting = bool(rows) and all(
        (int(r.get("ack_round", 0) or 0) > 0 and int(r.get("ack_round", 0) or 0) < obj_round)
        for r in rows
    )

    return {
        "obj_round": obj_round,
        "released_round": released_round,
        "verdes_count": verdes_count,
        "rojas_count": rojas_count,
        "faltan_count": faltan_count,
        "closed_count": closed_count,
        "expected_count": expected_count,
        "partial_pattern": partial_pattern,
        "bot_x_actual": bot_x_actual,
        "complete": complete,
        "data_quality": data_quality,
        "max_lag_s": max_lag_s,
        "avg_lag_s": avg_lag_s,
        "waiting_bots": waiting_bots,
        "stale_bots": stale_bots,
        "all_prev_waiting": all_prev_waiting,
        "status_state": str(st.get("status", "") or ""),
        "reason_state": str(st.get("reason", "") or ""),
        "completed_state": bool(st.get("completed", False)),
        "real_pending_bot": str(st.get("real_pending_bot", "") or ""),
        "real_pending_cycle": int(st.get("real_pending_cycle", 0) or 0),
    }


def _fmt_estado_round_live(estado_visual: str, estado_base: str = "") -> str:
    return Fore.YELLOW + Style.BRIGHT + "waiting" + Style.RESET_ALL


def _sync_round_manual_status() -> dict:
    try:
        st = _sync_round_safe_read_json(SYNC_ROUND_STATE_PATH) or {}
        rows_pack = _ack_live_build_rows()
        summary = _ack_live_calc_summary(rows_pack)

        obj_round = int(summary.get("obj_round", st.get("round_id", 1)) or 1)
        released_round = int(summary.get("released_round", st.get("released_round", obj_round)) or obj_round)
        closed_count = int(summary.get("closed_count", 0) or 0)
        expected_count = int(summary.get("expected_count", len(BOT_NAMES)) or len(BOT_NAMES))
        faltan_count = int(summary.get("faltan_count", max(0, expected_count - closed_count)) or 0)

        return {
            "obj_round": obj_round,
            "released_round": released_round,
            "closed_count": closed_count,
            "expected_count": expected_count,
            "faltan_count": faltan_count,
            "released_ok": released_round >= obj_round,
        }
    except Exception:
        return {
            "obj_round": 0,
            "released_round": 0,
            "closed_count": 0,
            "expected_count": len(BOT_NAMES),
            "faltan_count": len(BOT_NAMES),
            "released_ok": False,
        }


def _round_live_estado_display(row: dict) -> str:
    """
    ROUND LIVE muestra estados visuales de cierre/espera.
    Esta función NO modifica datos internos.
    Solo transforma la fila en texto visual.
    """
    try:
        row = row if isinstance(row, dict) else {}

        res = str(row.get("res") or row.get("resultado") or row.get("symbol") or "").strip()
        ack_round = int(row.get("ack_round", row.get("ack", 0)) or 0)
        gap = row.get("gap", 0)
        edad_s = row.get("edad_s", row.get("age_s", row.get("ack_age_s", None)))
        is_current = bool(row.get("is_current", row.get("current", True)))

        if isinstance(edad_s, str):
            edad_txt = edad_s.strip().lower()
            if edad_txt in ("", "--", "none", "null", "-"):
                edad_s = None
            else:
                try:
                    if edad_txt.endswith("ms"):
                        edad_s = float(edad_txt[:-2].strip()) / 1000.0
                    elif edad_txt.endswith("m"):
                        edad_s = float(edad_txt[:-1].strip()) * 60.0
                    elif edad_txt.endswith("s"):
                        edad_s = float(edad_txt[:-1].strip())
                    else:
                        edad_s = float(edad_txt)
                except Exception:
                    edad_s = None
        elif edad_s is not None:
            try:
                edad_s = float(edad_s)
            except Exception:
                edad_s = None

        loss_values = {"X", "✗", "PÉRDIDA", "PERDIDA", "LOSS"}
        win_values = {"✓", "GANANCIA", "WIN"}
        res_upper = res.upper()
        is_loss = res_upper in loss_values or res in ("X", "✗")
        is_win = res_upper in win_values or res == "✓"
        is_closed = is_loss or is_win

        if ack_round <= 0:
            return "pendiente_ack"
        if (not res) or (res in (".", "-", "--", "None", "null")):
            return "pendiente_resultado"
        if gap != 0:
            return "ronda_no_actual"
        if edad_s is None:
            if is_closed:
                return "cerrado_sin_edad"
            return "pendiente_resultado"
        if is_current and is_closed and edad_s <= ROUND_LIVE_INVEST_WINDOW_S:
            if is_loss:
                return "X_reciente"
            if is_win:
                return "✓_reciente"
            return "cerrado_reciente"
        if is_current and is_closed and edad_s > ROUND_LIVE_INVEST_WINDOW_S:
            return "cerrado_fuera_ventana"
        if is_current and not is_closed:
            return "esperando_cierre"
        return "esperando"
    except Exception:
        return "estado_error"
