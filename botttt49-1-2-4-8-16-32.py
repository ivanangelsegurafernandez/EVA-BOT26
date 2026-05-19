# -*- coding: utf-8 -*-
import asyncio
import json
import glob
import csv
import os
import sys


def _early_api_error_text_for_buy_selftest(exc) -> str:
    try:
        return str(exc or "")
    except Exception:
        return ""

def _early_is_insufficient_balance_error_for_buy_selftest(exc) -> bool:
    txt = _early_api_error_text_for_buy_selftest(exc).lower()
    return (
        "insufficientbalance" in txt
        or "insufficient balance" in txt
        or "balance (0.00" in txt
        or "account balance" in txt and "insufficient" in txt
    )

def _early_is_rate_limit_error_for_buy_selftest(exc) -> bool:
    txt = _early_api_error_text_for_buy_selftest(exc).lower()
    return (
        "ratelimit" in txt
        or "rate limit" in txt
        or "ticks_history" in txt and "limit" in txt
    )

if os.environ.get("RUN_BUY_ERROR_CLASSIFIER_SELFTEST") == "1":
    _digits = "".join(ch for ch in os.path.basename(__file__) if ch.isdigit())
    _nombre_bot_selftest = f"fulll{_digits[:2]}" if _digits else os.path.basename(__file__)
    assert _early_is_insufficient_balance_error_for_buy_selftest("API error: InsufficientBalance - Your account balance (0.00 USD) is insufficient to buy this contract (1.00 USD)")
    assert _early_is_insufficient_balance_error_for_buy_selftest(RuntimeError("InsufficientBalance"))
    assert not _early_is_insufficient_balance_error_for_buy_selftest("API error: RateLimit - You have reached the rate limit for ticks_history.")
    assert _early_is_rate_limit_error_for_buy_selftest("API error: RateLimit - You have reached the rate limit for ticks_history.")
    assert _early_is_rate_limit_error_for_buy_selftest(RuntimeError("rate limit"))
    print(f"✅ SELFTEST BUY_ERROR_CLASSIFIER OK | {_nombre_bot_selftest}")
    raise SystemExit(0)
from datetime import datetime, timezone
from statistics import mean
import pandas as pd
import time  # Added for timestamps in orden_real and BLOQUE 5
import random  # Added for jitter in BLOQUE 1.3
import itertools  # For req_counter in api_call
import math
import importlib
import warnings

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

websockets = _load_optional_module("websockets")
WEBSOCKETS_OK = websockets is not None

colorama = _load_optional_module("colorama")
if colorama is not None:
    Fore = colorama.Fore
    Back = colorama.Back
    Style = colorama.Style
    init = colorama.init
else:
    class _NoColor:
        def __getattr__(self, _name):
            return ""
    Fore = _NoColor()
    Back = _NoColor()
    Style = _NoColor()
    def init(*args, **kwargs):
        return None

pygame = _load_optional_module("pygame")
PYGAME_OK = pygame is not None

# === BLINDAJE: señales limpias ===
import signal
from contextlib import suppress
stop_event = asyncio.Event()

def handle_stop(sig, frame):
    # no tumbar de golpe; pedimos apagado ordenado
    if not stop_event.is_set():
        stop_event.set()

for _sig in (signal.SIGINT, signal.SIGTERM):
    with suppress(Exception):
        signal.signal(_sig, handle_stop)

# === /BLINDAJE ===

init(autoreset=True)

# Inicio de mixer blindado
try:
    if PYGAME_OK and not pygame.mixer.get_init():
        pygame.mixer.init()
except Exception as _e:
    print("Audio deshabilitado (mixer.init):", _e)

# Forzar que siempre use la carpeta donde está el script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# === PATCH SFX: audio seguro, canales y rate-limit ===
AUDIO_ENABLED = False
try:
    # Si ya inicializaste mixer arriba, sólo validamos canales
    if PYGAME_OK:
        pygame.mixer.set_num_channels(6)  # margen para solapamientos
        AUDIO_ENABLED = True
except Exception as _e:
    print("Audio deshabilitado (pygame.mixer):", _e)
    AUDIO_ENABLED = False

SFX_FILES = {
    "FELICITACIONES": "ia_scifi_01_felicitaciones_ivan_dry.wav",
    "LO_SIENTO": "ia_scifi_02_losiento_ivan_dry.wav",
    "PASO_A_REAL": "ia_scifi_03_paso_a_real_dry.wav",
    "REINTENTA": "ia_scifi_05_reintenta_dry.wav",
    "NO_CONCLUYO": "ia_scifi_06_no_concluyo_dry.wav",
    "NO_PASAR_REAL": "ia_scifi_07_no_pasar_real_dry.wav",
}
SFX = {}
_SFX_LAST_TS = {}
_SFX_MIN_INTERVAL = {
    "FELICITACIONES": 4.0,
    "LO_SIENTO": 4.0,
    "PASO_A_REAL": 2.0,   # ✅ más sensible
    "REINTENTA": 6.0,
    "NO_CONCLUYO": 10.0,
    "NO_PASAR_REAL": 6.0,
}

def _sfx_load_all():
    if not AUDIO_ENABLED or not PYGAME_OK:
        return
    for k, fname in SFX_FILES.items():
        p = os.path.join(script_dir, fname)
        try:
            if os.path.exists(p):
                SFX[k] = pygame.mixer.Sound(p)
            else:
                # Silencioso si no existe, no rompemos nada
                SFX[k] = None
        except Exception as e:
            print(f"No se pudo cargar SFX {k}: {e}")
            SFX[k] = None

def play_sfx(key: str, vol: float = 0.9):
    # Respeta MODO_SILENCIOSO y modo_manual (definidos en tu código)
    if not AUDIO_ENABLED or not PYGAME_OK:
        return
    if key not in SFX:
        return
    if SFX.get(key) is None:
        return
    # Rate-limit
    now = time.time()
    last = _SFX_LAST_TS.get(key, 0.0)
    min_iv = _SFX_MIN_INTERVAL.get(key, 4.0)
    if now - last < min_iv:
        return
    # Si el usuario forzó silencio (MANUAL), no sonar
    manual = False
    try:
        manual = bool(estado_bot.get("modo_manual"))
    except NameError:
        manual = False
    # Si el usuario forzó silencio (MANUAL), no sonar...
    # ...PERO no silenciamos el "PASO_A_REAL" ni sonidos de resultado (clave para tu lógica).
    if 'MODO_SILENCIOSO' in globals() and MODO_SILENCIOSO and manual and key not in ("PASO_A_REAL", "FELICITACIONES", "LO_SIENTO"):
        return

    try:
        ch = pygame.mixer.find_channel(True)
        if ch:
            SFX[key].set_volume(max(0.0, min(1.0, vol)))
            ch.play(SFX[key])
            _SFX_LAST_TS[key] = now
    except Exception as e:
        # Nunca rompemos lógica por un sonido
        if _print_once(f"sfx-{key}-err", ttl=60.0):
            print(f"SFX falló ({key}): {e}")

# Carga diferida para evitar bloquear import
_sfx_load_all()

# === /PATCH SFX ===

# ==================== CONFIG BÁSICA ====================
NOMBRE_BOT = "fulll49"
ARCHIVO_CSV = f"registro_enriquecido_{NOMBRE_BOT}.csv"
ARCHIVO_TOKEN = "token_actual.txt"  # Fuente única de verdad (coincide con 5R6M)
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
ACTIVOS = ["1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V"]
MARTINGALA_DEMO = [1, 2, 4, 8, 16]
MARTINGALA_REAL = [1, 2, 4, 8, 16]
# NOTA OPERATIVA:
# El nombre histórico del archivo puede incluir "-32",
# pero este bot opera oficialmente con 5 ciclos:
#     1-2-4-8-16
# No agregar C6 / 32 USD salvo cambio explícito de estrategia.
VELAS = 20
PAUSA_POST_OPERACION_S = 2  # Pausa uniforme tras cada operación con resultado definido (BLOQUE 1)
# ==================== VENTANA DE DECISIÓN IA ====================
# Objetivo: dar tiempo al MAESTRO + humano para decidir pasar a REAL ANTES del BUY.
# Ventana corta de decisión IA para evitar freno excesivo del ciclo.
# (0 para desactivar)
VENTANA_DECISION_IA_S = 35        # segundos (alineado con maestro)
# Debe estar alineado con MANUAL_REAL_DECISION_WINDOW_S del maestro.
# Si es menor, la orden manual REAL puede llegar tarde y cortar contrato DEMO.
VENTANA_DECISION_IA_POLL_S = 0.25 # granularidad de espera
# === Filtro avanzado (sin cambiar 13 features) ===
SCORE_MIN = 2.35            # score mínimo para aceptar un setup
SCORE_DROP_MAX = 0.70       # caída máxima tolerada al revalidar pre-buy
REVALIDAR_VELAS_N = 8       # velas mínimas para revalidación rápida
resultado_global = {"demo": 0.0, "real": 0.0}
ultimo_token = None
reinicio_forzado = asyncio.Event()
estado_bot = {
    "ciclo_en_progreso": False,
    "token_msg_mostrado": False,
    "intentos_saldo": 0,
    "interrumpir_ciclo": False,
    "ciclo_forzado": None,
    "reinicios_consecutivos": 0,
    "modo_manual": False,
    "barra_activa": False,
    "score_senal": None,
    "ciclo_actual": 1,
    "real_first_cycle_reset_pending": False,
    "real_cycle_guard_last_ts": 0.0,
    "sync_round_id": 1,
    "sync_wait": False,
    "pending_contract_resolution": False,
    "pending_contract_id": None,
    "pending_since_ts": 0.0,
    "pending_round_id": None,
}  # Added modo_manual and barra_activa
racha_actual_bot = 0  # racha del bot: >0 = racha de GANANCIAS, <0 = racha de PÉRDIDAS

# === Handshake con 5R6M ===
primer_ingreso_real = False  # Sonido solo 1 vez por ventana

# Variables persistentes para saldos últimos válidos
saldo_demo_last = 0.0
saldo_real_last = 0.0
saldo_demo_ok = False
saldo_real_ok = False
real_activado_en_bot = 0.0  # BLOQUE 5: Global for activation timestamp

# BLOQUE 2: Commit guard for REAL operations
REAL_COMMIT_WINDOW_S = 20
last_real_contract_id = None
real_buy_commit_until = 0.0

# Legado: no gobierna el ciclo REAL actual; manda ciclo_maestro/ciclo_forzado
RESET_CICLO_EN_ENTRADA_REAL = True  # legado visual/auditoría

def commit_guard_active() -> bool:
    return (last_real_contract_id is not None) and (time.time() < real_buy_commit_until)

def commit_guard_set(contract_id: int):
    global last_real_contract_id, real_buy_commit_until
    last_real_contract_id = contract_id
    real_buy_commit_until = time.time() + REAL_COMMIT_WINDOW_S

def commit_guard_clear():
    global last_real_contract_id, real_buy_commit_until
    last_real_contract_id = None
    real_buy_commit_until = 0.0

# >>> PATCH 1 — Helpers de orden de ciclo
ORDEN_DIR = "orden_real"  # misma carpeta usada por el maestro
# === IA ACK (handshake maestro→bot) ===
IA_ACK_DIR = "ia_ack"
try:
    os.makedirs(IA_ACK_DIR, exist_ok=True)
except Exception:
    pass

def leer_ia_ack(bot: str):
    path = os.path.join(IA_ACK_DIR, f"{bot}.json")
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

MAX_CICLOS = len(MARTINGALA_REAL)
# === LXV_SYNC_COLUMN: sincronización por ronda/columna ===
SYNC_ROUND_DIR = "sync_round"
SYNC_ROUND_STATE = os.path.join(SYNC_ROUND_DIR, "state.json")
SYNC_ROUND_STATE_REAL_HOLD_TTL_S = 180.0

try:
    os.makedirs(SYNC_ROUND_DIR, exist_ok=True)
except Exception:
    pass

def _sync_round_safe_read_json(path: str):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def _sync_round_write_json_atomic(path: str, payload: dict) -> bool:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception:
        return False
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def _sync_round_ack_path() -> str:
    return os.path.join(SYNC_ROUND_DIR, f"{NOMBRE_BOT}.json")

def leer_token_actual():
    """
    Lee token_actual.txt de forma segura.
    Devuelve contenido crudo o "DEMO" si no está disponible.
    """
    try:
        base_dir = globals().get("script_dir", os.path.dirname(os.path.abspath(__file__)))
        token_path = (
            globals().get("ARCHIVO_TOKEN")
            or globals().get("TOKEN_FILE")
            or globals().get("TOKEN_ACTUAL_FILE")
            or os.path.join(base_dir, "token_actual.txt")
        )
        token_path = os.path.abspath(str(token_path))
        if not os.path.exists(token_path):
            return "DEMO"
        with open(token_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return raw if raw else "DEMO"
    except Exception:
        return "DEMO"

def _token_real_ocupado(token) -> bool:
    t = str(token or "").strip()
    tu = t.upper()

    if tu in ("", "DEMO", "REAL:NONE", "REAL:NULL", "REAL:--", "REAL:"):
        return False

    if tu.startswith("REAL:"):
        owner = t.split(":", 1)[1].strip().lower()
        return owner in ("fulll45", "fulll46", "fulll47", "fulll48", "fulll49", "fulll50")

    return False

def _sync_round_resolve_start_round() -> int:
    st = _sync_round_safe_read_json(SYNC_ROUND_STATE) or {}
    try:
        released = int(st.get("released_round", 1) or 1)
    except Exception:
        released = 1
    return max(1, released)

def _sync_adopt_official_round(released_round, expected_release, old_round, motivo=""):
    try:
        rr = int(released_round or 0)
        exp = int(expected_release or 0)
        old = int(old_round or 0)
    except Exception:
        return old

    if rr >= exp and rr > old:
        return rr

    if rr >= exp:
        return max(rr, old)

    return old

def _sync_round_adopt_official_if_stale(motivo="pre_demo_buy"):
    try:
        st = _sync_round_safe_read_json(SYNC_ROUND_STATE) or {}
        released_round = int(st.get("released_round", 0) or 0)
        local_round = int(estado_bot.get("sync_round_id", 1) or 1)
    except Exception:
        return
    if released_round >= 100 and local_round < (released_round - 5):
        estado_bot["sync_round_id"] = int(released_round)
        try:
            now_ts = time.time()
            key = f"_SYNC_ROUND_ADOPT_OFFICIAL_LOG_TS_{motivo}"
            last_ts = float(globals().get(key, 0.0) or 0.0)
            if (now_ts - last_ts) >= 8.0:
                globals()[key] = now_ts
                print(Fore.CYAN + Style.BRIGHT + f"🧭 SYNC_ROUND_ADOPT_OFFICIAL: {NOMBRE_BOT} local={local_round} → oficial={released_round}")
        except Exception:
            pass



async def _sync_round_wait_post_real_rejoin(initial_grace_s: float = 8.0, poll_s: float = 0.5, round_id_real=None) -> None:
    """
    Bloquea cooperativamente al ex-owner REAL hasta que el maestro confirme
    la liberación común post_real_rejoin. Mientras espera no vuelve a DEMO,
    no busca señal, no compra y no escribe ACK DEMO falso.
    """
    start_ts = time.time()
    last_log_ts = 0.0
    seen_own_rejoin = False
    target_seen = 0

    initial_state = _sync_round_safe_read_json(SYNC_ROUND_STATE)
    if isinstance(initial_state, dict):
        try:
            start_released = int(initial_state.get("released_round", 0) or 0)
        except Exception:
            start_released = 0
    else:
        start_released = 0
    rr_candidates = [round_id_real]
    if isinstance(initial_state, dict):
        rr_candidates.extend([
            initial_state.get("last_real_round"),
            initial_state.get("real_pending_round"),
        ])
    real_round_seen = 0
    for candidate in rr_candidates:
        try:
            c = int(candidate or 0)
        except Exception:
            c = 0
        if c > 0:
            real_round_seen = c
            break
    expected_target = max(int(start_released or 0), int(real_round_seen + 1 if real_round_seen > 0 else 0))

    def _safe_exit_ready(state: dict, released_now: int) -> bool:
        if not isinstance(state, dict):
            return False
        if expected_target <= 0 or int(released_now) < int(expected_target):
            return False
        try:
            token_now = str(leer_token_actual() or "").strip()
        except Exception:
            return False
        if _token_real_ocupado(token_now):
            return False
        try:
            if _sync_bot_es_owner_real():
                return False
        except Exception:
            return False
        try:
            any_real, _owner, _reason = _sync_any_real_owner_active()
            if any_real:
                return False
        except Exception:
            return False
        status_now = str(state.get("status") or "").strip().lower()
        if status_now in {"holding_real_result", "holding_real_turn", "real_pending", "waiting_real_close"}:
            return False
        return True

    while True:
        st = _sync_round_safe_read_json(SYNC_ROUND_STATE)
        if not isinstance(st, dict):
            now_ts = time.time()
            if (now_ts - last_log_ts) >= 15.0:
                last_log_ts = now_ts
                print(Fore.CYAN + f"⏳ POST_REAL_REJOIN BLOQUEANTE: bot={NOMBRE_BOT} | released=? | expected={expected_target or '?'} | payload_maestro=? | token=? | acción=esperar_state_json_valido")
            await asyncio.sleep(max(0.2, min(float(poll_s), 1.0)))
            continue
        pr = st.get("post_real_rejoin")
        try:
            released = int(st.get("released_round", 0) or 0)
        except Exception:
            released = 0
        if released <= 0:
            now_ts = time.time()
            if (now_ts - last_log_ts) >= 15.0:
                last_log_ts = now_ts
                print(Fore.CYAN + f"⏳ POST_REAL_REJOIN BLOQUEANTE: bot={NOMBRE_BOT} | released=? | expected={expected_target or '?'} | payload_maestro={'si' if isinstance(pr, dict) else 'no'} | token=? | acción=esperar_released_round_valido")
            await asyncio.sleep(max(0.2, min(float(poll_s), 1.0)))
            continue
        if isinstance(pr, dict) and str(pr.get("bot") or "").strip() == str(NOMBRE_BOT):
            seen_own_rejoin = True
            try:
                target = int(pr.get("target_release", 0) or 0)
            except Exception:
                target = 0
            if target > 0:
                target_seen = max(int(target_seen or 0), int(target))
            if target > 0 and released >= target:
                sync_round = _sync_adopt_official_round(released, target, estado_bot.get("sync_round_id", target), motivo="post_real_rejoin_target")
                print(Fore.GREEN + Style.BRIGHT + f"✅ POST_REAL_REJOIN completado: {NOMBRE_BOT} sincronizado en ronda oficial #{sync_round}")
                try:
                    estado_bot["sync_round_id"] = int(sync_round or released or target or 1)
                except Exception:
                    pass
                return
            if (not bool(pr.get("active", False))) and target_seen > 0 and released >= target_seen:
                sync_round = _sync_adopt_official_round(released, target_seen, estado_bot.get("sync_round_id", target_seen), motivo="post_real_rejoin_inactive")
                print(Fore.GREEN + Style.BRIGHT + f"✅ POST_REAL_REJOIN completado: {NOMBRE_BOT} sincronizado en ronda oficial #{sync_round}")
                try:
                    estado_bot["sync_round_id"] = int(sync_round or released or target_seen or 1)
                except Exception:
                    pass
                return
            now_ts = time.time()
            if (now_ts - last_log_ts) >= 15.0:
                last_log_ts = now_ts
                espera = target if target > 0 else (target_seen if target_seen > 0 else expected_target if expected_target > 0 else "?")
                try:
                    token_now = str(leer_token_actual() or "?").strip()
                except Exception:
                    token_now = "?"
                print(Fore.CYAN + f"⏳ POST_REAL_REJOIN BLOQUEANTE: bot={NOMBRE_BOT} | released={released} | expected={espera} | payload_maestro=si | token={token_now} | acción=esperar_release_comun")
        else:
            if seen_own_rejoin and target_seen > 0 and released >= target_seen:
                sync_round = _sync_adopt_official_round(released, target_seen, estado_bot.get("sync_round_id", target_seen), motivo="post_real_rejoin_seen")
                print(Fore.GREEN + Style.BRIGHT + f"✅ POST_REAL_REJOIN completado: {NOMBRE_BOT} sincronizado en ronda oficial #{sync_round}")
                try:
                    estado_bot["sync_round_id"] = int(sync_round or released or target_seen or 1)
                except Exception:
                    pass
                return
            now_ts = time.time()
            elapsed = max(0.0, now_ts - start_ts)
            if elapsed >= 18.0 and _safe_exit_ready(st, released):
                sync_round = _sync_adopt_official_round(released, expected_target or released, estado_bot.get("sync_round_id", released), motivo="post_real_rejoin_safe_exit")
                try:
                    token_now = str(leer_token_actual() or "").strip()
                except Exception:
                    token_now = "?"
                print(Fore.GREEN + Style.BRIGHT + f"✅ POST_REAL_REJOIN SAFE_EXIT: bot={NOMBRE_BOT} | released={released} | expected={expected_target or released} | token={token_now}")
                print(Fore.GREEN + Style.BRIGHT + f"✅ POST_REAL_REJOIN completado por release ya avanzado: {NOMBRE_BOT} sincronizado en ronda oficial #{sync_round}")
                try:
                    estado_bot["sync_round_id"] = int(sync_round or released or 1)
                except Exception:
                    pass
                return
            if (now_ts - last_log_ts) >= 15.0:
                last_log_ts = now_ts
                espera = target_seen if target_seen > 0 else expected_target if expected_target > 0 else "?"
                try:
                    token_now = str(leer_token_actual() or "?").strip()
                except Exception:
                    token_now = "?"
                print(Fore.CYAN + f"⏳ POST_REAL_REJOIN BLOQUEANTE: bot={NOMBRE_BOT} | released={released} | expected={espera} | payload_maestro=no | token={token_now} | acción=esperar_o_salida_segura")
        await asyncio.sleep(max(0.2, min(float(poll_s), 1.0)))

def _sync_bot_es_owner_real() -> bool:
    try:
        tok = str(leer_token_actual() or "").strip().upper()
    except Exception:
        tok = ""

    expected = f"REAL:{str(NOMBRE_BOT).strip().upper()}"
    return tok == expected

def _sync_round_emit_close_ack(round_id: int, resultado: str, contract_id=None, asset=None, ciclo=None, modo_real_contrato=False) -> bool:
    res = str(resultado or "").upper().strip().replace("PERDIDA", "PÉRDIDA")
    if res not in ("GANANCIA", "PÉRDIDA", "INDEFINIDO"):
        return False
    rid = max(1, int(round_id or 1))
    prev = _sync_round_safe_read_json(_sync_round_ack_path()) or {}
    try:
        prev_round = int(prev.get("round_id", 0) or 0)
    except Exception:
        prev_round = 0
    if prev_round > rid:
        return False
    if bool(modo_real_contrato):
        print(Fore.CYAN + f"ℹ️ ACK SYNC omitido: contrato REAL de {NOMBRE_BOT}; no contamina columna DEMO.")
        return False
    mode_sync = "DEMO"
    payload = {
        "bot": NOMBRE_BOT,
        "round_id": rid,
        "ts": time.time(),
        "resultado": res,
        "contract_id": contract_id,
        "asset": asset,
        "ciclo": ciclo,
        "status": "closed",
        "sync_wait": False,
        "mode": mode_sync,
        "source": "ORDEN_REAL" if mode_sync == "REAL" else "SYNC_DEMO",
        "token": "DEMO" if mode_sync == "DEMO" else str(leer_token_actual() or ""),
    }
    return _sync_round_write_json_atomic(_sync_round_ack_path(), payload)


def _sync_round_emit_close_ack_confirmado(round_id, resultado, max_retries=3, **ack_kwargs) -> bool:
    res = str(resultado or "").upper().strip().replace("PERDIDA", "PÉRDIDA")
    rid = max(1, int(round_id or 1))
    valid_results = ("GANANCIA", "PÉRDIDA", "INDEFINIDO")
    if bool(ack_kwargs.get("modo_real_contrato", False)):
        try:
            _sync_round_emit_close_ack(rid, res, **ack_kwargs)
        except Exception:
            pass
        return False

    def _ack_confirmado() -> bool:
        data = _sync_round_safe_read_json(_sync_round_ack_path()) or {}
        if not isinstance(data, dict):
            return False
        try:
            ack_round = int(data.get("round_id", 0) or 0)
        except Exception:
            ack_round = 0
        ack_res = str(data.get("resultado", "") or "").upper().strip().replace("PERDIDA", "PÉRDIDA")
        return (
            str(data.get("bot", "") or "").strip() == str(NOMBRE_BOT)
            and ack_round == rid
            and str(data.get("status", "") or "").strip().lower() == "closed"
            and ack_res == res
            and ack_res in valid_results
            and bool(data.get("sync_wait", False)) is False
        )

    attempts = max(1, int(max_retries or 1))
    for attempt in range(attempts):
        try:
            _sync_round_emit_close_ack(rid, res, **ack_kwargs)
        except Exception:
            pass
        if _ack_confirmado():
            print(Fore.YELLOW + f"🧷 LXV_SYNC_COLUMN ACK cierre | {NOMBRE_BOT} | ronda #{rid} | {res}")
            return True
        if attempt < attempts - 1:
            time.sleep(0.2)

    try:
        st = _sync_round_safe_read_json(SYNC_ROUND_STATE) or {}
        try:
            released = int(st.get("released_round", rid) or rid)
        except Exception:
            released = rid
        modo_real = bool(ack_kwargs.get("modo_real_contrato", False))
        token_txt = ""
        try:
            token_txt = str(leer_token_actual() or "").strip()
        except Exception:
            token_txt = ""
        mode_txt = "REAL" if (modo_real or token_txt.upper().startswith("REAL")) else "DEMO"
        _sync_round_write_recovery_request(
            bot=NOMBRE_BOT,
            round_id=rid,
            next_round=rid + 1,
            released=released,
            reason="ack_close_missing_after_trade_result",
            mode=mode_txt,
            resultado=res,
            source="ACK_CLOSE_NO_CONFIRMADO",
            neutral=True,
            no_trade_result=True,
            quarantine=True,
            usable_for_real=False,
            usable_for_lxv=False,
            usable_for_training=False,
        )
    except Exception:
        pass
    print(Fore.YELLOW + Style.BRIGHT + f"⚠️ ACK_CLOSE_NO_CONFIRMADO | bot={NOMBRE_BOT} | ronda=#{rid} | resultado={res} | recovery_request=SI")
    return False

def _sync_round_write_wait_heartbeat(round_id: int, next_round: int):
    path = _sync_round_ack_path()
    prev = _sync_round_safe_read_json(path) or {}
    payload = dict(prev) if isinstance(prev, dict) else {}
    payload["bot"] = NOMBRE_BOT
    payload["round_id"] = int(round_id)
    payload["status"] = "closed"
    payload["sync_wait"] = True
    payload["waiting_release_round"] = int(next_round)
    payload["last_seen_ts"] = time.time()
    _sync_round_write_json_atomic(path, payload)


def _sync_round_write_release_heartbeat(round_id: int, next_round: int):
    path = _sync_round_ack_path()
    prev = _sync_round_safe_read_json(path) or {}
    payload = dict(prev) if isinstance(prev, dict) else {}
    payload["bot"] = NOMBRE_BOT
    payload["round_id"] = int(round_id)
    payload["status"] = "closed"
    payload["sync_wait"] = False
    payload["waiting_release_round"] = int(next_round)
    payload["last_seen_ts"] = time.time()
    _sync_round_write_json_atomic(path, payload)




def _sync_round_write_recovery_request(bot, round_id, next_round, released, reason, any_real_active=False, any_real_owner="", any_real_reason="", **extra_fields) -> bool:
    try:
        req_dir = os.path.join(SYNC_ROUND_DIR, "recovery_requests")
        os.makedirs(req_dir, exist_ok=True)
        path = os.path.join(req_dir, f"{str(bot or NOMBRE_BOT)}.json")
        payload = {
            "bot": str(bot or NOMBRE_BOT),
            "round_id": int(round_id),
            "next_round": int(next_round),
            "released": int(released),
            "reason": str(reason or ""),
            "ts": time.time(),
            "any_real_active": bool(any_real_active),
            "any_real_owner": str(any_real_owner or ""),
            "any_real_reason": str(any_real_reason or ""),
        }
        if isinstance(extra_fields, dict) and extra_fields:
            payload.update(extra_fields)

        # === BLINDAJE CENTRAL RECOVERY ARTIFICIAL ===
        try:
            src_txt = str(
                payload.get("source")
                or payload.get("src")
                or payload.get("reason")
                or payload.get("motivo")
                or ""
            ).lower()

            reason_txt = str(
                payload.get("reason")
                or payload.get("motivo")
                or payload.get("detalle")
                or ""
            ).lower()

            joined = f"{src_txt} {reason_txt}"

            recovery_like = any(k in joined for k in (
                "recovery",
                "incident",
                "incident_lock",
                "ack_close_missing",
                "stale_open",
                "sin_evidencia",
                "closed_sin_profit",
                "artificial",
                "quarantine",
                "cuarentena",
            ))

            neutral_like = bool(
                payload.get("neutral")
                or payload.get("no_trade_result")
                or payload.get("quarantine")
                or payload.get("recovery_artificial")
            )

            if recovery_like or neutral_like:
                payload["neutral"] = True
                payload["no_trade_result"] = True
                payload["quarantine"] = True
                payload["usable_for_real"] = False
                payload["usable_for_lxv"] = False
                payload["usable_for_training"] = False
                payload["recovery_artificial"] = True
                payload["real_block_reason"] = (
                    payload.get("real_block_reason")
                    or "recovery_artificial_no_usable_para_real_lxv_training"
                )
        except Exception:
            pass
        # === /BLINDAJE CENTRAL RECOVERY ARTIFICIAL ===
        return _sync_round_write_json_atomic(path, payload)
    except Exception:
        return False

def _sync_any_real_owner_active() -> tuple[bool, str, str]:
    """
    Devuelve (True, owner_bot, motivo) si hay REAL global activo o pendiente.
    """
    def _valid_bot(name: str) -> bool:
        b = str(name or "").strip()
        return b in {"fulll45", "fulll46", "fulll47", "fulll48", "fulll49", "fulll50"}

    now_ts = time.time()
    base_dir = globals().get("script_dir", os.path.dirname(os.path.abspath(__file__)))

    try:
        tok = str(leer_token_actual() or "").strip()
        if _token_real_ocupado(tok):
            owner = tok.split(":", 1)[1].strip() if ":" in tok else ""
            if _valid_bot(owner):
                return True, owner, "token_actual"
    except Exception:
        pass

    try:
        candidates = []
        if "_orden_real_candidate_paths" in globals():
            try:
                candidates.extend(_orden_real_candidate_paths() or [])
            except Exception:
                pass
        patterns = [
            os.path.join(base_dir, "orden_real", "*.json"),
            os.path.join(base_dir, "orden_real_*.json"),
            os.path.join(base_dir, "orden_real.json"),
        ]
        for pat in patterns:
            try:
                candidates.extend(glob.glob(pat))
            except Exception:
                continue
        seen = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            try:
                data = _sync_round_safe_read_json(path) or {}
                if not isinstance(data, dict) or bool(data.get("consumed", False)) or bool(data.get("closed", False)):
                    continue
                owner = str(data.get("bot") or data.get("target_bot") or data.get("owner_bot") or "").strip()
                if not _valid_bot(owner):
                    continue
                ts = float(data.get("ts") or data.get("created_ts") or data.get("created_at") or 0.0)
                ttl = float(data.get("ttl_s") or data.get("ttl") or globals().get("REAL_ORDER_TTL_S", 90) or 90)
                if ts > 0 and (now_ts - ts) <= max(1.0, ttl):
                    try:
                        tok = str(leer_token_actual() or "").strip().upper()
                    except Exception:
                        tok = ""
                    expected = f"REAL:{str(owner).strip().upper()}"
                    if tok != expected:
                        if _print_once(f"orden-real-sin-token-{owner}", ttl=30):
                            print(Fore.YELLOW + f"🟡 ORDEN_REAL_SIN_TOKEN_IGNORADA: {owner} no bloquea DEMO | token={tok or '--'}")
                        continue
                    return True, owner, "orden_real_viva"
            except Exception:
                continue
    except Exception:
        pass

    # real_owner_state.json por sí solo no bloquea DEMO: tras REAL:none debe considerarse libre.
    # Solo token REAL:fulllXX, orden REAL viva, REAL_CLOSE_PENDING o estado explícito de cierre retienen HOLD.

    try:
        fresh_close = max(float(globals().get("REAL_CLOSE_MAX_AGE_S", 120) or 120), 120.0)
        for name in ("real_close_pending.json", "real_close_state.json", "close_pending_real.json"):
            path = os.path.join(base_dir, name)
            try:
                data = _sync_round_safe_read_json(path) or {}
                if not isinstance(data, dict):
                    continue
                owner = str(data.get("bot") or data.get("owner_bot") or "").strip()
                ts = float(data.get("ts") or data.get("created_ts") or data.get("updated_ts") or 0.0)
                active_flag = data.get("active", True)
                if bool(data.get("consumed", False)) or bool(data.get("closed", False)) or active_flag is False:
                    continue
                if _valid_bot(owner) and ts > 0 and (now_ts - ts) <= fresh_close:
                    return True, owner, "real_close_pending"
            except Exception:
                continue
    except Exception:
        pass

    try:
        st = _sync_round_safe_read_json(SYNC_ROUND_STATE) or {}
        if isinstance(st, dict):
            status = str(st.get("real_status") or st.get("status") or "").strip().lower()
            owner = str(
                st.get("real_owner")
                or st.get("owner_real")
                or st.get("bot_real")
                or st.get("real_bot")
                or st.get("bot")
                or st.get("owner_bot")
                or ""
            ).strip()
            real_global_raw = st.get("real_global", False)
            real_global = bool(real_global_raw) or str(real_global_raw).strip().upper() in {"SI", "SÍ", "YES", "TRUE", "1", "ON"}
            close_pending_raw = st.get("real_close_pending", st.get("REAL_CLOSE_PENDING", False))
            close_pending = bool(close_pending_raw) or str(close_pending_raw).strip().upper() in {"SI", "SÍ", "YES", "TRUE", "1", "ON"}
            token_status = str(st.get("token_status") or "").strip()
            active_statuses = {
                "holding_real_result", "holding_real_turn", "real_active", "real_pending",
                "waiting_real_close", "closing", "active",
            }
            if _token_real_ocupado(token_status):
                token_owner = token_status.split(":", 1)[1].strip() if ":" in token_status else owner
                if _valid_bot(token_owner):
                    try:
                        tok_actual = str(leer_token_actual() or "").strip().upper()
                    except Exception:
                        tok_actual = ""
                    expected = f"REAL:{str(token_owner).strip().upper()}"
                    if tok_actual != expected:
                        if _print_once(f"sync-token-status-sin-token-real-{token_owner}", ttl=30):
                            print(Fore.YELLOW + f"🟡 SYNC_TOKEN_STATUS_SIN_TOKEN_REAL_IGNORADO: owner={token_owner} no bloquea DEMO | token_actual={tok_actual or '--'}")
                        return False, "", "sync_round_token_status_without_real_token_ignored"
                    return True, token_owner, "sync_round_token_status"
            if (real_global or close_pending or status in active_statuses) and _valid_bot(owner):
                state_ts = _sync_round_state_ts(st)
                state_age = (now_ts - state_ts) if state_ts > 0 else 999999.0
                ttl_state = float(globals().get("SYNC_ROUND_STATE_REAL_HOLD_TTL_S", 180.0) or 180.0)
                if state_ts <= 0 or state_age > ttl_state:
                    try:
                        stale_key = "sync_round_state_stale_ignorado_ts"
                        last = float(estado_bot.get(stale_key, 0.0) or 0.0)
                        if (now_ts - last) >= 60.0:
                            print(
                                Fore.YELLOW + Style.BRIGHT +
                                f"⚠️ SYNC_ROUND_STATE_STALE_IGNORADO | bot={NOMBRE_BOT} | owner={owner or '--'} "
                                f"| status={status or '--'} | age={state_age:.0f}s | token libre"
                            )
                            estado_bot[stale_key] = now_ts
                    except Exception:
                        pass
                else:
                    try:
                        tok = str(leer_token_actual() or "").strip().upper()
                    except Exception:
                        tok = ""
                    expected = f"REAL:{str(owner).strip().upper()}"
                    if tok != expected:
                        if _print_once(f"sync-state-sin-token-{owner}", ttl=30):
                            print(Fore.YELLOW + f"🟡 SYNC_ROUND_STATE_SIN_TOKEN_IGNORADO: owner={owner} no bloquea DEMO | token={tok or '--'}")
                        return False, "", "sync_round_state_without_token_ignored"
                    return True, owner, "sync_round_state"
    except Exception:
        pass

    return False, "", ""


async def _sync_wait_global_real_clear(contexto="", allow_owner=True):
    """HOLD no intrusivo: bloquea solo nuevas operaciones DEMO si otro bot está en REAL."""
    ctx = str(contexto or "").strip() or "pre_new_trade"
    was_holding = False
    while True:
        active, owner, reason = _sync_any_real_owner_active()
        owner = str(owner or "").strip()
        if not active:
            if was_holding:
                print(Fore.GREEN + Style.BRIGHT + f"✅ GLOBAL_REAL_CLEAR: {NOMBRE_BOT} reanuda DEMO | contexto={ctx}")
            return
        if bool(allow_owner) and owner == str(NOMBRE_BOT):
            return
        now_ts = time.time()
        key = f"global-real-hold:{ctx}"
        last_ts = float(estado_bot.get(key, 0.0) or 0.0)
        if (now_ts - last_ts) >= 12.0:
            print(
                Fore.YELLOW + Style.BRIGHT +
                f"⏸️ GLOBAL_REAL_HOLD_PRE_NEW_TRADE: {NOMBRE_BOT} detenido porque REAL activo={owner or '--'} "
                f"| contexto={ctx} | motivo={reason or '--'} | no inicia nueva compra DEMO"
            )
            estado_bot[key] = now_ts
        was_holding = True
        await asyncio.sleep(1.2 + random.uniform(0.0, 0.6))

def _sync_real_active_for_other_bot():
    active, owner, reason = _sync_any_real_owner_active()
    owner = str(owner or "").strip()
    return bool(active and owner and owner != str(NOMBRE_BOT)), owner, reason


def _selftest_sync_demo_hold_global():
    if str(os.environ.get("RUN_SYNC_DEMO_HOLD_GLOBAL_SELFTEST", "0")).strip() != "1":
        return
    import tempfile

    base_dir = tempfile.mkdtemp(prefix="sync_demo_hold_global_")
    old_script_dir = globals().get("script_dir")
    old_token = globals().get("ARCHIVO_TOKEN")
    old_sync_state = globals().get("SYNC_ROUND_STATE")
    try:
        globals()["script_dir"] = base_dir
        globals()["ARCHIVO_TOKEN"] = os.path.join(base_dir, "token_actual.txt")
        globals()["SYNC_ROUND_STATE"] = os.path.join(base_dir, "sync_round", "state.json")
        os.makedirs(os.path.join(base_dir, "orden_real"), exist_ok=True)

        def wr(path, data):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

        def wt(v):
            with open(globals()["ARCHIVO_TOKEN"], "w", encoding="utf-8") as f:
                f.write(v)

        now = time.time()
        wt("REAL:fulll47")
        ok, owner, reason = _sync_any_real_owner_active(); assert ok and owner=="fulll47" and reason=="token_actual"

        wt("REAL:none")
        ok, owner, reason = _sync_any_real_owner_active(); assert not ok

        wr(os.path.join(base_dir, "orden_real", "fulll47.json"), {"bot":"fulll47", "consumed":False, "created_ts":now, "ttl_s":90})
        ok, owner, reason = _sync_any_real_owner_active(); assert not ok

        wt("REAL:fulll47")
        ok, owner, reason = _sync_any_real_owner_active()
        assert ok
        assert owner == "fulll47"
        assert reason in ("token_actual", "orden_real_viva")

        os.remove(os.path.join(base_dir, "orden_real", "fulll47.json"))
        wt("REAL:none")
        wr(os.path.join(base_dir, "real_close_pending.json"), {"bot":"fulll47", "ts":now})
        ok, owner, reason = _sync_any_real_owner_active(); assert ok and owner=="fulll47" and reason=="real_close_pending"

        os.remove(os.path.join(base_dir, "real_close_pending.json"))

        wt("REAL:none")
        wr(os.path.join(base_dir, "sync_round", "state.json"), {"token_status": "REAL:fulll47", "real_global": False, "real_owner": "fulll47", "status": "waiting", "ts": now})
        ok, owner, reason = _sync_any_real_owner_active()
        assert not ok
        assert reason in (
            None,
            "",
            "sync_round_token_status_without_real_token_ignored",
            "sync_round_state_without_token_ignored",
        )

        wt("REAL:fulll47")
        wr(os.path.join(base_dir, "sync_round", "state.json"), {"token_status": "REAL:fulll47", "real_global": False, "real_owner": "fulll47", "status": "waiting", "ts": now})
        ok, owner, reason = _sync_any_real_owner_active()
        assert ok
        assert owner == "fulll47"
        assert reason in ("token_actual", "sync_round_token_status")

        wt("REAL:fulll49")
        wr(os.path.join(base_dir, "sync_round", "state.json"), {"token_status": "REAL:fulll47", "real_owner": "fulll47", "status": "waiting", "ts": now})
        ok, owner, reason = _sync_any_real_owner_active()
        assert not (ok and owner == "fulll47" and reason == "sync_round_token_status")

        wt("REAL:none")
        wr(os.path.join(base_dir, "sync_round", "state.json"), {"real_global": True, "real_owner": "fulll47", "status": "real_active", "ts": now})
        ok, owner, reason = _sync_any_real_owner_active()
        assert not ok
        assert reason in (None, "", "sync_round_state_without_token_ignored")

        wt("DEMO")
        ok, owner, reason = _sync_any_real_owner_active()
        assert not ok
        assert reason in (None, "", "sync_round_state_without_token_ignored")

        wt("")
        ok, owner, reason = _sync_any_real_owner_active()
        assert not ok
        assert reason in (None, "", "sync_round_state_without_token_ignored")

        wt("REAL:fulll47")
        wr(os.path.join(base_dir, "sync_round", "state.json"), {"real_global": True, "real_owner": "fulll47", "status": "real_active", "ts": now})
        ok, owner, reason = _sync_any_real_owner_active()
        assert ok
        assert owner == "fulll47"
        assert reason in ("token_actual", "sync_round_state")

        wr(os.path.join(base_dir, "sync_round", "state.json"), {"real_global": True, "real_owner": "fulll47", "status": "real_active", "ts": now - 9999})
        wt("REAL:none")
        ok, owner, reason = _sync_any_real_owner_active(); assert not ok

        print("SELFTEST SYNC_DEMO_HOLD_GLOBAL OK")
        raise SystemExit(0)
    finally:
        if old_script_dir is not None:
            globals()["script_dir"] = old_script_dir
        if old_token is not None:
            globals()["ARCHIVO_TOKEN"] = old_token
        if old_sync_state is not None:
            globals()["SYNC_ROUND_STATE"] = old_sync_state




SYNC_WAIT_POLL_S = 0.15
SYNC_WAIT_HEARTBEAT_S = 1.0
SYNC_WAIT_STALE_S = 45.0
SYNC_WAIT_MAX_IDLE_S = 60.0
SYNC_WAIT_ABSOLUTE_MAX_S = 90.0
SYNC_STANDBY_PRINT_COOLDOWN_S = 3.0


def _sync_round_state_ts(st: dict) -> float:
    if not isinstance(st, dict):
        return 0.0
    for key in ("ts", "updated_ts", "updated_at", "last_seen_ts", "started_at"):
        try:
            val = float(st.get(key, 0.0) or 0.0)
        except Exception:
            continue
        if val > 0:
            return val
    return 0.0


async def _sync_round_wait_release(round_id: int) -> int:
    
    def _sync_bot_es_owner_real() -> bool:
        try:
            tok = str(leer_token_actual() or "").strip().upper()
        except Exception:
            tok = ""

        expected = f"REAL:{str(NOMBRE_BOT).strip().upper()}"
        return tok == expected

    rid = max(1, int(round_id or 1))
    next_round = rid + 1
    print(Fore.YELLOW + Style.BRIGHT + f"⏸️ LXV_SYNC_COLUMN standby columna: {NOMBRE_BOT} ronda #{rid} esperando liberación #{next_round}...")
    estado_bot["sync_wait"] = True
    wait_start_ts = time.time()
    last_hb_ts = 0.0
    last_progress_ts = wait_start_ts
    last_released = None
    last_state_ts = 0.0
    first_wait_tick = True
    last_standby_print_ts = 0.0
    last_global_hold_print_ts = 0.0
    while not stop_event.is_set():
        st = _sync_round_safe_read_json(SYNC_ROUND_STATE) or {}
        try:
            released = int(st.get("released_round", 1) or 1)
        except Exception:
            released = 1
        state_ts = _sync_round_state_ts(st)
        now_ts = time.time()
        after_real_release = st.get("after_real_release") if isinstance(st, dict) else None
        if isinstance(after_real_release, dict) and bool(after_real_release.get("active", False)):
            try:
                after_released_to = int(after_real_release.get("released_to", 0) or 0)
            except Exception:
                after_released_to = 0
            if after_released_to >= next_round:
                official_round = _sync_adopt_official_round(after_released_to, next_round, rid, motivo="after_real_release")
                estado_bot["sync_wait"] = False
                estado_bot["sync_round_id"] = official_round
                try:
                    _sync_round_write_release_heartbeat(rid, official_round)
                except Exception:
                    pass
                print(Fore.GREEN + Style.BRIGHT + f"🔓 REAL_CLOSED_RELEASE_ALL detectado: {NOMBRE_BOT} sale de standby → ronda oficial #{official_round}")
                return official_round
        rollback_detected = bool(rid > 10 and next_round > 10 and released <= 1)
        if rollback_detected:
            if (now_ts - float(globals().get("_SYNC_STATE_ROLLBACK_LOG_TS", 0.0) or 0.0)) >= 8.0:
                globals()["_SYNC_STATE_ROLLBACK_LOG_TS"] = now_ts
                print(Fore.YELLOW + Style.BRIGHT + f"⚠️ SYNC_STATE_ROLLBACK_DETECTED: bot={NOMBRE_BOT} | rid={rid} | released={released} | esperando={next_round}")
            try:
                _sync_round_write_recovery_request(
                    bot=NOMBRE_BOT,
                    round_id=rid,
                    next_round=next_round,
                    released=released,
                    reason="released_round_rollback_detected",
                    any_real_active=False,
                    any_real_owner="",
                    any_real_reason="released_round_rollback_detected",
                )
            except Exception:
                pass
            try:
                _sync_round_write_wait_heartbeat(rid, next_round)
            except Exception:
                pass
            await asyncio.sleep(1.0)
            continue
        if first_wait_tick or (released != last_released) or (state_ts > 0 and state_ts != last_state_ts):
            last_progress_ts = now_ts
        should_write = first_wait_tick or (released != last_released) or ((now_ts - last_hb_ts) >= SYNC_WAIT_HEARTBEAT_S)
        if _sync_bot_es_owner_real():
            estado_bot["sync_wait"] = False
            try:
                _sync_round_write_release_heartbeat(rid, next_round)
            except Exception:
                pass
            print(
                Fore.GREEN + Style.BRIGHT +
                f"🔓 SYNC REAL OWNER: {NOMBRE_BOT} sale de standby para continuar REAL en ronda #{rid}; DEMO sigue en HOLD."
            )
            return rid
        if released >= next_round:
            official_round = _sync_adopt_official_round(released, next_round, rid, motivo="normal_release")
            estado_bot["sync_wait"] = False
            estado_bot["sync_round_id"] = official_round
            _sync_round_write_release_heartbeat(rid, official_round)
            print(Fore.GREEN + f"🔓 LXV_SYNC_COLUMN liberación detectada: ronda oficial #{official_round} (bot {NOMBRE_BOT})")
            print(Fore.GREEN + Style.BRIGHT + f"▶️ LXV_SYNC_COLUMN salida standby: {NOMBRE_BOT} → ronda oficial #{official_round}")
            return official_round
        stale_state = (state_ts > 0) and ((now_ts - state_ts) >= SYNC_WAIT_STALE_S)
        idle_s = now_ts - last_progress_ts
        total_wait_s = now_ts - wait_start_ts
        owner_real = False
        try:
            owner_real = bool(_sync_bot_es_owner_real())
        except Exception:
            owner_real = False
        pending_contract_resolution = bool(estado_bot.get("pending_contract_resolution", False))
        contrato_pendiente = bool(
            estado_bot.get("contrato_pendiente", False)
            or estado_bot.get("pending_contract", False)
            or estado_bot.get("contract_pending", False)
        )
        en_modo_real = False
        try:
            en_modo_real = str(estado_bot.get("tipo_cuenta", "")).upper() == "REAL" or str(estado_bot.get("modo", "")).upper() == "REAL"
        except Exception:
            en_modo_real = False
        if idle_s >= SYNC_WAIT_MAX_IDLE_S:
            print(Fore.YELLOW + f"⚠️ LXV_SYNC_COLUMN no_progress: {NOMBRE_BOT} ronda #{rid} released={released} esperando #{next_round}")
            last_progress_ts = now_ts
        if total_wait_s >= SYNC_WAIT_ABSOLUTE_MAX_S and (not owner_real) and (not pending_contract_resolution) and (not contrato_pendiente) and (not en_modo_real):
            any_real_active, any_real_owner, any_real_reason = _sync_any_real_owner_active()
            if any_real_active:
                if (now_ts - last_global_hold_print_ts) >= 10.0:
                    print(
                        Fore.YELLOW + Style.BRIGHT +
                        f"⏳ SYNC DEMO HOLD GLOBAL:\n"
                        f"bot={NOMBRE_BOT}\n"
                        f"ronda={rid}\n"
                        f"espera_release={next_round}\n"
                        f"released={released}\n"
                        f"owner_real={any_real_owner or '--'}\n"
                        f"motivo={any_real_reason or '--'}\n"
                        f"acción=no_escape_demo"
                    )
                    last_global_hold_print_ts = now_ts
                try:
                    _sync_round_write_wait_heartbeat(rid, next_round)
                    last_hb_ts = now_ts
                    last_released = released
                    last_state_ts = state_ts
                    first_wait_tick = False
                except Exception:
                    pass
                last_progress_ts = now_ts
                await asyncio.sleep(1.0)
                continue
            if released < next_round:
                try:
                    _sync_round_write_wait_heartbeat(rid, next_round)
                    last_hb_ts = now_ts
                    last_released = released
                    last_state_ts = state_ts
                    first_wait_tick = False
                except Exception:
                    pass
                _sync_round_write_recovery_request(
                    bot=NOMBRE_BOT,
                    round_id=rid,
                    next_round=next_round,
                    released=released,
                    reason="demo_wait_timeout_no_release",
                    any_real_active=any_real_active,
                    any_real_owner=any_real_owner,
                    any_real_reason=any_real_reason,
                )
                if (now_ts - last_global_hold_print_ts) >= 10.0:
                    print(
                        Fore.YELLOW + Style.BRIGHT +
                        f"⏳ SYNC DEMO HOLD RECOVERY:\n"
                        f"{NOMBRE_BOT} espera released_round >= {next_round};\n"
                        f"actual={released};\n"
                        f"recovery_request=SI;\n"
                        f"real_global={'SI' if any_real_active else 'NO'};\n"
                        f"owner={any_real_owner or '--'};\n"
                        f"no compra DEMO."
                    )
                    last_global_hold_print_ts = now_ts
                last_progress_ts = now_ts
                first_wait_tick = False
                await asyncio.sleep(1.0)
                continue
            estado_bot["sync_wait"] = False
            try:
                _sync_round_write_release_heartbeat(rid, next_round)
            except Exception:
                pass
            return next_round
        if (idle_s >= SYNC_WAIT_MAX_IDLE_S) and (
            stale_state or total_wait_s >= (SYNC_WAIT_STALE_S + SYNC_WAIT_MAX_IDLE_S)
        ):
            if _sync_bot_es_owner_real():
                estado_bot["sync_wait"] = False
                try:
                    _sync_round_write_release_heartbeat(rid, next_round)
                except Exception:
                    pass
                print(
                    Fore.GREEN + Style.BRIGHT +
                    f"🔓 SYNC REAL OWNER watchdog: {NOMBRE_BOT} continúa REAL en ronda #{rid}."
                )
                return rid

            print(
                Fore.YELLOW +
                f"⏳ SYNC watchdog: sigo esperando liberación real de #{next_round}; "
                f"no opero de nuevo en ronda #{rid}."
            )
            print(
                Fore.YELLOW +
                f"⏳ SYNC DEMO HOLD: {NOMBRE_BOT} espera released_round >= {next_round}; actual={released}; no recompra ronda {rid}"
            )
            last_progress_ts = time.time()
            first_wait_tick = True
            await asyncio.sleep(1.0)
            continue
        if should_write:
            _sync_round_write_wait_heartbeat(rid, next_round)
            last_hb_ts = now_ts
            last_released = released
            last_state_ts = state_ts
            first_wait_tick = False
        if _manual_real_order_targets_this_bot():
            estado_bot["sync_wait"] = False
            try:
                _sync_round_write_release_heartbeat(rid, next_round)
            except Exception:
                pass
            print(
                Fore.GREEN + Style.BRIGHT +
                f"🟢 MANUAL REAL override: {NOMBRE_BOT} sale de standby sync ronda #{rid} → #{next_round}."
                + Style.RESET_ALL
            )
            return next_round
        now_print = time.time()
        if (now_print - last_standby_print_ts) >= float(globals().get("SYNC_STANDBY_PRINT_COOLDOWN_S", 10.0)):
            last_standby_print_ts = now_print
            print(Fore.CYAN + f"… standby columna {NOMBRE_BOT}: ronda #{rid}, released_round={released}")
            last_standby_print_ts = now_ts
        await asyncio.sleep(SYNC_WAIT_POLL_S)
    estado_bot["sync_wait"] = False
    _sync_round_write_release_heartbeat(rid, next_round)
    return rid
# === /LXV_SYNC_COLUMN ===

# ✅ Asegura carpeta de órdenes (evita rarezas si el maestro aún no la creó)
try:
    os.makedirs(ORDEN_DIR, exist_ok=True)
except Exception:
    pass


_ORDER_REAL_OLD_WARN_LAST_TS = 0.0


def _orden_real_payload_vivo(data: dict | None) -> bool:
    try:
        d = data if isinstance(data, dict) else {}
        ts = float(d.get("created_ts") or d.get("ts") or 0.0)
        ttl = float(d.get("ttl_s") or d.get("ttl") or 45.0)
        if ts <= 0:
            return False
        return (time.time() - ts) <= max(1.0, ttl)
    except Exception:
        return False




def _orden_real_candidate_paths():
    paths = []
    try:
        orden_dir = globals().get("ORDEN_DIR", "orden_real")
        bot = str(globals().get("NOMBRE_BOT", "") or "").strip()
        if bot:
            paths.append(os.path.join(orden_dir, f"{bot}.json"))
    except Exception:
        pass
    paths.append("orden_real.json")
    return paths

def _manual_real_order_targets_this_bot() -> bool:
    try:
        payload = None
        for p in _orden_real_candidate_paths():
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                cand = json.load(f)
            if isinstance(cand, dict):
                payload = cand
                break
        if not isinstance(payload, dict):
            return False
        if not isinstance(payload, dict):
            return False

        bot = str(payload.get("bot") or payload.get("owner") or "").strip()
        source = str(payload.get("source") or "").upper().strip()
        manual_override = bool(payload.get("manual_override") or payload.get("force_exit_sync_wait"))

        if bot != NOMBRE_BOT:
            return False
        if source != "MANUAL":
            return False
        if not manual_override:
            return False

        ts = float(payload.get("created_ts") or payload.get("ts") or 0.0)
        ttl = float(payload.get("ttl_s") or 45.0)
        if ts <= 0:
            return False
        if (time.time() - ts) > ttl:
            return False

        try:
            tok = leer_token_actual()
            tok_s = str(tok or "").strip()
        except Exception:
            return False
        tok_u = tok_s.upper()
        if tok_u not in (NOMBRE_BOT.upper(), f"REAL:{NOMBRE_BOT}".upper()):
            return False

        return True
    except Exception:
        return False
def _warn_stale_real_order_cooldown():
    global _ORDER_REAL_OLD_WARN_LAST_TS
    try:
        now = float(time.time())
        if (now - float(_ORDER_REAL_OLD_WARN_LAST_TS or 0.0)) >= 6.0:
            _ORDER_REAL_OLD_WARN_LAST_TS = now
            print(Fore.YELLOW + Style.BRIGHT + "⏱️ Orden REAL vieja ignorada por TTL." + Style.RESET_ALL)
    except Exception:
        pass




def _safe_int_cycle(value, default=None):
    try:
        c = int(value)
        if 1 <= c <= len(MARTINGALA_REAL):
            return c
    except Exception:
        pass
    return default


def _leer_json_seguro(path):
    try:
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _orden_real_path_local():
    try:
        candidates = _orden_real_candidate_paths() if "_orden_real_candidate_paths" in globals() else []
        for p in candidates:
            if p:
                return p
    except Exception:
        pass
    return os.path.join(globals().get("ORDEN_DIR", "orden_real"), f"{NOMBRE_BOT}.json")


def _leer_orden_real_viva_para_bot():
    return _leer_json_seguro(_orden_real_path_local())


def _leer_owner_state_vivo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return _leer_json_seguro(os.path.join(base_dir, "real_owner_state.json"))


def _orden_real_bot_ok(orden, ciclo_esperado=None, max_age_s=90):
    try:
        if not isinstance(orden, dict) or not orden:
            return False, None, "orden_missing"
        bot_ord = str(orden.get("bot") or orden.get("owner") or "").strip()
        owner_ord = str(orden.get("owner") or orden.get("bot") or "").strip()
        if bot_ord != NOMBRE_BOT or owner_ord != NOMBRE_BOT:
            return False, None, "bot_owner_mismatch"
        campos = ["ciclo", "ciclo_orden", "ciclo_forzado", "marti_ciclo", "ciclo_real_oficial"]
        ciclos = []
        for k in campos:
            if k in orden:
                c = _safe_int_cycle(orden.get(k), None)
                if c is None:
                    return False, None, f"{k}_invalid"
                ciclos.append(c)
        if not ciclos:
            return False, None, "sin_ciclo"
        if len(set(ciclos)) != 1:
            return False, None, "ciclos_no_coinciden"
        ciclo_orden = ciclos[0]
        if ciclo_esperado is not None:
            ce = _safe_int_cycle(ciclo_esperado, None)
            if ce is None or ciclo_orden != ce:
                return False, ciclo_orden, "ciclo_esperado_mismatch"
        order_id = str(orden.get("order_id") or "").strip()
        if not order_id:
            return False, ciclo_orden, "order_id_missing"
        now = time.time()
        try:
            ts = float(orden.get("ts", orden.get("created_ts", 0)) or 0)
        except Exception:
            ts = 0
        try:
            ttl = float(orden.get("ttl_s", max_age_s) or max_age_s)
        except Exception:
            ttl = max_age_s
        max_age = max(3.0, min(float(max_age_s), ttl))
        if ts <= 0 or (now - ts) > max_age:
            return False, ciclo_orden, "orden_vencida"
        if bool(orden.get("consumed", False)):
            return False, ciclo_orden, "orden_consumida"
        return True, ciclo_orden, "ok"
    except Exception as e:
        return False, None, f"orden_err:{e}"


def _owner_state_confirma_ciclo(ciclo_esperado, max_age_s=120):
    try:
        data = _leer_owner_state_vivo()
        if not isinstance(data, dict) or not data:
            return False
        owner = str(data.get("owner_bot") or data.get("bot") or "").strip()
        if owner != NOMBRE_BOT:
            return False
        c = _safe_int_cycle(data.get("ciclo"), None)
        if c != int(ciclo_esperado):
            return False
        ts = float(data.get("assigned_ts", data.get("ts", 0)) or 0)
        if ts <= 0 or (time.time() - ts) > float(max_age_s):
            return False
        return True
    except Exception:
        return False


def _validar_pre_buy_real(ciclo_local):
    try:
        token_actual = leer_token_actual() if "leer_token_actual" in globals() else ""
    except Exception:
        token_actual = ""
    if str(token_actual).strip() != f"REAL:{NOMBRE_BOT}":
        return False, f"token={token_actual}"
    orden = _leer_orden_real_viva_para_bot()
    ok, ciclo_orden, motivo = _orden_real_bot_ok(orden, ciclo_esperado=ciclo_local)
    if not ok:
        return False, f"orden={ciclo_orden} motivo={motivo}"
    owner = _leer_owner_state_vivo()
    if isinstance(owner, dict) and owner:
        try:
            owner_bot = str(owner.get("owner_bot") or owner.get("bot") or "").strip()
            owner_ciclo = _safe_int_cycle(owner.get("ciclo"), None)
            owner_ts = float(owner.get("assigned_ts", owner.get("ts", 0)) or 0)
            if owner_ts > 0 and (time.time() - owner_ts) <= 120:
                if owner_bot != NOMBRE_BOT or owner_ciclo != int(ciclo_local):
                    return False, f"owner_state=C{owner_ciclo} owner={owner_bot}"
        except Exception:
            return False, "owner_state_err"
    return True, "ok"

def leer_orden_real(bot: str):
    """
    Devuelve (ciclo, ts, quiet, src) si existe orden fresca, o (None, None, 0, None) si no.
    """
    try:
        for ruta in _orden_real_candidate_paths():
            tmp = ruta + ".tmp"
            if not os.path.exists(ruta):
                continue
            with open(ruta, "r", encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as t:
                t.write(f.read())
            with open(tmp, "r", encoding="utf-8") as f:
                data = json.load(f)
            os.remove(tmp)
            payload_bot = str(data.get("bot") or "").strip()
            is_fallback = os.path.basename(ruta) == "orden_real.json"
            if payload_bot:
                if payload_bot != bot:
                    continue
            elif not is_fallback:
                continue
            if bool(data.get("consumed", False)):
                return None, None, 0, None
            raw_ciclo = (
                data.get("ciclo")
                or data.get("ciclo_orden")
                or data.get("ciclo_forzado")
                or data.get("marti_ciclo")
                or 1
            )
            cyc = int(raw_ciclo)
            ts = float(data.get("created_ts") or data.get("ts") or 0.0)
            ttl = float(data.get("ttl_s") or data.get("ttl") or 45.0)
            quiet = 1 if int(data.get("quiet", 0)) == 1 else 0
            src = str(data.get("source") or data.get("src") or "").upper() or None
            lim = max(1.0, ttl)
            if (not _orden_real_payload_vivo(data)) or (time.time() - ts > lim):
                _warn_stale_real_order_cooldown()
                return None, None, 0, None
            ciclo_norm = max(1, min(cyc, MAX_CICLOS))
            if cyc != ciclo_norm:
                print(Fore.YELLOW + f"⚠️ ciclo>MAX_CICLOS detectado, normalizado a C{ciclo_norm}")
            return ciclo_norm, ts, quiet, src
        return None, None, 0, None
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None, None, 0, None

# <<< PATCH 1

# >>> PATCH: WS robusto
WS_KW = dict(ping_interval=15, ping_timeout=10, close_timeout=5, max_queue=None)
# <<< PATCH

# >>> PATCH (cerca de tus globals) BLOQUE 10
MODO_SILENCIOSO = False
_last_log = {}

def _print_once(key: str, ttl: float = 25.0) -> bool:
    now = time.time()
    exp = _last_log.get(key, 0)
    if now < exp:
        return False
    _last_log[key] = now + ttl
    return True


BUY_INSUFFICIENT_BALANCE_PAUSE_S = 120.0
BUY_RATE_LIMIT_PAUSE_S = 45.0
BUY_INSUFFICIENT_BALANCE_MAX_FAST_RETRIES = 1

def _api_error_text(exc) -> str:
    try:
        return str(exc or "")
    except Exception:
        return ""

def _is_insufficient_balance_error(exc) -> bool:
    txt = _api_error_text(exc).lower()
    return (
        "insufficientbalance" in txt
        or "insufficient balance" in txt
        or "balance (0.00" in txt
        or "account balance" in txt and "insufficient" in txt
    )

def _is_rate_limit_error(exc) -> bool:
    txt = _api_error_text(exc).lower()
    return (
        "ratelimit" in txt
        or "rate limit" in txt
        or "ticks_history" in txt and "limit" in txt
    )

def _set_buy_pause(reason: str, seconds: float):
    try:
        until = time.time() + max(5.0, float(seconds or 0.0))
        estado_bot["buy_pause_until"] = until
        estado_bot["buy_pause_reason"] = str(reason or "buy_pause")
        return until
    except Exception:
        return time.time() + 30.0

async def _esperar_buy_pause_si_activa(contexto=""):
    try:
        until = float(estado_bot.get("buy_pause_until", 0.0) or 0.0)
        if until <= time.time():
            return False

        remaining = max(0.0, until - time.time())
        reason = str(estado_bot.get("buy_pause_reason", "buy_pause") or "buy_pause")

        if _print_once(f"buy_pause:{reason}", ttl=10.0):
            print(
                Fore.YELLOW + Style.BRIGHT +
                f"⏸️ BUY_PAUSE activa ({reason}) | {remaining:.0f}s restantes | {contexto} | ciclo se conserva"
            )

        await asyncio.sleep(min(remaining, 5.0))
        return True
    except Exception:
        return False

async def _manejar_buy_insufficient_balance(api_e, modo_real: bool, monto, ciclo):
    """
    Manejo seguro de InsufficientBalance.

    REGLAS:
    - No marcar GANANCIA.
    - No marcar PÉRDIDA.
    - No marcar INDEFINIDO.
    - No avanzar Martingala.
    - No escribir resultado en CSV.
    - No emitir ACK de cierre.
    - No tocar token_actual.txt.
    - No liberar real.lock.
    - Solo pausar compras y conservar el mismo ciclo.
    """
    try:
        modo_txt = "REAL" if modo_real else "DEMO"
        estado_bot["ciclo_forzado"] = ciclo
        estado_bot["token_msg_mostrado"] = False

        key = "insufficient_balance_real" if modo_real else "insufficient_balance_demo"
        until = _set_buy_pause(key, BUY_INSUFFICIENT_BALANCE_PAUSE_S)

        print(
            Fore.RED + Style.BRIGHT +
            f"⛔ SALDO_INSUFICIENTE_{modo_txt}: no se compró contrato | monto={monto} | "
            f"ciclo C{ciclo} conservado | pausa={BUY_INSUFFICIENT_BALANCE_PAUSE_S:.0f}s"
        )
        print(
            Fore.YELLOW + Style.BRIGHT +
            "🛡️ No se registra resultado, no avanza Martingala y no se emite ACK de cierre."
        )

        if not modo_real:
            print(
                Fore.YELLOW + Style.BRIGHT +
                "🔎 Revisar TOKEN DEMO / cuenta DEMO: Deriv reporta saldo 0.00 USD."
            )
        else:
            print(
                Fore.RED + Style.BRIGHT +
                "🚨 REAL sin saldo suficiente: no hubo contrato. Mantener ciclo y esperar control del maestro."
            )

        await asyncio.sleep(min(10.0, max(1.0, until - time.time())))
        return True
    except Exception:
        await asyncio.sleep(15.0)
        return True

def _selftest_buy_error_classifier():
    assert _is_insufficient_balance_error("API error: InsufficientBalance - Your account balance (0.00 USD) is insufficient to buy this contract (1.00 USD)")
    assert _is_insufficient_balance_error(RuntimeError("InsufficientBalance"))
    assert not _is_insufficient_balance_error("API error: RateLimit - You have reached the rate limit for ticks_history.")
    assert _is_rate_limit_error("API error: RateLimit - You have reached the rate limit for ticks_history.")
    assert _is_rate_limit_error(RuntimeError("rate limit"))
    print(f"✅ SELFTEST BUY_ERROR_CLASSIFIER OK | {NOMBRE_BOT}")

if os.environ.get("RUN_BUY_ERROR_CLASSIFIER_SELFTEST") == "1":
    _selftest_buy_error_classifier()
    raise SystemExit(0)

async def _desactivar_silencioso_en(seg=90):
    await asyncio.sleep(seg)
    global MODO_SILENCIOSO
    MODO_SILENCIOSO = False

async def _silencio_temporal(seg=90, fuente=None):
    global MODO_SILENCIOSO
    MODO_SILENCIOSO = True
    estado_bot["modo_manual"] = (str(fuente).upper() == "MANUAL")
    try:
        await asyncio.sleep(seg)
    finally:
        MODO_SILENCIOSO = False
        estado_bot["modo_manual"] = False

# <<< PATCH

# >>> PATCH (globals) BLOQUE 3
_contratos_procesados = set()
# <<< PATCH

# >>> PATCH (globals) BLOQUE 3 y BLOQUE 4
csv_lock = asyncio.Lock()
# <<< PATCH

# >>> PATCH: cooldown antirrebote BLOQUE 2 y 9
COOLDOWN_REAL_S = 12
# <<< PATCH

# >>> PATCH BLOQUE 4 y 8
REFRESCO_SALDO = 12
_last_saldo_ts = 0.0
PENDING_CONTRACT_FENCE_S = 35.0
INCIDENT_LOCK_RECOVERY_ATTEMPTS = 5
INCIDENT_LOCK_MAX_AGE_S = PENDING_CONTRACT_FENCE_S


def _set_pending_contract_resolution(round_id: int, contract_id=None, reason: str = "", token_usado=None, asset=None, direction=None, ciclo=None, **ctx):
    estado_bot["pending_contract_resolution"] = True
    estado_bot["pending_contract_id"] = contract_id
    estado_bot["pending_since_ts"] = float(time.time())
    estado_bot["pending_round_id"] = int(round_id or estado_bot.get("sync_round_id", 1) or 1)
    estado_bot["pending_reason"] = str(reason or "")
    estado_bot["pending_attempts"] = 0
    if token_usado is not None:
        estado_bot["pending_token_usado"] = token_usado
    elif estado_bot.get("pending_token_usado") is None:
        estado_bot["pending_token_usado"] = estado_bot.get("ultimo_token_usado") or None
    if asset is not None:
        estado_bot["pending_asset"] = asset
    if direction is not None:
        estado_bot["pending_direction"] = direction
    if ciclo is not None:
        estado_bot["pending_ciclo"] = ciclo
    elif estado_bot.get("pending_ciclo") is None:
        estado_bot["pending_ciclo"] = estado_bot.get("ciclo_actual") or 1
    if isinstance(ctx, dict) and ctx:
        for k, v in ctx.items():
            if str(k).startswith("pending_"):
                estado_bot[str(k)] = v
            else:
                estado_bot[f"pending_{k}"] = v
    if _print_once("pending-contract-set", ttl=6.0):
        cid_txt = f" contract_id={contract_id}" if contract_id is not None else ""
        reason_txt = reason or "unknown"
        print(Fore.YELLOW + Style.BRIGHT + f"🧱 Fence contrato incierto activado.{cid_txt} reason={reason_txt}")


def _clear_pending_contract_resolution(reason: str = "", resultado_final=None):
    if estado_bot.get("pending_contract_resolution") and _print_once("pending-contract-clear", ttl=6.0):
        print(Fore.GREEN + f"✅ Fence contrato incierto liberado ({reason or 'resolved'}).")
    estado_bot["pending_contract_resolution"] = False
    estado_bot["pending_contract_id"] = None
    estado_bot["pending_since_ts"] = 0.0
    estado_bot["pending_round_id"] = None
    estado_bot["pending_reason"] = ""
    estado_bot["pending_attempts"] = 0
    estado_bot["pending_token_usado"] = None
    estado_bot["pending_asset"] = None
    estado_bot["pending_direction"] = None
    estado_bot["pending_ciclo"] = None


def _emitir_sync_recovery_incident_demo_neutral(round_id, contract_id, ciclo, asset, direction, attempts, edad):
    try:
        rid = int(round_id or 0)
        cid = int(contract_id or 0)
        if rid <= 0 or cid <= 0:
            return False
        key = f"incident-demo-stale-open:{rid}:{cid}"
        if globals().get("_LAST_INCIDENT_DEMO_STALE_OPEN_KEY") == key:
            return True
        st = _sync_round_safe_read_json(SYNC_ROUND_STATE) or {}
        try:
            released = int(st.get("released_round", rid) or rid)
        except Exception:
            released = rid
        ok = _sync_round_write_recovery_request(
            bot=NOMBRE_BOT,
            round_id=rid,
            next_round=rid + 1,
            released=released,
            reason="incident_lock_demo_stale_open",
            contract_id=cid,
            ciclo=ciclo,
            asset=asset,
            direction=direction,
            attempts=int(attempts or 0),
            edad=float(edad or 0.0),
            neutral=True,
            no_trade_result=True,
            quarantine=True,
            source="INCIDENT_LOCK_STALE_OPEN_DEMO",
        )
        if ok:
            globals()["_LAST_INCIDENT_DEMO_STALE_OPEN_KEY"] = key
            print(
                Fore.CYAN + Style.BRIGHT +
                f"🧷 INCIDENT_LOCK_SYNC_RECOVERY_REQUEST | bot={NOMBRE_BOT} | ronda=#{rid} | next=#{rid + 1} | "
                f"reason=incident_lock_demo_stale_open | neutral=SI"
            )
        return bool(ok)
    except Exception:
        return False




def _emitir_sync_recovery_incident_demo_sin_evidencia_neutral(round_id, ciclo, asset, direction, attempts, edad):
    try:
        rid = int(round_id or 0)
        if rid <= 0:
            return False
        source = "INCIDENT_LOCK_SIN_EVIDENCIA_DEMO"
        key = f"incident-demo-sin-evidencia:{rid}:{source}:{ciclo}:{asset}:{direction}"
        if globals().get("_LAST_INCIDENT_DEMO_SIN_EVIDENCIA_KEY") == key:
            return True
        st = _sync_round_safe_read_json(SYNC_ROUND_STATE) or {}
        try:
            released = int(st.get("released_round", rid) or rid)
        except Exception:
            released = rid
        ok = _sync_round_write_recovery_request(
            bot=NOMBRE_BOT,
            round_id=rid,
            next_round=rid + 1,
            released=released,
            reason="incident_lock_demo_sin_evidencia",
            contract_id=None,
            ciclo=ciclo,
            asset=asset,
            direction=direction,
            attempts=int(attempts or 0),
            edad=float(edad or 0.0),
            neutral=True,
            no_trade_result=True,
            quarantine=True,
            source=source,
        )
        if ok:
            globals()["_LAST_INCIDENT_DEMO_SIN_EVIDENCIA_KEY"] = key
            print(
                Fore.CYAN + Style.BRIGHT +
                f"🧷 INCIDENT_LOCK_SYNC_RECOVERY_REQUEST | bot={NOMBRE_BOT} | ronda=#{rid} | next=#{rid + 1} | "
                f"reason=incident_lock_demo_sin_evidencia | neutral=SI"
            )
        return bool(ok)
    except Exception:
        return False


def _emitir_sync_recovery_incident_demo_closed_sin_profit_neutral(round_id, contract_id, ciclo, asset, direction, attempts, edad):
    try:
        rid = int(round_id or 0)
        cid = int(contract_id or 0)
        if rid <= 0 or cid <= 0:
            return False
        source = "INCIDENT_LOCK_CLOSED_SIN_PROFIT_DEMO"
        key = f"incident-demo-closed-sin-profit:{rid}:{cid}:{source}"
        if globals().get("_LAST_INCIDENT_DEMO_CLOSED_SIN_PROFIT_KEY") == key:
            return True
        st = _sync_round_safe_read_json(SYNC_ROUND_STATE) or {}
        try:
            released = int(st.get("released_round", rid) or rid)
        except Exception:
            released = rid
        ok = _sync_round_write_recovery_request(
            bot=NOMBRE_BOT,
            round_id=rid,
            next_round=rid + 1,
            released=released,
            reason="incident_lock_demo_closed_sin_profit",
            contract_id=cid,
            ciclo=ciclo,
            asset=asset,
            direction=direction,
            attempts=int(attempts or 0),
            edad=float(edad or 0.0),
            neutral=True,
            no_trade_result=True,
            quarantine=True,
            source=source,
        )
        if ok:
            globals()["_LAST_INCIDENT_DEMO_CLOSED_SIN_PROFIT_KEY"] = key
            print(
                Fore.CYAN + Style.BRIGHT +
                f"🧷 INCIDENT_LOCK_SYNC_RECOVERY_REQUEST | bot={NOMBRE_BOT} | ronda=#{rid} | next=#{rid + 1} | "
                f"reason=incident_lock_demo_closed_sin_profit | neutral=SI"
            )
        return bool(ok)
    except Exception:
        return False

def _emitir_sync_recovery_incident_demo_sold_sin_contexto_neutral(round_id, contract_id, ciclo, asset, direction, attempts, edad):
    try:
        rid = int(round_id or 0)
        cid = int(contract_id or 0)
        if rid <= 0 or cid <= 0:
            return False
        source = "INCIDENT_LOCK_SOLD_SIN_CONTEXTO_DEMO"
        key = f"incident-demo-sold-sin-contexto:{rid}:{cid}:{source}"
        if globals().get("_LAST_INCIDENT_DEMO_SOLD_SIN_CONTEXTO_KEY") == key:
            return True
        st = _sync_round_safe_read_json(SYNC_ROUND_STATE) or {}
        try:
            released = int(st.get("released_round", rid) or rid)
        except Exception:
            released = rid
        ok = _sync_round_write_recovery_request(
            bot=NOMBRE_BOT,
            round_id=rid,
            next_round=rid + 1,
            released=released,
            reason="incident_lock_demo_sold_sin_contexto",
            contract_id=cid,
            ciclo=ciclo,
            asset=asset,
            direction=direction,
            attempts=int(attempts or 0),
            edad=float(edad or 0.0),
            neutral=True,
            no_trade_result=True,
            quarantine=True,
            usable_for_real=False,
            usable_for_lxv=False,
            usable_for_training=False,
            source=source,
        )
        if ok:
            globals()["_LAST_INCIDENT_DEMO_SOLD_SIN_CONTEXTO_KEY"] = key
            print(
                Fore.CYAN + Style.BRIGHT +
                f"🧷 INCIDENT_LOCK_SYNC_RECOVERY_REQUEST | bot={NOMBRE_BOT} | ronda=#{rid} | next=#{rid + 1} | "
                f"reason=incident_lock_demo_sold_sin_contexto | neutral=SI"
            )
        return bool(ok)
    except Exception:
        return False



def _incident_lock_demo_sin_evidencia_permitido(pending_id, pending_round, token_ctx, elapsed, attempts):
    try:
        rid = int(pending_round or 0)
    except Exception:
        return False
    token_is_real = bool(token_ctx == TOKEN_REAL or str(token_ctx or "").strip().upper().startswith("REAL"))
    modo_is_real = bool(str(estado_bot.get("tipo_cuenta", "") or estado_bot.get("modo", "")).strip().upper() == "REAL")
    any_real_active, _owner_real, _real_reason = (True, "", "real_owner_check_failed")
    try:
        any_real_active, _owner_real, _real_reason = _sync_any_real_owner_active()
    except Exception:
        any_real_active = True
    try:
        real_close_pending_raw = globals().get("REAL_CLOSE_PENDING", False)
        real_close_pending_live = any(bool(v) for v in real_close_pending_raw.values()) if isinstance(real_close_pending_raw, dict) else bool(real_close_pending_raw)
    except Exception:
        real_close_pending_live = True
    try:
        real_order_live = bool(callable(globals().get("_manual_real_order_targets_this_bot")) and _manual_real_order_targets_this_bot())
    except Exception:
        real_order_live = True
    return (
        pending_id is None
        and rid > 0
        and not modo_is_real
        and not token_is_real
        and not any_real_active
        and not real_close_pending_live
        and not real_order_live
        and float(elapsed or 0.0) >= float(INCIDENT_LOCK_MAX_AGE_S)
        and int(attempts or 0) >= int(INCIDENT_LOCK_RECOVERY_ATTEMPTS)
    )

def _emitir_ack_sync_incident_demo_resuelto(round_id, resultado, contract_id=None, asset=None, ciclo=None, token_usado=None):
    """ACK DEMO específico para INCIDENT_LOCK resuelto; nunca emite ACK para token REAL."""
    try:
        res = str(resultado or "").strip().upper().replace("PERDIDA", "PÉRDIDA")
        token_is_real = bool(token_usado == TOKEN_REAL or str(token_usado or "").strip().upper().startswith("REAL"))
        if token_is_real or res not in ("GANANCIA", "PÉRDIDA"):
            return False
        return bool(_sync_round_emit_close_ack_confirmado(round_id, res, contract_id=contract_id, asset=asset, ciclo=ciclo, modo_real_contrato=False))
    except Exception:
        return False


def _incident_lock_demo_stale_open_permitido(poc, pending_id, pending_round, token_ctx, elapsed, attempts):
    try:
        cid = int(pending_id or 0)
        rid = int(pending_round or 0)
    except Exception:
        return False
    token_is_real = bool(token_ctx == TOKEN_REAL or str(token_ctx or "").strip().upper().startswith("REAL"))
    modo_is_real = bool(str(estado_bot.get("tipo_cuenta", "") or estado_bot.get("modo", "")).strip().upper() == "REAL")
    profit_final = poc.get("profit") if isinstance(poc, dict) else None
    resultado_txt = str((poc or {}).get("status") or (poc or {}).get("result") or (poc or {}).get("resultado") or "").strip().upper()
    has_trade_result = resultado_txt in ("GANANCIA", "PÉRDIDA", "PERDIDA", "WIN", "LOSS")
    any_real_active, _owner_real, _real_reason = (True, "", "real_owner_check_failed")
    try:
        any_real_active, _owner_real, _real_reason = _sync_any_real_owner_active()
    except Exception:
        any_real_active = True
    try:
        real_close_pending_raw = globals().get("REAL_CLOSE_PENDING", False)
        real_close_pending_live = any(bool(v) for v in real_close_pending_raw.values()) if isinstance(real_close_pending_raw, dict) else bool(real_close_pending_raw)
    except Exception:
        real_close_pending_live = True
    try:
        real_order_live = bool(callable(globals().get("_manual_real_order_targets_this_bot")) and _manual_real_order_targets_this_bot())
    except Exception:
        real_order_live = True
    return (
        cid > 0
        and rid > 0
        and not modo_is_real
        and not token_is_real
        and not any_real_active
        and not real_close_pending_live
        and not real_order_live
        and profit_final in (None, "")
        and not has_trade_result
        and not bool((poc or {}).get("is_sold", False))
        and float(elapsed or 0.0) >= float(INCIDENT_LOCK_MAX_AGE_S)
        and int(attempts or 0) >= int(INCIDENT_LOCK_RECOVERY_ATTEMPTS)
    )


def _incident_lock_demo_closed_sin_profit_permitido(contract_id, pending_round, token_ctx, elapsed, attempts):
    try:
        cid = int(contract_id or 0)
        rid = int(pending_round or 0)
    except Exception:
        return False
    token_is_real = bool(token_ctx == TOKEN_REAL or str(token_ctx or "").strip().upper().startswith("REAL"))
    modo_is_real = bool(str(estado_bot.get("tipo_cuenta", "") or estado_bot.get("modo", "")).strip().upper() == "REAL")
    any_real_active, _owner_real, _real_reason = (True, "", "real_owner_check_failed")
    try:
        any_real_active, _owner_real, _real_reason = _sync_any_real_owner_active()
    except Exception:
        any_real_active = True
    try:
        real_close_pending_raw = globals().get("REAL_CLOSE_PENDING", False)
        real_close_pending_live = any(bool(v) for v in real_close_pending_raw.values()) if isinstance(real_close_pending_raw, dict) else bool(real_close_pending_raw)
    except Exception:
        real_close_pending_live = True
    try:
        real_order_live = bool(callable(globals().get("_manual_real_order_targets_this_bot")) and _manual_real_order_targets_this_bot())
    except Exception:
        real_order_live = True
    return (
        cid > 0
        and rid > 0
        and not modo_is_real
        and not token_is_real
        and not any_real_active
        and not real_close_pending_live
        and not real_order_live
        and float(elapsed or 0.0) >= float(INCIDENT_LOCK_MAX_AGE_S)
        and int(attempts or 0) >= int(INCIDENT_LOCK_RECOVERY_ATTEMPTS)
    )


def _incident_lock_profit_confiable(*items):
    def _num(v):
        if v in (None, ""):
            return None
        try:
            return float(v)
        except Exception:
            return None

    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("profit", "profit_loss"):
            if key in item:
                val = _num(item.get(key))
                if val is not None:
                    return val, key
        sell = _num(item.get("sell_price"))
        buy = _num(item.get("buy_price"))
        if sell is not None and buy is not None:
            return sell - buy, "sell_price-buy_price"
    return None, None


async def buscar_contrato_reciente_deriv_seguro(ws, token_usado=None, contract_id=None, asset=None, direction=None, round_id=None):
    """Busca un contrato pendiente/reciente en Deriv sin inventar resultado."""
    async def _call(payload, expect=None, timeout=5.0):
        if ws is not None:
            return await api_call(ws, payload, expect_msg_type=expect, timeout=timeout)
        token = token_usado or estado_bot.get("pending_token_usado") or leer_token_actual()
        async with websockets.connect(DERIV_WS_URL, **WS_KW) as ws_tmp:
            await authorize_ws(ws_tmp, token, tries=2, timeout=6.0)
            return await api_call(ws_tmp, payload, expect_msg_type=expect, timeout=timeout)

    async def _consultar_poc(cid, source_hint="proposal_open_contract", row=None, row_is_open=None):
        try:
            cid_int = int(cid or 0)
        except Exception:
            return None
        if cid_int <= 0:
            return None
        poc = {}
        try:
            data = await _call({"proposal_open_contract": 1, "contract_id": cid_int}, "proposal_open_contract")
            poc = data.get("proposal_open_contract", {}) if isinstance(data, dict) else {}
        except Exception as e:
            if _print_once(f"incident-lock-poc-search-error-{source_hint}", ttl=12.0):
                print(Fore.YELLOW + f"INCIDENT_LOCK búsqueda proposal_open_contract({source_hint}) falló: {type(e).__name__}: {e}")
        profit, profit_source = _incident_lock_profit_confiable(poc, row)
        if poc:
            is_closed = bool(poc.get("is_sold", False))
            is_open = not is_closed
        else:
            is_open = bool(row_is_open) if row_is_open is not None else False
            is_closed = not is_open
        if poc and profit_source and "profit" not in poc:
            poc = dict(poc)
            poc["profit"] = profit
        return {
            "found": True,
            "source": source_hint,
            "contract_id": cid_int,
            "row": row,
            "poc": poc,
            "is_open": bool(is_open),
            "is_closed": bool(is_closed),
            "profit": profit,
            "profit_reliable": profit_source is not None,
            "profit_source": profit_source,
        }

    if contract_id is not None:
        direct = await _consultar_poc(contract_id, "proposal_open_contract")
        if direct and direct.get("poc"):
            return direct

    # Sin contract_id: buscar contrato fantasma en fuentes Deriv robustas antes de liberar.
    for source, payload, expect in (
        ("portfolio", {"portfolio": 1}, "portfolio"),
        ("profit_table", {"profit_table": 1, "limit": 10, "description": 1, "sort": "DESC"}, "profit_table"),
        ("statement", {"statement": 1, "limit": 20, "description": 1}, "statement"),
    ):
        try:
            data = await _call(payload, expect, timeout=6.0)
            rows = []
            if source == "portfolio":
                rows = ((data.get("portfolio") or {}).get("contracts") or []) if isinstance(data, dict) else []
            elif source == "profit_table":
                rows = ((data.get("profit_table") or {}).get("transactions") or []) if isinstance(data, dict) else []
            else:
                rows = ((data.get("statement") or {}).get("transactions") or []) if isinstance(data, dict) else []
            for row in rows or []:
                cid = row.get("contract_id") or row.get("id") or row.get("transaction_id")
                if not cid:
                    continue
                symbol_txt = str(row.get("symbol") or row.get("underlying") or row.get("display_name") or "").upper()
                desc_txt = str(row.get("shortcode") or row.get("longcode") or row.get("description") or "").upper()
                if asset and str(asset).upper() not in (symbol_txt + " " + desc_txt):
                    continue
                if direction and str(direction).upper() not in desc_txt:
                    # La dirección no siempre viene normalizada en statement/profit_table; no descartamos portfolio.
                    if source != "portfolio":
                        continue
                is_open = source == "portfolio" or str(row.get("status") or "").lower() == "open"
                enriched = await _consultar_poc(cid, source, row=row, row_is_open=is_open)
                if enriched:
                    return enriched
        except Exception as e:
            if _print_once(f"incident-lock-{source}-search-error", ttl=12.0):
                print(Fore.YELLOW + f"INCIDENT_LOCK búsqueda {source} falló: {type(e).__name__}: {e}")
    return {"found": False, "source": "none"}


async def _procesar_incident_lock_sold(poc, source="proposal_open_contract"):
    cid = estado_bot.get("pending_contract_id") or (poc or {}).get("contract_id")
    asset = estado_bot.get("pending_asset") or (poc or {}).get("symbol") or (poc or {}).get("underlying")
    direction = estado_bot.get("pending_direction") or (poc or {}).get("contract_type") or (poc or {}).get("contract_type_display")
    ciclo = estado_bot.get("pending_ciclo") or estado_bot.get("ciclo_actual") or estado_bot.get("ciclo_forzado") or 1
    token_ctx = estado_bot.get("pending_token_usado") or ultimo_token
    pending_round = estado_bot.get("pending_sync_round_id_original") or estado_bot.get("pending_round_id")
    attempts = int(estado_bot.get("pending_attempts", 0) or 0)
    since = float(estado_bot.get("pending_since_ts", 0.0) or 0.0)
    elapsed = max(0.0, time.time() - since) if since else 0.0

    estado_bot["pending_asset"] = asset
    estado_bot["pending_direction"] = direction
    estado_bot["pending_ciclo"] = ciclo
    estado_bot["pending_token_usado"] = token_ctx

    try:
        profit = float((poc or {}).get("profit"))
    except Exception:
        if _print_once(f"incident-sold-sin-contexto-{NOMBRE_BOT}-{cid}-{pending_round}-profit", ttl=10.0):
            print(Fore.RED + Style.BRIGHT + f"⚠️ INCIDENT_LOCK_SOLD_SIN_CONTEXTO_COMPLETO | bot={NOMBRE_BOT} | contract_id={cid} | falta=profit")
        return False
    resultado = "GANANCIA" if profit > 0 else "PÉRDIDA"
    criticos = {
        "pending_asset": asset,
        "pending_direction": direction,
        "pending_ciclo": ciclo,
        "pending_token_usado": token_ctx,
        "pending_round_id": pending_round,
        "contract_id": cid,
    }
    faltantes = [k for k, v in criticos.items() if v in (None, "", 0)]
    if faltantes:
        modo_is_real = bool(str(estado_bot.get("tipo_cuenta", "") or estado_bot.get("modo", "")).strip().upper() == "REAL")
        token_is_real = bool(token_ctx == TOKEN_REAL or str(token_ctx or "").strip().upper().startswith("REAL") or modo_is_real)
        if token_is_real:
            estado_bot["incident_lock_manual_review"] = "INCIDENT_LOCK_REAL_SOLD_SIN_CONTEXTO_REQUIERE_REVISION"
            if _print_once(f"incident-sold-sin-contexto-{NOMBRE_BOT}-{cid}-{pending_round}", ttl=10.0):
                print(
                    Fore.RED + Style.BRIGHT +
                    f"🚨 INCIDENT_LOCK_REAL_SOLD_SIN_CONTEXTO_REQUIERE_REVISION | bot={NOMBRE_BOT} | "
                    f"contract_id={cid} | NO se libera token REAL | revisión manual requerida"
                )
            return False

        ciclo_ctx = ciclo or estado_bot.get("ciclo_actual") or 1
        estado_bot["pending_contract_state"] = "COMPRA_NO_CONFIRMADA_SOLD_SIN_CONTEXTO_DEMO"
        estado_bot["pending_contract_action"] = "LIBERAR_SOLD_SIN_CONTEXTO_DEMO"
        estado_bot["ciclo_forzado"] = ciclo_ctx
        estado_bot["pending_ciclo"] = ciclo_ctx
        estado_bot["ciclo_actual"] = ciclo_ctx
        if _print_once(f"incident-sold-sin-contexto-{NOMBRE_BOT}-{cid}-{pending_round}", ttl=10.0):
            print(Fore.RED + Style.BRIGHT + f"⚠️ INCIDENT_LOCK_SOLD_SIN_CONTEXTO_COMPLETO | bot={NOMBRE_BOT} | contract_id={cid} | faltan={','.join(faltantes)}")
        print(
            Fore.YELLOW + Style.BRIGHT +
            f"🧯 INCIDENT_LOCK_SOLD_SIN_CONTEXTO_DEMO_LIBERADO | bot={NOMBRE_BOT} | ronda=#{pending_round} | C{ciclo_ctx} | "
            f"contract_id={cid} | acción=recovery_request_neutral | mismo_ciclo=SI"
        )
        _emitir_sync_recovery_incident_demo_sold_sin_contexto_neutral(
            pending_round, cid, ciclo_ctx, asset, direction, attempts, elapsed
        )
        _clear_pending_contract_resolution(
            reason="COMPRA_NO_CONFIRMADA_SOLD_SIN_CONTEXTO_DEMO",
            resultado_final="COMPRA_NO_CONFIRMADA_SOLD_SIN_CONTEXTO_DEMO",
        )
        estado_bot["ciclo_forzado"] = ciclo_ctx
        estado_bot["ciclo_actual"] = ciclo_ctx
        estado_bot["ciclo_en_progreso"] = False
        estado_bot["token_msg_mostrado"] = False
        if callable(globals().get("_sync_round_wait_release")) and pending_round:
            estado_bot["sync_round_id"] = await _sync_round_wait_release(pending_round)
        return False

    token_is_real = bool(token_ctx == TOKEN_REAL or str(token_ctx or "").strip().upper().startswith("REAL"))
    ack_ok = False

    try:
        if cid not in _contratos_procesados:
            await finalizar_contrato_bg(
                cid, 0, asset, direction,
                estado_bot.get("pending_monto", 0.0),
                estado_bot.get("pending_rsi9", 0.0),
                estado_bot.get("pending_rsi14", 0.0),
                estado_bot.get("pending_sma5", 0.0),
                estado_bot.get("pending_sma20", 0.0),
                estado_bot.get("pending_cruce", 0),
                estado_bot.get("pending_breakout", 0),
                estado_bot.get("pending_rsi_reversion", 0),
                ciclo,
                estado_bot.get("pending_payout", 0.0),
                estado_bot.get("pending_condiciones", {}),
                token_ctx,
                epoch_pretrade=estado_bot.get("pending_epoch_pretrade"),
                trade_uid=estado_bot.get("pending_trade_uid"),
                close_snapshot=estado_bot.get("pending_close_snapshot"),
            )
        _contratos_procesados.add(cid)
    except Exception as e:
        print(
            Fore.YELLOW + Style.BRIGHT +
            f"⚠️ INCIDENT_LOCK_SOLD_FINALIZAR_FALLO | bot={NOMBRE_BOT} | contract_id={cid} | error={type(e).__name__}: {e}"
        )
        return False

    if not token_is_real:
        ack_ok = _emitir_ack_sync_incident_demo_resuelto(pending_round, resultado, contract_id=cid, asset=asset, ciclo=ciclo, token_usado=token_ctx)

    _clear_pending_contract_resolution(reason="incident_lock_resuelto_sold")
    print(
        Fore.GREEN + Style.BRIGHT +
        f"✅ INCIDENT_LOCK_RECONCILED_SOLD_PROCESADO | bot={NOMBRE_BOT} | contract_id={cid} | "
        f"resultado={resultado} | C{ciclo} | ack={'SI' if ack_ok else 'NO'}"
    )
    return True



async def resolver_contrato_incierto_seguro(ws):
    """Recuperación robusta INCIDENT_LOCK: buscar fantasma, procesar closed real o liberar stale-open DEMO neutral."""
    if not estado_bot.get("pending_contract_resolution"):
        return True
    pending_id = estado_bot.get("pending_contract_id")
    since = float(estado_bot.get("pending_since_ts", 0.0) or 0.0)
    elapsed = max(0.0, time.time() - since)
    attempts = int(estado_bot.get("pending_attempts", 0) or 0)
    pending_round = int(estado_bot.get("pending_sync_round_id_original") or estado_bot.get("pending_round_id") or estado_bot.get("sync_round_id", 0) or 0)
    token_ctx = estado_bot.get("pending_token_usado")
    asset = estado_bot.get("pending_asset")
    direction = estado_bot.get("pending_direction")
    ciclo_actual = estado_bot.get("pending_ciclo") or estado_bot.get("ciclo_actual") or 1

    found = await buscar_contrato_reciente_deriv_seguro(ws, token_ctx, pending_id, asset, direction, pending_round)
    if found.get("found"):
        cid = found.get("contract_id") or pending_id
        if cid and not pending_id:
            estado_bot["pending_contract_id"] = cid
            pending_id = cid
        poc = found.get("poc") or {"contract_id": cid, "profit": found.get("profit"), "is_sold": bool(found.get("is_closed"))}
        if bool(found.get("profit_reliable", False)) and found.get("profit") is not None:
            poc = dict(poc or {})
            poc["contract_id"] = cid
            poc["profit"] = found.get("profit")
        if bool(found.get("is_closed")) or bool(poc.get("is_sold", False)):
            if bool(found.get("profit_reliable", False)) and found.get("profit") is not None:
                poc = dict(poc or {})
                poc["contract_id"] = cid
                poc["symbol"] = asset or poc.get("symbol")
                poc["contract_type"] = direction or poc.get("contract_type")
                poc["pending_round"] = pending_round
                return await _procesar_incident_lock_sold(poc, source=found.get("source"))
            if elapsed < float(INCIDENT_LOCK_MAX_AGE_S) or attempts < int(INCIDENT_LOCK_RECOVERY_ATTEMPTS):
                if _print_once("incident-lock-closed-sin-profit-wait", ttl=6.0):
                    print(Fore.YELLOW + f"⏳ INCIDENT_LOCK_CLOSED_SIN_PROFIT_WAIT | bot={NOMBRE_BOT} | contract_id={cid} | C{ciclo_actual} | attempts={attempts}/{INCIDENT_LOCK_RECOVERY_ATTEMPTS}")
                return False
            if _incident_lock_demo_closed_sin_profit_permitido(cid, pending_round, token_ctx, elapsed, attempts):
                estado_bot["pending_contract_state"] = "COMPRA_NO_CONFIRMADA_CLOSED_SIN_PROFIT_DEMO"
                estado_bot["pending_contract_action"] = "LIBERAR_CLOSED_SIN_PROFIT_DEMO"
                estado_bot["ciclo_forzado"] = ciclo_actual
                estado_bot["pending_ciclo"] = ciclo_actual
                estado_bot["ciclo_actual"] = ciclo_actual
                print(
                    Fore.YELLOW + Style.BRIGHT +
                    f"🧯 INCIDENT_LOCK_CLOSED_SIN_PROFIT_DEMO_LIBERADO | bot={NOMBRE_BOT} | ronda=#{pending_round} | C{ciclo_actual} | "
                    f"contract_id={int(cid)} | edad={elapsed:.0f}s | attempts={attempts}/{INCIDENT_LOCK_RECOVERY_ATTEMPTS} | "
                    f"acción=recovery_request_neutral | mismo_ciclo=SI"
                )
                _emitir_sync_recovery_incident_demo_closed_sin_profit_neutral(pending_round, int(cid), ciclo_actual, asset, direction, attempts, elapsed)
                _clear_pending_contract_resolution(
                    reason="COMPRA_NO_CONFIRMADA_CLOSED_SIN_PROFIT_DEMO",
                    resultado_final="COMPRA_NO_CONFIRMADA_CLOSED_SIN_PROFIT_DEMO",
                )
                estado_bot["ciclo_forzado"] = ciclo_actual
                estado_bot["pending_ciclo"] = ciclo_actual
                estado_bot["ciclo_actual"] = ciclo_actual
                estado_bot["ciclo_en_progreso"] = False
                estado_bot["token_msg_mostrado"] = False
                if callable(globals().get("_sync_round_wait_release")):
                    estado_bot["sync_round_id"] = await _sync_round_wait_release(pending_round)
                return False
            print(Fore.YELLOW + Style.BRIGHT + f"⚠️ INCIDENT_LOCK_CLOSED_SIN_PROFIT: se mantiene fence para revisión | bot={NOMBRE_BOT} | contract_id={cid}")
            return False
        if elapsed < float(INCIDENT_LOCK_MAX_AGE_S) or attempts < int(INCIDENT_LOCK_RECOVERY_ATTEMPTS):
            if _print_once("incident-lock-open-wait", ttl=6.0):
                print(Fore.YELLOW + f"⏳ INCIDENT_LOCK_OPEN_WAIT | bot={NOMBRE_BOT} | contract_id={cid} | C{ciclo_actual} | attempts={attempts}/{INCIDENT_LOCK_RECOVERY_ATTEMPTS}")
            return False
        poc_open = found.get("poc") or {"is_sold": False, "profit": None}
        if _incident_lock_demo_stale_open_permitido(poc_open, pending_id, pending_round, token_ctx, elapsed, attempts):
            estado_bot["pending_contract_state"] = "COMPRA_NO_CONFIRMADA_STALE_OPEN_DEMO"
            estado_bot["pending_contract_action"] = "LIBERAR_STALE_OPEN_DEMO"
            estado_bot["ciclo_forzado"] = ciclo_actual
            estado_bot["pending_ciclo"] = ciclo_actual
            estado_bot["ciclo_actual"] = ciclo_actual
            print(
                Fore.YELLOW + Style.BRIGHT +
                f"🧯 INCIDENT_LOCK_STALE_OPEN_DEMO_LIBERADO | bot={NOMBRE_BOT} | ronda=#{pending_round} | C{ciclo_actual} | "
                f"contract_id={int(pending_id)} | edad={elapsed:.0f}s | attempts={attempts}/{INCIDENT_LOCK_RECOVERY_ATTEMPTS} | "
                f"acción=recovery_request_neutral | mismo_ciclo=SI"
            )
            _emitir_sync_recovery_incident_demo_neutral(pending_round, int(pending_id), ciclo_actual, asset, direction, attempts, elapsed)
            _clear_pending_contract_resolution(reason="COMPRA_NO_CONFIRMADA_STALE_OPEN_DEMO", resultado_final="COMPRA_NO_CONFIRMADA_STALE_OPEN_DEMO")
            estado_bot["ciclo_forzado"] = ciclo_actual
            estado_bot["ciclo_en_progreso"] = False
            estado_bot["token_msg_mostrado"] = False
            if callable(globals().get("_sync_round_wait_release")):
                estado_bot["sync_round_id"] = await _sync_round_wait_release(pending_round)
            return False
        return False

    if elapsed < float(INCIDENT_LOCK_MAX_AGE_S) or attempts < int(INCIDENT_LOCK_RECOVERY_ATTEMPTS):
        if _print_once("incident-lock-ghost-wait", ttl=6.0):
            print(Fore.YELLOW + f"⏳ INCIDENT_LOCK_BUSQUEDA_FANTASMA | bot={NOMBRE_BOT} | contract_id={pending_id} | attempts={attempts}/{INCIDENT_LOCK_RECOVERY_ATTEMPTS}")
        return False

    # Sin evidencia después de edad+intentos: solo DEMO puede liberar neutral; nunca se inventa PÉRDIDA.
    token_is_real = bool(token_ctx == TOKEN_REAL or str(token_ctx or "").strip().upper().startswith("REAL"))
    if _incident_lock_demo_sin_evidencia_permitido(pending_id, pending_round, token_ctx, elapsed, attempts):
        estado_bot["pending_contract_state"] = "COMPRA_NO_CONFIRMADA_SIN_EVIDENCIA_DEMO"
        estado_bot["pending_contract_action"] = "LIBERAR_SIN_EVIDENCIA_DEMO"
        estado_bot["ciclo_forzado"] = ciclo_actual
        estado_bot["pending_ciclo"] = ciclo_actual
        estado_bot["ciclo_actual"] = ciclo_actual
        print(
            Fore.YELLOW + Style.BRIGHT +
            f"🧯 INCIDENT_LOCK_SIN_EVIDENCIA_DEMO_LIBERADO | bot={NOMBRE_BOT} | ronda=#{pending_round} | C{ciclo_actual} | "
            f"edad={elapsed:.0f}s | attempts={attempts}/{INCIDENT_LOCK_RECOVERY_ATTEMPTS} | "
            f"acción=recovery_request_neutral | mismo_ciclo=SI"
        )
        _emitir_sync_recovery_incident_demo_sin_evidencia_neutral(pending_round, ciclo_actual, asset, direction, attempts, elapsed)
        _clear_pending_contract_resolution(
            reason="COMPRA_NO_CONFIRMADA_SIN_EVIDENCIA_DEMO",
            resultado_final="COMPRA_NO_CONFIRMADA_SIN_EVIDENCIA_DEMO",
        )
        estado_bot["ciclo_forzado"] = ciclo_actual
        estado_bot["pending_ciclo"] = ciclo_actual
        estado_bot["ciclo_actual"] = ciclo_actual
        if callable(globals().get("_sync_round_wait_release")):
            estado_bot["sync_round_id"] = await _sync_round_wait_release(pending_round)
        return False
    if pending_id and not token_is_real and pending_round > 0:
        poc_empty = {"is_sold": False, "profit": None}
        if _incident_lock_demo_stale_open_permitido(poc_empty, pending_id, pending_round, token_ctx, elapsed, attempts):
            estado_bot["pending_contract_state"] = "COMPRA_NO_CONFIRMADA_STALE_OPEN_DEMO"
            estado_bot["pending_contract_action"] = "LIBERAR_STALE_OPEN_DEMO"
            estado_bot["ciclo_forzado"] = ciclo_actual
            _emitir_sync_recovery_incident_demo_neutral(pending_round, int(pending_id), ciclo_actual, asset, direction, attempts, elapsed)
            _clear_pending_contract_resolution(reason="COMPRA_NO_CONFIRMADA_STALE_OPEN_DEMO", resultado_final="COMPRA_NO_CONFIRMADA_STALE_OPEN_DEMO")
            estado_bot["ciclo_forzado"] = ciclo_actual
            return False
    print(Fore.YELLOW + Style.BRIGHT + f"⚠️ INCIDENT_LOCK_SIN_EVIDENCIA: se mantiene fence para revisión | bot={NOMBRE_BOT} | contract_id={pending_id}")
    return False


async def _pending_contract_fence_tick(ws):
    if not estado_bot.get("pending_contract_resolution"):
        return True

    since = float(estado_bot.get("pending_since_ts", 0.0) or 0.0)
    elapsed = max(0.0, time.time() - since)
    attempts = int(estado_bot.get("pending_attempts", 0) or 0) + 1
    estado_bot["pending_attempts"] = attempts

    try:
        return await resolver_contrato_incierto_seguro(ws)
    except Exception as e:
        if _print_once("pending-contract-resolver-error", ttl=8.0):
            print(Fore.YELLOW + f"INCIDENT_LOCK resolver falló: {type(e).__name__}: {e}")

    if elapsed < float(PENDING_CONTRACT_FENCE_S):
        if _print_once("pending-contract-wait", ttl=6.0):
            rem = max(0.0, float(PENDING_CONTRACT_FENCE_S) - elapsed)
            print(
                Fore.YELLOW +
                f"⏳ FENCE ACTIVO | {NOMBRE_BOT} | ciclo=C{estado_bot.get('ciclo_actual', '?')} "
                f"| restante={rem:.0f}s | compra_bloqueada=SI | mismo_ciclo=SI"
            )
        await asyncio.sleep(1.0)
        return False

    print(Fore.RED + Style.BRIGHT + "⚠️ Fence contrato incierto agotó timeout. Se conserva INCIDENT_LOCK para recuperación robusta; no se inventa resultado.")
    reinicio_forzado.set()
    return False
# <<< PATCH

# >>> BLOQUE A: Buffer de logs para no romper la barra
log_buffer = []

def _buffer_log(msg: str):
    log_buffer.append(msg)

def _flush_log_buffer():
    if not log_buffer:
        return
    print()
    for m in log_buffer:
        print(m)
    log_buffer.clear()

# <<< BLOQUE A

# >>> BLOQUE B: Key para commit notice
def _commit_notice_key():
    return f"commit-guard-{last_real_contract_id or 'cooldown'}"

# <<< BLOQUE B

# >>> BLOQUE C: Separadores limpios para consola
def sep_saldos():
    """Separador discreto para bloques de saldo."""
    print(Fore.GREEN + "─" * 60)

def sep_ciclo():
    """Separador discreto para inicio/fin de ciclos de martingala."""
    print(Fore.BLUE + "─" * 60)

# <<< BLOQUE C

# ==================== UTILIDADES ====================
# Header único para el CSV enriquecido (incluye racha_actual y es_rebote)
# === HEADER FINAL CORREGIDO (23 columnas exactas) ===
CSV_HEADER = [
    "fecha", "activo", "direction", "monto", "resultado", "ganancia_perdida",
    "rsi_9", "rsi_14", "sma_5", "sma_20",
    "cruce_sma", "breakout", "rsi_reversion", "racha_actual", "es_rebote", "ciclo_martingala",
    "payout_total",          # nuevo: USD total retornado (stake + profit)
    "payout_multiplier",     # nuevo: ratio total/stake (independiente del monto)
    "puntaje_estrategia",
    "result_bin",            # 1 o 0 solo en filas cerradas
    "trade_status",          # "PRE_TRADE" o "CERRADO"
    "modo_cuenta",
    "epoch",
    "ts",
    "ia_prob_en_juego",
    "ia_prob_source",
    "ia_decision_id",
    "ia_gate_real",
    "ia_modo_ack",
    "ia_ready_ack"
]
CLOSE_SNAPSHOT_COLS = [f"close_{i}" for i in range(20)]
CSV_HEADER = CSV_HEADER + CLOSE_SNAPSHOT_COLS
# =============================================================================
# CSV — helpers robustos (evita columnas corridas + asegura puntaje 0..1)
# =============================================================================
def _to_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return default
            s = s.replace(",", ".")
            return float(s)
        return float(x)
    except Exception:
        return default

def _warn_close_snapshot_insuficiente(closes, total: int = 20, min_valid: int = 10, cooldown_s: float = 120.0):
    try:
        valid_closes = sum(1 for c in list(closes or []) if isinstance(c, (int, float)) and math.isfinite(float(c)) and float(c) > 0.0)
    except Exception:
        valid_closes = 0
    if valid_closes >= int(min_valid):
        return
    now = time.time()
    last = float(globals().get("_last_warn_close_snapshot_ts", 0.0) or 0.0)
    if (now - last) < float(cooldown_s):
        return
    globals()["_last_warn_close_snapshot_ts"] = now
    print(Fore.YELLOW + f"[WARN] close_snapshot insuficiente: {valid_closes}/{int(total)}")

def _extract_close_snapshot(velas, n: int = 20):
    closes = []
    try:
        seq = list(velas or [])
        if not seq:
            return [None] * int(n)
        seq = seq[-int(n):]
        seq = list(reversed(seq))  # close_0 = más reciente
        for v in seq:
            c = None
            if isinstance(v, dict):
                c = v.get("close", v.get("c"))
            elif isinstance(v, bool):
                c = None
            elif isinstance(v, str):
                c = v.strip()
            else:
                c = v
            try:
                cf = float(c)
                if math.isfinite(cf):
                    closes.append(cf)
                else:
                    closes.append(None)
            except Exception:
                closes.append(None)
        while len(closes) < int(n):
            closes.append(None)
    except Exception:
        closes = [None] * int(n)
    return closes[:int(n)]

def _norm_puntaje_01(condiciones, total_cond=3):
    """
    Acepta:
      - 0..1 ya normalizado
      - enteros 0..3
      - strings tipo "2/3"
    Devuelve float en [0,1].
    """
    try:
        if isinstance(condiciones, str) and "/" in condiciones:
            a, b = condiciones.split("/", 1)
            a = _to_float(a, 0.0)
            b = _to_float(b, float(total_cond))
            if b <= 0:
                return 0.0
            v = a / b
        else:
            v = _to_float(condiciones, 0.0)
            # si viene 2 o 3, lo llevamos a 2/3, 3/3
            if v > 1.0001 and total_cond > 0:
                v = v / float(total_cond)
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return float(v)
    except Exception:
        return 0.0

def _write_row_dict_atomic(archivo_csv: str, row_dict: dict):
    """
    Escribe SIEMPRE respetando el orden de CSV_HEADER.
    """
    row = [row_dict.get(col, "") for col in CSV_HEADER]
    write_csv_atomic(archivo_csv, row)

def _build_trade_uid(epoch_val, symbol, direccion, ciclo, token, ts_iso=None):
    try:
        ep = int(float(epoch_val or 0))
    except Exception:
        ep = int(time.time())
    cyc = int(ciclo) if ciclo is not None else 1
    sym = str(symbol or "").strip().upper()
    direc = str(direccion or "").strip().upper()
    tok = str(token or "NA").strip().upper()
    ts_part = str(ts_iso or "").strip()
    return f"{NOMBRE_BOT}|{ep}|C{cyc}|{sym}|{direc}|{tok}|{ts_part}"



def _resolver_modo_cuenta_csv(token_actual=None, modo_actual=None):
    """
    V18:
    Devuelve DEMO o REAL para escribir evidencia explícita en CSV.
    No afecta compra, token ni estrategia.
    """
    try:
        try:
            if "TOKEN_REAL" in globals() and token_actual == TOKEN_REAL:
                return "REAL"
            if "TOKEN_DEMO" in globals() and token_actual == TOKEN_DEMO:
                return "DEMO"
        except Exception:
            pass
        txt = str(modo_actual or token_actual or "").strip().upper()
        if txt.startswith("REAL") or txt == "CR" or txt.startswith("CR"):
            return "REAL"
        if "REAL" in txt:
            return "REAL"
        return "DEMO"
    except Exception:
        return "DEMO"


def _trade_key_from_row(row: dict) -> str:
    rid = str((row or {}).get("ia_decision_id", "") or "").strip()
    if rid:
        return rid
    parts = [
        str((row or {}).get("activo", "") or "").strip().upper(),
        str((row or {}).get("direction", "") or "").strip().upper(),
        str((row or {}).get("epoch", "") or "").strip(),
        str((row or {}).get("ciclo_martingala", "") or "").strip(),
        str((row or {}).get("ts", "") or "").strip(),
    ]
    return "|".join(parts)

def _audit_csv_trade_metrics(archivo_csv: str) -> tuple[int, int, int]:
    try:
        rec = {}
        with open(archivo_csv, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                status = str(row.get("trade_status", "")).strip().upper()
                if status not in {"PRE_TRADE", "PENDIENTE", "CERRADO"}:
                    continue
                key = _trade_key_from_row(row)
                if not key:
                    continue
                cur = rec.get(key, {"has_pre": False, "has_close": False, "rb": "", "ts": ""})
                cur["has_pre"] = bool(cur["has_pre"] or status in {"PRE_TRADE", "PENDIENTE"})
                if status == "CERRADO":
                    cur["has_close"] = True
                    cur["rb"] = str(row.get("result_bin", "")).strip()
                cur["ts"] = str(row.get("ts", cur.get("ts", "")) or cur.get("ts", ""))
                rec[key] = cur

        total_cerrados = 0
        ganancias = 0
        pendientes = 0
        for v in rec.values():
            rb = str(v.get("rb", "")).strip()
            if bool(v.get("has_close", False)) and rb in {"0", "1"}:
                total_cerrados += 1
                if rb == "1":
                    ganancias += 1
            elif bool(v.get("has_pre", False)):
                pendientes += 1
        return int(total_cerrados), int(ganancias), int(pendientes)
    except Exception:
        return 0, 0, 0

# === FIN HEADER FINAL ===
def write_pretrade_snapshot(
    archivo_csv,
    symbol=None,
    direccion=None,
    monto=None,
    rsi9=None,
    rsi14=None,
    sma5=None,
    sma20=None,
    cruce=None,
    breakout=None,
    rsi_reversion=None,
    ciclo=None,
    payout=None,
    condiciones=None,
    racha_actual_bot=0,
    **kwargs
):
    """
    PRE_TRADE snapshot consistente y tolerante:
    - Acepta llamada POSICIONAL (old) y llamada por KW (new) como la tuya.
    - Detecta payout como multiplier (<=3.5) o payout_total (>3.5).
    - puntaje_estrategia SIEMPRE 0..1
    - RETORNA epoch_val para GateWin/ACK.
    """

    # -------------------------
    # Aliases (tu llamada usa nombres distintos)
    # -------------------------
    if symbol is None:
        symbol = kwargs.get("activo") or kwargs.get("symbol")
    if direccion is None:
        direccion = kwargs.get("direccion") or kwargs.get("direction")
    if monto is None:
        monto = kwargs.get("monto") or kwargs.get("amount")

    if rsi9 is None:
        rsi9 = kwargs.get("rsi_9")
    if rsi14 is None:
        rsi14 = kwargs.get("rsi_14")
    if sma5 is None:
        sma5 = kwargs.get("sma_5")
    if sma20 is None:
        sma20 = kwargs.get("sma_20")

    if cruce is None:
        cruce = kwargs.get("cruce_sma")
    if breakout is None:
        breakout = kwargs.get("breakout")
    if rsi_reversion is None:
        rsi_reversion = kwargs.get("rsi_reversion")

    if ciclo is None:
        ciclo = kwargs.get("ciclo_martingala") or kwargs.get("ciclo")

    # condiciones/score puede venir como "puntaje_estrategia"
    if condiciones is None:
        condiciones = kwargs.get("puntaje_estrategia") or kwargs.get("condiciones")

    # racha previa real (PRE-TRADE)
    racha_prev = kwargs.get("racha_actual", racha_actual_bot)
    try:
        racha_prev = int(float(racha_prev))
    except Exception:
        racha_prev = int(racha_actual_bot) if isinstance(racha_actual_bot, (int, float)) else 0

    # es_rebote puede venir ya calculado
    es_rebote_in = kwargs.get("es_rebote", None)
    if es_rebote_in is None:
        es_rebote_flag = 1 if (racha_prev <= -4) else 0
    else:
        try:
            es_rebote_flag = 1 if int(float(es_rebote_in)) == 1 else 0
        except Exception:
            es_rebote_flag = 1 if (racha_prev <= -4) else 0

    # -------------------------
    # monto float
    # -------------------------
    try:
        monto_f = float(monto or 0.0)
    except Exception:
        monto_f = 0.0

    # -------------------------
    # payout robusto
    # -------------------------
    payout_total_f = 0.0
    payout_mult_f = 0.0
    try:
        p = float(payout) if payout not in (None, "", "nan", "NaN") else 0.0
        # si NaN/inf, lo anulamos
        try:
            if not math.isfinite(p):
                p = 0.0
            if not math.isfinite(monto_f):
                monto_f = 0.0
        except Exception:
            pass

        if p > 0 and p <= 3.5:
            payout_mult_f = p
            payout_total_f = (monto_f * payout_mult_f) if monto_f > 0 else 0.0
        elif p > 3.5:
            payout_total_f = p
            payout_mult_f = (payout_total_f / monto_f) if monto_f > 0 else 0.0
    except Exception:
        payout_total_f = 0.0
        payout_mult_f = 0.0

    # -------------------------
    # puntaje 0..1
    # -------------------------
    try:
        puntaje01 = _norm_puntaje_01(condiciones)
    except Exception:
        puntaje01 = 0.0

    now = datetime.now(timezone.utc)
    epoch_val = int(now.timestamp())
    ts_val = now.isoformat()
    trade_uid = str(kwargs.get("trade_uid", "") or "").strip()
    if not trade_uid:
        trade_uid = _build_trade_uid(epoch_val, symbol, direccion, ciclo, kwargs.get("token", "NA"), ts_iso=ts_val)
    close_snapshot = kwargs.get("close_snapshot", None)
    closes = _extract_close_snapshot(close_snapshot, n=20)
    _warn_close_snapshot_insuficiente(closes)

    modo_cuenta_csv = _resolver_modo_cuenta_csv(
        token_actual=kwargs.get("token"),
        modo_actual="REAL" if kwargs.get("token") == globals().get("TOKEN_REAL") else "DEMO",
    )

    row_dict = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activo": symbol,
        "direction": direccion,
        "monto": float(monto_f),
        "resultado": "PENDIENTE",
        "ganancia_perdida": "",
        "rsi_9": rsi9,
        "rsi_14": rsi14,
        "sma_5": sma5,
        "sma_20": sma20,
        "cruce_sma": int(cruce) if cruce is not None else "",
        "breakout": int(breakout) if breakout is not None else "",
        "rsi_reversion": int(rsi_reversion) if rsi_reversion is not None else "",
        "racha_actual": int(racha_prev),
        "es_rebote": int(es_rebote_flag),
        "ciclo_martingala": int(ciclo) if ciclo is not None else 1,
        "payout_total": float(round(payout_total_f, 2)),
        "payout_multiplier": float(round(payout_mult_f, 6)),
        "puntaje_estrategia": float(round(float(puntaje01), 6)),
        "result_bin": "",
        "trade_status": "PRE_TRADE",
        "modo_cuenta": modo_cuenta_csv,
        "epoch": int(epoch_val),
        "ts": ts_val,
        "ia_prob_en_juego": "",
        "ia_prob_source": "",
        "ia_decision_id": trade_uid,
        "ia_gate_real": "",
        "ia_modo_ack": "",
        "ia_ready_ack": "",
    }
    for i, c in enumerate(closes):
        row_dict[f"close_{i}"] = "" if c is None else float(c)

    _write_row_dict_atomic(archivo_csv, row_dict)
    return epoch_val

def write_token_atomic(path: str, content: str):
    """
    Escritura atómica robusta para tokens (ARCHIVO_TOKEN).
    - Reintenta en Windows si el archivo está bloqueado por otro proceso.
    - Limpia .tmp si queda colgado.
    """
    tmp = path + ".tmp"
    last_err = None

    # 1) escribir tmp
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(Fore.RED + f"⚠️ Token: no pude escribir tmp: {e}")
        return

    # 2) replace atómico con reintentos
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.06 + 0.04 * attempt)
        except Exception as e:
            last_err = e
            break

    print(Fore.RED + f"⚠️ Token: os.replace falló: {last_err}")
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass
def release_real_token_if_owned():
    """
    Libera el token REAL solo si el archivo todavía dice REAL:<este bot>.
    Evita pisar al MAESTRO si ya reasignó REAL a otro bot.
    """
    expected = f"REAL:{NOMBRE_BOT}"
    try:
        with open(ARCHIVO_TOKEN, "r", encoding="utf-8", errors="replace") as f:
            cur = (f.read() or "").strip()
    except Exception:
        return False

    # CAS: solo escribo si sigo siendo el dueño
    if cur == expected:
        try:
            write_token_atomic(ARCHIVO_TOKEN, "REAL:none")
            return True
        except Exception:
            return False

    return False

def write_csv_atomic(path: str, row):
    """
    Escritura atómica + auto-reparación de filas inconsistentes (columnas corridas / len != header).
    Garantía:
      - Header final SIEMPRE = CSV_HEADER
      - Cada fila SIEMPRE se escribe con len(CSV_HEADER) columnas (pad/truncate)
      - Evita que un CSV roto haga que pandas luego "skip" filas.
    """
    import os, csv, time

    def _norm_len(r, target_len: int):
        if r is None:
            return [""] * target_len
        r = list(r)
        if len(r) < target_len:
            r = r + ([""] * (target_len - len(r)))
        elif len(r) > target_len:
            r = r[:target_len]
        return r

    # ---------- Lock cross-process (maestro/bot) ----------
    lock_path = path + ".lock"
    fd = None
    start = time.time()
    try:
        while time.time() - start < 5:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except FileExistsError:
                time.sleep(0.05)
        # si no se pudo lockear, igual continuamos (no matamos al bot)
    except Exception:
        fd = None

    num_cols = len(CSV_HEADER)
    tmp = path + ".tmp"

    rows_to_write = []
    old_header = []
    data_rows = []
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0

    needs_repair = False

    if file_exists:
        try:
            with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                old_header = next(reader, None) or []
                data_rows = [r for r in reader]

            # Detectar filas "mutantes" aun si header coincide
            if old_header == CSV_HEADER:
                for r in data_rows[:300]:
                    if len(r) != num_cols:
                        needs_repair = True
                        break

            # Normalizar filas respecto a header viejo para evitar IndexError
            if old_header:
                data_rows = [_norm_len(r, len(old_header)) for r in data_rows]

            if old_header == CSV_HEADER:
                # Header igual: solo normalizamos longitudes
                rows_to_write = [_norm_len(r, num_cols) for r in data_rows]
            else:
                # Header distinto: remapeo por nombre si se puede
                idx = {name: i for i, name in enumerate(old_header)} if old_header else {}
                remapped = []
                for r in data_rows:
                    new_r = [""] * num_cols
                    mapped_any = False
                    for j, col in enumerate(CSV_HEADER):
                        if col in idx and idx[col] < len(r):
                            new_r[j] = r[idx[col]]
                            mapped_any = True
                    if not mapped_any:
                        new_r = _norm_len(r, num_cols)
                    remapped.append(_norm_len(new_r, num_cols))
                rows_to_write = remapped
                needs_repair = True  # header cambiado implica reescritura correctiva
        except Exception:
            # Si está muy roto, no frenamos: recreamos desde cero con la fila nueva
            rows_to_write = []
            needs_repair = True

    new_row = _norm_len(row, num_cols)

    # Guard anti-duplicado:
    # - Si NO hace falta reparar, y la última fila coincide, salimos.
    # - Si SÍ hace falta reparar, igual reescribimos (sin re-agregar duplicado).
    append_new = True
    if rows_to_write and rows_to_write[-1] == new_row:
        if not needs_repair:
            # CSV ya está sano, no hagas nada
            if fd is not None:
                try: os.close(fd)
                except: pass
                try: os.remove(lock_path)
                except: pass
            return
        append_new = False  # reparo pero no duplico

    # Escritura atómica con retries (sin mover el original a .bak antes)
    last_err = None
    for _ in range(3):
        try:
            with open(tmp, "w", newline="", encoding="utf-8", errors="replace") as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADER)
                for r in rows_to_write:
                    w.writerow(_norm_len(r, num_cols))
                if append_new:
                    w.writerow(new_row)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            last_err = None
            break
        except Exception as e:
            last_err = e
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            time.sleep(0.05)

    # Fallback final: append directo
    if last_err is not None:
        try:
            file_exists = os.path.exists(path) and os.path.getsize(path) > 0
            with open(path, "a", newline="", encoding="utf-8", errors="replace") as f:
                w = csv.writer(f)
                if not file_exists:
                    w.writerow(CSV_HEADER)
                w.writerow(new_row)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    # release lock
    if fd is not None:
        try: os.close(fd)
        except: pass
        try: os.remove(lock_path)
        except: pass
# ============================================================================
# PATCH CSV (SOLO) — Completar es_rebote y ciclo_martingala si vienen vacíos
# - No toca estrategia, no toca trading, no toca IA.
# - Solo asegura que el CSV enriquecido SIEMPRE tenga estas 2 columnas completas.
# ============================================================================
_CSV_REPARADO_1VEZ = False

def _to_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default

def infer_ciclo_por_monto(monto):
    """
    Si ciclo_martingala viene vacío, lo inferimos por el monto comparando con
    MARTINGALA_REAL / MARTINGALA_DEMO. Si no calza, devolvemos 1.
    """
    try:
        m = float(monto)
    except Exception:
        return 1

    secuencias = []
    try:
        if isinstance(MARTINGALA_REAL, (list, tuple)) and len(MARTINGALA_REAL) > 0:
            secuencias.append(MARTINGALA_REAL)
    except Exception:
        pass
    try:
        if isinstance(MARTINGALA_DEMO, (list, tuple)) and len(MARTINGALA_DEMO) > 0:
            secuencias.append(MARTINGALA_DEMO)
    except Exception:
        pass

    # Match exacto
    for seq in secuencias:
        for i, v in enumerate(seq):
            try:
                if abs(m - float(v)) <= 1e-9:
                    return i + 1
            except Exception:
                continue

    # Tolerancia por redondeos
    for seq in secuencias:
        for i, v in enumerate(seq):
            try:
                if abs(m - float(v)) <= 0.01:
                    return i + 1
            except Exception:
                continue

    return 1

def reparar_csv_esrebote_ciclo(archivo):
    """
    Repara SOLO filas donde es_rebote o ciclo_martingala están vacíos.
    No recalcula nada más. No altera columnas existentes.
    """
    try:
        if not os.path.exists(archivo):
            return

        with open(archivo, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        if not rows:
            return

        header = rows[0]
        if ("es_rebote" not in header) or ("ciclo_martingala" not in header):
            return

        idx_es = header.index("es_rebote")
        idx_ci = header.index("ciclo_martingala")
        idx_monto = header.index("monto") if "monto" in header else None

        changed = False
        fixed = [header]

        for r in rows[1:]:
            if not r:
                continue

            # Normaliza largo (sin mover columnas)
            if len(r) < len(header):
                r = r + [""] * (len(header) - len(r))
            elif len(r) > len(header):
                r = r[:len(header)]

            # Completar es_rebote si vacío
            if isinstance(r[idx_es], str) and r[idx_es].strip() == "":
                r[idx_es] = "0"
                changed = True

            # Completar ciclo_martingala si vacío
            if isinstance(r[idx_ci], str) and r[idx_ci].strip() == "":
                ciclo = 1
                if idx_monto is not None:
                    ciclo = infer_ciclo_por_monto(r[idx_monto])
                r[idx_ci] = str(int(ciclo))
                changed = True

            fixed.append(r)

        if changed:
            tmp = archivo + ".tmp_fix"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(fixed)
            os.replace(tmp, archivo)
            print(Fore.YELLOW + "🧽 CSV reparado: es_rebote/ciclo_martingala completados (solo columnas vacías).")

    except Exception as e:
        print(Fore.RED + f"⚠️ No se pudo reparar CSV ({archivo}): {e}")

def _modo_selftest_import_seguro():
    keys = (
        "CHATGPT_SELFTEST",
        "RUN_ROUND_ALIGN_SELFTEST",
        "RUN_SYNC_RECOVERY_RELEASE_SELFTEST",
        "RUN_ACK_HEARTBEAT_FRESHNESS_SELFTEST",
        "RUN_LXV_4V2X_CANDIDATE_PANEL_SELFTEST",
        "RUN_ROUND_DRIFT_AHEAD_SELFTEST",
        "RUN_SYNC_DEMO_HOLD_GLOBAL_SELFTEST",
        "RUN_CSV_MODO_CUENTA_SELFTEST",
        "RUN_BUY_ERROR_CLASSIFIER_SELFTEST",
    )
    for k in keys:
        if str(os.environ.get(k, "")).strip().lower() in ("1", "true", "yes", "si", "sí"):
            return True
    return False


def cargar_tokens():
    """
    tokens_usuario.txt:
        línea 1 = TOKEN_DEMO
        línea 2 = TOKEN_REAL
    """
    ruta = "tokens_usuario.txt"
    intento = 0
    while not stop_event.is_set():
        try:
            if not os.path.exists(ruta):
                if _modo_selftest_import_seguro():
                    print("CHATGPT_SELFTEST activo: tokens_usuario.txt no existe, usando tokens dummy.")
                    return "DEMO_SELFTEST_TOKEN", "REAL_SELFTEST_TOKEN"

                intento += 1
                if intento % 3 == 1:
                    print("tokens_usuario.txt no existe. Esperando a que la GUI lo genere...")
                time.sleep(3)
                continue

            with open(ruta, "r", encoding="utf-8") as f:
                lineas = [ln.strip() for ln in f.readlines()]

            if len(lineas) < 2 or not lineas[0] or not lineas[1]:
                if _modo_selftest_import_seguro():
                    print("CHATGPT_SELFTEST activo: tokens_usuario.txt inválido, usando tokens dummy.")
                    return "DEMO_SELFTEST_TOKEN", "REAL_SELFTEST_TOKEN"

                intento += 1
                if intento % 5 == 1:
                    print("tokens_usuario.txt inválido (faltan líneas o están vacías). Reintentando...")
                time.sleep(3)
                continue

            demo, real = lineas[0], lineas[1]
            print(f"Tokens cargados desde archivo: DEMO={demo[:4]}*** REAL={real[:4]}***")
            return demo, real

        except Exception as e:
            if _modo_selftest_import_seguro():
                print(f"CHATGPT_SELFTEST activo: error leyendo tokens_usuario.txt ({e}), usando tokens dummy.")
                return "DEMO_SELFTEST_TOKEN", "REAL_SELFTEST_TOKEN"

            intento += 1
            if intento % 5 == 1:
                print(f"Error leyendo tokens_usuario.txt: {e}. Reintentando en 3s...")
            time.sleep(3)

    print("Detención solicitada durante carga de tokens. Usando tokens dummy de parada segura.")
    return "DEMO_STOP_TOKEN", "REAL_STOP_TOKEN"

TOKEN_DEMO, TOKEN_REAL = cargar_tokens()

def reset_csv_and_total():
    """
    Borra el CSV si existe, lo recrea con el encabezado actualizado y
    resetea el total acumulado de DEMO (no REAL).
    Solo úsalo manualmente si quieres empezar una sesión limpia.
    """
    if os.path.exists(ARCHIVO_CSV):
        os.remove(ARCHIVO_CSV)
    with open(ARCHIVO_CSV, "w", newline="", encoding="utf-8", errors="replace") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
    resultado_global["demo"] = 0.0
    print(Fore.YELLOW + "CSV limpiado manualmente y total DEMO resetado para sesión nueva.")

# Crea el CSV si no existe (con header actualizado, sin borrar histórico existente)
if not os.path.exists(ARCHIVO_CSV):
    with open(ARCHIVO_CSV, "w", newline="", encoding="utf-8", errors="replace") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

def leer_token_desde_archivo():
    """
    Lee ARCHIVO_TOKEN. Si contiene 'REAL:fulll45' -> autoriza con TOKEN_REAL, si no -> TOKEN_DEMO.
    """
    try:
        with open(ARCHIVO_TOKEN, "r", encoding="utf-8", errors="replace") as f:
            linea = f.read().strip()
            if linea == f"REAL:{NOMBRE_BOT}":
                return TOKEN_REAL
    except:
        pass
    return TOKEN_DEMO

def calcular_rsi(cierres, periodo=14):
    if len(cierres) < periodo + 1:
        return 50
    ganancias, perdidas = [], []
    for i in range(1, periodo + 1):
        delta = cierres[-i] - cierres[-i - 1]
        (ganancias if delta > 0 else perdidas).append(abs(delta))
    media_gan = mean(ganancias) if ganancias else 0.0001
    media_per = mean(perdidas) if perdidas else 0.0001
    rs = media_gan / media_per
    return round(100 - (100 / (1 + rs)), 2)

def evaluar_estrategia(velas):
    # Normaliza a float por si Deriv devuelve strings
    cierres = [float(v["close"]) for v in velas]
    open_ = float(velas[-1]["open"])
    close = float(velas[-1]["close"])

    sma5 = sum(cierres[-5:]) / 5
    if len(cierres) >= 20:
        sma20 = sum(cierres[-20:]) / 20
    else:
        sma20 = sum(cierres) / max(1, len(cierres))

    rsi9 = calcular_rsi(cierres, 9)
    rsi14 = calcular_rsi(cierres, 14)

    high_prev = float(velas[-2]["high"])
    low_prev = float(velas[-2]["low"])

    breakout = (close > high_prev) or (close < low_prev)
    cruce_sma = ((sma5 > sma20 and close > sma5) or (sma5 < sma20 and close < sma5))
    rsi_reversion = ((rsi14 < 30 and rsi9 > rsi14) or (rsi14 > 70 and rsi9 < rsi14))

    direccion = "CALL" if close > open_ else "PUT"
    condiciones = int(breakout) + int(cruce_sma) + int(rsi_reversion)

    # Importante: mantenemos el orden de retorno que tu bot ya espera
    return condiciones, direccion, rsi9, rsi14, sma5, sma20, breakout, cruce_sma, rsi_reversion


def puntuar_setups(condiciones, direccion, rsi9, rsi14, sma5, sma20, breakout, cruce_sma, rsi_reversion):
    """
    Score interno para elegir MEJOR activo entre candidatos válidos (sin cambiar 13 features).
    Mantiene la regla base (>=2/3), pero evita tomar el primer símbolo "aceptable".
    """
    try:
        score = float(condiciones)

        # Alineación de tendencia con la dirección sugerida
        tendencia_call = (sma5 > sma20)
        tendencia_put = (sma5 < sma20)
        alineado = (direccion == "CALL" and tendencia_call) or (direccion == "PUT" and tendencia_put)
        if alineado:
            score += 0.75

        # Fortaleza del cruce (distancia relativa entre medias)
        den = max(abs(float(sma20)), 1e-9)
        gap = abs(float(sma5) - float(sma20)) / den
        score += min(0.50, gap * 25.0)

        # Confirmaciones de setup
        if breakout:
            score += 0.35
        if rsi_reversion:
            score += 0.25

        # Penalización suave si RSI está en zona "gris" (menos edge)
        if 45.0 <= float(rsi14) <= 55.0:
            score -= 0.15

        return float(score)
    except Exception:
        return float(condiciones or 0)


def setup_pasa_filtro(score: float, condiciones: int) -> bool:
    """Gate de calidad: mantiene >=2/3 y exige score mínimo."""
    try:
        return (int(condiciones) >= 2) and (float(score) >= float(SCORE_MIN))
    except Exception:
        return False
# ==================== WS HELPERS ====================
# BLOQUE 1: api_call wrapper
_req_counter = itertools.count(1)

async def api_call(ws, payload: dict, expect_msg_type: str = None, timeout=10.0):
    """
    Envia payload con req_id y espera respuesta con el mismo req_id.
    Si expect_msg_type se especifica, valida msg_type (con aliases defensivos).
    Lanza RuntimeError ante errores del API Deriv.
    """
    rid = next(_req_counter)
    payload = dict(payload)
    payload["req_id"] = rid

    await ws.send(json.dumps(payload))

    aliases = {
        "candles": {"history"},
        "history": {"candles"},
    }

    deadline = time.time() + float(timeout)

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"Timeout esperando respuesta para req_id={rid}")

        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Mensaje corrupto/partial: ignora y sigue escuchando
            continue

        # Errores explícitos del API
        if data.get("error"):
            err = data["error"]
            raise RuntimeError(f"API error: {err.get('code')} - {err.get('message')}")

        # Filtra por req_id
        if data.get("req_id") != rid:
            continue

        # Si espero un msg_type específico, valido (con aliases)
        if expect_msg_type:
            mt = data.get("msg_type")
            if mt != expect_msg_type and mt not in aliases.get(expect_msg_type, set()):
                continue

        return data

async def authorize_ws(ws, token: str, tries: int = 3, timeout: float = 8.0):
    """Authorize robusto: reintenta antes de rendirse (reduce timeouts)."""
    last = None
    for i in range(tries):
        try:
            await api_call(ws, {"authorize": token}, expect_msg_type=None, timeout=timeout)
            return
        except Exception as e:
            last = e
            await asyncio.sleep(0.4 + 0.4 * i + random.uniform(0.0, 0.3))
    raise last

# BLOQUE 2: obtener_velas con cooldown
_symbol_cooldown = {}  # symbol -> epoch hasta el que está en pausa

# Salud WS
_ws_fail_streak = 0  # cuántas 1006/errores seguidos en esta pasada
ws_reset_needed = asyncio.Event()  # señal para que el loop principal reabra WS

def _es_error_transitorio_ws(exc: Exception) -> bool:
    """Errores de red/WS que deben reintentarse sin tumbar el ciclo."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, websockets.exceptions.ConnectionClosed, OSError)):
        return True
    msg = str(exc).lower()
    return (
        "connectionclosed" in msg
        or "timeout" in msg
        or "timed out" in msg
        or "se agotó el tiempo" in msg
        or "winerror 121" in msg
    )

async def obtener_velas(ws, symbol, token, reintentos=4):
    global _ws_fail_streak
    # respeta cooldown por símbolo
    until = _symbol_cooldown.get(symbol, 0)
    if time.time() < until:
        return []
    delay = 0.8
    for intento in range(reintentos):
        try:
            data = await api_call(ws, {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": VELAS,
                "end": "latest",
                "start": 1,
                "style": "candles",
                "granularity": 60
            }, expect_msg_type="candles", timeout=12.0)
            candles = data.get("candles", [])
            # Éxito: resetea racha de fallas WS
            if candles:
                _ws_fail_streak = 0
            return candles or []
        except websockets.exceptions.ConnectionClosed as e:
            # 1006/close: marca cooldown corto al símbolo y sube racha global
            _symbol_cooldown[symbol] = time.time() + 20
            _ws_fail_streak += 1
            if _print_once(f"ws-obt-closed-{symbol}", ttl=8):
                print(Fore.YELLOW + f"WS cerrado ({getattr(e, 'code', '???')}) en {symbol}. Reintento {intento+1}/{reintentos}...")
        except (asyncio.TimeoutError, json.JSONDecodeError):
            if _print_once(f"ws-obt-timeout-{symbol}", ttl=8):
                print(Fore.YELLOW + f"Timeout/JSON en velas {symbol}. Reintentando...")
        except RuntimeError as api_e:
            msg = str(api_e)
            if "RateLimit" in msg or "market" in msg.lower():
                pass  # retry suave
            else:
                # error “duro”: enfría más tiempo y abandona
                _symbol_cooldown[symbol] = time.time() + 90
                if _print_once(f"cool-{symbol}", ttl=60):
                    print(Fore.YELLOW + f"{symbol} en cooldown 90s por error: {api_e}")
                return []
        except Exception as e:
            if _print_once(f"ws-obt-err-{symbol}", ttl=8):
                print(Fore.RED + f"Error velas {symbol}: {e}. Reintentando...")
        # Fallback: desde el 3er intento usa una conexión efímera dedicada
        if intento >= 2:
            try:
                async with websockets.connect(DERIV_WS_URL, **WS_KW) as ws2:
                    await authorize_ws(ws2, token, tries=2, timeout=6.0)
                    data2 = await api_call(ws2, {
                        "ticks_history": symbol,
                        "adjust_start_time": 1,
                        "count": VELAS,
                        "end": "latest",
                        "start": 1,
                        "style": "candles",
                        "granularity": 60
                    }, expect_msg_type="candles", timeout=12.0)
                    candles2 = data2.get("candles", [])
                    if candles2:
                        _ws_fail_streak = 0
                    return candles2 or []
            except Exception as e2:
                # si también falla, seguimos con backoff
                if _is_rate_limit_error(e2):
                    _set_buy_pause("rate_limit_ticks_history", BUY_RATE_LIMIT_PAUSE_S)
                if _print_once(f"ws-efimera-{symbol}", ttl=8):
                    print(Fore.YELLOW + f"Fallback efímero falló en {symbol}: {e2}")
        await asyncio.sleep(delay + random.uniform(0.0, 0.5))  # Jitter para evitar rate-limits
        delay = min(delay * 1.5, 3.0)
    return []

async def check_token_and_reconnect(ws, current_token):
    global ultimo_token
    global primer_ingreso_real, real_activado_en_bot
    token_desde_archivo = leer_token_desde_archivo()
    if token_desde_archivo != current_token:
        # BLOQUE 2 y 9: Anti-rebote + commit guard
        if current_token == TOKEN_REAL and token_desde_archivo == TOKEN_DEMO:
            # SOLO ignorar "rebote a DEMO" si hay ciclo en progreso.
            if estado_bot.get("ciclo_en_progreso") and (commit_guard_active() or (time.time() - real_activado_en_bot < COOLDOWN_REAL_S)):
                key = _commit_notice_key()
                if not estado_bot.get("barra_activa", False) and _print_once(key, ttl=180):
                    print(Fore.YELLOW + "Commit REAL o cooldown activo: ignorando rebote a DEMO.")
                return ws, current_token

        if estado_bot["ciclo_en_progreso"]:
            # Corte en caliente: interrumpe el reloj YA
            if not estado_bot.get("token_msg_mostrado", False):
                if not (MODO_SILENCIOSO and estado_bot.get("modo_manual")):
                    print(Fore.MAGENTA + Style.BRIGHT + "Cambio de token detectado: cortando ciclo y aplicando de inmediato.")
                    print(Fore.YELLOW + Style.BRIGHT + "⚠️ REAL/DEMO cambió mientras había contrato abierto. Esperando cierre seguro antes de nueva compra." + Style.RESET_ALL)
                estado_bot["token_msg_mostrado"] = True
            estado_bot["interrumpir_ciclo"] = True
            reinicio_forzado.set()
            return ws, current_token  # dejar que esperar_resultado lo desprenda
        else:
            print(Fore.YELLOW + Style.BRIGHT + f"Token cambió a {'REAL' if token_desde_archivo == TOKEN_REAL else 'DEMO'}. Reconectando...")
            try:
                await ws.close()
            except:
                pass
            ws = await websockets.connect(DERIV_WS_URL, **WS_KW)  # BLOQUE 1.2
            await authorize_ws(ws, token_desde_archivo)
            await asyncio.sleep(0.6 + random.uniform(0.0, 0.5))  # BLOQUE 4: micro-cooldown
            if token_desde_archivo == TOKEN_REAL:
                # >>> PATCH 4 — Al entrar a REAL, preconfigura el ciclo forzado
                if not primer_ingreso_real:
                    print(Fore.LIGHTRED_EX + Back.WHITE + Style.BRIGHT + f"\n{NOMBRE_BOT.upper()} ENTRÓ EN CUENTA REAL {datetime.now().strftime('%H:%M:%S')}")
                    # SFX: PASO_A_REAL (reemplaza racha_detectada.wav)
                    try:
                        play_sfx("PASO_A_REAL", vol=0.9)
                    except Exception:
                        pass
                    primer_ingreso_real = True
                    real_activado_en_bot = time.time()  # BLOQUE 5 and 2: Set activation time
                    cyc, _, quiet, src = leer_orden_real(NOMBRE_BOT)  # BLOQUE 7: Relee fresh
                    estado_bot["real_first_cycle_reset_pending"] = True
                    ciclo_maestro = None
                    try:
                        if cyc is not None:
                            cyc_int = int(cyc)
                            if 1 <= cyc_int <= MAX_CICLOS:
                                ciclo_maestro = cyc_int
                    except Exception:
                        ciclo_maestro = None

                    ciclo_forzado_prev = estado_bot.get("ciclo_forzado")
                    try:
                        ciclo_forzado_prev = int(ciclo_forzado_prev) if ciclo_forzado_prev is not None else None
                    except Exception:
                        ciclo_forzado_prev = None
                    if ciclo_forzado_prev is not None and not (1 <= ciclo_forzado_prev <= MAX_CICLOS):
                        ciclo_forzado_prev = None

                    orden = _leer_orden_real_viva_para_bot()
                    ok_ord, ciclo_orden, motivo_orden = _orden_real_bot_ok(orden)
                    if not ok_ord:
                        print(Fore.YELLOW + "🚨 TOKEN_REAL_SIN_ORDEN_VALIDA: no compro, espero orden maestro")
                        await asyncio.sleep(2)
                        reinicio_forzado.set()
                        return ws, current_token
                    ciclo_retenido = _safe_int_cycle(estado_bot.get("ciclo_forzado") or estado_bot.get("ciclo_actual") or 1, 1)
                    if int(ciclo_orden) < int(ciclo_retenido) and not _owner_state_confirma_ciclo(ciclo_orden):
                        print(Fore.CYAN + Style.BRIGHT + f"🚨 CICLO_REAL_MISMATCH_CRITICO: orden=C{ciclo_orden} < retenido=C{ciclo_retenido}. NO COMPRA. Esperando orden corregida.")
                        await asyncio.sleep(2)
                        reinicio_forzado.set()
                        return ws, current_token
                    if int(ciclo_orden) < int(ciclo_retenido):
                        print(Fore.GREEN + f"✅ HANDSHAKE REAL OK: orden=C{ciclo_orden} confirmada por owner_state fresco")
                    estado_bot["ciclo_forzado"] = int(ciclo_orden)
                    estado_bot["ciclo_actual"] = int(ciclo_orden)
                    print(Fore.GREEN + f"✅ HANDSHAKE REAL OK: orden=C{ciclo_orden} token=REAL:{NOMBRE_BOT}")

                    # Silenciar ruido guiado por maestro (BLOQUE 3)
                    if quiet or (str(src).upper() == "MANUAL"):
                        asyncio.create_task(_silencio_temporal(90, fuente=src))
                    else:
                        asyncio.create_task(_desactivar_silencioso_en(90))
                    reinicio_forzado.set()
                else:
                    if not (MODO_SILENCIOSO and estado_bot.get("modo_manual")) and not estado_bot.get("barra_activa", False):
                        if _print_once("rea-REAL", ttl=180):
                            print(Fore.YELLOW + "Reafirmación de REAL (sin reset de martingala)")
                    cyc, _, quiet, src = leer_orden_real(NOMBRE_BOT)  # BLOQUE 7: Relee fresh
                    ciclo_maestro = None
                    try:
                        if cyc is not None:
                            cyc_int = int(cyc)
                            if 1 <= cyc_int <= MAX_CICLOS:
                                ciclo_maestro = cyc_int
                    except Exception:
                        ciclo_maestro = None

                    ciclo_forzado_prev = estado_bot.get("ciclo_forzado")
                    try:
                        ciclo_forzado_prev = int(ciclo_forzado_prev) if ciclo_forzado_prev is not None else None
                    except Exception:
                        ciclo_forzado_prev = None
                    if ciclo_forzado_prev is not None and not (1 <= ciclo_forzado_prev <= MAX_CICLOS):
                        ciclo_forzado_prev = None

                    orden = _leer_orden_real_viva_para_bot()
                    ok_ord, ciclo_orden, _motivo_orden = _orden_real_bot_ok(orden)
                    if ok_ord:
                        ciclo_retenido = _safe_int_cycle(estado_bot.get("ciclo_forzado") or estado_bot.get("ciclo_actual") or 1, 1)
                        if int(ciclo_orden) < int(ciclo_retenido) and not _owner_state_confirma_ciclo(ciclo_orden):
                            print(Fore.CYAN + Style.BRIGHT + f"🚨 CICLO_REAL_MISMATCH_CRITICO: orden=C{ciclo_orden} < retenido=C{ciclo_retenido}. NO COMPRA. Esperando orden corregida.")
                            await asyncio.sleep(2)
                            reinicio_forzado.set()
                            return ws, current_token
                        estado_bot["ciclo_forzado"] = int(ciclo_orden)
                        estado_bot["ciclo_actual"] = int(ciclo_orden)
                        print(Fore.GREEN + f"✅ HANDSHAKE REAL OK: orden=C{ciclo_orden} token=REAL:{NOMBRE_BOT}")
                    else:
                        print(Fore.YELLOW + "🚨 TOKEN_REAL_SIN_ORDEN_VALIDA: no compro, espero orden maestro")
                        await asyncio.sleep(2)
                        reinicio_forzado.set()
                        return ws, current_token

                    if quiet or (str(src).upper() == "MANUAL"):
                        asyncio.create_task(_silencio_temporal(90, fuente=src))
                    else:
                        asyncio.create_task(_desactivar_silencioso_en(90))
                # <<< PATCH 4
            else:
                # Saliste de REAL: prepara el sonido para la próxima ventana
                primer_ingreso_real = False
                estado_bot["real_first_cycle_reset_pending"] = False
                reinicio_forzado.set()
            ultimo_token = token_desde_archivo  # mantén vigilante y lazo alineados
            return ws, token_desde_archivo
    else:
        if token_desde_archivo == TOKEN_REAL:
            # ✅ FIX: si el bot arranca/reconecta y ya está en REAL, igual debe "hablar" 1 vez
            if not primer_ingreso_real:
                if not estado_bot.get("barra_activa", False):
                    try:
                        print(
                            Fore.LIGHTRED_EX + Back.WHITE + Style.BRIGHT
                            + f"\n{NOMBRE_BOT.upper()} INICIÓ EN CUENTA REAL"
                            + Style.RESET_ALL
                        )
                    except Exception:
                        pass

                # Audio PASO_A_REAL (blindado)
                try:
                    play_sfx("PASO_A_REAL", vol=0.95)
                except Exception:
                    pass

                primer_ingreso_real = True
                estado_bot["real_first_cycle_reset_pending"] = True
                cyc, _, _quiet, _src = leer_orden_real(NOMBRE_BOT)
                ciclo_maestro = None
                try:
                    if cyc is not None:
                        cyc_int = int(cyc)
                        if 1 <= cyc_int <= MAX_CICLOS:
                            ciclo_maestro = cyc_int
                except Exception:
                    ciclo_maestro = None

                ciclo_forzado_prev = estado_bot.get("ciclo_forzado")
                try:
                    ciclo_forzado_prev = int(ciclo_forzado_prev) if ciclo_forzado_prev is not None else None
                except Exception:
                    ciclo_forzado_prev = None
                if ciclo_forzado_prev is not None and not (1 <= ciclo_forzado_prev <= MAX_CICLOS):
                    ciclo_forzado_prev = None

                orden = _leer_orden_real_viva_para_bot()
                ok_ord, ciclo_orden, _motivo_orden = _orden_real_bot_ok(orden)
                if not ok_ord:
                    print(Fore.YELLOW + "🚨 TOKEN_REAL_SIN_ORDEN_VALIDA: no compro, espero orden maestro")
                    await asyncio.sleep(2)
                    return ws, current_token
                ciclo_retenido = _safe_int_cycle(estado_bot.get("ciclo_forzado") or estado_bot.get("ciclo_actual") or 1, 1)
                if int(ciclo_orden) < int(ciclo_retenido) and not _owner_state_confirma_ciclo(ciclo_orden):
                    print(Fore.CYAN + Style.BRIGHT + f"🚨 CICLO_REAL_MISMATCH_CRITICO: orden=C{ciclo_orden} < retenido=C{ciclo_retenido}. NO COMPRA. Esperando orden corregida.")
                    await asyncio.sleep(2)
                    return ws, current_token
                if int(ciclo_orden) < int(ciclo_retenido):
                    print(Fore.GREEN + f"✅ HANDSHAKE REAL OK: orden=C{ciclo_orden} confirmada por owner_state fresco")
                estado_bot["ciclo_forzado"] = int(ciclo_orden)
                estado_bot["ciclo_actual"] = int(ciclo_orden)
                print(Fore.GREEN + f"✅ HANDSHAKE REAL OK: orden=C{ciclo_orden} token=REAL:{NOMBRE_BOT}")
                try:
                    real_activado_en_bot = time.time()
                except Exception:
                        pass

            else:
                # Mantén tu mensaje de reafirmación como estaba (sin tocar otras lógicas)
                if not (MODO_SILENCIOSO and estado_bot.get("modo_manual")) and not estado_bot.get("barra_activa", False):
                    if _print_once("rea-REAL", ttl=180):
                        print(Fore.YELLOW + "Reafirmación de REAL (sin reset de martingala)")

        ultimo_token = token_desde_archivo  # mantén vigilante y lazo alineados
        return ws, current_token

async def vigilar_token():
    """Dispara reinicio cuando cambia el archivo token_actual.txt"""
    global ultimo_token
    while not stop_event.is_set():
        await asyncio.sleep(2)
        token_desde_archivo = leer_token_desde_archivo()
        if token_desde_archivo != ultimo_token:
            # BLOQUE 2 y 9: Anti-rebote + commit guard in watcher
            if ultimo_token == TOKEN_REAL and token_desde_archivo == TOKEN_DEMO:
                # SOLO ignorar "rebote a DEMO" si hay ciclo en progreso.
                if estado_bot.get("ciclo_en_progreso") and (commit_guard_active() or (time.time() - real_activado_en_bot < COOLDOWN_REAL_S)):
                    key = _commit_notice_key()
                    if not estado_bot.get("barra_activa", False) and _print_once(key, ttl=180):
                        print(Fore.YELLOW + "Commit REAL o cooldown activo: ignorando rebote a DEMO.")
                    continue
                        
            if estado_bot["ciclo_en_progreso"]:
                if not estado_bot.get("token_msg_mostrado", False):
                    if not (MODO_SILENCIOSO and estado_bot.get("modo_manual")):
                        print(Fore.MAGENTA + Style.BRIGHT + "Cambio de token detectado: cortando ciclo y aplicando de inmediato.")
                        print(Fore.YELLOW + Style.BRIGHT + "⚠️ REAL/DEMO cambió mientras había contrato abierto. Esperando cierre seguro antes de nueva compra." + Style.RESET_ALL)
                    estado_bot["token_msg_mostrado"] = True
                estado_bot["interrumpir_ciclo"] = True
                reinicio_forzado.set()
            else:
                ultimo_token = token_desde_archivo
                reinicio_forzado.set()

async def consultar_saldo_real(ws):
    global saldo_real_last
    try:
        data = await api_call(ws, {"balance": 1}, expect_msg_type="balance", timeout=6.0)
        b = data.get("balance", {}).get("balance")
        if b is not None:
            saldo_real_last = float(b)
            return saldo_real_last
        if _print_once("saldo-real-empty-main", ttl=20):
            print(Fore.YELLOW + "Balance REAL no disponible (respuesta vacía). Intento conexión dedicada...")
    except Exception as e:
        if _print_once("saldo-real-error-main", ttl=20):
            print(Fore.YELLOW + f"Balance por ws actual falló ({e}). Intento conexión dedicada...")
    # Conexión dedicada
    try:
        async with websockets.connect(DERIV_WS_URL, **WS_KW) as ws2:
            await authorize_ws(ws2, TOKEN_REAL, tries=2, timeout=6.0)
            data2 = await api_call(ws2, {"balance": 1}, expect_msg_type="balance", timeout=6.0)
            b2 = data2.get("balance", {}).get("balance")
            if b2 is not None:
                saldo_real_last = float(b2)
                return saldo_real_last
    except Exception as e2:
        if _print_once("saldo-real-error-dedicada", ttl=20):
            print(Fore.RED + Style.BRIGHT + f"[ERROR] al consultar saldo REAL (dedicada): {e2}")
    if _print_once("saldo-real-no-disponible-final", ttl=20):
        print(Fore.YELLOW + "Balance REAL no disponible. Uso último valor válido y **no compro** si no alcanza.")
    return saldo_real_last

# ==================== LÓGICA DE OPERACIÓN ====================
async def buscar_estrategia(ws, ciclo, token):
    await _sync_wait_global_real_clear("pre_search")
    if await _esperar_buy_pause_si_activa(contexto=f"pre_scan C{ciclo}"):
        return "REINTENTAR", None, None, None, None, None, None, None, None, None, None
    print(Fore.MAGENTA + Style.BRIGHT + f"\nBuscando señal válida para Martingala #{ciclo}")
    for intento in range(1, 11):
        if reinicio_forzado.is_set():
            return "REINTENTAR", None, None, None, None, None, None, None, None, None, None
        if MODO_SILENCIOSO and estado_bot.get("modo_manual"):
            if intento in (1, 5, 10):
                print(Fore.YELLOW + f"Intento #{intento} (silencioso)...")
        else:
            print(Fore.YELLOW + f"Intento #{intento}...")
        errores_intento = []
        activos_invalidos = []
        mejores = []
        for symbol in ACTIVOS:
            velas = await obtener_velas(ws, symbol, token, reintentos=4)
            await asyncio.sleep(0.12 + random.uniform(0.0, 0.18))
            if reinicio_forzado.is_set():
                return "REINTENTAR", None, None, None, None, None, None, None, None, None, None
            try:
                if len(velas) < VELAS:
                    activos_invalidos.append(symbol)
                    continue
                condiciones, direccion, rsi9, rsi14, sma5, sma20, breakout, cruce, rsi_reversion = evaluar_estrategia(velas)
                if condiciones >= 2:
                    score = puntuar_setups(condiciones, direccion, rsi9, rsi14, sma5, sma20, breakout, cruce, rsi_reversion)
                    if setup_pasa_filtro(score, condiciones):
                        close_snapshot = _extract_close_snapshot(velas, n=20)
                        mejores.append((score, condiciones, symbol, direccion, rsi9, rsi14, sma5, sma20, breakout, cruce, rsi_reversion, close_snapshot))
                    else:
                        activos_invalidos.append(symbol)
                else:
                    activos_invalidos.append(symbol)
            except Exception as e:
                errores_intento.append(symbol)

        if mejores:
            # Prioridad: mayor score; desempate por más condiciones
            mejores.sort(key=lambda x: (x[0], x[1]), reverse=True)
            score, condiciones, symbol, direccion, rsi9, rsi14, sma5, sma20, breakout, cruce, rsi_reversion, close_snapshot = mejores[0]
            estado_bot["score_senal"] = float(score)
            print(Fore.GREEN + Style.BRIGHT + f"Estrategia válida en {symbol} | Dirección: {direccion} | Condiciones: {condiciones}/3 | Score={score:.3f}")
            return symbol, direccion, rsi9, rsi14, sma5, sma20, breakout, cruce, condiciones, rsi_reversion, close_snapshot

        if errores_intento:
            print(Fore.RED + f"Error WS en activos: {', '.join(errores_intento)} | Intento #{intento}")
        if activos_invalidos:
            msg_sil = (MODO_SILENCIOSO and estado_bot.get("modo_manual"))
            if not msg_sil:
                print(Fore.YELLOW + f"Ningún activo válido en intento #{intento}. Esperando 15s...")
            elif intento in (1, 5, 10):
                print(Fore.YELLOW + f"Sin activo válido (intento #{intento}, silencioso). Esperando 15s...")
        # Nueva lógica: si todos salieron inválidos y la racha de 1006 es alta, pide reconexión
        if len(activos_invalidos) == len(ACTIVOS) and _ws_fail_streak >= len(ACTIVOS):
            if _print_once("ws-reopen-needed", ttl=15):
                print(Fore.YELLOW + Style.BRIGHT + "Múltiples 1006 detectados en barrido. Señalando reconexión limpia del WS...")
            ws_reset_needed.set()
            # No seguimos martillando: pequeño respiro
            await asyncio.sleep(1.0 + random.uniform(0.0, 0.5))  # Jitter
        await asyncio.sleep(15 + random.uniform(0.0, 0.5))  # Jitter para pausas
    print(Fore.RED + Style.BRIGHT + f"No se encontró activo válido tras 10 intentos para Martingala #{ciclo}. Reintentando MISMO ciclo...")
    try:
        play_sfx("REINTENTA", vol=0.8)
    except Exception:
        pass
    await asyncio.sleep(30)
    return "REINTENTAR", None, None, None, None, None, None, None, None, None, None

async def esperar_resultado(ws, contract_id, symbol, direccion, monto, rsi9, rsi14, sma5, sma20, cruce, breakout, rsi_reversion, ciclo, payout, condiciones, token_usado_buy, epoch_pretrade=None, trade_uid=None, close_snapshot=None):
    # ✅ SIEMPRE cerramos/logueamos con el token real del BUY (aunque el maestro cambie token_actual.txt)
    token_antes = token_usado_buy
    print(Fore.CYAN + "=" * 80)
    estado_bot["barra_activa"] = True
    try:
        for i in range(60):
            # ¿Pediron corte inmediato? Desprendemos y liberamos YA.
            if estado_bot.get("interrumpir_ciclo"):
                remaining = 60 - i
                print(Fore.MAGENTA + Style.BRIGHT + "\nToken cambió: finalizo contrato en segundo plano y libero el ciclo.")
                # No reutilizar 'ws' para evitar choques de recv: usa una conexión propia
                asyncio.create_task(finalizar_contrato_bg(
                    contract_id, remaining, symbol, direccion, monto,
                    rsi9, rsi14, sma5, sma20, cruce, breakout, rsi_reversion,
                    ciclo, payout, condiciones, token_antes, epoch_pretrade=epoch_pretrade, trade_uid=trade_uid, close_snapshot=close_snapshot
                ))
                estado_bot["interrumpir_ciclo"] = False
                estado_bot["ciclo_en_progreso"] = False
                estado_bot["token_msg_mostrado"] = False
                return "INDEFINIDO", 0.0  # libera al loop para reautorizar ya
            if MODO_SILENCIOSO and estado_bot.get("modo_manual"):
                if i in (0, 29, 59):
                    barra = (
                        f"\r[{'█' * (i + 1)}{' ' * (59 - i)}] "
                        f"{i + 1:02d}s | C{ciclo} {symbol} {direccion} (silencioso)"
                    )
                    sys.stdout.write(barra)
                    sys.stdout.flush()
            else:
                barra = (
                    f"\r[{'█' * (i + 1)}{' ' * (59 - i)}] "
                    f"{i + 1:02d}s | C{ciclo} {symbol} {direccion}"
                )
                sys.stdout.write(barra)
                sys.stdout.flush()
            await asyncio.sleep(1 + random.uniform(0.0, 0.1))  # Pequeño jitter para stability
        print("\n" + "=" * 80)
        print(Fore.CYAN + Style.BRIGHT + "\nFinalizando contrato...")
        try:
            data = await api_call(ws, {"proposal_open_contract": 1, "contract_id": contract_id}, expect_msg_type="proposal_open_contract")
            poc = data.get("proposal_open_contract", {})
            profit = float(poc.get("profit", 0.0))
            resultado = "GANANCIA" if profit > 0 else "PÉRDIDA"
            # === PATCH SFX resultado principal ===
            try:
                if token_antes == TOKEN_REAL:
                    if resultado == "GANANCIA":
                        play_sfx("FELICITACIONES", vol=1.0)
                    else:
                        play_sfx("LO_SIENTO", vol=0.9)
            except Exception:
                pass
            # === /PATCH SFX ===
            color = Fore.GREEN if profit > 0 else Fore.RED
            print(color + Style.BRIGHT + f"{resultado}: {profit:.2f} USD")
            # >>> PATCH BLOQUE 3 y 5
            if contract_id in _contratos_procesados:
                return resultado, profit
            _contratos_procesados.add(contract_id)
            # <<< PATCH
            # Registrar resultado SOLO si es definido, con features enriquecidas
            try:
                global racha_actual_bot
                # 1) Actualizar racha del bot
                racha_anterior = racha_actual_bot
                if resultado == "GANANCIA":
                    racha_actual_bot = racha_actual_bot + 1 if racha_actual_bot > 0 else 1
                else:  # "PÉRDIDA"
                    racha_actual_bot = racha_actual_bot - 1 if racha_actual_bot < 0 else -1
                # 2) Detectar rebote (PRE-TRADE, sin fuga):
                #    Si veníamos de racha negativa larga ANTES del trade, marcamos rebote potencial.
                es_rebote_flag = 1 if (racha_anterior <= -4) else 0

                # 3) Escribir fila en CSV
                # ==========================================================
                # payout robusto (CIERRE NORMAL):
                # - si payout <= 3.5 => es payout_multiplier (ratio_total)
                # - si payout > 3.5  => es payout_total (USD)
                # Resultado SIEMPRE coherente:
                #   payout_total_f y ratio_total
                # ==========================================================
                payout_total_f = 0.0
                ratio_total = 0.0
                # monto
                try:
                    monto_f = float(monto) if monto not in (None, "", "nan", "NaN") else 0.0
                except Exception:
                    monto_f = 0.0

                # payout (puede venir como multiplier o como total)
                try:
                    p = float(payout) if payout not in (None, "", "nan", "NaN") else 0.0
                except Exception:
                    p = 0.0
                # si p es NaN/inf, lo anulamos
                try:
                    if not math.isfinite(p):
                        p = 0.0
                    if not math.isfinite(monto_f):
                        monto_f = 0.0
                except Exception:
                    pass                   
                try:
                    if p > 0 and p <= 3.5:
                        # payout viene como multiplier (1.95 etc.)
                        ratio_total = p
                        payout_total_f = (monto_f * ratio_total) if monto_f > 0 else 0.0
                    elif p > 3.5:
                        # payout viene como total (USD)
                        payout_total_f = p
                        ratio_total = (payout_total_f / monto_f) if monto_f > 0 else 0.0
                    else:
                        payout_total_f = 0.0
                        ratio_total = 0.0
                except Exception:
                    payout_total_f = 0.0
                    ratio_total = 0.0

                now = datetime.now(timezone.utc)
                epoch_val = int(epoch_pretrade) if epoch_pretrade is not None else int(now.timestamp())
                ts_val = now.isoformat()
                
                async with csv_lock:
                    # ==========================
                    # CIERRE CERRADO (DICT MODERNO)
                    # ==========================
                    puntaje01 = _norm_puntaje_01(condiciones)  # helper REAL del bot
                    ack_ctx = estado_bot.get("ack_ctx", {}) if isinstance(estado_bot.get("ack_ctx", {}), dict) else {}
                    ia_prob_en_juego = ack_ctx.get("ia_prob_en_juego", "")
                    ia_prob_source = str(ack_ctx.get("ia_prob_source", "") or "").strip()
                    ia_ready_ack = bool(ack_ctx.get("ia_ready_ack", False))
                    if isinstance(ia_prob_en_juego, (int, float)):
                        ia_prob_source = ia_prob_source or "HUD"
                        ia_ready_ack = True
                    else:
                        ia_prob_source = ia_prob_source or "NO_READY"

                    trade_uid_final = str(trade_uid or "").strip()
                    if not trade_uid_final:
                        trade_uid_final = _build_trade_uid(epoch_val, symbol, direccion, ciclo, token_antes, ts_iso=ts_val)
                    modo_cuenta_csv = _resolver_modo_cuenta_csv(
                        token_actual=token_antes,
                        modo_actual="REAL" if token_antes == TOKEN_REAL else "DEMO",
                    )
                    row_dict = {
                        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "activo": symbol,
                        "direction": direccion,
                        "monto": float(monto_f),
                        "resultado": resultado,
                        "ganancia_perdida": float(f"{profit:.2f}"),
                        "rsi_9": rsi9,
                        "rsi_14": rsi14,
                        "sma_5": sma5,
                        "sma_20": sma20,
                        "cruce_sma": int(cruce),
                        "breakout": int(breakout),
                        "rsi_reversion": int(rsi_reversion),
                        "racha_actual": int(racha_anterior),
                        "es_rebote": int(es_rebote_flag),
                        "ciclo_martingala": int(ciclo),
                        "payout_total": float(round(payout_total_f, 2)),
                        "payout_multiplier": float(round(float(ratio_total), 6)),
                        "puntaje_estrategia": float(round(float(puntaje01), 6)),
                        "result_bin": 1 if resultado == "GANANCIA" else 0 if resultado == "PÉRDIDA" else "",
                        "trade_status": "CERRADO",
                        "modo_cuenta": modo_cuenta_csv,
                        "epoch": int(epoch_val),
                        "ts": ts_val,
                        "ia_prob_en_juego": ia_prob_en_juego,
                        "ia_prob_source": ia_prob_source,
                        "ia_decision_id": trade_uid_final,
                        "ia_gate_real": ack_ctx.get("ia_gate_real", ""),
                        "ia_modo_ack": ack_ctx.get("ia_modo_ack", ""),
                        "ia_ready_ack": ia_ready_ack,
                    }
                    closes = _extract_close_snapshot(close_snapshot, n=20)
                    _warn_close_snapshot_insuficiente(closes)
                    for i, c in enumerate(closes):
                        row_dict[f"close_{i}"] = "" if c is None else float(c)
                    _write_row_dict_atomic(ARCHIVO_CSV, row_dict)

            except Exception as csv_e:
                print(Fore.RED + f"[ERROR] al escribir CSV: {csv_e}")
            # Calcular y mostrar % de éxito acumulado (solo cierres auditables)
            try:
                total_cerrados, ganancias, pendientes = _audit_csv_trade_metrics(ARCHIVO_CSV)

                if total_cerrados:
                    porcentaje_exito = (ganancias / total_cerrados) * 100
                    print(f"Éxito acumulado en {ARCHIVO_CSV}: {ganancias}/{total_cerrados} = {porcentaje_exito:.2f}%")
                else:
                    print(
                        f"Éxito acumulado en {ARCHIVO_CSV}: sin cierres auditables aún "
                        f"(pendientes={pendientes})"
                    )
            except Exception as e:
                print(f"No se pudo calcular % de éxito: {type(e).__name__}: {e!r}")

            # Acumular profit separado
            if token_antes == TOKEN_REAL:
                resultado_global["real"] += profit
            else:
                resultado_global["demo"] += profit
            # Si fue GANANCIA en REAL -> reproducir sonido (sin tocar token)
            if resultado == "GANANCIA" and token_antes == TOKEN_REAL:
                try:
                    if PYGAME_OK:
                        pygame.mixer.music.load("ganabot.wav")
                        pygame.mixer.music.play()
                except Exception:
                    pass
                print(Fore.GREEN + Style.BRIGHT + "GANANCIA en cuenta REAL! (token lo maneja 5R6M; sigo en sesión)")
            # BLOQUE 2: Clear commit guard after REAL result
            if token_antes == TOKEN_REAL:
                commit_guard_clear()
            # >>> PATCH BLOQUE 5
            print(Fore.CYAN + f"Ciclo #{ciclo} | {symbol} {direccion} | payout={float(payout or 0):.2f} | {resultado} {profit:+.2f} USD")
            # <<< PATCH
            return resultado, profit
        except websockets.exceptions.ConnectionClosed:
            if _print_once("no-close-frame", ttl=15):
                print(Fore.YELLOW + "WS cerrado sin close frame (resolverá en background). Mismo ciclo.")
            try:
                play_sfx("REINTENTA", vol=0.8)
            except Exception:
                pass
            return "INDEFINIDO", 0.0
        except Exception as e:
            print(Fore.RED + Style.BRIGHT + f"[ERROR] Resultado INDEFINIDO: {e}. Reintentando mismo ciclo...")
            try:
                play_sfx("REINTENTA", vol=0.8)
            except Exception:
                pass
            return "INDEFINIDO", 0.0
    finally:
        estado_bot["barra_activa"] = False
        _flush_log_buffer()

async def finalizar_contrato_bg(contract_id, remaining, symbol, direccion, monto,
                                rsi9, rsi14, sma5, sma20, cruce, breakout, rsi_reversion,
                                ciclo, payout, condiciones, token_usado, epoch_pretrade=None, trade_uid=None, close_snapshot=None):
    """
    Finaliza un contrato en background cuando hubo cambio de token / reinicio.
    Importante IA:
    - es_rebote debe ser PRE-TRADE (racha previa <= -4), NO depender del resultado (sin fuga).
    """
    try:
        if remaining and remaining > 0:
            await asyncio.sleep(remaining)

        # === Consultar contrato ===
        async with websockets.connect(DERIV_WS_URL, **WS_KW) as ws_bg:
            await api_call(ws_bg, {"authorize": token_usado}, expect_msg_type=None)
            data = await api_call(
                ws_bg,
                {"proposal_open_contract": 1, "contract_id": contract_id},
                expect_msg_type="proposal_open_contract"
            )

        poc = data.get("proposal_open_contract", {}) if isinstance(data, dict) else {}
        profit = float(poc.get("profit", 0.0) or 0.0)
        resultado = "GANANCIA" if profit > 0 else "PÉRDIDA"

        # === SFX BG (solo REAL) ===
        try:
            if token_usado == TOKEN_REAL:
                if resultado == "GANANCIA":
                    play_sfx("FELICITACIONES", vol=1.0)
                else:
                    play_sfx("LO_SIENTO", vol=0.9)
        except Exception:
            pass

        # === Evitar doble commit por mismo contrato ===
        if contract_id in _contratos_procesados:
            return
        _contratos_procesados.add(contract_id)

        # === IA / racha / es_rebote (SIN FUGA) ===
        try:
            global racha_actual_bot

            racha_anterior = int(racha_actual_bot)

            # actualizar racha con el resultado (esto es post-trade, OK)
            if resultado == "GANANCIA":
                racha_actual_bot = racha_actual_bot + 1 if racha_actual_bot > 0 else 1
            else:
                racha_actual_bot = racha_actual_bot - 1 if racha_actual_bot < 0 else -1

            # es_rebote PRE-TRADE: venías de 4+ pérdidas antes de este trade
            es_rebote_flag = 1 if (racha_anterior <= -4) else 0

        except Exception:
            racha_anterior = int(racha_actual_bot) if "racha_actual_bot" in globals() else 0
            es_rebote_flag = 1 if (racha_anterior <= -4) else 0

        # === Escribir fila resultado en CSV enriquecido ===
        now = datetime.now(timezone.utc)
        epoch_val = int(epoch_pretrade) if epoch_pretrade is not None else int(now.timestamp())
        ts_val = now.isoformat()

        # ==========================================================
        # payout robusto:
        # - si payout <= 3.5 => es payout_multiplier (ratio_total)
        # - si payout > 3.5  => es payout_total (USD)
        # Guardamos SIEMPRE:
        #   payout_total = monto * payout_multiplier
        #   payout_multiplier = payout_total / monto
        # ==========================================================
        payout_total = 0.0
        payout_ratio_total = 0.0

        # monto
        try:
            monto_f = float(monto) if monto not in (None, "", "nan", "NaN") else 0.0
        except Exception:
            monto_f = 0.0

        # payout (puede venir como multiplier o como total)
        try:
            p = float(payout) if payout not in (None, "", "nan", "NaN") else 0.0
        except Exception:
            p = 0.0

        # si p es NaN/inf, lo anulamos
        try:
            if not math.isfinite(p):
                p = 0.0
            if not math.isfinite(monto_f):
                monto_f = 0.0
        except Exception:
            pass

        try:
            if p > 0 and p <= 3.5:
                # payout viene como multiplier (1.95 etc.)
                payout_ratio_total = p
                payout_total = (monto_f * payout_ratio_total) if monto_f > 0 else 0.0
            elif p > 3.5:
                # payout viene como total (15.62 etc.)
                payout_total = p
                payout_ratio_total = (payout_total / monto_f) if monto_f > 0 else 0.0
            else:
                payout_total = 0.0
                payout_ratio_total = 0.0
        except Exception:
            payout_total = 0.0
            payout_ratio_total = 0.0

        async with csv_lock:
            # ==========================
            # CIERRE BG CERRADO (DICT MODERNO)
            # ==========================
            try:
                monto_f = float(monto or 0.0)
            except Exception:
                monto_f = 0.0
            try:
                payout_total_f = float(payout_total or 0.0)
            except Exception:
                payout_total_f = 0.0
            try:
                payout_mult_f = float(payout_ratio_total or 0.0)
            except Exception:
                payout_mult_f = 0.0
            puntaje01 = _norm_puntaje_01(condiciones)  # helper REAL del bot
            trade_uid_final = str(trade_uid or "").strip()
            if not trade_uid_final:
                trade_uid_final = _build_trade_uid(epoch_val, symbol, direccion, ciclo, token_usado, ts_iso=ts_val)
            modo_cuenta_csv = _resolver_modo_cuenta_csv(
                token_actual=token_usado,
                modo_actual="REAL" if token_usado == TOKEN_REAL else "DEMO",
            )
            row_dict = {
                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "activo": symbol,
                "direction": direccion,
                "monto": float(monto_f),
                "resultado": resultado,
                "ganancia_perdida": float(f"{profit:.2f}"),
                "rsi_9": rsi9,
                "rsi_14": rsi14,
                "sma_5": sma5,
                "sma_20": sma20,
                "cruce_sma": int(cruce),
                "breakout": int(breakout),
                "rsi_reversion": int(rsi_reversion),
                "racha_actual": int(racha_anterior),
                "es_rebote": int(es_rebote_flag),
                "ciclo_martingala": int(ciclo),
                "payout_total": float(round(payout_total_f, 2)),
                "payout_multiplier": float(round(payout_mult_f, 6)),
                "puntaje_estrategia": float(round(float(puntaje01), 6)),
                "result_bin": 1 if resultado == "GANANCIA" else 0 if resultado == "PÉRDIDA" else "",
                "trade_status": "CERRADO",
                "modo_cuenta": modo_cuenta_csv,
                "epoch": int(epoch_val),
                "ts": ts_val,
                "ia_decision_id": trade_uid_final,
            }
            closes = _extract_close_snapshot(close_snapshot, n=20)
            _warn_close_snapshot_insuficiente(closes)
            for i, c in enumerate(closes):
                row_dict[f"close_{i}"] = "" if c is None else float(c)
            _write_row_dict_atomic(ARCHIVO_CSV, row_dict)
        # === Logs ===
        msg = Fore.CYAN + f"Contrato #{contract_id} finalizado en background: {resultado} {profit:.2f} USD"
        if estado_bot.get("barra_activa", False):
            _buffer_log(msg)
        else:
            print(msg)

        # Clear commit guard cuando REAL finaliza en BG
        if token_usado == TOKEN_REAL:
            commit_guard_clear()

        msg2 = Fore.CYAN + (
            f"Ciclo #{ciclo} | {symbol} {direccion} | payout={float(payout or 0):.2f} | "
            f"{resultado} {profit:+.2f} USD"
        )
        if estado_bot.get("barra_activa", False):
            _buffer_log(msg2)
        else:
            print(msg2)

        try:
            _clear_pending_contract_resolution(reason="bg_resolved")
            print(Fore.GREEN + Style.BRIGHT + "✅ Fence contrato incierto liberado (bg_resolved)." + Style.RESET_ALL)
        except Exception:
            pass

        try:
            estado_bot["ciclo_en_progreso"] = False
            estado_bot["token_msg_mostrado"] = False
        except Exception:
            pass

        try:
            reinicio_forzado.set()
        except Exception:
            pass

    except Exception as e:
        msg = Fore.YELLOW + f"finalizar_contrato_bg: {type(e).__name__}: {e!r}"
        if estado_bot.get("barra_activa", False):
            _buffer_log(msg)
        else:
            print(msg)
        return

async def leer_csv():
    """Lee el archivo CSV y devuelve los registros."""
    registros = []
    try:
        with open(ARCHIVO_CSV, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                print(Fore.YELLOW + "CSV vacío o sin encabezado.")
                return registros
            for row in reader:
                registros.append(row)
        print(Fore.GREEN + f"Leídos {len(registros)} registros del CSV.")
        return registros
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"[ERROR] al leer CSV: {e}")
        return []

def _fmt_saldo_display(value, ok: bool, label: str, cached: bool = False) -> str:
    pref = f"Saldo cuenta {label}" + (" (cached)" if cached else "")
    if not ok:
        return f"{pref}: N/D (último saldo no disponible todavía)"
    try:
        return f"{pref}: {float(value):.2f} USD"
    except Exception:
        return f"{pref}: N/D"

async def mostrar_saldos():
    global saldo_demo_last, saldo_real_last, saldo_demo_ok, saldo_real_ok, _last_saldo_ts
    print(Fore.GREEN + Style.BRIGHT + "\nConsultando Saldos")

    # BLOQUE 8: Rate-limit with cache
    if time.time() - _last_saldo_ts < REFRESCO_SALDO:
        print(Fore.LIGHTBLUE_EX + Style.BRIGHT + _fmt_saldo_display(saldo_demo_last, saldo_demo_ok, "DEMO", cached=True))
        print(Fore.YELLOW + Style.BRIGHT + _fmt_saldo_display(saldo_real_last, saldo_real_ok, "REAL", cached=True))
        print(Fore.GREEN + "─" * 80)
        return

    saldo_demo = saldo_demo_last
    saldo_real = saldo_real_last

    # DEMO
    try:
        async with websockets.connect(DERIV_WS_URL, **WS_KW) as ws:  # BLOQUE 1.2
            await authorize_ws(ws, TOKEN_DEMO, tries=2, timeout=6.0)
            data = await api_call(ws, {"balance": 1}, expect_msg_type="balance")
            b = data.get("balance", {}).get("balance")
            if b is not None:
                saldo_demo = float(b)
                saldo_demo_last = saldo_demo
                saldo_demo_ok = True
            else:
                if _print_once("saldo-demo-empty", ttl=REFRESCO_SALDO):
                    print(Fore.YELLOW + "Balance DEMO no disponible, usando último valor válido.")
    except Exception as e:
        if _print_once("saldo-demo-error", ttl=REFRESCO_SALDO):
            print(Fore.YELLOW + Style.BRIGHT + f"[WARN] saldo DEMO: {type(e).__name__}: {e!r}")
            print(Fore.YELLOW + "Balance DEMO no disponible, usando último valor válido.")

    # REAL
    try:
        async with websockets.connect(DERIV_WS_URL, **WS_KW) as ws:  # BLOQUE 1.2
            await authorize_ws(ws, TOKEN_REAL, tries=2, timeout=6.0)
            data = await api_call(ws, {"balance": 1}, expect_msg_type="balance")
            b = data.get("balance", {}).get("balance")
            if b is not None:
                saldo_real = float(b)
                saldo_real_last = saldo_real
                saldo_real_ok = True
            else:
                if _print_once("saldo-real-empty", ttl=REFRESCO_SALDO):
                    print(Fore.YELLOW + "Balance REAL no disponible, usando último valor válido.")
    except Exception as e:
        if _print_once("saldo-real-error", ttl=REFRESCO_SALDO):
            print(Fore.YELLOW + Style.BRIGHT + f"[WARN] saldo REAL: {type(e).__name__}: {e!r}")
            print(Fore.YELLOW + "Balance REAL no disponible, usando último valor válido.")

    print(Fore.LIGHTBLUE_EX + Style.BRIGHT + _fmt_saldo_display(saldo_demo, saldo_demo_ok, "DEMO"))
    print(Fore.YELLOW + Style.BRIGHT + _fmt_saldo_display(saldo_real, saldo_real_ok, "REAL"))
    print(Fore.GREEN + "─" * 80)
    print(Fore.GREEN + "─" * 80)
    _last_saldo_ts = time.time()


# ==================== LOOP PRINCIPAL ====================
async def ejecutar_panel():
    global ultimo_token
    global _ws_fail_streak

    # Eliminado: reset_csv_and_total() para acumular histórico completo
    await mostrar_saldos()
    # =================== PATCH CSV (SOLO) ===================
    global _CSV_REPARADO_1VEZ
    if not _CSV_REPARADO_1VEZ:
        reparar_csv_esrebote_ciclo(ARCHIVO_CSV)
        _CSV_REPARADO_1VEZ = True
    # ================= FIN PATCH CSV (SOLO) =================
   
    
    async def _cerrar_ws(_ws):
        try:
            if _ws is not None:
                await _ws.close()
        except Exception:
            pass

    async def _abrir_ws(token: str, tries: int = 4):
        last = None
        for intento in range(1, tries + 1):
            try:
                _ws = await websockets.connect(DERIV_WS_URL, **WS_KW)
                await authorize_ws(_ws, token)
                return _ws
            except Exception as e:
                last = e
                if _es_error_transitorio_ws(e):
                    espera = min(6.0, 0.8 * intento + random.uniform(0.0, 0.6))
                    if _print_once(f"ws-open-retry-{intento}", ttl=2):
                        print(Fore.YELLOW + f"WS/NET inestable al abrir sesión ({type(e).__name__}). Reintento {intento}/{tries} en {espera:.1f}s...")
                    await asyncio.sleep(espera)
                    continue
                raise
        raise last

    ws = None
    try:
        current_token = leer_token_desde_archivo()
        ultimo_token = current_token  # ✅ evita reinicio fantasma del watcher al inicio
        ws = await _abrir_ws(current_token)
        estado_bot["sync_round_id"] = _sync_round_resolve_start_round()

        indefinidos_consecutivos = 0  # Contador para indefinidos por ciclo

        while not stop_event.is_set():

            # ========= REINICIO FORZADO (token / watcher) =========
            if reinicio_forzado.is_set():
                estado_bot["reinicios_consecutivos"] += 1
                if estado_bot["reinicios_consecutivos"] > 5:
                    print(Fore.RED + "Demasiados reinicios consecutivos. Fallback a ciclo #1 + backoff 5s.")
                    estado_bot["ciclo_forzado"] = 1
                    estado_bot["reinicios_consecutivos"] = 0
                    await asyncio.sleep(5)

                print(Fore.YELLOW + Style.BRIGHT + "Reinicio forzado detectado. (reconectando sin salir)")
                reinicio_forzado.clear()
                indefinidos_consecutivos = 0

                await _cerrar_ws(ws)
                ws = None
                new_token = leer_token_desde_archivo()
                ws = await _abrir_ws(new_token)

                current_token = new_token
                ultimo_token = new_token
                await asyncio.sleep(0.6 + random.uniform(0.0, 0.5))
                continue

            # ========= ARRANQUE DE MARTINGALA =========
            modo_real = (current_token == TOKEN_REAL)
            if modo_real:
                if not estado_bot.get("barra_activa", False) and _print_once("real-start-msg", ttl=120):
                    hora = ""
                    try:
                        hora = datetime.now().strftime("%H:%M:%S")
                    except Exception:
                        hora = ""
                    print(
                        Fore.LIGHTRED_EX + Back.WHITE + Style.BRIGHT
                        + f"\n🚨 {NOMBRE_BOT.upper()} MODO REAL ACTIVADO {hora} 🚨"
                        + Style.RESET_ALL
                    )

            if not modo_real:
                _sync_round_adopt_official_if_stale(motivo="pre_demo_buy")

            martingala = MARTINGALA_REAL if modo_real else MARTINGALA_DEMO

            sep_ciclo()
            ciclo_orden, _ts, _quiet, _src = leer_orden_real(NOMBRE_BOT)
            ciclo_forzado = estado_bot.get("ciclo_forzado")
            ciclo_maestro = None
            try:
                if ciclo_orden is not None:
                    ciclo_orden_int = int(ciclo_orden)
                    if 1 <= ciclo_orden_int <= MAX_CICLOS:
                        ciclo_maestro = ciclo_orden_int
            except Exception:
                ciclo_maestro = None

            try:
                ciclo_forzado = int(ciclo_forzado) if ciclo_forzado is not None else None
            except Exception:
                ciclo_forzado = None
            if ciclo_forzado is not None and not (1 <= ciclo_forzado <= MAX_CICLOS):
                ciclo_prev = ciclo_forzado
                ciclo_forzado = max(1, min(ciclo_forzado, MAX_CICLOS))
                print(Fore.YELLOW + f"⚠️ ciclo>MAX_CICLOS detectado, normalizado a C{ciclo_forzado} (retenido=C{ciclo_prev})")

            ciclo = ciclo_maestro if ciclo_maestro is not None else ciclo_forzado
            if modo_real:
                now_guard = time.time()
                last_guard = float(estado_bot.get("real_cycle_guard_last_ts", 0.0) or 0.0)
                if (now_guard - last_guard) >= 8.0:
                    if ciclo_maestro is None and ciclo_forzado is not None:
                        print(Fore.YELLOW + "REAL sin orden viva del maestro, usando retenido")
                    elif ciclo_maestro is None and ciclo_forzado is None:
                        print(Fore.YELLOW + "🚨 TOKEN_REAL_SIN_ORDEN_VALIDA: no compro, espero orden maestro.")
                        await asyncio.sleep(2)
                        continue
                    estado_bot["real_cycle_guard_last_ts"] = now_guard
            if modo_real and ciclo_maestro is None:
                print(Fore.YELLOW + "🚨 TOKEN_REAL_SIN_ORDEN_VALIDA: no compro, espero orden maestro.")
                await asyncio.sleep(2)
                continue
            if modo_real and ciclo_maestro is not None and ciclo_forzado is not None and int(ciclo_maestro) < int(ciclo_forzado):
                if not _owner_state_confirma_ciclo(ciclo_maestro):
                    print(
                        Fore.CYAN + Style.BRIGHT +
                        f"🚨 CICLO_REAL_MISMATCH_CRITICO: orden=C{ciclo_maestro} < retenido=C{ciclo_forzado}. NO COMPRA. Esperando orden corregida."
                    )
                    await asyncio.sleep(2)
                    continue
                print(Fore.GREEN + f"✅ HANDSHAKE REAL OK: orden=C{ciclo_maestro} confirmada por owner_state fresco")
            if modo_real and estado_bot.get("real_first_cycle_reset_pending"):
                if ciclo_maestro is not None:
                    print(Fore.YELLOW + f"Primer ciclo REAL confirmado por maestro en C{ciclo_maestro}")
                elif ciclo_forzado is not None:
                    print(Fore.YELLOW + "🚨 TOKEN_REAL_SIN_ORDEN_VALIDA: no compro, espero orden maestro.")
                    await asyncio.sleep(2)
                    continue
                else:
                    print(Fore.YELLOW + "🚨 TOKEN_REAL_SIN_ORDEN_VALIDA: no compro, espero orden maestro.")
                    await asyncio.sleep(2)
                    continue
                estado_bot["real_first_cycle_reset_pending"] = False

            estado_bot["ciclo_forzado"] = None
            estado_bot["reinicios_consecutivos"] = 0
            N = len(martingala)
            indefinidos_consecutivos = 0

            while ciclo <= N and (not stop_event.is_set()):

                monto = martingala[ciclo - 1]
                estado_bot["ciclo_actual"] = int(ciclo)

                # Sync token/WS con el maestro (sin perder ciclo)
                ws, current_token = await check_token_and_reconnect(ws, current_token)

                if reinicio_forzado.is_set():
                    estado_bot["ciclo_forzado"] = ciclo
                    proximo = estado_bot.get("ciclo_forzado") or ciclo
                    print(Fore.YELLOW + Style.BRIGHT + f"Reinicio forzado durante ciclo. Ciclo actual #{ciclo} → siguiente #{proximo}.")
                    reinicio_forzado.clear()
                    await asyncio.sleep(2)
                    indefinidos_consecutivos = 0
                    break

                modo_real = (current_token == TOKEN_REAL)
                martingala = MARTINGALA_REAL if modo_real else MARTINGALA_DEMO

                # Fence conservador: si hay contrato incierto pendiente, no imprimir cabecera ni comprar
                if estado_bot.get("pending_contract_resolution"):
                    if not await _pending_contract_fence_tick(ws):
                        continue
                    if not modo_real:
                        await _sync_wait_global_real_clear("post_fence_before_retry")

                if not modo_real:
                    await _sync_wait_global_real_clear("pre_search")

                print(Fore.CYAN + Style.BRIGHT + "=" * 80)
                titulo = f"{NOMBRE_BOT.upper()} | MODO {'REAL' if modo_real else 'DEMO'} | CICLO #{ciclo}/{len(martingala)}"
                print(Fore.CYAN + Style.BRIGHT + titulo.center(80))
                print(Fore.CYAN + Style.BRIGHT + "=" * 80)

                # Fence conservador: si hay contrato incierto pendiente, no permitir nuevas compras
                if estado_bot.get("pending_contract_resolution"):
                    if not await _pending_contract_fence_tick(ws):
                        continue
                    if not modo_real:
                        await _sync_wait_global_real_clear("post_fence_before_retry")

                # Salud WS (si buscar_estrategia detectó 1006 masivos)
                if ws_reset_needed.is_set():
                    await _cerrar_ws(ws)
                    ws = await _abrir_ws(current_token)
                    _ws_fail_streak = 0
                    ws_reset_needed.clear()
                    if _print_once("ws-reopened", ttl=20):
                        print(Fore.CYAN + Style.BRIGHT + "WS reabierto por salud. Retomando MISMO ciclo.")
                    await asyncio.sleep(0.6 + random.uniform(0.0, 0.5))

                # ========= BUSCAR SEÑAL =========
                symbol, direccion, rsi9, rsi14, sma5, sma20, breakout, cruce, condiciones, rsi_reversion, close_snapshot = await buscar_estrategia(ws, ciclo, current_token)

                if symbol == "REINTENTAR" or symbol is None:
                    continue

                if not all([direccion, rsi9 is not None, rsi14 is not None]):
                    print(Fore.YELLOW + "Datos de estrategia incompletos. Reintentando ciclo.")
                    continue

                # Rechequeo token justo antes de avanzar
                ws, current_token = await check_token_and_reconnect(ws, current_token)

                if reinicio_forzado.is_set():
                    estado_bot["ciclo_forzado"] = ciclo
                    print(Fore.YELLOW + Style.BRIGHT + f"Reinicio forzado tras buscar estrategia. Mantengo ciclo #{ciclo}.")
                    reinicio_forzado.clear()
                    await asyncio.sleep(2)
                    indefinidos_consecutivos = 0
                    break

                modo_real_now = (current_token == TOKEN_REAL)
                if modo_real_now != modo_real:
                    estado_bot["ciclo_forzado"] = ciclo
                    print(Fore.YELLOW + Style.BRIGHT + "Token cambió justo antes de validar saldo/compra. Reinicio limpio para mantener sincronía con el maestro.")
                    reinicio_forzado.set()
                    break

                # ========= SALDO REAL (si aplica) =========
                if modo_real:
                    saldo = await consultar_saldo_real(ws)
                    if saldo < monto:
                        estado_bot["intentos_saldo"] += 1
                        if estado_bot["intentos_saldo"] > 3:
                            print(Fore.RED + Style.BRIGHT + "Saldo no recuperado tras 3 intentos. Paso a DEMO.")
                            try:
                                play_sfx("NO_PASAR_REAL", vol=0.9)
                            except Exception:
                                pass
                            # ✅ Liberación segura (CAS): solo si aún soy el dueño del REAL
                            release_real_token_if_owned()
                            estado_bot["intentos_saldo"] = 0
                            reinicio_forzado.set()
                        else:
                            print(Fore.RED + Style.BRIGHT + f"Saldo REAL insuficiente: {saldo:.2f} < {monto:.2f}. Espero y reintento MISMO ciclo ({estado_bot['intentos_saldo']}/3).")
                            await asyncio.sleep(15 + random.uniform(0.0, 0.5))
                        continue

                # ========= REVALIDACIÓN PRE-BUY =========
                if not modo_real:
                    await _sync_wait_global_real_clear("pre_revalidation")
                try:
                    score_sel = estado_bot.get("score_senal")
                    velas_rv = await obtener_velas(ws, symbol, current_token, reintentos=2)
                    if velas_rv and len(velas_rv) >= int(REVALIDAR_VELAS_N):
                        cond2, dir2, rsi9_2, rsi14_2, sma5_2, sma20_2, br2, cr2, rev2 = evaluar_estrategia(velas_rv)
                        score2 = puntuar_setups(cond2, dir2, rsi9_2, rsi14_2, sma5_2, sma20_2, br2, cr2, rev2)
                        piso = float(SCORE_MIN)
                        if isinstance(score_sel, (int, float)):
                            piso = max(piso, float(score_sel) - float(SCORE_DROP_MAX))

                        if (dir2 != direccion) or (int(cond2) < 2) or (float(score2) < piso):
                            print(Fore.YELLOW + Style.BRIGHT + f"Revalidación falló en {symbol}: dir {direccion}->{dir2}, cond={cond2}, score={score2:.3f}<piso {piso:.3f}. Reintentando ciclo...")
                            await asyncio.sleep(2.0 + random.uniform(0.0, 0.5))
                            continue

                        # refresca snapshot para compra/log consistentes
                        direccion, rsi9, rsi14, sma5, sma20, breakout, cruce, rsi_reversion, condiciones = dir2, rsi9_2, rsi14_2, sma5_2, sma20_2, br2, cr2, rev2, cond2
                        estado_bot["score_senal"] = float(score2)
                except Exception:
                    pass

                # ========= PROPOSAL =========
                if not modo_real:
                    await _sync_wait_global_real_clear("pre_demo_buy")
                try:
                    data_proposal = await api_call(ws, {
                        "proposal": 1,
                        "amount": float(monto),
                        "basis": "stake",
                        "contract_type": direccion,
                        "currency": "USD",
                        "duration": 1,
                        "duration_unit": "m",
                        "symbol": symbol
                    }, expect_msg_type="proposal", timeout=8.0)
                except RuntimeError as api_e:
                    _symbol_cooldown[symbol] = time.time() + 60
                    print(Fore.RED + Style.BRIGHT + f"[ERROR] Propuesta: {api_e}. {symbol} en cooldown 60s.")
                    estado_bot["token_msg_mostrado"] = False
                    await asyncio.sleep(8 + random.uniform(0.0, 0.5))
                    continue
                except Exception as e:
                    if _es_error_transitorio_ws(e):
                        if _print_once("proposal-transient", ttl=8):
                            print(Fore.YELLOW + Style.BRIGHT + f"[WARN] Propuesta inestable ({type(e).__name__}). Reabro WS y mantengo ciclo #{ciclo}.")
                        await _cerrar_ws(ws)
                        ws = await _abrir_ws(current_token)
                        await asyncio.sleep(0.6 + random.uniform(0.0, 0.4))
                        continue
                    raise

                # Si token cambió DURANTE proposal → NO compramos, reinicio limpio
                if reinicio_forzado.is_set():
                    estado_bot["ciclo_forzado"] = ciclo
                    print(Fore.YELLOW + Style.BRIGHT + f"Token cambió durante proposal. Cancelo compra y reinicio en ciclo #{ciclo}.")
                    reinicio_forzado.clear()
                    await asyncio.sleep(1.2)
                    break

                proposal = data_proposal.get("proposal", {})
                payout = float(proposal.get("payout", 0.0))
                ask_price = float(proposal.get("ask_price", monto))
                payout_ratio = (payout / float(monto)) if float(monto) > 0 else 0.0

                if payout_ratio < 0.70:
                    print(Fore.YELLOW + Style.BRIGHT + f"Payout de {payout_ratio*100:.1f}% demasiado bajo. Reintentando mismo ciclo...")
                    try:
                        play_sfx("NO_PASAR_REAL", vol=0.9)
                    except Exception:
                        pass
                    estado_bot["token_msg_mostrado"] = False
                    await asyncio.sleep(15 + random.uniform(0.0, 0.5))
                    continue

                print(Fore.CYAN + Style.BRIGHT + f"[{symbol}] Martingala #{ciclo} - {direccion} - {monto} USD")
                # === PRE-TRADE SNAPSHOT (para inferencia real del Maestro) ===
                epoch_pre = None
                now_pre = datetime.now(timezone.utc)
                ts_pre = now_pre.isoformat()
                trade_uid = _build_trade_uid(int(now_pre.timestamp()), symbol, direccion, ciclo, current_token, ts_iso=ts_pre)
                try:
                    # es_rebote PRE-TRADE: venías de 4+ pérdidas ANTES de este trade
                    es_rebote_pre = 1.0 if int(racha_actual_bot) <= -4 else 0.0

                    epoch_pre = write_pretrade_snapshot(
                        ARCHIVO_CSV,
                        fecha=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        activo=symbol,
                        direccion=direccion,              # CALL/PUT
                        monto=float(monto),
                        rsi_9=float(rsi9),
                        rsi_14=float(rsi14),
                        sma_5=float(sma5),
                        sma_20=float(sma20),
                        cruce_sma=float(int(cruce)),
                        breakout=float(int(breakout)),
                        rsi_reversion=float(int(rsi_reversion)),
                        racha_actual=int(racha_actual_bot),     # racha vigente ANTES del trade
                        es_rebote=float(es_rebote_pre),         # ✅ SIN FUGA (pre-trade real)
                        ciclo_martingala=int(ciclo),
                        payout=float(payout),
                        puntaje_estrategia=float(condiciones),  # tu score
                        token=current_token,
                        trade_uid=trade_uid,
                        close_snapshot=close_snapshot,
                    )
                except Exception:
                    epoch_pre = None

                # === /PRE-TRADE SNAPSHOT ===
                # ==================== VENTANA DE DECISIÓN IA (GATEWIN) ====================
                # Ya escribimos el PRE-TRADE snapshot. Ahora damos tiempo para que:
                # 1) el MAESTRO calcule/muestre la prob IA
                # 2) tú elijas bot/ciclo
                # 3) si el MAESTRO asigna REAL (token), el watcher lo detecte y dispare reinicio_forzado
                # Resultado: evitamos comprar en DEMO cuando justo tocaba REAL.
                if VENTANA_DECISION_IA_S > 0:
                    t0 = time.time()
                    ack_visto = False

                    while (time.time() - t0) < VENTANA_DECISION_IA_S:
                        if reinicio_forzado.is_set():
                            break
                        # Doble seguro: si el token cambió durante GateWin, corta ya
                        try:
                            tok_now = leer_token_desde_archivo()
                            if tok_now != current_token:
                                reinicio_forzado.set()
                                break
                        except Exception:
                            pass
                        # ✅ Leer ACK del maestro (si llega, lo mostramos una sola vez)
                        if (not ack_visto) and epoch_pre:
                            ack = leer_ia_ack(NOMBRE_BOT)
                            try:
                                ep_ack = int(float(ack.get("epoch", 0) or 0)) if ack else 0
                                ep_pre = int(float(epoch_pre or 0))
                                # tolera pequeños desfases de epoch para no dejar telemetría en NO_READY
                                epoch_ok = bool(ep_ack >= max(0, ep_pre - 2))
                                if ack and epoch_ok:
                                    p = ack.get("prob", None)
                                    p_hud = ack.get("prob_hud", None)
                                    p_play = ack.get("prob_en_juego", None)
                                    has_prob_hud = ack.get("has_prob_hud", None)
                                    has_prob_play = ack.get("has_prob_en_juego", None)
                                    if isinstance(has_prob_play, bool):
                                        p_show = p_play if has_prob_play else None
                                    elif isinstance(has_prob_hud, bool):
                                        p_show = p_hud if has_prob_hud else p
                                    else:
                                        p_show = p_hud if isinstance(p_hud, (int, float)) else p

                                    auc = float(ack.get("auc", 0.0) or 0.0)
                                    modo = ack.get("modo", "OFF")
                                    thr_real = ack.get("real_thr", None)
                                    reliable_ack = bool(ack.get("reliable", False))
                                    ready_ack = bool(ack.get("ia_ready", False))
                                    modo_norm = str(modo or "OFF").strip().upper()
                                    # Si hay prob visible, no forzar vacío solo por modo OFF transitorio.
                                    if modo_norm == "OFF" and (not isinstance(p_show, (int, float))):
                                        p_show = None
                                    auc_txt = f"{auc:.3f}" if (reliable_ack and 0.0 < auc < 1.0 and modo_norm != "OFF") else "N/A"

                                    estado_bot["ack_ctx"] = {
                                        "ia_prob_en_juego": p_show if isinstance(p_show, (int, float)) else "",
                                        "ia_prob_source": str(ack.get("prob_source", "")) or ("HUD" if isinstance(p_show, (int, float)) else "NO_READY"),
                                        "ia_decision_id": str(ack.get("decision_id", "")),
                                        "ia_gate_real": float(thr_real) if isinstance(thr_real, (int, float)) else "",
                                        "ia_modo_ack": str(modo),
                                        "ia_ready_ack": bool(ready_ack or isinstance(p_show, (int, float))),
                                    }

                                    if isinstance(p_show, (int, float)):
                                        if isinstance(thr_real, (int, float)):
                                            print(f"🤖 IA ACK ({NOMBRE_BOT}) → {p_show*100:.1f}% | Gate REAL={float(thr_real)*100:.1f}% | AUC={auc_txt} | modo={modo}")
                                        else:
                                            print(f"🤖 IA ACK ({NOMBRE_BOT}) → {p_show*100:.1f}% | AUC={auc_txt} | modo={modo}")
                                    else:
                                        print(f"🤖 IA ACK ({NOMBRE_BOT}) → (sin prob) | AUC={auc_txt} | modo={modo}")

                                    ack_visto = True
                            except Exception:
                                pass

                        await asyncio.sleep(VENTANA_DECISION_IA_POLL_S)

                    # Si el token cambió durante la ventana, NO compramos con estado viejo.
                    if reinicio_forzado.is_set():
                        estado_bot["ciclo_forzado"] = ciclo
                        print(
                            Fore.YELLOW + Style.BRIGHT +
                            f"[VENTANA IA] Token cambió durante la decisión. Reintentando ciclo #{ciclo} (sin comprar)."
                        )
                        reinicio_forzado.clear()
                        await asyncio.sleep(0.8)
                        continue

# ==================== /VENTANA DE DECISIÓN IA ====================

                if not modo_real:
                    active_other, owner_other, _reason_other = _sync_real_active_for_other_bot()
                    if active_other:
                        print(
                            Fore.RED + Style.BRIGHT +
                            f"⛔ DEMO_BUY_BLOCKED_BY_REAL_ACTIVE: bot={NOMBRE_BOT} owner={owner_other or '--'} "
                            f"contexto=pre_buy_final acción=esperar"
                        )
                    await _sync_wait_global_real_clear("pre_buy_final")
                    active_other, owner_other, _reason_other = _sync_real_active_for_other_bot()
                    if active_other:
                        continue

                if modo_real:
                    ok_prebuy, why_prebuy = _validar_pre_buy_real(estado_bot.get("ciclo_actual", ciclo))
                    if not ok_prebuy:
                        print(f"🚨 REAL_BUY_ABORT_CICLO_DESALINEADO: local=C{estado_bot.get('ciclo_actual', ciclo)} {why_prebuy}")
                        await asyncio.sleep(2)
                        reinicio_forzado.set()
                        continue

                try:
                    data_buy = await api_call(ws, {
                        "buy": 1,
                        "price": float(ask_price),
                        "parameters": {
                            "amount": float(monto),
                            "basis": "stake",
                            "contract_type": direccion,
                            "currency": "USD",
                            "duration": 1,
                            "duration_unit": "m",
                            "symbol": symbol
                        }
                    }, expect_msg_type="buy", timeout=8.0)
                except RuntimeError as api_e:
                    if _is_insufficient_balance_error(api_e):
                        await _manejar_buy_insufficient_balance(
                            api_e=api_e,
                            modo_real=bool(modo_real),
                            monto=monto,
                            ciclo=ciclo,
                        )
                        continue

                    if _is_rate_limit_error(api_e):
                        _set_buy_pause("rate_limit_buy", BUY_RATE_LIMIT_PAUSE_S)
                        print(
                            Fore.YELLOW + Style.BRIGHT +
                            f"⏳ RATE_LIMIT_BUY: {api_e} | pausa={BUY_RATE_LIMIT_PAUSE_S:.0f}s | ciclo C{ciclo} conservado"
                        )
                        await asyncio.sleep(10 + random.uniform(0.0, 0.5))
                        continue

                    print(Fore.RED + Style.BRIGHT + f"[ERROR] Compra: {api_e}. Reintentando mismo ciclo...")
                    estado_bot["token_msg_mostrado"] = False
                    await asyncio.sleep(10 + random.uniform(0.0, 0.5))
                    continue
                except Exception as e:
                    if _es_error_transitorio_ws(e):
                        _set_pending_contract_resolution(
                            round_id=int(estado_bot.get("sync_round_id", 1) or 1),
                            contract_id=None,
                            reason=f"buy_transient_{type(e).__name__}",
                            token_usado=current_token,
                            asset=symbol,
                            direction=direccion,
                            ciclo=ciclo,
                            monto=monto,
                            rsi9=rsi9,
                            rsi14=rsi14,
                            sma5=sma5,
                            sma20=sma20,
                            cruce=cruce,
                            breakout=breakout,
                            rsi_reversion=rsi_reversion,
                            payout=payout,
                            condiciones=condiciones,
                            epoch_pretrade=epoch_pre,
                            trade_uid=trade_uid,
                            close_snapshot=close_snapshot,
                        )
                        if _print_once("buy-transient", ttl=8):
                            print(Fore.YELLOW + Style.BRIGHT + f"[WARN] Compra inestable ({type(e).__name__}). Reabro WS y mantengo ciclo #{ciclo}.")
                        await _cerrar_ws(ws)
                        ws = await _abrir_ws(current_token)
                        await asyncio.sleep(0.6 + random.uniform(0.0, 0.4))
                        continue
                    raise

                contract_id = data_buy["buy"]["contract_id"]
                _clear_pending_contract_resolution(reason="buy_confirmed")

                # ✅ Ciclo en progreso significa: YA hay contrato abierto
                estado_bot["ciclo_en_progreso"] = True

                # Commit guard REAL
                if modo_real:
                    commit_guard_set(contract_id)

                # Si justo hubo cambio de token y pidieron reinicio, forzamos detach inmediato
                if reinicio_forzado.is_set():
                    estado_bot["interrumpir_ciclo"] = True

                # ========= RESULTADO =========
                resultado, profit = await esperar_resultado(
                    ws, contract_id, symbol, direccion, monto,
                    rsi9, rsi14, sma5, sma20, cruce, breakout, rsi_reversion,
                    ciclo, payout, condiciones, current_token, epoch_pre, trade_uid=trade_uid, close_snapshot=close_snapshot
                )

                if resultado == "INDEFINIDO":
                    _set_pending_contract_resolution(round_id=int(estado_bot.get("sync_round_id", 1) or 1), contract_id=contract_id, reason="resultado_indefinido", token_usado=current_token, asset=symbol, direction=direccion, ciclo=ciclo, monto=monto, rsi9=rsi9, rsi14=rsi14, sma5=sma5, sma20=sma20, cruce=cruce, breakout=breakout, rsi_reversion=rsi_reversion, payout=payout, condiciones=condiciones, epoch_pretrade=epoch_pre, trade_uid=trade_uid, close_snapshot=close_snapshot)
                    print(Fore.YELLOW + "INDEFINIDO: WS/Token restart. Se mantiene MISMO ciclo (BG resolverá).")
                    indefinidos_consecutivos += 1

                    if indefinidos_consecutivos > 5:
                        print(Fore.YELLOW + Style.BRIGHT + f"⚠️ Demasiados INDEFINIDOS: reinicio técnico de conexión, se conserva C{ciclo}.")
                        try:
                            play_sfx("NO_CONCLUYO", vol=0.9)
                        except Exception:
                            pass
                        estado_bot["ciclo_forzado"] = ciclo
                        estado_bot["pending_ciclo"] = ciclo
                        estado_bot["ciclo_actual"] = ciclo
                        estado_bot["forzar_mismo_ciclo_por_indefinido"] = True
                        indefinidos_consecutivos = 0
                        try:
                            await _cerrar_ws(ws)
                        except Exception:
                            pass
                        ws = await _abrir_ws(current_token)
                        estado_bot["ciclo_en_progreso"] = False
                        estado_bot["token_msg_mostrado"] = False
                        continue

                    await asyncio.sleep(random.uniform(0.5, 1.2))
                    await _cerrar_ws(ws)
                    ws = await _abrir_ws(current_token)
                    estado_bot["ciclo_en_progreso"] = False
                    estado_bot["token_msg_mostrado"] = False
                    continue

                # Resultado definido
                _clear_pending_contract_resolution(reason="resultado_definido")
                indefinidos_consecutivos = 0
                estado_bot["intentos_saldo"] = 0
                estado_bot["ciclo_en_progreso"] = False
                estado_bot["token_msg_mostrado"] = False

                # LXV_SYNC_COLUMN: cierre definido -> ACK inmediato para maestro/HUD
                round_id_local = int(estado_bot.get("sync_round_id", 1) or 1)
                _sync_round_emit_close_ack_confirmado(
                    round_id_local,
                    resultado,
                    contract_id=contract_id,
                    asset=symbol,
                    ciclo=ciclo,
                    modo_real_contrato=bool(modo_real),
                )

                print(Back.BLUE + Style.BRIGHT + f"\nTotal DEMO: {resultado_global['demo']:.2f} USD | Total REAL: {resultado_global['real']:.2f} USD")
                await mostrar_saldos()
                sep_ciclo()

                # LXV_SYNC_COLUMN: standby hasta liberación global (solo DEMO)
                if not modo_real:
                    estado_bot["sync_round_id"] = await _sync_round_wait_release(round_id_local)
                    await _sync_wait_global_real_clear("post_release_before_continue")
                else:
                    print(Fore.CYAN + f"ℹ️ Contrato REAL cerrado: {NOMBRE_BOT} no espera release DEMO de columna.")

                # ========= MODO REAL =========
                if modo_real:
                    if resultado == "GANANCIA":
                        print(Fore.GREEN + Style.BRIGHT + "✅ GANANCIA en REAL. Turno terminado. Volviendo a DEMO...\n")

                        try:
                            release_real_token_if_owned()
                        except Exception:
                            pass
                        try:
                            globals()["primer_ingreso_real"] = False
                        except Exception:
                            pass
                        try:
                            globals()["real_activado_en_bot"] = 0.0
                        except Exception:
                            pass
                        try:
                            commit_guard_clear()
                        except Exception:
                            pass

                        await _sync_round_wait_post_real_rejoin()
                        reinicio_forzado.set()
                        break

                    try:
                        ciclo_actual_real = int(ciclo or estado_bot.get("ciclo_actual", 1) or 1)
                    except Exception:
                        ciclo_actual_real = 1

                    print(
                        Fore.RED + Style.BRIGHT +
                        f"❌ PÉRDIDA en REAL C{ciclo_actual_real}. Turno terminado. Volviendo a DEMO. Próximo ciclo lo decide el MAESTRO.\n"
                    )

                    try:
                        release_real_token_if_owned()
                    except Exception:
                        pass
                    try:
                        globals()["primer_ingreso_real"] = False
                    except Exception:
                        pass
                    try:
                        globals()["real_activado_en_bot"] = 0.0
                    except Exception:
                        pass
                    try:
                        commit_guard_clear()
                    except Exception:
                        pass

                    estado_bot["token_msg_mostrado"] = False
                    estado_bot["ciclo_en_progreso"] = False

                    await _sync_round_wait_post_real_rejoin()
                    reinicio_forzado.set()
                    break


                # ========= DEMO =========
                print(Fore.YELLOW + f"Pausa de {PAUSA_POST_OPERACION_S}s antes de continuar...")
                await asyncio.sleep(PAUSA_POST_OPERACION_S + random.uniform(0.0, 0.5))

                if resultado == "GANANCIA":
                    print(Fore.CYAN + Style.BRIGHT + "Ganancia en DEMO. Fin de Martingala.\n")
                    break
                else:
                    ciclo += 1

            # si salimos del inner por reinicio_forzado, el outer lo procesará arriba
            if stop_event.is_set():
                break

    except Exception as e:
        if _es_error_transitorio_ws(e):
            ciclo_ref = int(estado_bot.get("ciclo_actual", 1) or 1)
            estado_bot["ciclo_forzado"] = max(1, min(MAX_CICLOS, ciclo_ref))
            reinicio_forzado.set()
            print(Fore.YELLOW + Style.BRIGHT + f"[WARN] WS/NET transitorio ({type(e).__name__}). Blindaje activo: reintento en ciclo #{estado_bot['ciclo_forzado']}.")
            await asyncio.sleep(1.2 + random.uniform(0.0, 0.5))
        else:
            print(Fore.RED + Style.BRIGHT + f"[ERROR] Fallo general: {type(e).__name__}: {e!r}")
            await asyncio.sleep(10 + random.uniform(0.0, 0.5))
    finally:
        try:
            await _cerrar_ws(ws)
        except Exception:
            pass

async def monitor():
    while not stop_event.is_set():
        await ejecutar_panel()
        if stop_event.is_set():
            break
        await asyncio.sleep(2)


def _selftest_csv_modo_cuenta():
    if str(os.environ.get("RUN_CSV_MODO_CUENTA_SELFTEST", "0")).strip() != "1":
        return False
    assert _resolver_modo_cuenta_csv("REAL:fulll50") == "REAL"
    assert _resolver_modo_cuenta_csv("REAL") == "REAL"
    assert _resolver_modo_cuenta_csv("DEMO") == "DEMO"
    assert _resolver_modo_cuenta_csv("") == "DEMO"
    print("✅ SELFTEST CSV_MODO_CUENTA OK")
    return True


async def main():
    if not WEBSOCKETS_OK:
        print(Fore.YELLOW + "[WARN] websockets no disponible; bot en espera sin operar.")
        while not stop_event.is_set():
            await asyncio.sleep(2)
        return

    # watcher del token (CRÍTICO para GateWin)
    try:
        asyncio.create_task(vigilar_token())
    except Exception as e:
        try:
            print(Fore.YELLOW + f"[WARN] no pude iniciar vigilar_token(): {e!r}")
        except Exception:
            print(f"[WARN] no pude iniciar vigilar_token(): {e!r}")

    # loop principal
    await monitor()

if __name__ == "__main__":
    if _selftest_csv_modo_cuenta():
        sys.exit(0)
    _selftest_sync_demo_hold_global()

    while True:
        try:
            # stop_event puede quedar ligado a un loop anterior si hubo reinicio.
            # Lo recreamos antes de cada corrida para evitar errores de loop cerrado.
            try:
                if "stop_event" in globals():
                    stop_event = asyncio.Event()
            except Exception:
                pass

            asyncio.run(main())

            try:
                if "stop_event" in globals() and stop_event.is_set():
                    print(Fore.YELLOW + "\n🛑 Bot detenido de forma ordenada. No se reinicia.")
                    break
            except Exception:
                pass

            print(Fore.YELLOW + "\n⚠️ main() terminó sin error. Reiniciando en 5 segundos...")
            time.sleep(5)

        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n🛑 Bot detenido manualmente por Ctrl+C.")
            break

        except SystemExit:
            print(Fore.YELLOW + "\n🛑 Bot recibió SystemExit. Salida ordenada.")
            break

        except BaseException as e:
            try:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                print(Fore.RED + "\n🚨 ERROR EXTERNO NO CONTROLADO EN BOT")
                print(Fore.RED + f"Tipo: {type(e).__name__}")
                print(Fore.RED + f"Detalle: {e}")
                import traceback
                traceback.print_exc()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                pass

            print(Fore.YELLOW + "\n🔁 El bot NO se cerrará. Reiniciando en 5 segundos...")
            time.sleep(5)


def _run_bot_real_cycle_guard_selftest():
    if str(os.environ.get("RUN_BOT_REAL_CYCLE_GUARD_SELFTEST", "0")).strip() != "1":
        return False
    print("[SELFTEST] RUN_BOT_REAL_CYCLE_GUARD_SELFTEST=1")
    o1 = {}
    ok1, _c1, m1 = _orden_real_bot_ok(o1)
    print(f"[SELFTEST] caso1 token REAL + sin orden -> ok={ok1} motivo={m1}")
    retenido = 3
    orden_c1 = {"bot": NOMBRE_BOT, "owner": NOMBRE_BOT, "ciclo": 1, "order_id": "x", "ts": time.time(), "ttl_s": 90}
    ok2, c2, m2 = _orden_real_bot_ok(orden_c1)
    allow2 = not (ok2 and c2 < retenido and not _owner_state_confirma_ciclo(c2))
    print(f"[SELFTEST] caso2 orden C1 vs retenido C3 sin owner fresco -> acepta={allow2} (esperado False) motivo={m2}")
    orig_leer = globals().get("_leer_orden_real_viva_para_bot")
    orig_tok = globals().get("leer_token_actual")
    orig_owner = globals().get("_leer_owner_state_vivo")
    try:
        globals()["_leer_orden_real_viva_para_bot"] = lambda: orden_c1
        globals()["leer_token_actual"] = lambda: f"REAL:{NOMBRE_BOT}"
        globals()["_leer_owner_state_vivo"] = lambda: {}
        ok3, m3 = _validar_pre_buy_real(3)
        print(f"[SELFTEST] caso3 prebuy local C3 + orden C1 -> ok={ok3} motivo={m3}")
        orden_c3 = {"bot": NOMBRE_BOT, "owner": NOMBRE_BOT, "ciclo": 3, "order_id": "y", "ts": time.time(), "ttl_s": 90}
        globals()["_leer_orden_real_viva_para_bot"] = lambda: orden_c3
        globals()["_leer_owner_state_vivo"] = lambda: {"owner_bot": NOMBRE_BOT, "ciclo": 3, "assigned_ts": time.time()}
        ok4, m4 = _validar_pre_buy_real(3)
        print(f"[SELFTEST] caso4 prebuy local C3 + orden C3 + owner C3 -> ok={ok4} motivo={m4}")
    finally:
        if orig_leer is not None: globals()["_leer_orden_real_viva_para_bot"] = orig_leer
        if orig_tok is not None: globals()["leer_token_actual"] = orig_tok
        if orig_owner is not None: globals()["_leer_owner_state_vivo"] = orig_owner
    return True

if __name__ == "__main__" and _run_bot_real_cycle_guard_selftest():
    raise SystemExit(0)
