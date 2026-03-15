#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None


# =========================
# Configuración
# =========================
BOT_FILES = [
    "registro_enriquecido_fulll45.csv",
    "registro_enriquecido_fulll46.csv",
    "registro_enriquecido_fulll47.csv",
    "registro_enriquecido_fulll48.csv",
    "registro_enriquecido_fulll49.csv",
    "registro_enriquecido_fulll50.csv",
]

MAIN_FILES = ["dataset_incremental.csv", "model_meta_v2.json"]
OPTIONAL_FILES = [
    "ia_signals_log.csv",
    "feature_names_v2.pkl",
    "scaler_v2.pkl",
    "modelo_xgb_v2.pkl",
    "ia_temporal_degradation_report.json",
]

PRED_THRESHOLDS = [0.60, 0.70, 0.75, 0.80]
ECE_BINS = 10
MIN_MADURE_SAMPLE = 120
MIN_SEGMENT_SAMPLE = 60

OUT_JSON = "auditor_reporte_integral.json"
OUT_TEMPORAL = "auditor_resumen_temporal.csv"
OUT_BOT = "auditor_resumen_por_bot.csv"
OUT_THR = "auditor_resumen_por_threshold.csv"
OUT_DATASET = "auditor_resumen_dataset.csv"


@dataclass
class InputAudit:
    usados: list[str]
    faltantes_principales: list[str]
    faltantes_opcionales: list[str]


def leer_csv_robusto(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception:
            continue
    return None


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def normalizar_resultado(v: Any) -> str:
    if v is None:
        return "INDEFINIDO"
    s = str(v).strip().upper()
    if any(x in s for x in ("GAN", "WIN", "✓", "✔", "✅")):
        return "GANANCIA"
    if any(x in s for x in ("PERD", "PÉRD", "LOSS", "✗", "❌")):
        return "PÉRDIDA"
    return "INDEFINIDO"


def normalizar_trade_status(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().upper()
    if s in ("CERRADO", "CLOSED", "SETTLED", "SOLD", "EXPIRED"):
        return "CERRADO"
    if s in ("PRE_TRADE", "PENDIENTE", "PENDING", "OPEN", "ABIERTO", "IN_PROGRESS"):
        return "NO_CERRADO"
    return s


def parsear_tiempo(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    if "epoch" in df.columns:
        e = pd.to_numeric(df["epoch"], errors="coerce")
        if e.notna().sum() > 0:
            return e.fillna(method="ffill").fillna(method="bfill").fillna(np.arange(n))
    for c in ("timestamp", "ts", "fecha", "datetime", "time"):
        if c in df.columns:
            t = pd.to_datetime(df[c], errors="coerce", utc=True)
            if t.notna().sum() > 0:
                return pd.Series((t.view("int64") // 10**9), index=df.index).fillna(method="ffill").fillna(method="bfill").fillna(np.arange(n))
    return pd.Series(np.arange(n), index=df.index, dtype=float)


def elegir_columna_prob(df: pd.DataFrame) -> str | None:
    for c in ("ia_prob_en_juego", "prob_ia_oper", "prob_ia"):
        if c in df.columns:
            return c
    return None


def construir_result_bin(df: pd.DataFrame) -> pd.Series:
    if "result_bin" in df.columns:
        y = pd.to_numeric(df["result_bin"], errors="coerce")
        return pd.Series(np.where(y >= 0.5, 1, np.where(y < 0.5, 0, np.nan)), index=df.index)
    if "resultado" in df.columns:
        norm = df["resultado"].map(normalizar_resultado)
        return norm.map({"GANANCIA": 1, "PÉRDIDA": 0}).astype(float)
    return pd.Series(np.nan, index=df.index)


def ece_aprox(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float | None:
    if y.size == 0 or p.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(p))
    out = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < len(edges) - 2 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        out += abs(float(np.mean(p[m])) - float(np.mean(y[m]))) * (float(m.sum()) / total)
    return float(out)


def metricas_thresholds(y: np.ndarray, p: np.ndarray, thresholds: list[float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    n = len(y)
    for t in thresholds:
        m = p >= float(t)
        cov = (float(m.sum()) / float(n)) if n > 0 else None
        prec = float(np.mean(y[m])) if m.sum() > 0 else None
        out[f"precision_at_{int(t*100):03d}"] = prec
        out[f"cobertura_at_{int(t*100):03d}"] = cov
    return out


def calcular_metricas_prediccion(df: pd.DataFrame) -> dict[str, Any]:
    prob_col = elegir_columna_prob(df)
    y = construir_result_bin(df)
    if prob_col is None:
        return {"n_total": int(len(df)), "n_util": 0, "error": "sin_columna_prob"}

    p = pd.to_numeric(df[prob_col], errors="coerce").clip(0, 1)
    util = y.isin([0, 1]) & p.notna()
    yu = y[util].astype(float).to_numpy()
    pu = p[util].astype(float).to_numpy()

    if yu.size == 0:
        base = {
            "n_total": int(len(df)),
            "n_util": 0,
            "avg_pred": None,
            "win_rate_real": None,
            "inflacion_pp": None,
            "brier": None,
            "ece_aprox": None,
        }
        base.update(metricas_thresholds(np.array([]), np.array([]), PRED_THRESHOLDS))
        return base

    avg_pred = float(np.mean(pu))
    win = float(np.mean(yu))
    infl = (avg_pred - win) * 100.0
    brier = float(np.mean((pu - yu) ** 2))
    ece = ece_aprox(yu, pu, bins=ECE_BINS)

    out = {
        "n_total": int(len(df)),
        "n_util": int(yu.size),
        "avg_pred": avg_pred,
        "win_rate_real": win,
        "inflacion_pp": infl,
        "brier": brier,
        "ece_aprox": ece,
    }
    out.update(metricas_thresholds(yu, pu, PRED_THRESHOLDS))
    return out


def banda_payout(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if x < 0.7:
        return "baja"
    if x < 1.0:
        return "media"
    return "alta"


def safe_group_wr(df: pd.DataFrame, col: str) -> dict[str, float]:
    if col not in df.columns or "result_bin_norm" not in df.columns:
        return {}
    d = df[df["result_bin_norm"].isin([0, 1])].copy()
    if d.empty:
        return {}
    grp = d.groupby(col)["result_bin_norm"].mean()
    return {str(k): float(v) for k, v in grp.items()}


def safe_group_pnl(df: pd.DataFrame, col: str) -> dict[str, float]:
    if col not in df.columns or "ganancia_perdida" not in df.columns:
        return {}
    d = df.copy()
    d["ganancia_perdida"] = pd.to_numeric(d["ganancia_perdida"], errors="coerce").fillna(0.0)
    grp = d.groupby(col)["ganancia_perdida"].sum()
    return {str(k): float(v) for k, v in grp.items()}


def calcular_metricas_inversion(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["total_trades"] = int(len(df))

    if "trade_status" in df.columns:
        st = df["trade_status"].map(normalizar_trade_status)
        d_closed = df[st == "CERRADO"].copy()
    else:
        d_closed = df.copy()

    out["total_closed"] = int(len(d_closed))
    rb = d_closed["result_bin_norm"] if "result_bin_norm" in d_closed.columns else pd.Series([], dtype=float)
    out["win_rate_closed"] = float(rb[rb.isin([0, 1])].mean()) if rb.isin([0, 1]).sum() > 0 else None

    if "ganancia_perdida" in d_closed.columns:
        gp = pd.to_numeric(d_closed["ganancia_perdida"], errors="coerce").dropna()
        out["pnl_total"] = float(gp.sum()) if len(gp) else None
        out["pnl_promedio"] = float(gp.mean()) if len(gp) else None
    else:
        out["pnl_total"] = None
        out["pnl_promedio"] = None

    out["win_rate_por_bot"] = safe_group_wr(d_closed, "bot")
    out["pnl_por_bot"] = safe_group_pnl(d_closed, "bot")
    out["win_rate_por_ciclo_martingala"] = safe_group_wr(d_closed, "ciclo_martingala")
    out["pnl_por_ciclo_martingala"] = safe_group_pnl(d_closed, "ciclo_martingala")
    out["win_rate_por_activo"] = safe_group_wr(d_closed, "symbol")

    if "payout" in d_closed.columns:
        d_closed["payout_band"] = d_closed["payout"].map(banda_payout)
        out["win_rate_por_banda_payout"] = safe_group_wr(d_closed, "payout_band")
    else:
        out["win_rate_por_banda_payout"] = {}

    out["win_rate_por_modo_ia_ack"] = safe_group_wr(d_closed, "modo_ia_ack")

    if "ia_gate_real" in d_closed.columns:
        g = d_closed["ia_gate_real"].astype(str).str.upper().str.strip()
        out["win_rate_real_gate"] = float(d_closed[g.isin(["1", "TRUE", "SI", "SÍ", "YES"])]["result_bin_norm"].mean()) if (g.isin(["1", "TRUE", "SI", "SÍ", "YES"]).sum() > 0) else None
        out["win_rate_no_gate"] = float(d_closed[~g.isin(["1", "TRUE", "SI", "SÍ", "YES"])]["result_bin_norm"].mean()) if ((~g.isin(["1", "TRUE", "SI", "SÍ", "YES"]).sum()) > 0) else None
    else:
        out["win_rate_real_gate"] = None
        out["win_rate_no_gate"] = None

    out["tasa_cerrados"] = float(len(d_closed) / max(1, len(df)))
    if "resultado_norm" in df.columns:
        out["tasa_indefinido"] = float((df["resultado_norm"] == "INDEFINIDO").mean())
    else:
        out["tasa_indefinido"] = None
    return out


def split_blocks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(df)
    if n == 0:
        return {"first_20": df, "mid_20": df, "last_20": df, "first_half": df, "second_half": df}
    p20 = max(1, int(n * 0.2))
    p40 = int(n * 0.4)
    p60 = int(n * 0.6)
    return {
        "first_20": df.iloc[:p20],
        "mid_20": df.iloc[p40:p60],
        "last_20": df.iloc[-p20:],
        "first_half": df.iloc[: n // 2],
        "second_half": df.iloc[n // 2 :],
    }


def ventanas_rodantes(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    n = len(df)
    if n < 100:
        w = 5
    elif n < 1000:
        w = 7
    else:
        w = 10
    out = []
    for i in range(w):
        a = int(i * n / w)
        b = int((i + 1) * n / w)
        out.append((f"win_{i+1:02d}", df.iloc[a:b]))
    return out


def auditar_dataset(df_inc: pd.DataFrame | None) -> dict[str, Any]:
    if df_inc is None:
        return {"disponible": False}
    d = normalizar_columnas(df_inc)
    out: dict[str, Any] = {"disponible": True, "filas_dataset_incremental": int(len(d)), "columnas_dataset_incremental": int(len(d.columns))}
    out["pct_duplicados_exactos"] = float(d.duplicated().mean()) if len(d) else 0.0
    out["pct_nulos_por_columna"] = {c: float(d[c].isna().mean()) for c in d.columns}

    num = d.select_dtypes(include=[np.number])
    low_var = []
    for c in num.columns:
        v = float(num[c].fillna(0).var())
        if v < 1e-8:
            low_var.append(c)
    out["columnas_varianza_casi_cero"] = low_var

    if "result_bin" in d.columns:
        y = pd.to_numeric(d["result_bin"], errors="coerce")
        valid = y.isin([0, 1])
        out["ratio_etiquetas_validas_result_bin"] = float(valid.mean()) if len(y) else None
        out["distribucion_clases_0_1"] = {
            "class_0": int((y == 0).sum()),
            "class_1": int((y == 1).sum()),
        }
    else:
        out["ratio_etiquetas_validas_result_bin"] = None
        out["distribucion_clases_0_1"] = {}

    main_cols = [c for c in ("payout", "volatilidad", "hora_bucket", "racha_actual") if c in d.columns]
    drift = {}
    for c in main_cols:
        x = pd.to_numeric(d[c], errors="coerce")
        if x.notna().sum() < 20:
            continue
        q = max(1, int(len(x) * 0.2))
        ini = x.iloc[:q].dropna()
        fin = x.iloc[-q:].dropna()
        if len(ini) == 0 or len(fin) == 0:
            continue
        drift[c] = float(fin.mean() - ini.mean())
    out["drift_inicio_vs_final"] = drift
    return out


def cargar_model_meta(path: str = "model_meta_v2.json") -> dict[str, Any]:
    if not os.path.exists(path):
        return {"disponible": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        return {"disponible": False, "error": "json_invalido"}
    return {
        "disponible": True,
        "n_samples": m.get("n_samples", m.get("n")),
        "reliable": m.get("reliable"),
        "warmup_mode": m.get("warmup_mode"),
        "positive_rate": m.get("positive_rate"),
        "threshold": m.get("threshold"),
        "model_family": m.get("model_family", m.get("model_type")),
        "trained_at": m.get("trained_at"),
    }


def construir_veredicto_estabilidad(pred_first: dict[str, Any], pred_final: dict[str, Any], model_meta: dict[str, Any]) -> dict[str, Any]:
    n1 = int(pred_first.get("n_util", 0) or 0)
    n2 = int(pred_final.get("n_util", 0) or 0)
    if n1 < MIN_MADURE_SAMPLE or n2 < MIN_MADURE_SAMPLE:
        return {"estado": "INSUFICIENTE_MUESTRA", "motivo": f"n_inicio={n1}, n_final={n2}"}

    wr_drop_pp = (float(pred_first.get("win_rate_real") or 0) - float(pred_final.get("win_rate_real") or 0)) * 100.0
    infl_up_pp = float(pred_final.get("inflacion_pp") or 0) - float(pred_first.get("inflacion_pp") or 0)
    brier_up = float(pred_final.get("brier") or 0) - float(pred_first.get("brier") or 0)
    p75_drop_pp = (float(pred_first.get("precision_at_075") or 0) - float(pred_final.get("precision_at_075") or 0)) * 100.0

    warm = bool(model_meta.get("warmup_mode", False)) if isinstance(model_meta, dict) else False
    if warm and n2 < (MIN_MADURE_SAMPLE * 2):
        return {"estado": "INSUFICIENTE_MUESTRA", "motivo": "modelo_en_warmup"}

    if infl_up_pp >= 8.0 and wr_drop_pp >= 5.0:
        estado = "SOBRECONFIANZA"
    elif wr_drop_pp >= 10.0 and (brier_up >= 0.05 or p75_drop_pp >= 10.0):
        estado = "DEGRADACIÓN_FUERTE"
    elif wr_drop_pp >= 6.0 or p75_drop_pp >= 6.0 or (brier_up >= 0.03 and infl_up_pp >= 4.0):
        estado = "DEGRADACIÓN_MODERADA"
    elif wr_drop_pp >= 3.0 or infl_up_pp >= 3.0:
        estado = "DEGRADACIÓN_LEVE"
    else:
        estado = "ESTABLE"

    return {
        "estado": estado,
        "wr_drop_pp": wr_drop_pp,
        "infl_up_pp": infl_up_pp,
        "brier_up": brier_up,
        "p75_drop_pp": p75_drop_pp,
    }


def cargar_registros_enriquecidos() -> tuple[pd.DataFrame, InputAudit]:
    frames = []
    usados = []
    falt_princ = []

    for f in BOT_FILES:
        d = leer_csv_robusto(f)
        if d is None:
            falt_princ.append(f)
            continue
        d = normalizar_columnas(d)
        d["_source_file"] = f
        d["bot"] = d.get("bot", os.path.splitext(f)[0].replace("registro_enriquecido_", ""))
        d["tiempo_orden"] = parsear_tiempo(d)
        d["resultado_norm"] = d.get("resultado", "").map(normalizar_resultado) if "resultado" in d.columns else "INDEFINIDO"
        d["result_bin_norm"] = construir_result_bin(d)
        if "trade_status" in d.columns:
            d["trade_status"] = d["trade_status"].map(normalizar_trade_status)
        frames.append(d)
        usados.append(f)

    falt_opt = [f for f in OPTIONAL_FILES if not os.path.exists(f)]
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_df.empty:
        all_df = all_df.sort_values("tiempo_orden").reset_index(drop=True)
    return all_df, InputAudit(usados, falt_princ, falt_opt)


def generar_reportes(reporte: dict[str, Any], temporal_rows: list[dict[str, Any]], bot_rows: list[dict[str, Any]], thr_rows: list[dict[str, Any]], ds_rows: list[dict[str, Any]]) -> None:
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    pd.DataFrame(temporal_rows).to_csv(OUT_TEMPORAL, index=False, encoding="utf-8")
    pd.DataFrame(bot_rows).to_csv(OUT_BOT, index=False, encoding="utf-8")
    pd.DataFrame(thr_rows).to_csv(OUT_THR, index=False, encoding="utf-8")
    pd.DataFrame(ds_rows).to_csv(OUT_DATASET, index=False, encoding="utf-8")


def imprimir_resumen(reporte: dict[str, Any]) -> None:
    pred = reporte.get("prediccion_global", {})
    ver = reporte.get("veredicto", {})
    model = reporte.get("model_meta", {})
    bots = reporte.get("bot_estabilidad", [])

    top_est = bots[:3]
    top_deg = list(reversed(bots[-3:])) if len(bots) >= 3 else bots

    ini = reporte.get("inicio_maduro", {})
    fin = reporte.get("final_maduro", {})

    print("\n" + "=" * 72)
    print("AUDITOR EJECUTIVO — ESTABILIDAD DE CALIDAD")
    print("=" * 72)
    print(f"Trades útiles: {pred.get('n_util', 0)} | Bots auditados: {reporte.get('bots_auditados', 0)}")
    print(f"Modelo: reliable={model.get('reliable')} warmup={model.get('warmup_mode')} n={model.get('n_samples')}")
    print(f"Dataset: filas={reporte.get('dataset', {}).get('filas_dataset_incremental')} dup%={reporte.get('dataset', {}).get('pct_duplicados_exactos')}")
    print(f"VEREDICTO GLOBAL: {ver.get('estado')} | detalle={ver}")
    print("--- Inicio maduro vs Final maduro ---")
    print(f"avg_pred: {ini.get('avg_pred')} -> {fin.get('avg_pred')}")
    print(f"win_rate_real: {ini.get('win_rate_real')} -> {fin.get('win_rate_real')}")
    print(f"inflacion_pp: {ini.get('inflacion_pp')} -> {fin.get('inflacion_pp')}")
    print(f"brier: {ini.get('brier')} -> {fin.get('brier')}")
    print(f"precision@0.70: {ini.get('precision_at_070')} -> {fin.get('precision_at_070')}")
    print(f"precision@0.75: {ini.get('precision_at_075')} -> {fin.get('precision_at_075')}")
    print("Top 3 bots estables:", [x.get("bot") for x in top_est])
    print("Top 3 bots degradados:", [x.get("bot") for x in top_deg])
    print("Advertencias críticas:")
    for w in reporte.get("advertencias", []):
        print(" -", w)




def _run_lightweight_without_pd_np() -> int:
    """Fallback mínimo para entornos sin pandas/numpy: no rompe y genera salidas base."""
    import csv

    usados = []
    falt = []
    rows = []
    for f in BOT_FILES:
        if not os.path.exists(f):
            falt.append(f)
            continue
        usados.append(f)
        try:
            with open(f, "r", encoding="utf-8", errors="replace", newline="") as fh:
                rr = csv.DictReader(fh)
                for r in rr:
                    r = {str(k).strip().lower(): v for k, v in r.items()}
                    rows.append(r)
        except Exception:
            continue

    n_total = len(rows)
    n_util = 0
    p_vals = []
    y_vals = []
    for r in rows:
        p = None
        for c in ("ia_prob_en_juego", "prob_ia_oper", "prob_ia"):
            if c in r and str(r.get(c, "")).strip() != "":
                try:
                    p = float(r[c])
                    break
                except Exception:
                    p = None
        y = None
        if str(r.get("result_bin", "")).strip() != "":
            try:
                y = 1.0 if float(r.get("result_bin")) >= 0.5 else 0.0
            except Exception:
                y = None
        if y is None:
            ytxt = normalizar_resultado(r.get("resultado"))
            if ytxt == "GANANCIA":
                y = 1.0
            elif ytxt == "PÉRDIDA":
                y = 0.0
        if p is not None and y is not None:
            n_util += 1
            p_vals.append(max(0.0, min(1.0, p)))
            y_vals.append(y)

    avg_pred = (sum(p_vals) / len(p_vals)) if p_vals else None
    win = (sum(y_vals) / len(y_vals)) if y_vals else None
    infl = ((avg_pred - win) * 100.0) if (avg_pred is not None and win is not None) else None

    reporte = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "modo": "lightweight_no_pandas_numpy",
        "inputs": {
            "usados": usados,
            "faltantes_principales": falt,
            "faltantes_opcionales": [f for f in OPTIONAL_FILES if not os.path.exists(f)],
            "principales_esperados": BOT_FILES + MAIN_FILES,
            "opcionales_esperados": OPTIONAL_FILES,
        },
        "prediccion_global": {
            "n_total": n_total,
            "n_util": n_util,
            "avg_pred": avg_pred,
            "win_rate_real": win,
            "inflacion_pp": infl,
        },
        "advertencias": [
            "pandas/numpy no disponibles: auditor ejecutado en modo lightweight",
            "instala pandas+numpy para métricas completas (ECE/Brier/rolling avanzados)",
        ],
        "archivos_salida": [OUT_JSON, OUT_TEMPORAL, OUT_BOT, OUT_THR, OUT_DATASET],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    # CSV mínimos requeridos
    with open(OUT_TEMPORAL, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["segmento", "n_total", "n_util", "avg_pred", "win_rate_real", "inflacion_pp"]); w.writerow(["global", n_total, n_util, avg_pred, win, infl])
    with open(OUT_BOT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["bot", "n_total", "n_util", "win_rate_real"])
    with open(OUT_THR, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["threshold", "precision", "cobertura"])
    with open(OUT_DATASET, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["metrica", "valor"]); w.writerow(["modo", "lightweight_no_pandas_numpy"])

    print("=" * 72)
    print("AUDITOR EJECUTIVO — ESTABILIDAD DE CALIDAD (LIGHTWEIGHT)")
    print("=" * 72)
    print(f"Trades útiles: {n_util} / {n_total}")
    print(f"avg_pred={avg_pred} win_rate_real={win} inflacion_pp={infl}")
    print("Advertencia: pandas/numpy no disponibles; reporte resumido generado.")
    return 0


def main() -> int:
    if (pd is None) or (np is None):
        return _run_lightweight_without_pd_np()

    df, audit = cargar_registros_enriquecidos()
    df_inc = leer_csv_robusto("dataset_incremental.csv")
    meta = cargar_model_meta("model_meta_v2.json")

    advertencias = []
    if audit.faltantes_principales:
        advertencias.append("faltan registros principales: " + ", ".join(audit.faltantes_principales))
    if df.empty:
        advertencias.append("no hay registros enriquecidos utilizables")

    pred_global = calcular_metricas_prediccion(df) if not df.empty else {"n_total": 0, "n_util": 0}
    inv_global = calcular_metricas_inversion(df) if not df.empty else {"total_trades": 0}

    blocks = split_blocks(df) if not df.empty else {}
    temporal_rows = []
    for name, part in blocks.items():
        m = calcular_metricas_prediccion(part) if not part.empty else {"n_total": 0, "n_util": 0}
        temporal_rows.append({"segmento": name, **m})

    for wname, wdf in ventanas_rodantes(df) if not df.empty else []:
        m = calcular_metricas_prediccion(wdf)
        temporal_rows.append({"segmento": wname, **m})

    bot_rows = []
    if not df.empty and "bot" in df.columns:
        for b, g in df.groupby("bot"):
            m = calcular_metricas_prediccion(g)
            bot_rows.append({"bot": str(b), **m})
        bot_rows.sort(key=lambda x: (x.get("inflacion_pp") or 999, -(x.get("win_rate_real") or 0)))

    thr_rows = []
    if not df.empty:
        prob_col = elegir_columna_prob(df)
        y = df["result_bin_norm"] if "result_bin_norm" in df.columns else pd.Series(np.nan, index=df.index)
        if prob_col is not None:
            p = pd.to_numeric(df[prob_col], errors="coerce")
            util = y.isin([0, 1]) & p.notna()
            yu = y[util].astype(float).to_numpy()
            pu = p[util].astype(float).to_numpy()
            mts = metricas_thresholds(yu, pu, PRED_THRESHOLDS) if len(yu) else {}
            for t in PRED_THRESHOLDS:
                thr_rows.append({
                    "threshold": t,
                    "precision": mts.get(f"precision_at_{int(t*100):03d}"),
                    "cobertura": mts.get(f"cobertura_at_{int(t*100):03d}"),
                })

    ds = auditar_dataset(df_inc)
    ds_rows = [{"metrica": k, "valor": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v} for k, v in ds.items()]

    # Inicio maduro / final maduro
    ini = blocks.get("first_20", pd.DataFrame())
    fin = blocks.get("last_20", pd.DataFrame())
    ini_m = calcular_metricas_prediccion(ini) if not ini.empty else {}
    fin_m = calcular_metricas_prediccion(fin) if not fin.empty else {}
    verdict = construir_veredicto_estabilidad(ini_m, fin_m, meta)

    if bool(meta.get("warmup_mode", False)):
        advertencias.append("modelo actual sigue en warmup")
    if ds.get("pct_duplicados_exactos", 0) and float(ds.get("pct_duplicados_exactos", 0)) > 0.15:
        advertencias.append("dataset_incremental con duplicación elevada")
    if len(bot_rows) > 0:
        bad = [b["bot"] for b in bot_rows if b.get("n_util", 0) >= MIN_SEGMENT_SAMPLE and (b.get("inflacion_pp") or 0) > 8]
        if bad:
            advertencias.append("sobreconfianza creciente en bots: " + ", ".join(bad[:3]))

    reporte = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "usados": audit.usados,
            "faltantes_principales": audit.faltantes_principales,
            "faltantes_opcionales": audit.faltantes_opcionales,
            "principales_esperados": BOT_FILES + MAIN_FILES,
            "opcionales_esperados": OPTIONAL_FILES,
        },
        "prediccion_global": pred_global,
        "inversion_global": inv_global,
        "dataset": ds,
        "model_meta": meta,
        "inicio_maduro": ini_m,
        "final_maduro": fin_m,
        "veredicto": verdict,
        "bot_estabilidad": bot_rows,
        "advertencias": advertencias,
        "clasificacion_reglas": {
            "ESTABLE": "sin caída material de WR/precision ni aumento relevante de inflacion/brier",
            "DEGRADACIÓN_LEVE": "caída moderada pequeña o inflación al alza leve",
            "DEGRADACIÓN_MODERADA": "caída material de WR/precision o brier+inflación empeorando",
            "DEGRADACIÓN_FUERTE": "caída fuerte de WR con deterioro combinado de calibración",
            "SOBRECONFIANZA": "sube inflación mientras cae WR real",
            "INSUFICIENTE_MUESTRA": "n útil insuficiente en comparación madura",
        },
        "archivos_salida": [OUT_JSON, OUT_TEMPORAL, OUT_BOT, OUT_THR, OUT_DATASET],
    }

    generar_reportes(reporte, temporal_rows, bot_rows, thr_rows, ds_rows)
    imprimir_resumen(reporte)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
