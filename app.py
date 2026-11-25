# -*- coding: utf-8 -*-
"""
Modelos de previsão para o índice S&P 500 — previsão probabilística com validação rigorosa
=====================================================================
Modelos de previsão para o índice S&P 500 — rigorous probabilistic forecasting

INSTALAÇÃO / INSTALL (uma vez / once):
    pip install streamlit pandas numpy scipy scikit-learn plotly yfinance
    pip install xlrd shap      # opcionais / optional: Shiller .xls local file, SHAP

EXECUÇÃO / RUN:
    streamlit run app.py

Arquitetura / Architecture
    Modelo 1 (tático, até 2 anos): previsão DIRETA por horizonte (21/63/126/252/504 pregões)
        de toda a distribuição (quantis P1..P99). Membros: benchmark histórico, GJR-GARCH-t,
        HAR-RV(+VIX), Ridge, Quantile Regression Forest e GBM quantílico, combinados em ensemble
        com encolhimento ao benchmark (escolhido em validação aninhada) e calibração conformal
        adaptativa (ACI). Validação: purged walk-forward, CRPS, PIT, cobertura, Clark-West e
        Diebold-Mariano contra o benchmark.
    Modelo 2 (estrutural, até 2040): retorno = crescimento de lucros + reversão do múltiplo (CAPE)
        + dividendos; parâmetros bayesianos (encolhimento à média histórica longa); choques por
        filtered historical simulation GJR-GARCH-t com bootstrap em blocos; cenários de estresse.
Conteúdo educacional/quantitativo; não constitui recomendação de investimento.
"""
import importlib.util
import inspect
import io
import re
import time
import types
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from scipy import optimize, signal, stats
from scipy.interpolate import PchipInterpolator
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

HAS_SHAP = importlib.util.find_spec("shap") is not None

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Modelos de previsão para o índice S&P 500", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
START_FULL = "1950-01-01"          # histórico completo baixado (^GSPC existe desde 1950+)
LONG_END_YEAR = 2040
N_SIMS = 5000                      # Monte Carlo (Modelo 2)
SEED = 42
HORIZONS = (21, 63, 126, 252, 504)  # pregões à frente: ~1m, 3m, 6m, 1a, 2a
QS = np.array([0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
               0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99])
GBM_KNOTS = np.array([0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975])
MEMBERS = ["garch", "har", "ridge", "qrf", "gbm"]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())   # ~250 pregões/ano

QUALITY = {   # compromisso velocidade x precisão
    "fast": dict(step=10, n_trees=60, gbm_iter=50, folds=4),
    "standard": dict(step=5, n_trees=100, gbm_iter=80, folds=5),
    "full": dict(step=3, n_trees=200, gbm_iter=150, folds=6),
}

# moeda -> (ticker yfinance, inverter?)  | taxa final = unidades da moeda por 1 USD
CCY = {
    "USD": (None, False), "BRL": ("BRL=X", False), "EUR": ("EURUSD=X", True),
    "GBP": ("GBPUSD=X", True), "JPY": ("JPY=X", False), "CNY": ("CNY=X", False),
    "CHF": ("CHF=X", False), "CAD": ("CAD=X", False), "AUD": ("AUDUSD=X", True),
}
SYMBOL = {"USD": "US$", "BRL": "R$", "EUR": "€", "GBP": "£", "JPY": "¥",
          "CNY": "CN¥", "CHF": "CHF", "CAD": "C$", "AUD": "A$"}

FEATS_TECH = ["ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "dist_ema50", "dist_ema200",
              "ema_spread", "bb_pctb", "bb_width", "dd_252", "rv_5", "rv_21", "rv_63",
              "vix_log", "vix_rel", "vrp", "trend_z"]
FEATS_EXT = ["log_cape", "cape_z", "ey_minus_y", "dy", "mcap_z", "curve_10_2", "credit_sp",
             "credit_chg_63", "dff", "dff_chg_126", "unrate", "unrate_chg_6m", "cpi_yoy",
             "dxy_mom_63", "rec_prob"]
FEATS = FEATS_TECH + FEATS_EXT

_PLOTLY_W = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters \
    else {"use_container_width": True}
_TABLE_W = {"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters \
    else {"use_container_width": True}

_LANG = "pt"
TXT = {}   # preenchido mais abaixo (pt, en)


def tr(key, **kw):
    """Texto da interface no idioma escolhido (pt/en)."""
    s = TXT[key][0 if _LANG == "pt" else 1]
    return s.format(**kw) if kw else s


# ----------------------------------------------------------------------------
# Formatação / UI
# ----------------------------------------------------------------------------
def fnum(v, d=2):
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "—"
    s = f"{v:,.{d}f}"
    if _LANG == "pt":
        s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    return s


def fsci(v, d=2):
    """Notação científica (ex.: 1,23e-02) no padrão do idioma."""
    if v is None or not np.isfinite(v):
        return "—"
    s = f"{v:.{d}e}"
    return s.replace(".", ",") if _LANG == "pt" else s


def fmoney(v, ccy, d=0):
    return f"{SYMBOL[ccy]} {fnum(v, d)}"


def fpct(v, d=1):
    return "—" if v is None or not np.isfinite(v) else f"{fnum(v * 100, d)}%"


def fdate(d):
    d = pd.Timestamp(d)
    return d.strftime("%d/%m/%Y") if _LANG == "pt" else d.strftime("%b %d, %Y")


def show(fig):
    st.plotly_chart(fig, **_PLOTLY_W)


def table(df):
    st.dataframe(df, hide_index=True, **_TABLE_W)


def _ceil_div(a, b):
    return -(-int(a) // int(b))


# ----------------------------------------------------------------------------
# Métricas probabilísticas (quantis em log-retorno)
# ----------------------------------------------------------------------------
def pinball(Q, y):
    e = y[:, None] - Q
    return np.maximum(QS[None, :] * e, (QS[None, :] - 1.0) * e)


_EDGES = np.r_[0.0, QS, 1.0]
CRPS_W = (_EDGES[2:] - _EDGES[:-2]) / 2.0     # pesos trapezoidais da integral sobre os níveis


def crps_q(Q, y):
    """CRPS aproximado: 2 x integral da pinball loss sobre os níveis de quantil (trapézios)."""
    return 2.0 * (pinball(Q, y) * CRPS_W[None, :]).sum(axis=1)


def qmean(Q):
    """Média implícita da distribuição discretizada (integral da função quantílica)."""
    mid = ((Q[:, 1:] + Q[:, :-1]) / 2.0 * np.diff(QS)[None, :]).sum(axis=1)
    return mid + QS[0] * Q[:, 0] + (1.0 - QS[-1]) * Q[:, -1]


def pit_values(Q, y):
    return np.array([np.interp(y[i], Q[i], QS) for i in range(len(y))])


def cdf_at(q_row, x):
    """P(Y <= x) por interpolação da função quantílica de uma linha."""
    return float(np.interp(x, q_row, QS))


# ----------------------------------------------------------------------------
# Testes estatísticos contra o benchmark (erros sobrepostos -> HAC de Newey-West)
# ----------------------------------------------------------------------------
def _nw_var_mean(d, lag):
    d = np.asarray(d, float)
    n = len(d)
    dc = d - d.mean()
    v = dc @ dc / n
    for l in range(1, min(lag, n - 1) + 1):
        v += 2.0 * (1.0 - l / (lag + 1.0)) * (dc[l:] @ dc[:-l]) / n
    return v / n


def dm_test(d, lag):
    """Diebold-Mariano (ajuste HLN). d = perda_benchmark - perda_modelo; H1: E[d] > 0."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return np.nan, np.nan
    var = _nw_var_mean(d, lag)
    if not var > 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(var)
    hh = lag + 1
    stat *= np.sqrt(max((n + 1 - 2 * hh + hh * (hh - 1) / n) / n, 1e-6))
    return float(stat), float(1.0 - stats.t.cdf(stat, n - 1))


def cw_test(y, f_model, f_bench, lag):
    """Clark-West (modelos aninhados). H1: o modelo tem MSPE menor que o benchmark."""
    y, fm, fb = (np.asarray(a, float) for a in (y, f_model, f_bench))
    f = (y - fb) ** 2 - ((y - fm) ** 2 - (fb - fm) ** 2)
    n = len(f)
    if n < 20:
        return np.nan, np.nan
    var = _nw_var_mean(f, lag)
    if not var > 0:
        return np.nan, np.nan
    stat = f.mean() / np.sqrt(var)
    return float(stat), float(1.0 - stats.norm.cdf(stat))


def r2_os(y, f_model, f_bench):
    """R² fora da amostra de Campbell-Thompson contra o benchmark."""
    den = np.sum((y - f_bench) ** 2)
    return float(1.0 - np.sum((y - f_model) ** 2) / den) if den > 0 else np.nan


# ----------------------------------------------------------------------------
# Conformal adaptativo (ACI, Gibbs & Candès 2021) com atraso de feedback = horizonte
# ----------------------------------------------------------------------------
def aci_run(Q, y, d, alpha, gamma=0.01):
    """
    Q: (n, nq) quantis ordenados; y: (n,) realizações; d: atraso (em origens) até o resultado ser
    conhecido; alpha: nível de erro nominal. Em cada origem t, o intervalo usa o alpha_t atualizado
    apenas com erros já resolvidos (origem t-d). Retorna (alpha_t, lo, hi, err, alpha_próxima_origem).
    """
    n = len(y)
    amin, amax = max(0.002, alpha / 5.0), min(0.95, alpha * 3.0)   # limita o ajuste a [α/5, 3α]
    a = np.empty(n)
    lo = np.empty(n)
    hi = np.empty(n)
    err = np.full(n, np.nan)
    at = float(alpha)
    for t in range(n):
        if t - d >= 0 and np.isfinite(err[t - d]):
            at = float(min(max(at + gamma * (alpha - err[t - d]), amin), amax))
        a[t] = at
        lo[t] = np.interp(at / 2.0, QS, Q[t])
        hi[t] = np.interp(1.0 - at / 2.0, QS, Q[t])
        if np.isfinite(y[t]):
            err[t] = 0.0 if lo[t] <= y[t] <= hi[t] else 1.0
    nxt = at
    if n - d >= 0 and n - d < n and np.isfinite(err[n - d]):
        nxt = float(min(max(at + gamma * (alpha - err[n - d]), amin), amax))
    return a, lo, hi, err, nxt


def damp_alpha(a_raw, alpha, n_eff, n_ref=40.0):
    """Amortece o ajuste conformal quando há poucas observações independentes (n_eff)."""
    kappa = float(min(1.0, max(n_eff, 0.0) / n_ref))
    return float(alpha + kappa * (a_raw - alpha))


# ----------------------------------------------------------------------------
# Volatilidade: GJR-GARCH(1,1) com t de Student (MLE; filtro vetorizado via lfilter)
# ----------------------------------------------------------------------------
def _t_logpdf(z, nu):
    c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
    return c - (nu + 1.0) / 2.0 * np.log1p(z * z / (nu - 2.0))


def gjr_variance(eps, p, v0):
    """s2[t] = Var(eps[t] | info t-1), com s2[n] = previsão para t=n (eps em %)."""
    om, al, ga, be = p
    c = (al + ga * (eps < 0)) * eps ** 2
    u = np.empty(len(eps) + 1)
    u[0] = v0
    u[1:] = om + c
    return signal.lfilter([1.0], [1.0, -be], u)


def _garch_nll(theta, R, v0):
    mu, om, al, ga, be, nu = theta
    eps = R - mu
    s2 = gjr_variance(eps, (om, al, ga, be), v0)[:-1]
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e10
    return float(-(_t_logpdf(eps / np.sqrt(s2), nu) - 0.5 * np.log(s2)).sum())


def garch_fit(r, x0=None):
    """Ajusta GJR-GARCH-t aos log-retornos diários `r` (decimais). Retorna dict de parâmetros."""
    R = 100.0 * np.asarray(r, float)
    R = R[np.isfinite(R)]
    v0 = float(R.var())
    start = x0 if x0 is not None else [R.mean(), v0 * 0.03, 0.03, 0.08, 0.90, 8.0]
    bounds = [(-0.5, 0.5), (1e-6, 10.0), (0.0, 0.5), (0.0, 0.8), (0.0, 0.9995), (2.5, 60.0)]
    cons = [{"type": "ineq", "fun": lambda p: 0.9995 - (p[2] + p[3] / 2.0 + p[4])}]
    ok, theta = False, None
    try:
        res = optimize.minimize(_garch_nll, start, args=(R, v0), method="SLSQP", bounds=bounds,
                                constraints=cons, options={"maxiter": 250, "ftol": 1e-9})
        theta = np.asarray(res.x, float)
        ok = bool(np.isfinite(res.fun) and res.fun < 1e9 and theta[2] + theta[3] / 2 + theta[4] < 1)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:   # fallback estável (persistência 0,99)
        theta = np.array([R.mean(), v0 * 0.01, 0.04, 0.06, 0.92, 8.0])
    pers = float(theta[2] + theta[3] / 2.0 + theta[4])
    return {"theta": theta, "v0": v0, "persist": pers, "V": float(theta[1] / (1.0 - pers)),
            "converged": ok}


def garch_s2n(g, r):
    """Variância (%²) do próximo pregão dado o filtro até t, para todas as linhas de `r`."""
    R = 100.0 * np.asarray(r, float)
    th = g["theta"]
    return gjr_variance(R - th[0], th[1:5], g["v0"])[1:]


def garch_hvar(s2n, g, h):
    """Variância (decimal²) acumulada em h dias, com reversão à variância incondicional."""
    p, V = g["persist"], g["V"]
    f = h if abs(1.0 - p) < 1e-8 else (1.0 - p ** h) / (1.0 - p)
    return np.maximum(h * V + (s2n - V) * f, 1e-8) / 1e4


# ----------------------------------------------------------------------------
# HAR-RV(+VIX) e pilha NNLS de variância (volatilidade como alvo principal)
# ----------------------------------------------------------------------------
def har_design(rv, vix):
    s = pd.Series(rv)
    d = s.to_numpy()
    w = s.rolling(5, min_periods=5).mean().to_numpy()
    m = s.rolling(22, min_periods=22).mean().to_numpy()
    vv = np.where(np.isfinite(vix), (np.nan_to_num(vix) / 100.0) ** 2 / 252.0, m)
    return np.column_stack([np.ones(len(rv)), d, w, m, vv])


# ----------------------------------------------------------------------------
# Features point-in-time (só usam informação disponível no fechamento de t)
# ----------------------------------------------------------------------------
def qi(q):
    return int(np.argmin(np.abs(QS - q)))


def expanding_trend_z(close, min_n=756):
    """Tendência log-linear em JANELA EXPANSIVA (sem look-ahead): z, ajuste, σ e inclinação por data."""
    idx = close.index
    t = (idx - idx[0]).days.to_numpy(float) / 365.25
    y = np.log(close.to_numpy(float))
    n = np.arange(1, len(y) + 1, dtype=float)
    sx, sy = np.cumsum(t), np.cumsum(y)
    xm, ym = sx / n, sy / n
    cxx = np.cumsum(t * t) - n * xm * xm
    cxy = np.cumsum(t * y) - n * xm * ym
    cyy = np.cumsum(y * y) - n * ym * ym
    with np.errstate(divide="ignore", invalid="ignore"):
        b = cxy / cxx
        a = ym - b * xm
        s = np.sqrt(np.maximum(cyy - b * cxy, 0.0) / np.maximum(n - 2.0, 1.0))
        fit = a + b * t
        z = (y - fit) / s
    z[n < min_n] = np.nan
    return pd.Series(z, idx), pd.Series(fit, idx), pd.Series(s, idx), pd.Series(b, idx)


def static_features(close, rv, vix):
    logp = np.log(close)
    f = pd.DataFrame(index=close.index)
    for k in (5, 21, 63, 126, 252):
        f[f"ret_{k}"] = logp.diff(k)
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    f["dist_ema50"] = close / ema50 - 1
    f["dist_ema200"] = close / ema200 - 1
    f["ema_spread"] = ema50 / ema200 - 1
    ma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    up, lo = ma20 + 2 * sd20, ma20 - 2 * sd20
    f["bb_pctb"] = (close - lo) / (up - lo)
    f["bb_width"] = (up - lo) / ma20
    f["dd_252"] = logp - logp.rolling(252, min_periods=60).max()
    for w in (5, 21, 63):
        f[f"rv_{w}"] = np.sqrt(252.0 * rv.rolling(w).mean())
    f["vix_log"] = np.log(vix)
    f["vix_rel"] = vix / vix.rolling(63).mean() - 1
    f["vrp"] = vix - 100.0 * f["rv_21"]                    # prêmio de risco de variância (pts de vol)
    z, fit, s, _ = expanding_trend_z(close)
    f["trend_z"] = z
    f["trend_fit"] = fit
    f["trend_sd"] = s
    return f.replace([np.inf, -np.inf], np.nan)


def build_panel(data, fx_mode, ccy, train_start):
    """
    Painel de modelagem. fx_mode='spot': modela o S&P em USD (câmbio só na exibição, taxa atual).
    fx_mode='series': modela a série já convertida (embute a deriva cambial histórica).
    Retorna (P, o0) onde o0 = primeira origem de treino/avaliação.
    """
    spx = data["spx"]
    close_usd = spx["Close"]
    if fx_mode == "spot" or ccy == "USD":
        close = close_usd.copy()
        r2 = np.log(close).diff() ** 2
        ok = (spx["High"] > spx["Low"]) & (spx["High"] >= spx["Close"] * 0.999) \
            & (spx["Low"] <= spx["Close"] * 1.001)
        pk = np.log(spx["High"] / spx["Low"]) ** 2 / (4.0 * np.log(2.0))   # Parkinson
        rv = pk.where(ok, r2)
    else:
        close = (close_usd * data["fx"][ccy]).dropna()
        rv = np.log(close).diff() ** 2
    rv = rv.fillna(0.0)
    vix = data["vix"].reindex(close.index)
    feats = static_features(close, rv, vix)
    ext = data["ext"].reindex(close.index)
    P = pd.DataFrame({"close": close, "rv": rv, "vix": vix}).join(feats)
    for c in FEATS_EXT:
        P[c] = ext[c] if c in ext.columns else np.nan
    o0 = int(max(np.searchsorted(P.index.values, np.datetime64(pd.Timestamp(train_start))), 300))
    return P, o0


# ----------------------------------------------------------------------------
# Motor de previsão direta de quantis por horizonte
# ----------------------------------------------------------------------------
def _prep(P):
    close = P["close"].to_numpy(float)
    logp = np.log(close)
    return logp, np.r_[0.0, np.diff(logp)], P["rv"].to_numpy(float), P["vix"].to_numpy(float)


def vol_arrays(P, g):
    _, r, rv, vix = _prep(P)
    return {"s2n": garch_s2n(g, r), "Xh": har_design(rv, vix), "cs": np.r_[0.0, np.cumsum(rv)]}


def _design(P, cols, arrs):
    gv = np.sqrt(arrs["s2n"] / 1e4 * 252.0)                 # vol anualizada prevista (GARCH)
    return np.column_stack([P[cols].to_numpy(float), gv])


def _transform(XA, st_, impute):
    X = np.clip(XA, st_["lo"], st_["hi"])
    return np.where(np.isfinite(X), X, st_["med"]) if impute else X


def qrf_quantiles(Ltr, ztr, Lte, chunk=400):
    """Quantile Regression Forest (Meinshausen): quantis ponderados pelos pesos de folha."""
    n_tr, T = Ltr.shape
    order = np.argsort(ztr)
    zs = ztr[order]
    out = np.empty((Lte.shape[0], len(QS)))
    for a in range(0, Lte.shape[0], chunk):
        Lc = Lte[a:a + chunk]
        W = np.zeros((Lc.shape[0], n_tr))
        for t in range(T):
            same = Lc[:, t][:, None] == Ltr[:, t][None, :]
            W += same / np.maximum(same.sum(axis=1, keepdims=True), 1)
        cw = np.cumsum(W[:, order] / T, axis=1)
        for j, q in enumerate(QS):
            out[a:a + chunk, j] = zs[np.minimum((cw < q - 1e-12).sum(axis=1), n_tr - 1)]
    return out


def knots_to_grid(K, zref):
    """Interpola os quantis nos nós do GBM para a grade QS; caudas por deslocamento do benchmark."""
    K = np.sort(K, axis=1)
    inner = (QS >= GBM_KNOTS[0] - 1e-12) & (QS <= GBM_KNOTS[-1] + 1e-12)
    out = np.empty((K.shape[0], len(QS)))
    out[:, inner] = PchipInterpolator(GBM_KNOTS, K, axis=1)(QS[inner])
    ii = np.where(inner)[0]
    for j in range(ii[0]):
        out[:, j] = out[:, ii[0]] - (zref[ii[0]] - zref[j])
    for j in range(ii[-1] + 1, len(QS)):
        out[:, j] = out[:, ii[-1]] + (zref[j] - zref[ii[-1]])
    return out


def ridge_oof_quantiles(Xs, zs, gap, alpha, nb=5):
    """Quantis dos resíduos do Ridge FORA DA DOBRA (blocos contíguos com purga): evita dispersão otimista."""
    n = len(zs)
    edges = np.linspace(0, n, nb + 1).astype(int)
    res = np.full(n, np.nan)
    for b in range(nb):
        a, c = edges[b], edges[b + 1]
        tr_idx = np.r_[0:max(a - gap, 0), min(c + gap, n):n].astype(int)
        if len(tr_idx) < 50:
            continue
        res[a:c] = zs[a:c] - Ridge(alpha=alpha).fit(Xs[tr_idx], zs[tr_idx]).predict(Xs[a:c])
    ok = np.isfinite(res)
    return np.quantile(res[ok], QS) if ok.sum() > 50 else None


def fit_bundle(P, cut, cfg):
    """Ajusta todos os membros usando APENAS linhas < cut (alvos de t exigem t+h <= cut-1: purga)."""
    o0, step, seed = cfg["o0"], cfg["step"], cfg["seed"]
    if cut - o0 < 900:
        raise ValueError("insufficient_history")
    Pc = P.iloc[:cut]
    logp, r, rv, vix = _prep(Pc)
    g = garch_fit(r[1:])
    arrs = vol_arrays(Pc, g)
    cols = [c for c in FEATS if Pc[c].iloc[o0:cut].notna().mean() > 0.4]
    XA = _design(Pc, cols, arrs)
    ru = np.arange(o0, cut - HORIZONS[0], step)
    lo, hi = np.nanquantile(XA[ru], 0.01, axis=0), np.nanquantile(XA[ru], 0.99, axis=0)
    med = np.nanmedian(np.clip(XA[ru], lo, hi), axis=0)
    Xi_u = np.where(np.isfinite(np.clip(XA[ru], lo, hi)), np.clip(XA[ru], lo, hi), med)
    mu_s, sd_s = Xi_u.mean(axis=0), Xi_u.std(axis=0)
    sd_s[sd_s == 0] = 1.0
    st_ = {"lo": lo, "hi": hi, "med": med, "mu": mu_s, "sd": sd_s}
    cs = arrs["cs"]
    bundle = {"cut": cut, "cols": cols, "st": st_, "garch": g, "h": {}, "step": step}
    for h in HORIZONS:
        rows = np.arange(o0, cut - h, step)
        if len(rows) < 60:
            raise ValueError("insufficient_history")
        y = logp[rows + h] - logp[rows]
        Vg = garch_hvar(arrs["s2n"][rows], g, h)
        Xh = arrs["Xh"][rows]
        mh = (cs[rows + h + 1] - cs[rows + 1]) / h
        ok = np.isfinite(Xh).all(axis=1) & np.isfinite(mh)
        coef = np.linalg.lstsq(Xh[ok], mh[ok], rcond=None)[0]
        floor = float(0.25 * np.median(Xh[ok, 3]))
        pred = Xh @ coef
        Vh = h * np.where(np.isfinite(pred), np.maximum(pred, floor), floor)
        A = np.column_stack([Vg, Vh, h * Xh[:, 4]])
        Sh = cs[rows + h + 1] - cs[rows + 1]
        okA = np.isfinite(A).all(axis=1) & np.isfinite(Sh) & ok
        w, _ = optimize.nnls(A[okA], Sh[okA])
        if w.sum() <= 0:
            w = np.array([0.5, 0.5, 0.0])
        Vs = np.maximum(A @ w, floor * h)
        rows, y, Vg, Vh, Vs = rows[okA], y[okA], Vg[okA], Vh[okA], Vs[okA]
        mu_h = float(y.mean() / h)
        base = h * mu_h
        zg, zh, zs = ((y - base) / np.sqrt(V) for V in (Vg, Vh, Vs))
        Xw = _transform(XA[rows], st_, False)
        Xi = _transform(XA[rows], st_, True)
        Xs = (Xi - mu_s) / sd_s
        msl = int(min(max(40, 3 * _ceil_div(h, step)), max(len(rows) // 5, 5)))
        rd = Ridge(alpha=150.0).fit(Xs, zs)
        eq_r = ridge_oof_quantiles(Xs, zs, _ceil_div(h, step), 150.0)
        rf = RandomForestRegressor(n_estimators=cfg["n_trees"], min_samples_leaf=msl, max_features=0.33,
                                   max_samples=0.7, n_jobs=-1, random_state=seed).fit(Xi, zs)
        gbm = [HistGradientBoostingRegressor(
            loss="quantile", quantile=float(q), learning_rate=0.06, max_iter=cfg["gbm_iter"], max_depth=3,
            min_samples_leaf=msl, l2_regularization=1.0, max_bins=64, early_stopping=False,
            random_state=seed).fit(Xw, zs) for q in GBM_KNOTS]
        bundle["h"][h] = {
            "mu": mu_h, "har": coef, "floor": floor, "nnls": w,
            "bench": np.quantile(y, QS), "zq_g": np.quantile(zg, QS), "zq_h": np.quantile(zh, QS),
            "zq_s": np.quantile(zs, QS), "ridge": rd, "ridge_eq": eq_r if eq_r is not None else np.quantile(zs - rd.predict(Xs), QS),
            "rf": rf, "Ltr": rf.apply(Xi).astype(np.int32), "z": zs, "gbm": gbm, "n_train": int(len(rows)),
        }
    return bundle


def predict_bundle(b, P, rows):
    """Quantis de log-retorno (n_rows x nq) por horizonte e membro, usando só features até cada linha."""
    g = b["garch"]
    arrs = vol_arrays(P, g)
    XA = _design(P, b["cols"], arrs)[rows]
    st_ = b["st"]
    Xw = _transform(XA, st_, False)
    Xi = _transform(XA, st_, True)
    Xs = (Xi - st_["mu"]) / st_["sd"]
    out, meta = {}, {}
    for h in HORIZONS:
        hb = b["h"][h]
        Vg = garch_hvar(arrs["s2n"][rows], g, h)
        Xh = arrs["Xh"][rows]
        pr = Xh @ hb["har"]
        Vh = h * np.where(np.isfinite(pr), np.maximum(pr, hb["floor"]), hb["floor"])
        A = np.column_stack([Vg, Vh, h * np.where(np.isfinite(Xh[:, 4]), Xh[:, 4], hb["floor"])])
        Vs = np.maximum(A @ hb["nnls"], hb["floor"] * h)
        base = h * hb["mu"]
        sd = lambda V: np.sqrt(V)[:, None]  # noqa: E731
        Qd = {"bench": np.tile(hb["bench"], (len(rows), 1))}
        Qd["garch"] = base + sd(Vg) * hb["zq_g"][None, :]
        Qd["har"] = base + sd(Vh) * hb["zq_h"][None, :]
        Qd["ridge"] = base + sd(Vs) * (hb["ridge"].predict(Xs)[:, None] + hb["ridge_eq"][None, :])
        Qd["qrf"] = base + sd(Vs) * qrf_quantiles(hb["Ltr"], hb["z"], hb["rf"].apply(Xi))
        K = np.column_stack([m.predict(Xw) for m in hb["gbm"]])
        Qd["gbm"] = base + sd(Vs) * knots_to_grid(K, hb["zq_s"])
        out[h] = {k: np.sort(v, axis=1) for k, v in Qd.items()}
        meta[h] = {"Vs": Vs, "Vg": Vg, "Vh": Vh}
    return out, meta


def combine(Qm, w, lam, Qb):
    Qavg = sum(w[m] * Qm[m] for m in MEMBERS if w.get(m, 0.0) > 0)
    return lam * Qavg + (1.0 - lam) * Qb


def select_ensemble(P, cfg):
    """
    Validação ANINHADA: treina em ~65% do histórico e, no bloco restante (purgado), escolhe pesos dos
    membros (∝ 1/CRPS³), o encolhimento λ ao benchmark e o alpha do ACI. Nada disso toca o bloco de teste.
    """
    n, o0, step = len(P), cfg["o0"], cfg["step"]
    inner = o0 + int(0.65 * (n - o0))
    b = fit_bundle(P, inner, cfg)
    rows_all = np.arange(inner, n - HORIZONS[0], step)
    pred, _ = predict_bundle(b, P, rows_all)
    logp = np.log(P["close"].to_numpy(float))
    sel = {}
    for h in HORIZONS:
        rows = rows_all[rows_all <= n - 1 - h]
        k = len(rows)
        d = _ceil_div(h, step)
        if k < 40:
            sel[h] = {"w": {m: 1.0 / len(MEMBERS) for m in MEMBERS}, "lam": 0.5, "a90": 0.10, "a50": 0.50,
                      "n_val": int(k), "n_eff": 0.0, "crps": {}, "crps_bench": np.nan}
            continue
        y = logp[rows + h] - logp[rows]
        Qm = {m: pred[h][m][:k] for m in MEMBERS}
        Qb = pred[h]["bench"][:k]
        crps = {m: float(crps_q(Qm[m], y).mean()) for m in MEMBERS}
        cmin = min(crps.values())
        temp = 80.0 * min(1.0, k * step / h / 100.0)        # poucas janelas independentes -> pesos ~iguais
        raw = np.array([np.exp(-temp * (crps[m] / cmin - 1.0)) for m in MEMBERS])
        wd = dict(zip(MEMBERS, raw / raw.sum()))
        cb_t = crps_q(Qb, y)
        n_eff = k * step / h
        dmean, dse = {}, {}
        for lm in LAMBDAS:                       # ganho de CRPS sobre o benchmark e seu erro-padrão
            dl = cb_t - crps_q(combine(Qm, wd, lm, Qb), y)
            dmean[lm] = float(dl.mean())
            dse[lm] = float(dl.std(ddof=1) / np.sqrt(max(n_eff, 2.0)))
        lbest = max(LAMBDAS, key=lambda v: dmean[v])
        # regra de 1 erro-padrão: sem ganho comprovado, encolhe tudo para o benchmark (λ=0)
        lam = 0.0 if dmean[lbest] <= 0 else min(v for v in LAMBDAS if dmean[v] >= dmean[lbest] - dse[lbest])
        best_c = float(cb_t.mean() - dmean[lam])
        Qe = combine(Qm, wd, lam, Qb)
        sel[h] = {"w": wd, "lam": float(lam), "n_val": int(k), "crps": crps,
                  "crps_bench": float(crps_q(Qb, y).mean()), "crps_ens": best_c,
                  "n_eff": float(k * step / h),
                  "a90": damp_alpha(aci_run(Qe, y, d, 0.10, cfg["gamma"])[-1], 0.10, k * step / h),
                  "a50": damp_alpha(aci_run(Qe, y, d, 0.50, cfg["gamma"])[-1], 0.50, k * step / h)}
    return sel


def fit_package(P, cfg):
    """Ajuste completo com dados em P (linhas [0, len(P))): seleção aninhada + ajuste final."""
    return {"sel": select_ensemble(P, cfg), "bundle": fit_bundle(P, len(P), cfg),
            "cut": len(P), "date": P.index[-1]}


def predict_package(pkg, P, rows):
    """Quantis do ensemble (e membros) para as linhas `rows` de P."""
    pred, meta = predict_bundle(pkg["bundle"], P, rows)
    out = {}
    for h in HORIZONS:
        s = pkg["sel"][h]
        Qm = {m: pred[h][m] for m in MEMBERS}
        out[h] = {"ens": combine(Qm, s["w"], s["lam"], pred[h]["bench"]), "bench": pred[h]["bench"],
                  "members": Qm, "Vs": meta[h]["Vs"]}
    return out


def run_validation(P, cfg):
    """Purged walk-forward (refit a cada bloco, com seleção aninhada em cada refit)."""
    n, o0, step = len(P), cfg["o0"], cfg["step"]
    K = cfg["folds"]
    first = o0 + 2520
    if n - first < 600:
        raise ValueError("insufficient_history")
    edges = np.linspace(first, n, K + 1).astype(int)
    logp = np.log(P["close"].to_numpy(float))
    out = {h: {"t": [], "y": [], "ens": [], "bench": [], "mem": {m: [] for m in MEMBERS}} for h in HORIZONS}
    folds = []
    for k in range(K):
        a, bnd = int(edges[k]), int(edges[k + 1])
        pkg = fit_package(P.iloc[:a], cfg)
        rows_all = np.arange(o0 + _ceil_div(a - o0, step) * step, bnd, step)
        rows_all = rows_all[rows_all <= n - 1 - HORIZONS[0]]
        folds.append({"cut": a, "date": P.index[a - 1], "sel": pkg["sel"]})
        if len(rows_all) == 0:
            continue
        pred, _ = predict_bundle(pkg["bundle"], P, rows_all)
        for h in HORIZONS:
            rows = rows_all[rows_all <= n - 1 - h]
            kk = len(rows)
            if kk == 0:
                continue
            s = pkg["sel"][h]
            Qm = {m: pred[h][m][:kk] for m in MEMBERS}
            o = out[h]
            o["t"].append(rows)
            o["y"].append(logp[rows + h] - logp[rows])
            o["bench"].append(pred[h]["bench"][:kk])
            o["ens"].append(combine(Qm, s["w"], s["lam"], pred[h]["bench"][:kk]))
            for m in MEMBERS:
                o["mem"][m].append(Qm[m])
    res = {}
    for h in HORIZONS:
        o = out[h]
        if not o["t"]:
            continue
        res[h] = {"t": np.concatenate(o["t"]), "y": np.concatenate(o["y"]),
                  "ens": np.vstack(o["ens"]), "bench": np.vstack(o["bench"]),
                  "mem": {m: np.vstack(o["mem"][m]) for m in MEMBERS}}
        res[h]["dates"] = P.index[res[h]["t"]]
    return {"h": res, "folds": folds, "step": step, "gamma": cfg["gamma"]}


def summarize_validation(val):
    """Métricas vs benchmark: R²_OS, CRPSS, Clark-West, Diebold-Mariano, cobertura, PIT, ACI."""
    rows, det = [], {}
    step, gamma = val["step"], val["gamma"]
    i05, i95, i25, i75 = qi(0.05), qi(0.95), qi(0.25), qi(0.75)
    for h, s in val["h"].items():
        y, Qe, Qb = s["y"], s["ens"], s["bench"]
        d = _ceil_div(h, step)
        ce, cb = crps_q(Qe, y), crps_q(Qb, y)
        me, mb = qmean(Qe), qmean(Qb)
        _, p_cw = cw_test(y, me, mb, d - 1)
        _, p_dm = dm_test(cb - ce, d - 1)
        pit = pit_values(Qe, y)
        a90 = aci_run(Qe, y, d, 0.10, gamma)
        a50 = aci_run(Qe, y, d, 0.50, gamma)
        cov90 = float(np.mean((y >= Qe[:, i05]) & (y <= Qe[:, i95])))
        cov50 = float(np.mean((y >= Qe[:, i25]) & (y <= Qe[:, i75])))
        skill = 1.0 - ce.mean() / cb.mean()
        cm = {m: float(1.0 - crps_q(s["mem"][m], y).mean() / cb.mean()) for m in MEMBERS}
        rows.append({"h": h, "n": int(len(y)), "n_eff": float(len(y) * step / h),
                     "r2os": r2_os(y, me, mb), "mse": float(np.mean((y - me) ** 2)), "mse_b": float(np.mean((y - mb) ** 2)),
                     "crpss": float(skill), "p_cw": p_cw, "p_dm": p_dm,
                     "cov90": cov90, "cov50": cov50, "cov90_aci": float(1.0 - np.nanmean(a90[3])),
                     "cov50_aci": float(1.0 - np.nanmean(a50[3])),
                     "pit_ks": float(stats.kstwo.sf(stats.kstest(pit, "uniform").statistic,
                                                    int(max(len(y) * step / h, 5)))),
                     "verdict": ("inconclusive" if len(y) * step / h < 30 else
                                 "yes" if (skill > 0 and np.isfinite(p_dm) and p_dm < 0.10) else "no")})
        k_roll = max(int(252 / step), 20)
        inside = ((y >= Qe[:, i05]) & (y <= Qe[:, i95])).astype(float)
        det[h] = {"pit_hist": np.histogram(pit, bins=10, range=(0, 1))[0], "members": cm,
                  "dates": s["dates"], "roll_cov": pd.Series(inside, index=s["dates"]).rolling(k_roll).mean(),
                  "crps_ens": float(ce.mean()), "crps_bench": float(cb.mean())}
    return pd.DataFrame(rows), det


# ----------------------------------------------------------------------------
# Dados: yfinance + FRED + Shiller (fallbacks + defasagem de publicação)
# ----------------------------------------------------------------------------
FRED_IDS = ["DGS10", "DFF", "T10Y2Y", "T10Y3M", "BAMLH0A0HYM2", "BAA10Y", "UNRATE", "CPIAUCNS", "GDP", "USREC"]
SHILLER_XLS = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
SHILLER_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
# Defasagem (dias após o INÍCIO do período de referência) até o dado ser publicado. FRED serve a
# última vintage (revisada), então usamos defasagens conservadoras como aproximação point-in-time.
PUB_LAG = {"UNRATE": 38, "CPIAUCNS": 45, "GDP": 120, "SHILLER": 35}


# ---- Configuração de rede (ajuste aqui se necessário) ------------------------------------------
AUTO_INSTALL_XLRD = True   # se faltar um leitor de .xls, tenta `pip install xlrd` uma vez (False desativa)
YF_GAP = 1.0               # pausa mínima (s) entre requisições ao Yahoo: evita a limitação de taxa
YF_BACKOFF = 2.5           # espera (s) crescente entre tentativas após uma resposta vazia
YF_COOLDOWN = 10.0         # espera (s) antes da 2ª rodada de tentativas dos itens que falharam
COMP_PATH = Path(__file__).resolve().parent / ".market_components.pkl" if "__file__" in globals() \
    else Path(".market_components.pkl")
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/124.0 Safari/537.36", "Accept": "text/csv,application/octet-stream,*/*"}
# Início conhecido de cada série (pedir desde 1950 para séries que só começam depois pode dar resposta vazia).
INCEPTION = {"^GSPC": "1950-01-01", "^SP500TR": "1988-01-04", "^VIX": "1990-01-02", "DX-Y.NYB": "1971-01-04"}
# Alternativas no FRED quando o Yahoo falha: VIX, dólar amplo e câmbio diário (série, inverter?).
FRED_FX = {"BRL": ("DEXBZUS", False), "EUR": ("DEXUSEU", True), "GBP": ("DEXUSUK", True), "JPY": ("DEXJPUS", False),
           "CNY": ("DEXCHUS", False), "CHF": ("DEXSZUS", False), "CAD": ("DEXCAUS", False), "AUD": ("DEXUSAL", True)}
SHILLER_MIRRORS = ["https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
                   "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv",
                   "https://datahub.io/core/s-and-p-500/r/data.csv"]
_LAST_CALL = {"t": 0.0}
_PIP_TRIED = {"xlrd": False}


def _pace():
    wait = YF_GAP - (time.time() - _LAST_CALL["t"])
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL["t"] = time.time()


def _norm_ohlc(raw):
    """Saída do yfinance (colunas planas ou MultiIndex, índice com/sem fuso) -> OHLC float limpo."""
    if raw is None or len(raw) == 0:
        raise ValueError("resposta vazia")
    out = pd.DataFrame(index=pd.to_datetime(raw.index).tz_localize(None))
    for c in ("Open", "High", "Low", "Close"):
        if c not in raw.columns:
            if c == "Close":
                raise ValueError("sem coluna Close")
            out[c] = np.nan
            continue
        col = raw[c]
        col = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
        out[c] = pd.to_numeric(col.to_numpy(), errors="coerce")
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    for c in ("Open", "High", "Low"):
        out[c] = out[c].fillna(out["Close"])
    return out[~out.index.duplicated()].sort_index().astype(float)


def _download(ticker, start, end, critical=False):
    """
    OHLC diário. Tentativas com pausa crescente (o Yahoo limita a taxa e responde vazio de forma intermitente):
    (1) desde max(start, início da série); (2) period='max'; (3) Ticker.history. O S&P 500 (crítico) ainda tenta
    desde 1970 e, por último, desde 2010 (histórico curto).
    """
    first = max(pd.Timestamp(start), pd.Timestamp(INCEPTION.get(ticker, "2000-01-01")))
    plans = [("range", first.strftime("%Y-%m-%d")), ("max", None)]
    if critical and first.year < 1970:
        plans.append(("range", "1970-01-01"))
    plans.append(("hist", None))                              # Ticker.history: caminho de código diferente do download()
    if critical:
        plans.append(("range", "2010-01-01"))
    errs = []
    for i, (kind, st_) in enumerate(plans):
        if i > 0 and YF_BACKOFF > 0:
            time.sleep(YF_BACKOFF * min(i, 3))
        _pace()
        try:
            if kind == "range":
                raw = yf.download(ticker, start=st_, end=end, auto_adjust=False, progress=False, threads=False)
            elif kind == "max":
                raw = yf.download(ticker, period="max", auto_adjust=False, progress=False, threads=False)
            else:
                raw = yf.Ticker(ticker).history(period="max", auto_adjust=False)
            out = _norm_ohlc(raw)
            out = out[(out.index >= pd.Timestamp(start)) & (out.index < pd.Timestamp(end))]
            if len(out) < 50:
                raise ValueError("série muito curta")
            return out
        except Exception as e:  # noqa: BLE001
            errs.append(f"{kind}{'' if st_ is None else ' ' + st_}: {str(e)[:60]}")
            if YF_BACKOFF > 0 and re.search(r"rate|429|too many", str(e), re.I):
                time.sleep(YF_BACKOFF * 4)
    raise RuntimeError(f"{ticker} (yfinance {getattr(yf, '__version__', '?')}): " + " | ".join(errs[-3:]))


def _http_get(url, timeout=15):
    """GET com timeout e 3 estratégias: requests -> curl_cffi (mesma pilha do yfinance) -> urllib."""
    errs = []
    try:
        import requests  # noqa: PLC0415
        r = requests.get(url, timeout=timeout, headers=_UA)
        r.raise_for_status()
        return r.content
    except Exception as e:  # noqa: BLE001
        errs.append(f"requests: {str(e)[:70]}")
    try:
        from curl_cffi import requests as creq  # noqa: PLC0415
        r = creq.get(url, timeout=timeout, impersonate="chrome")
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}")
        return r.content
    except Exception as e:  # noqa: BLE001
        errs.append(f"curl_cffi: {str(e)[:70]}")
    try:
        import urllib.request  # noqa: PLC0415
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except Exception as e:  # noqa: BLE001
        errs.append(f"urllib: {str(e)[:70]}")
    raise RuntimeError(" | ".join(errs))


def _looks_html(raw):
    head = raw[:300].lstrip().lower()
    return head.startswith(b"<!doctype") or b"<html" in head


def _cache_read():
    """Cache local por componente (VIX, câmbio, séries do FRED, Shiller...): completa o que falhar depois."""
    try:
        import pickle  # noqa: PLC0415
        with open(COMP_PATH, "rb") as f:
            blob = pickle.load(f)
        return blob if isinstance(blob, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _cache_write(cache):
    try:
        import pickle  # noqa: PLC0415
        tmp = Path(str(COMP_PATH) + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(cache, f)
        tmp.replace(COMP_PATH)
    except Exception:  # noqa: BLE001
        pass


def _read_fred(sid, start="1947-01-01"):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
    raw = _http_get(url, 20)
    if _looks_html(raw):
        raise ValueError("resposta HTML (acesso bloqueado?)")
    df = pd.read_csv(io.BytesIO(raw), na_values=[".", ""])
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df.iloc[:, 0])).dropna().sort_index()
    if len(s) < 10:
        raise ValueError("série muito curta")
    return s[~s.index.duplicated()]


def _finalize_shiller(out):
    out = out[~out.index.duplicated()].sort_index()
    if out["RP"].isna().all() and out["CPI"].notna().any():
        out["RP"] = out["P"] * out["CPI"].dropna().iloc[-1] / out["CPI"]
    out = out.dropna(subset=["P", "CAPE"])
    if len(out) < 600 or not out["CAPE"].between(2, 100).all():
        raise ValueError("layout inesperado do arquivo de Shiller")
    out["E10n"] = out["P"] / out["CAPE"]          # média de 10 anos dos lucros reais, em US$ do mês
    out["E10r"] = out["RP"] / out["CAPE"]         # idem, em US$ reais constantes
    return out


def parse_shiller_excel(raw):
    """
    Layout de ie_data.xls (aba Data, sem cabeçalho): col 0=Data(AAAA.MM), 1=P, 2=D, 3=E, 4=CPI, 6=GS10; o CAPE
    costuma estar na coluna 12, mas é DETECTADO (mediana 10-25, valores 2-100) caso o layout mude.
    O preço real é recalculado (P x CPI atual / CPI): não depende das colunas 'Real'.
    """
    recs, dates = [], []
    for _, r in raw.iterrows():
        try:
            d = float(r.iloc[0])
        except (TypeError, ValueError):
            continue
        if not 1800 < d < 2100:
            continue
        yr = int(d)
        mo = int(round((d - yr) * 100))
        if not 1 <= mo <= 12:
            continue
        dates.append(pd.Timestamp(year=yr, month=mo, day=1))
        recs.append(pd.to_numeric(r.iloc[:24], errors="coerce").to_numpy(float))
    if len(recs) < 600 or len(recs[0]) < 13:
        raise ValueError("layout inesperado do arquivo de Shiller")
    A = pd.DataFrame(np.vstack(recs), index=dates)

    def looks_like_cape(c):
        v = A[c].dropna()
        return len(v) > 600 and bool(v.between(2, 100).all()) and 8 <= float(v.median()) <= 30

    # 1) pelo texto do cabeçalho ('CAPE' / 'P/E10'), da esquerda para a direita; 2) heurística por proximidade da col. 12
    head = raw.iloc[:15].astype(str)
    by_name = [c for c in range(min(raw.shape[1], 40)) if head.iloc[:, c].str.contains(r"cape|p/e10", case=False, regex=True).any()]
    order = [c - 0 for c in by_name] + sorted(range(8, A.shape[1]), key=lambda c: (abs(c - 12), c))
    cc = next((c for c in order if c < A.shape[1] and looks_like_cape(c)), None)
    if cc is None:
        raise ValueError("coluna do CAPE não encontrada no arquivo de Shiller")
    out = pd.DataFrame({"P": A[1], "D": A[2], "E": A[3], "CPI": A[4], "GS10": A[6], "RP": np.nan, "RD": np.nan,
                        "RE": np.nan, "CAPE": A[cc]})
    return _finalize_shiller(out.astype(float))


def parse_shiller_csv(df):
    """Espelho CSV (datasets/s-and-p-500): Date, SP500, Dividend, Earnings, Consumer Price Index,
    Long Interest Rate, Real Price, Real Dividend, Real Earnings, PE10."""
    m = {str(c).strip().lower(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n in m:
                return pd.to_numeric(df[m[n]], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({"P": col("sp500", "p"), "D": col("dividend", "d"), "E": col("earnings", "e"),
                        "CPI": col("consumer price index", "cpi"), "GS10": col("long interest rate"),
                        "RP": col("real price"), "RD": col("real dividend"), "RE": col("real earnings"),
                        "CAPE": col("pe10", "cape")})
    dt = pd.to_datetime(df[m["date"]])
    out.index = pd.DatetimeIndex(dt.dt.to_period("M").dt.to_timestamp())
    return _finalize_shiller(out.astype(float))


def _pip_install(pkg):
    """Instala um pacote pequeno e conhecido no ambiente atual (uma única vez por sessão)."""
    if not AUTO_INSTALL_XLRD or _PIP_TRIED.get(pkg):
        return False
    _PIP_TRIED[pkg] = True
    try:
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", pkg],
                           capture_output=True, timeout=240)
        importlib.invalidate_caches()
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _read_excel_any(content):
    """Lê a aba 'Data' escolhendo o motor explicitamente (xlrd / calamine / openpyxl); instala o xlrd se faltar."""
    errs = []
    for attempt in range(2):
        for eng in ("xlrd", "calamine", "openpyxl"):
            try:
                try:
                    return pd.read_excel(io.BytesIO(content), sheet_name="Data", header=None, engine=eng)
                except ValueError as e:
                    if "sheet" in str(e).lower() or "worksheet" in str(e).lower():
                        return pd.read_excel(io.BytesIO(content), sheet_name=0, header=None, engine=eng)
                    raise
            except ImportError:
                errs.append(f"{eng}: não instalado")
            except Exception as e:  # noqa: BLE001
                errs.append(f"{eng}: {str(e)[:60]}")
        if attempt == 0 and importlib.util.find_spec("xlrd") is None and _pip_install("xlrd"):
            continue
        break
    raise RuntimeError("sem leitor de .xls (pip install xlrd) | " + " | ".join(errs[-3:]))


def read_shiller(override=""):
    """Tenta: arquivo/URL do usuário -> Yale (.xls) -> espelhos em CSV (não exigem leitor de Excel)."""
    errs = []
    for src in ([override.strip()] if override and override.strip() else []) + [SHILLER_XLS]:
        try:
            if str(src).lower().startswith(("http://", "https://")):
                content = _http_get(src, 30)
            else:
                with open(src, "rb") as f:
                    content = f.read()
            if _looks_html(content):
                raise ValueError("resposta HTML (link antigo?)")
            if str(src).lower().split("?")[0].endswith(".csv"):
                return parse_shiller_csv(pd.read_csv(io.BytesIO(content)))
            return parse_shiller_excel(_read_excel_any(content))
        except Exception as e:  # noqa: BLE001
            errs.append(f"{str(src)[-34:]}: {str(e)[:120]}")
    for url in SHILLER_MIRRORS:
        try:
            raw = _http_get(url, 25)
            if _looks_html(raw):
                raise ValueError("resposta HTML")
            return parse_shiller_csv(pd.read_csv(io.BytesIO(raw)))
        except Exception as e:  # noqa: BLE001
            errs.append(f"{url.split('/')[2]}: {str(e)[:80]}")
    raise RuntimeError(" || ".join(errs)[:420])


# ----------------------------------------------------------------------------
# Probabilidade de recessão (probit sobre a curva de juros), reestimada ANUALMENTE só com
# rótulos já resolvidos (12m à frente + 24m de atraso de divulgação do NBER): sem look-ahead
# ----------------------------------------------------------------------------
def fit_probit(x, y):
    def nll(p):
        eta = p[0] + p[1] * x
        return -(y * stats.norm.logcdf(eta) + (1 - y) * stats.norm.logcdf(-eta)).sum() + 1e-3 * float(p @ p)
    return optimize.minimize(nll, [-1.0, -0.3], method="BFGS").x


def recession_probit_pit(idx, spread, usrec):
    sp_m = spread.groupby(spread.index.to_period("M")).last()
    rec = usrec.copy()
    rec.index = rec.index.to_period("M")
    y_m = rec.rolling(12).max().shift(-12)            # houve recessão nos 12 meses seguintes?
    dfm = pd.concat([sp_m.rename("s"), y_m.rename("y")], axis=1).dropna()
    sp_d = spread.reindex(idx, method="ffill").shift(1)    # divulgado com 1 dia de atraso
    out = pd.Series(np.nan, index=idx)
    for yr in range(spread.index[0].year + 12, idx[-1].year + 1):
        tr_ = dfm[dfm.index <= pd.Period(f"{yr - 1}-12", "M") - 24]
        if len(tr_) < 120 or tr_["y"].sum() < 8:
            continue
        a, b = fit_probit(tr_["s"].to_numpy(float), tr_["y"].to_numpy(float))
        m = np.asarray(idx.year == yr)
        out[m] = stats.norm.cdf(a + b * sp_d[m])
    return out


def pseudo_shiller(close, cpi=None):
    """Substituto quando não há dados de lucros: 'CAPE' = preço real / média de 10 anos do preço real."""
    m = close.groupby(close.index.to_period("M")).mean()
    idx = m.index.to_timestamp()
    P = pd.Series(m.to_numpy(), index=idx)
    if cpi is not None:
        c = cpi.reindex(idx, method="ffill").bfill()
    else:
        c = pd.Series(np.exp(0.03 * (idx - idx[0]).days.to_numpy() / 365.25), index=idx)   # 3% a.a. fixo
    RP = P * c.iloc[-1] / c
    e10r = RP.rolling(120, min_periods=60).mean()
    out = pd.DataFrame({"P": P, "D": np.nan, "E": np.nan, "CPI": c, "GS10": np.nan, "RP": RP, "RD": np.nan,
                        "RE": np.nan, "CAPE": RP / e10r}).dropna(subset=["CAPE"])
    out["E10n"] = out["P"] / out["CAPE"]
    out["E10r"] = e10r.reindex(out.index)
    return out


def build_ext(idx, close, shiller, fred, dxy):
    """Features exógenas point-in-time (valuation, macro/crédito, recessão) no calendário do S&P."""
    ext = pd.DataFrame(index=idx)

    def daily(s, lag=1):
        return s.reindex(idx, method="ffill").shift(lag)         # publicado com 1 pregão de atraso

    def monthly(s, days):
        s2 = s.copy()
        s2.index = s2.index + pd.Timedelta(days=days)             # disponível só após a divulgação
        return s2.reindex(idx, method="ffill")

    dgs10 = daily(fred["DGS10"]) if "DGS10" in fred else None
    sh = shiller                                                  # --- valuation (Shiller ou pseudo-Shiller)
    avail = pd.DatetimeIndex(sh.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=PUB_LAG["SHILLER"]))

    def sh_daily(s):
        s = s.copy()
        s.index = avail
        return s.reindex(idx, method="ffill")

    lc = np.log(sh["CAPE"])
    e10 = sh_daily(sh["E10n"])
    n10 = min(len(sh) - 1, 120)                                   # se o arquivo está defasado, o lucro de 10a cresce na taxa recente
    g10 = float(np.clip(np.log(sh["E10n"].iloc[-1] / sh["E10n"].iloc[-1 - n10]) / (n10 / 12.0), 0.0, 0.08))
    stale = np.maximum((idx - avail[-1]).days.to_numpy(float), 0.0) / 365.25
    ext["log_cape"] = np.log(close / (e10 * np.exp(g10 * stale)))  # CAPE diário = preço de hoje / lucro 10a publicado
    ext["cape_z"] = (ext["log_cape"] - sh_daily(lc.expanding(120).mean())) / sh_daily(lc.expanding(120).std())
    ext["dy"] = sh_daily(sh["D"]) / close
    y10 = dgs10 / 100.0 if dgs10 is not None else sh_daily(sh["GS10"]) / 100.0
    ext["ey_minus_y"] = sh_daily(sh["E"]) / close - y10
    if "GDP" in fred:                                             # proxy "Buffett": S&P / PIB nominal
        ratio = np.log(close) - np.log(monthly(fred["GDP"], PUB_LAG["GDP"]))
        ext["mcap_z"] = (ratio - ratio.expanding(1260).mean()) / ratio.expanding(1260).std()
    if "T10Y2Y" in fred:
        ext["curve_10_2"] = daily(fred["T10Y2Y"])
    hy, baa = fred.get("BAMLH0A0HYM2"), fred.get("BAA10Y")        # spread de crédito: HY se tiver >=15 anos, senão Baa-10a
    cs = hy if (hy is not None and len(hy) >= 3750) else (baa if baa is not None else hy)
    if cs is not None:
        c = daily(cs)
        ext["credit_sp"], ext["credit_chg_63"] = c, c - c.shift(63)
    if "DFF" in fred:
        dff = daily(fred["DFF"])
        ext["dff"], ext["dff_chg_126"] = dff, dff - dff.shift(126)
    if "UNRATE" in fred:
        u = fred["UNRATE"]
        ext["unrate"] = monthly(u, PUB_LAG["UNRATE"])
        ext["unrate_chg_6m"] = monthly(u - u.shift(6), PUB_LAG["UNRATE"])
    if "CPIAUCNS" in fred:
        ext["cpi_yoy"] = monthly(fred["CPIAUCNS"].pct_change(12), PUB_LAG["CPIAUCNS"])
    if dxy is not None:
        dx = dxy.reindex(idx, method="ffill")
        ext["dxy_mom_63"] = np.log(dx / dx.shift(63))
    sp = fred.get("T10Y3M", fred.get("T10Y2Y"))
    if sp is not None and "USREC" in fred:
        ext["rec_prob"] = recession_probit_pit(idx, sp, fred["USREC"])
    return ext.replace([np.inf, -np.inf], np.nan)


def _close(obj):
    return obj["Close"] if isinstance(obj, pd.DataFrame) else obj


@st.cache_data(ttl=1800, show_spinner=False)
def load_market_data(start, today, shiller_override=""):
    """
    Baixa e prepara todos os dados. Cada fonte é um COMPONENTE independente com cache local: se uma atualização
    falha (o Yahoo limita a taxa de forma intermitente), usa-se o último valor salvo e o app vai completando o que
    falta a cada abertura. Falhas sem cache viram fallback (FRED, proxies) + aviso com o motivo.
    """
    end = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache, warns, stale, fails = _cache_read(), [], [], {}
    t0 = time.time()

    def fetch(name, fn, skip=False):
        if not skip:
            try:
                obj = fn()
                cache[name] = {"saved": pd.Timestamp.now(), "obj": obj}
                return obj
            except Exception as e:  # noqa: BLE001
                fails[name] = str(e)[:170]
        ent = cache.get(name)
        if ent is not None:
            stale.append((name, pd.Timestamp(ent["saved"]).strftime("%Y-%m-%d"), fails.get(name, "")[:90]))
            return ent["obj"]
        return None

    spx = fetch("spx", lambda: _download("^GSPC", start, end, critical=True))     # sem S&P não há app
    if spx is None:
        raise RuntimeError(fails.get("spx", "sem dados do S&P 500"))
    idx = spx.index
    close = spx["Close"]
    if idx[0] > pd.Timestamp("1951-01-01"):
        warns.append(("w_short_hist", {"d": idx[0].strftime("%Y-%m-%d")}))

    # ---- Yahoo: séries opcionais (1ª rodada; 2ª rodada após uma pausa para o que falhou)
    ticks = {"tr": "^SP500TR", "vix": "^VIX", **{f"fx:{c}": CCY[c][0] for c in CCY if CCY[c][0]}, "dxy": "DX-Y.NYB"}
    got = {nm: fetch(nm, lambda tk=tk: _download(tk, start, end)) for nm, tk in ticks.items()}
    missing = [nm for nm, v in got.items() if v is None]
    if missing and time.time() - t0 < 150:
        if YF_COOLDOWN > 0:
            time.sleep(YF_COOLDOWN)
        for nm in missing:
            got[nm] = fetch(nm, lambda tk=ticks[nm]: _download(tk, start, end))

    # ---- FRED: séries do modelo + alternativas para o que o Yahoo não entregou
    fred, fred_fail, streak, deadline = {}, [], 0, time.time() + 90

    def fred_get(sid):
        nonlocal streak
        skip = streak >= 3 or time.time() > deadline            # rede bloqueada: não espera cada timeout
        s = fetch("fred:" + sid, lambda: _read_fred(sid), skip=skip)
        streak = 0 if (s is not None and "fred:" + sid not in fails) else streak + (0 if skip else 1)
        return s

    for sid in FRED_IDS:
        s = fred_get(sid)
        if s is None:
            fred_fail.append(sid)
        else:
            fred[sid] = s
    if fred_fail:
        warns.append(("w_fred", {"sid": ", ".join(fred_fail), "err": fails.get("fred:" + fred_fail[0], "")[:110]}))

    # ---- retorno total
    if got["tr"] is not None:
        tr = _close(got["tr"]).reindex(idx, method="ffill")                      # NaN antes de 1988
    else:
        yrs = (idx - idx[0]).days.to_numpy() / 365.25
        tr = pd.Series(close.to_numpy() * np.exp(0.018 * yrs), index=idx)
        warns.append(("w_tr", {"err": fails.get("tr", "")[:90]}))
    # ---- VIX (Yahoo -> FRED VIXCLS -> proxy de volatilidade realizada)
    vsrc = got["vix"] if got["vix"] is not None else fred_get("VIXCLS")
    if vsrc is not None:
        vix = _close(vsrc).reindex(idx, method="ffill")
    else:
        rv = np.log(close).diff().rolling(21).std() * np.sqrt(252) * 100
        vix = rv.bfill().clip(lower=8)
        warns.append(("w_vix", {"err": fails.get("vix", "")[:90]}))
    # ---- câmbio (Yahoo -> FRED diário -> removido do seletor)
    fx = pd.DataFrame(index=idx)
    fx["USD"] = 1.0
    for code, (tk, inv) in CCY.items():
        if tk is None:
            continue
        obj = got.get(f"fx:{code}")
        if obj is None and code in FRED_FX:
            obj, inv = fred_get(FRED_FX[code][0]), FRED_FX[code][1]
        if obj is None:
            warns.append(("w_fx", {"ccy": code, "err": fails.get(f"fx:{code}", "")[:90]}))
            continue
        s = _close(obj)
        s = s[s > 0]
        fx[code] = (1.0 / s if inv else s).reindex(idx, method="ffill")          # sem bfill: sem dados antes do início
    # ---- dólar (Yahoo DXY -> FRED dólar amplo)
    dsrc = got["dxy"] if got["dxy"] is not None else fred_get("DTWEXBGS")
    dxy = _close(dsrc) if dsrc is not None else None
    if dxy is None:
        warns.append(("w_dxy", {}))

    if "BAMLH0A0HYM2" in fred and len(fred["BAMLH0A0HYM2"]) < 3750 and "BAA10Y" in fred:
        warns.append(("w_hy_short", {}))

    # ---- Shiller (arquivo/URL -> Yale -> espelhos CSV -> proxy de preços)
    proxy = False
    shiller = fetch("shiller", lambda: read_shiller(shiller_override))
    if shiller is None:
        proxy = True
        shiller = pseudo_shiller(close, fred.get("CPIAUCNS"))
        warns.append(("w_shiller", {"err": fails.get("shiller", "")[:230]}))
    elif (pd.Timestamp(today) - shiller.index[-1]).days > 120:
        warns.append(("w_shiller_stale", {"d": shiller.index[-1].strftime("%Y-%m")}))

    # ---- avisos de dados vindos do cache (agrupados por data de gravação)
    def label(names):
        fxn = sum(n.startswith("fx:") for n in names)
        frn = sum(n.startswith("fred:") for n in names)
        other = {"spx": "S&P 500", "tr": "^SP500TR", "vix": "VIX", "dxy": "DXY", "shiller": "Shiller"}
        parts = [other[n] for n in names if n in other]
        parts += ([f"{fxn} câmbio(s)" if _LANG == "pt" else f"{fxn} FX pair(s)"] if fxn else [])
        parts += ([f"{frn} série(s) FRED" if _LANG == "pt" else f"{frn} FRED series"] if frn else [])
        return ", ".join(parts)

    by_date = {}
    for nm, dt, err in stale:
        by_date.setdefault(dt, []).append((nm, err))
    for dt, items in by_date.items():
        warns.append(("w_cached", {"what": label([n for n, _ in items]), "d": dt,
                                   "err": next((e for _, e in items if e), "")[:100]}))

    ext = build_ext(idx, close, shiller, fred, dxy)
    _cache_write(cache)
    return {"spx": spx, "tr": tr, "vix": vix, "fx": fx, "ext": ext, "shiller": shiller,
            "shiller_proxy": proxy, "warnings": warns}


# ----------------------------------------------------------------------------
# Z-Score de valuation sem look-ahead + retornos futuros condicionais
# ----------------------------------------------------------------------------
def zscore_status(z, thr):
    return "over" if z >= thr else "under" if z <= -thr else "fair"


def forward_table(z, close, thr):
    """Retorno futuro (anualizado) condicional ao Z-Score point-in-time na origem."""
    logp = np.log(close)
    rows = []
    for yrs, h in ((1, 252), (3, 756), (5, 1260)):
        fwd = (logp.shift(-h) - logp) / yrs
        for bucket, mask in (("under", z <= -thr), ("fair", (z > -thr) & (z < thr)), ("over", z >= thr)):
            v = fwd[mask & z.notna()].dropna()
            rows.append({"bucket": bucket, "yrs": yrs, "n": int(len(v)),
                         "median": float(v.median()) if len(v) else np.nan,
                         "p_neg": float((v < 0).mean()) if len(v) else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Modelo 2 — fundamentos: retorno = crescimento de lucros + reversão do múltiplo + dividendos
# ----------------------------------------------------------------------------
def fund_annual(sh):
    """Observações anuais (dezembro): x = log CAPE, le = log lucro real de 10 anos, lrp = log preço real."""
    dec = sh[sh.index.month == 12].dropna(subset=["CAPE", "E10r", "RP"])
    out = pd.DataFrame({"x": np.log(dec["CAPE"]), "le": np.log(dec["E10r"]), "lrp": np.log(dec["RP"])})
    out.index = dec.index.year
    return out


def _slope(x):
    """Inclinação OLS de Δx em x e seu erro-padrão. Não depende do intercepto (evita a razão −a/b instável)."""
    y, xl = np.diff(x), x[:-1]
    xc = xl - xl.mean()
    sxx = float(xc @ xc)
    b = float(xc @ (y - y.mean()) / sxx)
    res = (y - y.mean()) - b * xc
    return b, float(np.sqrt(float(res @ res) / max(len(y) - 2, 1) / sxx)), len(y)


def bayes_params(ann, start_year, n0, rev, N, seed=SEED):
    """
    Posterior (normal) dos 3 parâmetros do Modelo 2, com PRIOR = amostra histórica completa (peso de n0 anos)
    e verossimilhança da amostra escolhida: velocidade de reversão do log-CAPE (κ), média de longo prazo
    (x̄) e crescimento real do lucro (g). Sorteios independentes e LIMITADOS a faixas plausíveis, calculadas
    na própria escala da série (funciona com CAPE de Shiller ou com o proxy de preços).
    """
    rng = np.random.default_rng(seed)
    x_all = ann["x"].to_numpy()
    x_sel = ann.loc[ann.index >= start_year, "x"].to_numpy()
    if len(x_sel) < 25:
        x_sel = x_all
    (bf, sef, nf), (bs, ses, ns_) = _slope(x_all), _slope(x_sel)

    def kendall(b, n):                                   # correção de viés de pequenas amostras no AR(1)
        phi = 1.0 + b
        return min(phi + (1.0 + 3.0 * phi) / n, 0.995) - 1.0

    bf_c, bs_c = kendall(bf, nf), kendall(bs, ns_)
    p0, p1 = (n0 / nf) / sef ** 2, 1.0 / ses ** 2        # precisão do prior (n0 anos) e da amostra
    b_mean, b_sd = (p0 * bf_c + p1 * bs_c) / (p0 + p1), 1.0 / np.sqrt(p0 + p1)
    b = np.clip(rng.normal(b_mean, b_sd, N), -0.30, -0.02)      # meia-vida entre ~2,5 e ~35 anos
    kappa = rev * -np.log1p(b)

    def mean_se(x, slope):                               # média de longo prazo com amostra efetiva (persistência)
        phi = float(np.clip(1.0 + slope, 0.0, 0.97))
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(max(len(x) * (1 - phi) / (1 + phi), 3.0)))

    (mf, sf), (ms, ss) = mean_se(x_all, bf_c), mean_se(x_sel, bs_c)
    q0, q1 = (n0 / nf) / sf ** 2, 1.0 / ss ** 2
    lo_x, hi_x = np.quantile(x_all, [0.05, 0.95])                # x̄ não sai da faixa histórica típica
    xbar = np.clip(rng.normal((q0 * mf + q1 * ms) / (q0 + q1), 1.0 / np.sqrt(q0 + q1), N), lo_x, hi_x)

    dle_all = np.diff(ann["le"].to_numpy())
    dle_sel = np.diff(ann.loc[ann.index >= start_year, "le"].to_numpy())
    if len(dle_sel) < 25:
        dle_sel = dle_all
    m_p = (n0 * dle_all.mean() + len(dle_sel) * dle_sel.mean()) / (n0 + len(dle_sel))
    s_p = float(np.sqrt((dle_all.var() * n0 + dle_sel.var() * len(dle_sel)) / (n0 + len(dle_sel))))
    g = np.clip(rng.normal(m_p, s_p / np.sqrt(n0 + len(dle_sel)), N),
                dle_all.mean() - 0.02, dle_all.mean() + 0.02)    # ±2 p.p. em torno do histórico completo
    return {"g": g, "kappa": kappa, "xbar": xbar, "phi": 1.0 + b, "g_mean": float(m_p),
            "mean_ols_phi": float(1.0 + bs), "n_sel": int(len(x_sel)), "n_all": int(len(x_all)),
            # a pressão de reversão satura nos extremos históricos de valuation (não se extrapola além deles)
            "x_eff": (float(x_all.min() - 0.1), float(x_all.max() + 0.1)),
            "x_range": (float(x_all.min() - 0.3), float(x_all.max() + 0.3))}


def payout_ratio(sh):
    """Dividendo / lucro-10a médio dos últimos 30 anos (dy = ρ·exp(-x))."""
    r = (sh["D"] / sh["E10n"]).dropna()
    return float(np.clip(r.iloc[-360:].mean(), 0.2, 0.9)) if len(r) > 60 else 0.5


@st.cache_data(show_spinner=False)
def fhs_pool(close, seed=SEED):
    """
    GJR-GARCH-t no histórico diário completo -> resíduos padronizados (pool FHS). Para a SIMULAÇÃO de longo
    prazo, a persistência efetiva (com o pool empírico, assimétrico) é limitada a 0,995 e o intercepto é
    recalibrado para que a variância de longo prazo seja a histórica (variance targeting): evita que a
    variância simulada exploda ao longo de 14 anos.
    """
    r = np.r_[0.0, np.diff(np.log(close.to_numpy(float)))]
    g = garch_fit(r[1:])
    R = 100.0 * r
    th = g["theta"]
    s2 = gjr_variance(R - th[0], th[1:5], g["v0"])
    zz = (R[1:] - th[0]) / np.sqrt(s2[1:len(R)])                 # ε_t / σ_t (t = 1..n-1)
    zz = zz[np.isfinite(zz)]
    zz = (zz - zz.mean()) / zz.std()
    q = float(np.mean((zz < 0) * zz ** 2))                       # E[1(z<0) z²]: >0,5 com cauda esquerda pesada
    al, ga, be = th[2], th[3], th[4]
    p_eff = float(al + ga * q + be)
    k = min(1.0, 0.995 / p_eff)
    al_s, ga_s, be_s = al * k, ga * k, be * k
    v_target = float(np.var(R[1:]))                              # variância histórica (%²/dia)
    om_s = v_target * (1.0 - (al_s + ga_s * q + be_s))
    return {"garch": g, "pool": zz, "s2_next": float(s2[-1]), "sim": (om_s, al_s, ga_s, be_s),
            "p_eff": p_eff, "ann_vol": float(np.sqrt(v_target * 252.0) / 100.0)}


def simulate_long(par, N=N_SIMS, seed=SEED):
    """
    Filtered Historical Simulation: choques = resíduos padronizados (bootstrap em blocos de 21 dias)
    escalados pela variância GJR-GARCH-t simulada. Drift diário = (g + π + κ(x̄ − x)) / 252, e o log-CAPE
    x evolui com o retorno simulado (reversão do múltiplo condicionada à trajetória).
    """
    rng = np.random.default_rng(seed)
    g_nom = par["bayes"]["g"][:N] + par["pi"] + par["adj"]
    kappa, xbar = par["bayes"]["kappa"][:N], par["bayes"]["xbar"][:N]
    days = pd.date_range(par["date0"] + BDAY, f"{LONG_END_YEAR}-12-31", freq=BDAY)
    S = len(days)
    pm = days.to_period("M")
    snap = np.where(np.r_[np.asarray(pm[1:] != pm[:-1]), True])[0]
    om, al, ga, be = par["sim"]
    pool, B, npool = par["pool"], 21, len(par["pool"])
    lo_e, hi_e = par["bayes"]["x_eff"]
    x = np.full(N, par["x0"])
    s2 = np.full(N, par["s2_next"])
    lv = np.zeros(N)
    ptr = np.zeros(N, dtype=int)
    rho, use_tr = par["rho"], par["use_tr"]
    out = np.empty((len(snap), N))
    k = 0
    for s in range(S):
        if s % B == 0:
            ptr = rng.integers(0, npool - B, N)
        eps = np.sqrt(s2) / 100.0 * pool[ptr]
        ptr = ptr + 1
        xe = np.clip(x, lo_e, hi_e)                              # a pressão de reversão satura nos extremos históricos
        r = (g_nom + kappa * (xbar - xe)) / 252.0 + eps
        x = x + r - g_nom / 252.0                                # o estado segue o preço (sem truncar: mantém a identidade)
        lv += r + (rho * np.exp(-xe) / 252.0 if use_tr else 0.0)
        e = 100.0 * eps
        s2 = om + (al + ga * (e < 0)) * e * e + be * s2
        if k < len(snap) and s == snap[k]:
            out[k] = lv
            k += 1
    lev = par["level0"] * np.exp(out)
    pcts = {p: np.percentile(lev, p, axis=1) for p in (5, 10, 25, 50, 75, 90, 95)}
    return {"dates": days[snap], "pct": pcts, "final": lev[-1], "level0": par["level0"],
            "date0": par["date0"], "n": N}


def decomposition(bp, x0, rho, pi, adj, horizon=10):
    """Contribuições esperadas (a.a., log) nos próximos `horizon` anos com os parâmetros médios."""
    x0 = float(np.clip(x0, *bp["x_eff"]))
    g = bp["g"].mean() + pi + adj
    kap, xb = float(bp["kappa"].mean()), float(bp["xbar"].mean())
    ks = np.arange(0, horizon + 1)
    xk = xb + np.exp(-kap * ks) * (x0 - xb)
    return {"earn": float(g), "mult": float((xk[-1] - x0) / horizon),
            "div": float(np.mean(rho * np.exp(-xk[:-1]))), "x_end": float(xk[-1])}


def stress_paths(close, horizon=1260):
    """Trajetórias de estresse históricas (relativas ao pico) para reaplicar ao nível atual."""
    ev = {"dotcom": "2000-03-24", "gfc": "2007-10-09", "covid": "2020-02-19"}
    out = {}
    for k, d in ev.items():
        i0 = int(np.searchsorted(close.index.values, np.datetime64(pd.Timestamp(d))))
        if i0 + 60 >= len(close):
            continue
        seg = close.iloc[i0:i0 + horizon + 1].to_numpy(float)
        rel = seg / seg[0]
        j = int(np.argmin(rel))
        rec = np.where(rel[j:] >= 1.0)[0]
        out[k] = {"rel": rel, "dd": float(rel.min() - 1.0), "days_to_trough": j,
                  "days_to_recover": int(j + rec[0]) if len(rec) else None}
    return out


def valuation_evidence(ann, hs=(1, 3, 5, 10), min_obs=40):
    """Previsibilidade do retorno real por valuation (log-CAPE): R² fora da amostra e Clark-West."""
    x, lrp = ann["x"].to_numpy(), ann["lrp"].to_numpy()
    n = len(x)
    rows = []
    for H in hs:
        y = np.full(n, np.nan)
        y[:n - H] = (lrp[H:] - lrp[:n - H]) / H
        ys, fm, fb = [], [], []
        for t in range(min_obs, n - H):
            s_ = np.arange(0, t - H + 1)                 # origens já resolvidas em t
            if len(s_) < 25:
                continue
            beta = np.linalg.lstsq(np.column_stack([np.ones(len(s_)), x[s_]]), y[s_], rcond=None)[0]
            ys.append(y[t])
            fm.append(beta[0] + beta[1] * x[t])
            fb.append(y[s_].mean())
        if len(ys) < 20:
            continue
        ys, fm, fb = map(np.array, (ys, fm, fb))
        _, p = cw_test(ys, fm, fb, H - 1)
        full = np.linalg.lstsq(np.column_stack([np.ones(n - H), x[:n - H]]), y[:n - H], rcond=None)[0]
        rows.append({"H": H, "n": int(len(ys)), "r2os": r2_os(ys, fm, fb), "p_cw": p, "slope": float(full[1])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Leque (fan chart) a partir dos quantis por horizonte (interpolação suave em h)
# ----------------------------------------------------------------------------
def fan_from_quantiles(Qh, sel, p_last, last_date, mult=1.0, use_aci=True):
    """
    Qh: {h: vetor de quantis do log-retorno}. Bandas de 90%/50% com os alphas do ACI (amortecido).
    A curva em h é interpolada em y = q/√h (PCHIP), preservando a escala de difusão.
    """
    ah = np.array(HORIZONS)
    i50 = qi(0.5)
    A = np.zeros((len(ah), 5))
    for j, h in enumerate(ah):
        a90 = sel[h]["a90"] if use_aci else 0.10
        a50 = sel[h]["a50"] if use_aci else 0.50
        q = Qh[h]
        A[j] = np.sort([np.interp(a90 / 2, QS, q), np.interp(a50 / 2, QS, q), q[i50],
                        np.interp(1 - a50 / 2, QS, q), np.interp(1 - a90 / 2, QS, q)])
    hh = np.arange(0, HORIZONS[-1] + 1)
    cols = np.empty((len(hh), 5))
    for k in range(5):
        f = PchipInterpolator(ah, A[:, k] / np.sqrt(ah))(np.clip(hh, ah[0], ah[-1]))
        cols[:, k] = f * np.sqrt(hh)
    cols = np.sort(cols, axis=1)
    dates = pd.DatetimeIndex([last_date]).append(pd.date_range(last_date + BDAY, periods=HORIZONS[-1], freq=BDAY))
    lev = p_last * np.exp(cols) * mult
    return pd.DataFrame(lev, index=dates, columns=["lo90", "lo50", "med", "hi50", "hi90"])


# ----------------------------------------------------------------------------
# Auditoria de features: SHAP (se instalado) + importância por permutação OOS + "canários" de ruído
# ----------------------------------------------------------------------------
def feature_audit(P, cfg, h, n_noise=5):
    """Se uma feature real não supera os canários de ruído aleatório, o modelo está se apoiando em ruído."""
    o0, step, seed = cfg["o0"], cfg["step"], cfg["seed"]
    n = len(P)
    logp = np.log(P["close"].to_numpy(float))
    rows = np.arange(o0, n - h, step)
    y = logp[rows + h] - logp[rows]
    cols = [c for c in FEATS if P[c].iloc[o0:].notna().mean() > 0.4]
    rng = np.random.default_rng(seed)
    X = np.column_stack([P[cols].to_numpy(float)[rows], rng.standard_normal((len(rows), n_noise))])
    names = cols + [f"noise_{i + 1}" for i in range(n_noise)]
    scale = np.sqrt(h / 252.0) * np.clip(P["rv_63"].to_numpy(float)[rows], 0.05, 1.0)
    z = (y - y.mean()) / np.where(np.isfinite(scale), scale, np.nanmedian(scale))
    cut = int(0.7 * len(rows))
    tr = np.arange(0, max(cut - _ceil_div(h, step), 50))
    te = np.arange(cut, len(rows))
    lo, hi = np.nanquantile(X[tr], 0.01, axis=0), np.nanquantile(X[tr], 0.99, axis=0)
    med = np.nanmedian(np.clip(X[tr], lo, hi), axis=0)
    prep = lambda A: np.where(np.isfinite(np.clip(A, lo, hi)), np.clip(A, lo, hi), med)  # noqa: E731
    Xtr, Xte = prep(X[tr]), prep(X[te])
    msl = int(min(max(40, 3 * _ceil_div(h, step)), max(len(tr) // 5, 5)))
    rf = RandomForestRegressor(n_estimators=cfg["n_trees"], min_samples_leaf=msl, max_features=0.33,
                               max_samples=0.7, n_jobs=-1, random_state=seed).fit(Xtr, z[tr])
    pi = permutation_importance(rf, Xte, z[te], n_repeats=5, random_state=seed, scoring="r2", n_jobs=1)
    shap_v = np.full(len(names), np.nan)
    if HAS_SHAP:
        try:
            import shap  # noqa: PLC0415
            sv = shap.TreeExplainer(rf).shap_values(Xte[: min(len(Xte), 400)], check_additivity=False)
            shap_v = np.abs(np.asarray(sv)).mean(axis=0)
        except Exception:  # noqa: BLE001
            shap_v = np.full(len(names), np.nan)
    df = pd.DataFrame({"feature": names, "perm": pi.importances_mean, "perm_sd": pi.importances_std,
                       "shap": shap_v, "noise": [nm.startswith("noise_") for nm in names]})
    ceiling = float(df.loc[df["noise"], "perm"].max())
    df["above_noise"] = (~df["noise"]) & (df["perm"] > max(ceiling, 0.0))
    return {"table": df.sort_values("perm", ascending=False).reset_index(drop=True), "ceiling": ceiling,
            "r2_oos": float(rf.score(Xte, z[te])), "h": h, "has_shap": bool(HAS_SHAP and np.isfinite(shap_v).any())}


# ----------------------------------------------------------------------------
# Monitor de drift (PSI) — distribuição recente vs. distribuição de treino
# ----------------------------------------------------------------------------
def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.clip(np.histogram(ref, bins=edges)[0] / len(ref), 1e-4, None)
    q = np.clip(np.histogram(cur, bins=edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def drift_table(P, o0, window=126):
    """
    PSI da janela recente vs. todo o histórico de treino, comparado com a distribuição de PSIs de
    janelas históricas do MESMO tamanho (features persistentes geram PSI alto sem haver drift real):
    o status usa o percentil do PSI atual na própria história da feature.
    """
    rows = []
    for c in FEATS:
        s = P[c].iloc[o0:].dropna().to_numpy(float)
        if len(s) < 600 + window:
            continue
        ref, cur = s[:-window], s[-window:]
        v = psi(ref, cur)
        hist = np.array([psi(ref, s[a:a + window]) for a in range(0, len(ref) - window, 63)])
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(v) or len(hist) < 8:
            continue
        pct = float(np.mean(hist <= v))
        rows.append({"feature": c, "psi": v, "pct": pct,
                     "status": "ok" if pct < 0.90 else "warn" if pct < 0.98 else "drift"})
    if not rows:
        return pd.DataFrame(columns=["feature", "psi", "pct", "status"])
    return pd.DataFrame(rows).sort_values(["pct", "psi"], ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Track record ao vivo: cada previsão (data de origem + quantis) é gravada e avaliada depois
# ----------------------------------------------------------------------------
QCOLS = [f"q{q:.3f}" for q in QS]
LOG_COLS = ["logged_at", "origin_date", "h", "version", "p0"] + QCOLS + ["a90", "a50"]
LOG_PATH = Path(__file__).resolve().parent / "forecast_log.csv" if "__file__" in globals() \
    else Path("forecast_log.csv")


def read_log(path=None):
    path = Path(path or LOG_PATH)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLS)
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=LOG_COLS)


def _log_key(d):
    return d["origin_date"].astype(str) + "|" + d["h"].astype(str) + "|" + d["version"].astype(str)


def merge_log(new, path=None):
    """Acrescenta registros novos (idempotente por origem+horizonte+versão). Retorna nº de linhas novas."""
    path = Path(path or LOG_PATH)
    old = read_log(path)
    add = new[~_log_key(new).isin(set(_log_key(old)))] if len(old) else new
    if add.empty:
        return 0
    pd.concat([old, add[LOG_COLS]], ignore_index=True).to_csv(path, index=False)
    return int(len(add))


def make_log_rows(pred, sel, p0, origin, version):
    rows = []
    for h in HORIZONS:
        q = pred[h]["ens"][-1]
        rows.append([pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), pd.Timestamp(origin).strftime("%Y-%m-%d"),
                     h, version, float(p0)] + [float(v) for v in q] + [float(sel[h]["a90"]), float(sel[h]["a50"])])
    return pd.DataFrame(rows, columns=LOG_COLS)


def evaluate_log(log, close):
    """Confronta cada previsão registrada com o resultado realizado (quando já ocorreu)."""
    if log.empty:
        return pd.DataFrame()
    idx = close.index
    logp = np.log(close.to_numpy(float))
    out = []
    for _, r in log.iterrows():
        pos = int(idx.get_indexer([pd.Timestamp(r["origin_date"])], method="pad")[0])
        h = int(r["h"])
        if pos < 0:
            continue
        q = np.sort(r[QCOLS].to_numpy(float))
        a90, a50 = float(r["a90"]), float(r["a50"])
        rec = {"origin": pd.Timestamp(r["origin_date"]), "h": h, "version": r["version"],
               "median": float(q[qi(0.5)]), "resolved": pos + h <= len(idx) - 1}
        if rec["resolved"]:
            y = float(logp[pos + h] - logp[pos])
            lo, hi = np.interp(a90 / 2, QS, q), np.interp(1 - a90 / 2, QS, q)
            rec.update({"target": idx[pos + h], "realized": y, "pit": float(np.interp(y, q, QS)),
                        "crps": float(crps_q(q[None, :], np.array([y]))[0]),
                        "in90": bool(lo <= y <= hi), "in90_raw": bool(q[qi(0.05)] <= y <= q[qi(0.95)])})
        out.append(rec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Gráficos (todos os rótulos passam por tr(): Português/Inglês)
# ----------------------------------------------------------------------------
def _log_ticks(lo=1, hi=10 ** 7):
    """Marcas 1-2-5 legíveis para eixo logarítmico (o Plotly mostraria '2' e '5' sem a potência de 10)."""
    vals, d = [], lo
    while d <= hi:
        vals += [v for v in (d, 2 * d, 5 * d) if v <= hi]
        d *= 10
    return vals, [fnum(v, 0) for v in vals]


def _title(text, size=16):
    return dict(text=text, x=0.0, xanchor="left", y=0.985, yanchor="top", font=dict(size=size))


def fig_base(title, ylabel, log=False, height=540, legend_rows=2):
    """
    Título no topo e legenda abaixo do eixo x (nunca disputam o mesmo espaço). As margens são calculadas
    pelo nº de linhas da legenda, e a legenda é posicionada em pixels relativos ao eixo.
    """
    t, b = 64, 70 + 24 * legend_rows
    H = height + 24 * legend_rows
    plot_h = H - t - b
    fig = go.Figure()
    fig.update_layout(
        title=_title(title), height=H, hovermode="x unified", separators=",." if _LANG == "pt" else ".,",
        yaxis_title=ylabel, yaxis_type="log" if log else "linear", font=dict(size=13),
        legend=dict(orientation="h", yanchor="top", y=-48.0 / plot_h, xanchor="left", x=0.0, font=dict(size=12)),
        margin=dict(l=10, r=10, t=t, b=b), xaxis_rangeslider_visible=False)
    if log:
        vals, texts = _log_ticks()
        fig.update_yaxes(tickvals=vals, ticktext=texts)
    else:
        fig.update_yaxes(tickformat=",.0f")
    return fig


def _window(s, years):
    return s if not years else s[s.index >= s.index[-1] - pd.DateOffset(years=years)]


def _ylabel(ccy):
    return tr("axis_level", ccy=ccy)


def fig_overview(disp, ccy, years, log):
    c = disp
    ema50, ema200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    w = lambda s: _window(s, years)  # noqa: E731
    fig = fig_base(tr("ov_chart_title", ccy=ccy), _ylabel(ccy), log)
    fig.add_trace(go.Scatter(x=w(ma20).index, y=w(ma20 - 2 * sd20), line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=w(ma20).index, y=w(ma20 + 2 * sd20), name=tr("lg_boll"), fill="tonexty",
                             fillcolor="rgba(148,163,184,0.18)", line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=w(c).index, y=w(c), name="S&P 500", line=dict(color="#111827", width=1.6),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=w(ema50).index, y=w(ema50), name=tr("lg_ema50"),
                             line=dict(color="#f59e0b", width=1.3), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=w(ema200).index, y=w(ema200), name=tr("lg_ema200"),
                             line=dict(color="#dc2626", width=1.3), hovertemplate="%{y:,.0f}"))
    return fig


def fig_gauge(z, thr, title):
    zc = float(np.clip(z, -3.5, 3.5))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=zc, number={"valueformat": ".2f"}, title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [-3.5, 3.5]}, "bar": {"color": "#111827", "thickness": 0.25},
               "steps": [{"range": [-3.5, -thr], "color": "rgba(22,163,74,0.45)"},
                         {"range": [-thr, thr], "color": "rgba(234,179,8,0.35)"},
                         {"range": [thr, 3.5], "color": "rgba(220,38,38,0.45)"}]}))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def fig_trend(P, thr, ylabel, years, log):
    fit, sd = P["trend_fit"], P["trend_sd"]
    f = np.exp(fit)
    up, lo = np.exp(fit + thr * sd), np.exp(fit - thr * sd)
    w = lambda s: _window(s.dropna(), years)  # noqa: E731
    fig = fig_base(tr("trend_title"), ylabel, log, 420)
    fig.add_trace(go.Scatter(x=w(lo).index, y=w(lo), line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=w(up).index, y=w(up), name=tr("lg_fairzone", t=fnum(thr, 1)), fill="tonexty",
                             fillcolor="rgba(234,179,8,0.18)", line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=w(f).index, y=w(f), name=tr("lg_trend_pit"),
                             line=dict(color="#2563eb", width=2, dash="dash"), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=w(P["close"]).index, y=w(P["close"]), name="S&P 500",
                             line=dict(color="#111827", width=1.6), hovertemplate="%{y:,.0f}"))
    return fig


def add_fan(fig, fan, label_med, color="37,99,235", show_names=True):
    x = fan.index
    edge = dict(color=f"rgba({color},0.45)", width=1, dash="dot")
    fig.add_trace(go.Scatter(x=x, y=fan["lo90"], mode="lines", line=edge, showlegend=False,
                             hovertemplate="%{y:,.0f}", name=tr("lg_p5")))
    fig.add_trace(go.Scatter(x=x, y=fan["hi90"], mode="lines", fill="tonexty", fillcolor=f"rgba({color},0.12)",
                             line=edge, name=tr("lg_band90"), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=fan["lo50"], mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=fan["hi50"], mode="lines", fill="tonexty", fillcolor=f"rgba({color},0.22)",
                             line=dict(width=0), name=tr("lg_band50"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=fan["med"], mode="lines", line=dict(color=f"rgb({color})", width=2.6),
                             name=label_med, hovertemplate="%{y:,.0f}"))


def fig_fan(hist, fan, bench_fan, ccy, years, show_bench):
    fig = fig_base(tr("m1_chart_title", ccy=ccy), _ylabel(ccy))
    h = _window(hist, years)
    fig.add_trace(go.Scatter(x=h.index, y=h, name=tr("lg_hist"), line=dict(color="#111827", width=1.6),
                             hovertemplate="%{y:,.0f}"))
    if show_bench and bench_fan is not None:
        for k, nm in (("lo90", tr("lg_bench")), ("hi90", None)):
            fig.add_trace(go.Scatter(x=bench_fan.index, y=bench_fan[k], mode="lines", name=nm or "",
                                     showlegend=nm is not None, line=dict(color="rgba(107,114,128,0.9)", width=1.2,
                                                                          dash="dash"), hovertemplate="%{y:,.0f}"))
    add_fan(fig, fan, tr("lg_median"))
    return fig


def fig_long(x, pc, hist, ccy, label, log, stress=None):
    fig = fig_base(tr("m2_chart_title", label=label, ccy=ccy, year=LONG_END_YEAR), _ylabel(ccy), log, 580,
                   legend_rows=3 if stress else 2)
    for p, nm in ((5, None), (95, None)):
        fig.add_trace(go.Scatter(x=x, y=pc[p], mode="lines", showlegend=False, hovertemplate="%{y:,.0f}",
                                 line=dict(color="rgba(100,116,139,0.5)", width=1, dash="dot"),
                                 name="P5" if p == 5 else "P95"))
    fig.add_trace(go.Scatter(x=x, y=pc[10], name=tr("lg_pess"), mode="lines",
                             line=dict(color="rgb(220,38,38)", width=2), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=pc[50], name=tr("lg_base"), mode="lines", fill="tonexty",
                             fillcolor="rgba(220,38,38,0.16)", line=dict(color="rgb(37,99,235)", width=2.8),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=pc[90], name=tr("lg_opt"), mode="lines", fill="tonexty",
                             fillcolor="rgba(22,163,74,0.16)", line=dict(color="rgb(22,163,74)", width=2),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name=tr("lg_hist"), line=dict(color="#111827", width=1.6),
                             hovertemplate="%{y:,.0f}"))
    for nm, (sx, sy) in (stress or {}).items():
        fig.add_trace(go.Scatter(x=sx, y=sy, name=tr("stress_" + nm), mode="lines",
                                 line=dict(width=1.6, dash="dash"), hovertemplate="%{y:,.0f}"))
    return fig


def fig_backtest(hist, fan, realized, bench_fan, ccy, origin):
    fig = fig_base(tr("bt_chart_title", d=fdate(origin)), _ylabel(ccy))
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name=tr("lg_train"), line=dict(color="#6b7280", width=1.5),
                             hovertemplate="%{y:,.0f}"))
    if bench_fan is not None:
        for k, nm in (("lo90", tr("lg_bench")), ("hi90", None)):
            fig.add_trace(go.Scatter(x=bench_fan.index, y=bench_fan[k], mode="lines", name=nm or "",
                                     showlegend=nm is not None, line=dict(color="rgba(107,114,128,0.9)", width=1.2,
                                                                          dash="dash"), hovertemplate="%{y:,.0f}"))
    add_fan(fig, fan, tr("lg_forecast"))
    if realized is not None and len(realized):
        fig.add_trace(go.Scatter(x=realized.index, y=realized, name=tr("lg_actual"),
                                 line=dict(color="#111827", width=2.4), hovertemplate="%{y:,.0f}"))
    xs = pd.Timestamp(origin).strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=xs, x1=xs, y0=0, y1=1, yref="paper",
                  line=dict(color="rgba(100,116,139,0.8)", dash="dot", width=1.5))
    fig.add_annotation(x=xs, y=1, yref="paper", text=tr("lg_origin"), showarrow=False, yanchor="bottom")
    return fig


def fig_pit(counts, h):
    tot = max(float(np.sum(counts)), 1.0)
    fig = go.Figure(go.Bar(x=[f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)], y=np.asarray(counts) / tot,
                           marker_color="rgba(37,99,235,0.7)", name="PIT"))
    fig.add_shape(type="line", x0=-0.5, x1=9.5, y0=0.1, y1=0.1, line=dict(color="#dc2626", dash="dash"))
    fig.update_layout(title=_title(tr("pit_title", h=h), 14), height=320, margin=dict(l=10, r=10, t=60, b=10),
                      yaxis_tickformat=".0%", separators=",." if _LANG == "pt" else ".,", showlegend=False)
    return fig


def fig_skill(summ):
    labels = [f"{h}d" for h in summ["h"]]
    colors = ["rgba(22,163,74,0.75)" if v == "yes" else "rgba(234,179,8,0.75)" if v == "inconclusive"
              else "rgba(220,38,38,0.7)" for v in summ["verdict"]]
    fig = go.Figure(go.Bar(x=labels, y=summ["crpss"] * 100.0, marker_color=colors))
    fig.update_layout(title=_title(tr("skill_title"), 14), height=320, margin=dict(l=10, r=10, t=60, b=10),
                      yaxis_title="CRPSS (%)", separators=",." if _LANG == "pt" else ".,", showlegend=False)
    return fig


def fig_roll_cov(roll, nominal=0.90):
    fig = go.Figure(go.Scatter(x=roll.index, y=roll.values, mode="lines", line=dict(color="#2563eb", width=1.8),
                               name=tr("lg_roll_cov")))
    fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=nominal, y1=nominal,
                  line=dict(color="#dc2626", dash="dash"))
    fig.update_layout(title=_title(tr("roll_title"), 14), height=300, margin=dict(l=10, r=10, t=60, b=10),
                      yaxis_tickformat=".0%", yaxis_range=[0, 1.02], showlegend=False,
                      separators=",." if _LANG == "pt" else ".,")
    return fig


def fig_importance(tab, top=15):
    d = tab.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=d["perm"], y=d["feature"], orientation="h", error_x=dict(array=d["perm_sd"]),
                           marker_color=["rgba(148,163,184,0.9)" if nz else "rgba(37,99,235,0.8)" for nz in d["noise"]]))
    fig.update_layout(title=_title(tr("imp_title"), 14), height=420, margin=dict(l=10, r=10, t=60, b=10),
                      xaxis_title=tr("imp_axis"), separators=",." if _LANG == "pt" else ".,", showlegend=False)
    return fig


def fig_psi(tab, top=15):
    d = tab.head(top).iloc[::-1]
    colors = ["rgba(220,38,38,0.8)" if s == "drift" else "rgba(234,179,8,0.8)" if s == "warn"
              else "rgba(22,163,74,0.7)" for s in d["status"]]
    fig = go.Figure(go.Bar(x=d["pct"], y=d["feature"], orientation="h", marker_color=colors))
    fig.update_layout(title=_title(tr("psi_title"), 14), height=420, margin=dict(l=10, r=10, t=60, b=10),
                      xaxis_tickformat=".0%", xaxis_range=[0, 1], showlegend=False,
                      separators=",." if _LANG == "pt" else ".,")
    return fig


# ----------------------------------------------------------------------------
# Textos da interface: (Português, English)
# ----------------------------------------------------------------------------
TXT.update({
    # ---- geral
    "title": ("📈 Modelos de previsão para o índice S&P 500", "📈 Modelos de previsão para o índice S&P 500"),
    "subtitle": ("Previsão probabilística por horizonte (até 2 anos) e modelo estrutural até 2040 · dados desde {d} · "
                 "conteúdo educacional, não é recomendação de investimento.",
                 "Probabilistic forecasts by horizon (up to 2 years) and a structural model to 2040 · data since {d} · "
                 "educational content, not investment advice."),
    "disclaimer": ("Modelos estatísticos baseiam-se em dados passados e não garantem resultados futuros. Nenhuma parte "
                   "deste aplicativo constitui recomendação de investimento.",
                   "Statistical models rely on past data and do not guarantee future results. Nothing in this app is "
                   "investment advice."),
    "err_data": ("Não foi possível baixar o S&P 500 ({err}). O que tentar: (1) verificar a conexão; (2) atualizar o yfinance "
                 "(`pip install -U yfinance`); (3) aguardar alguns minutos (o Yahoo pode limitar acessos) e tentar de novo.",
                 "Could not download the S&P 500 ({err}). What to try: (1) check your connection; (2) update yfinance "
                 "(`pip install -U yfinance`); (3) wait a few minutes (Yahoo may rate-limit) and try again."),
    "retry": ("🔄 Tentar novamente", "🔄 Try again"),
    "w_cache": ("Sem resposta do Yahoo ({err}). Usando os últimos dados salvos localmente ({d}): podem estar desatualizados.",
                "No response from Yahoo ({err}). Using the last locally saved data ({d}): it may be outdated."),
    "w_short_hist": ("O histórico do S&P 500 foi baixado só desde {d} (o Yahoo recusou datas anteriores): resultados menos robustos.",
                     "S&P 500 history was only downloaded since {d} (Yahoo refused earlier dates): results are less robust."),
    "spin_data": ("Baixando dados (yfinance, FRED, Shiller)...", "Downloading data (yfinance, FRED, Shiller)..."),
    "spin_m1": ("Treinando o Modelo 1 (seleção aninhada + ajuste final)...", "Training Model 1 (nested selection + final fit)..."),
    "spin_m2": ("Simulando o Modelo 2...", "Simulating Model 2..."),
    "spin_val": ("Rodando a validação purged walk-forward...", "Running purged walk-forward validation..."),
    "spin_bt": ("Treinando com dados anteriores a {y} e projetando...", "Training on data before {y} and forecasting..."),
    "spin_aud": ("Auditando as features...", "Auditing features..."),
    "tab1": ("📊 Visão Geral & Alertas", "📊 Overview & Alerts"),
    "tab2": ("🎯 Modelo 1 — Curto/Médio Prazo", "🎯 Model 1 — Short/Medium Term"),
    "tab3": ("🔭 Modelo 2 — Longo Prazo {y}", "🔭 Model 2 — Long Term {y}"),
    "tab4": ("🧪 Backtesting Histórico", "🧪 Historical Backtesting"),
    "tab5": ("✅ Validação & Diagnóstico", "✅ Validation & Diagnostics"),
    "tab6": ("📒 Track Record ao Vivo", "📒 Live Track Record"),
    "pa": ("a.a.", "p.a."),
    "years": ("anos", "years"),
    "win": ("Janela do gráfico", "Chart window"),
    "win_all": ("Tudo", "All"),
    "win_n": ("{n} anos", "{n} years"),
    "logscale": ("Escala logarítmica", "Log scale"),
    "n_days": ("{n} pregões", "{n} trading days"),
    "not_yet": ("ainda não", "not yet"),
    "axis_level": ("Nível do índice ({ccy})", "Index level ({ccy})"),
    "band90": ("Faixa 90%: {lo} – {hi}", "90% range: {lo} – {hi}"),
    # ---- sidebar
    "sb_controls": ("⚙️ Controles", "⚙️ Controls"),
    "sb_ccy": ("Moeda de visualização", "Display currency"),
    "sb_ccy_help": ("Cada ponto histórico é multiplicado pelo câmbio exato do mesmo dia.",
                    "Each historical point is multiplied by that same day's exact FX rate."),
    "sb_fxmode": ("Projeção em moeda estrangeira", "Forecasts in foreign currency"),
    "sb_fxmode_help": ("Recomendado: modelar em USD (onde estão VIX, valuation e macro) e converter pela taxa atual. "
                       "A alternativa embute a deriva cambial histórica nas projeções.",
                       "Recommended: model in USD (where VIX, valuation and macro live) and convert at today's rate. "
                       "The alternative embeds the historical FX drift in the forecasts."),
    "fx_spot": ("Modelar em USD e converter pela taxa atual", "Model in USD and convert at the current rate"),
    "fx_series": ("Modelar a série já convertida (embute deriva cambial)", "Model the already-converted series (embeds FX drift)"),
    "sb_m1": ("Modelo 1 — curto/médio prazo", "Model 1 — short/medium term"),
    "sb_start1": ("Histórico de treino desde", "Training history since"),
    "sb_start1_help": ("1990 = era do VIX; inclui 1990, 2000, 2008 e 2020.", "1990 = VIX era; includes 1990, 2000, 2008 and 2020."),
    "sb_quality": ("Qualidade do ajuste", "Fit quality"),
    "sb_quality_help": ("Mais qualidade = mais árvores e mais origens de treino (mais lento).",
                        "Higher quality = more trees and training origins (slower)."),
    "q_fast": ("Rápido", "Fast"), "q_standard": ("Padrão", "Standard"), "q_full": ("Completo", "Full"),
    "sb_refit": ("Re-treinar a cada N pregões", "Re-train every N trading days"),
    "sb_refit_help": ("Os modelos pesados só são re-treinados a cada N pregões; entre re-treinos, só as features são atualizadas.",
                      "Heavy models are only re-trained every N trading days; in between, only features are updated."),
    "sb_vix": ("Limiar de alta volatilidade (VIX)", "High-volatility threshold (VIX)"),
    "sb_m2": ("Modelo 2 — longo prazo", "Model 2 — long term"),
    "sb_index": ("Índice base", "Base index"),
    "idx_price": ("Preço (^GSPC)", "Price (^GSPC)"), "idx_tr": ("Retorno total (^SP500TR)", "Total return (^SP500TR)"),
    "sb_real": ("Valores reais (poder de compra de hoje)", "Real values (today's purchasing power)"),
    "sb_real_help": ("Deflaciona as projeções pela inflação esperada abaixo.", "Deflates projections by the expected inflation below."),
    "sb_infl": ("Inflação esperada (% a.a.)", "Expected inflation (% p.a.)"),
    "sb_infl_help": ("Soma-se ao crescimento real do lucro para obter a deriva nominal.",
                     "Added to real earnings growth to obtain the nominal drift."),
    "sb_fund_start": ("Amostra dos fundamentos desde", "Fundamentals sample since"),
    "sb_fund_start_help": ("Período usado para estimar reversão do múltiplo e crescimento do lucro. O prior é sempre o histórico completo (1871+).",
                           "Period used to estimate multiple reversion and earnings growth. The prior is always the full history (1871+)."),
    "sb_prior": ("Peso do prior histórico (anos equivalentes)", "Weight of the historical prior (equivalent years)"),
    "sb_prior_help": ("Quanto o histórico longo encolhe a estimativa da amostra escolhida (encolhimento bayesiano).",
                      "How strongly the long history shrinks the chosen-sample estimate (Bayesian shrinkage)."),
    "sb_rev": ("Intensidade da reversão do múltiplo (%)", "Multiple-reversion intensity (%)"),
    "sb_rev_help": ("100% = estimativa bayesiana; 0% = o múltiplo não reverte (teste de sensibilidade).",
                    "100% = Bayesian estimate; 0% = the multiple does not revert (sensitivity test)."),
    "sb_adj": ("Ajuste manual ao crescimento do lucro (p.p. a.a.)", "Manual adjustment to earnings growth (pp p.a.)"),
    "sb_adj_help": ("Teste de sensibilidade; 0 = sem ajuste.", "Sensitivity test; 0 = no adjustment."),
    "sb_alert": ("Alerta de valuation", "Valuation alert"),
    "sb_zthr": ("Limiar do Z-Score (± desvios)", "Z-Score threshold (± std. dev.)"),
    "sb_autolog": ("Registrar previsões (track record)", "Log forecasts (track record)"),
    "sb_autolog_help": ("Grava cada previsão em forecast_log.csv para avaliação futura (apenas ao modelar em USD).",
                        "Saves each forecast to forecast_log.csv for later evaluation (only when modelling in USD)."),
    "sb_adv": ("Dados avançados", "Advanced data"),
    "sb_shiller": ("Arquivo/URL do Shiller (opcional)", "Shiller file/URL (optional)"),
    "sb_shiller_help": ("Caminho local ou URL de ie_data.xls (ou .csv) caso o download automático falhe. Limpe o cache após mudar.",
                        "Local path or URL of ie_data.xls (or .csv) in case the automatic download fails. Refresh data after changing."),
    "sb_refresh": ("🔄 Atualizar dados e modelos", "🔄 Refresh data and models"),
    "sb_warn": ("⚠️ {n} aviso(s) de dados", "⚠️ {n} data warning(s)"),
    "w_tr": ('^SP500TR indisponível agora ({err}). Retorno total APROXIMADO (+1,8% a.a. de dividendos).', '^SP500TR unavailable right now ({err}). Total return APPROXIMATED (+1.8% p.a. dividends).'),
    "w_vix": ('^VIX indisponível agora ({err}). Usando a volatilidade realizada de 21 dias como proxy.', '^VIX unavailable right now ({err}). Using 21-day realized volatility as a proxy.'),
    "w_fx": ('Câmbio {ccy} indisponível agora ({err}); moeda fora do seletor até a próxima atualização.', 'FX {ccy} unavailable right now ({err}); currency left out of the selector until the next refresh.'),
    "w_fred": ('FRED: séries indisponíveis ({sid}). Motivo: {err}. As features que dependem delas ficam ausentes.', 'FRED: series unavailable ({sid}). Reason: {err}. Features that depend on them are missing.'),
    "w_hy_short": ("O histórico do spread high yield (ICE BofA) no FRED é curto; usando o spread Baa−Treasury 10a (desde 1986) como proxy de crédito.",
                   "The FRED high-yield spread (ICE BofA) history is short; using the Baa−10y Treasury spread (since 1986) as the credit proxy."),
    "w_cached": ("Atualização falhou para {what} ({err}); usando o último dado salvo em {d}. O app tenta de novo na próxima abertura.",
                 "Refresh failed for {what} ({err}); using the last data saved on {d}. The app retries on the next load."),
    "sb_retry_data": ("🔄 Tentar baixar de novo os dados faltantes", "🔄 Retry downloading the missing data"),
    "w_dxy": ('DXY (DX-Y.NYB) indisponível agora; feature do dólar ausente.', 'DXY (DX-Y.NYB) unavailable right now; dollar feature missing.'),
    "w_shiller": ("Dados de Shiller indisponíveis ({err}). Para ler o arquivo .xls instale o leitor (`pip install xlrd`) ou informe um arquivo/URL em 'Dados avançados'. Enquanto isso o Modelo 2 usa um proxy de valuation baseado só em preços.", "Shiller data unavailable ({err}). To read the .xls file install the reader (`pip install xlrd`) or provide a file/URL under 'Advanced data'. Meanwhile Model 2 uses a price-only valuation proxy."),
    "w_shiller_stale": ("O arquivo de Shiller termina em {d}: o CAPE é atualizado pelo preço, mas o denominador de lucros está defasado.",
                        "The Shiller file ends in {d}: CAPE is updated with price, but the earnings denominator is stale."),
    # ---- visão geral
    "kpi_day": ("no dia", "today"), "kpi_ytd": ("No ano (YTD)", "Year to date"), "kpi_12m": ("12 meses", "12 months"),
    "kpi_vix_help": ("Limiar de alta volatilidade: {t}", "High-volatility threshold: {t}"),
    "kpi_rec": ("Prob. de recessão (12m)", "Recession prob. (12m)"),
    "kpi_rec_help": ("Probit sobre a curva de juros (10a−3m), reestimado anualmente só com rótulos já resolvidos.",
                     "Probit on the yield curve (10y−3m), re-estimated yearly using only resolved labels."),
    "kpi_last": ("Último pregão", "Last trading day"),
    "regime_now": ("**Regime de mercado atual:** {r}", "**Current market regime:** {r}"),
    "reg_calm": ("Expansão calma (baixa volatilidade, sem sinal de recessão)", "Calm expansion (low volatility, no recession signal)"),
    "reg_vol": ("Expansão com volatilidade elevada", "Expansion with elevated volatility"),
    "reg_late": ("Fim de ciclo (probabilidade de recessão elevada)", "Late cycle (elevated recession probability)"),
    "reg_stress": ("Estresse (alta volatilidade e abaixo da MME200)", "Stress (high volatility and below the 200-day EMA)"),
    "val_alert_title": ("Alerta de valuation (sem look-ahead)", "Valuation alert (no look-ahead)"),
    "gauge_trend": ("Z vs. tendência (janela expansiva)", "Z vs. trend (expanding window)"),
    "gauge_cape": ("Z do CAPE (histórico até a data)", "CAPE Z (history to date)"),
    "gauge_comb": ("Z combinado", "Combined Z"),
    "cape_na": ("CAPE indisponível: indicador omitido da média.", "CAPE unavailable: indicator omitted from the average."),
    "status_over": ("🔴 **Sobrecomprado / Risco Elevado** — Z = {z}", "🔴 **Overbought / Elevated Risk** — Z = {z}"),
    "status_under": ("🟢 **Subavaliado / Oportunidade** — Z = {z}", "🟢 **Undervalued / Opportunity** — Z = {z}"),
    "status_fair": ("🟡 **Mercado Justo** — Z = {z}", "🟡 **Fair Market** — Z = {z}"),
    "trend_caption": ("Tendência log-linear em janela expansiva (ajustada só com dados até cada data): {g} a.a.; σ dos resíduos "
                      "(log): {sd}. Resíduos são autocorrelacionados: o Z pode permanecer extremo por anos — é um termômetro, não um gatilho.",
                      "Log-linear trend on an expanding window (fit only with data up to each date): {g} p.a.; residual σ (log): "
                      "{sd}. Residuals are autocorrelated: Z can stay extreme for years — a thermometer, not a trigger."),
    "ov_chart_title": ("S&P 500 em {ccy} — preço, médias e Bandas de Bollinger", "S&P 500 in {ccy} — price, averages and Bollinger Bands"),
    "trend_title": ("Preço vs. tendência estrutural (janela expansiva) e bandas de Z-Score",
                    "Price vs. structural trend (expanding window) and Z-Score bands"),
    "lg_boll": ("Bandas de Bollinger (20, 2σ)", "Bollinger Bands (20, 2σ)"),
    "lg_ema50": ("MME 50", "EMA 50"), "lg_ema200": ("MME 200", "EMA 200"),
    "lg_fairzone": ("±{t}σ (zona de 'Mercado Justo')", "±{t}σ ('Fair Market' zone)"),
    "lg_trend_pit": ("Tendência (só dados passados)", "Trend (past data only)"),
    "fwd_title": ("Retornos futuros históricos por faixa de Z-Score (point-in-time)", "Historical forward returns by Z-Score bucket (point-in-time)"),
    "fwd_caption": ("Z calculado só com dados até a origem. Janelas sobrepostas: as observações não são independentes — evidência descritiva, não prova.",
                    "Z computed only with data up to the origin. Overlapping windows: observations are not independent — descriptive evidence, not proof."),
    "col_bucket": ("Faixa de Z", "Z bucket"), "col_yrs": ("Horizonte (anos)", "Horizon (years)"),
    "col_med_ann": ("Retorno mediano a.a.", "Median return p.a."), "col_pneg": ("% negativos", "% negative"),
    "bucket_under": ("Subavaliado (Z ≤ −limiar)", "Undervalued (Z ≤ −threshold)"), "bucket_fair": ("Justo", "Fair"),
    "bucket_over": ("Sobrecomprado (Z ≥ limiar)", "Overbought (Z ≥ threshold)"),
    "sum_title": ("Resumo dos modelos", "Models summary"),
    "sum_m1": ("Modelo 1 · mediana em {d}", "Model 1 · median on {d}"),
    "sum_m2": ("Modelo 2 · base (P50) em {y}", "Model 2 · base (P50) in {y}"),
    "sum_m2_cap": ("P10: {lo} · P90: {hi}", "P10: {lo} · P90: {hi}"),
    "sum_skill": ("Horizontes com habilidade comprovada", "Horizons with proven skill"),
    "sum_skill_help": ("CRPSS > 0 e Diebold-Mariano p < 0,10 em validação purged walk-forward (com amostra efetiva suficiente).",
                       "CRPSS > 0 and Diebold-Mariano p < 0.10 in purged walk-forward validation (with enough effective sample)."),
    "sum_skill_na": ("Execute a validação na aba 'Validação & Diagnóstico'.", "Run the validation in the 'Validation & Diagnostics' tab."),
    "sum_drift": ("Monitor de drift", "Drift monitor"), "drift_ok": ("✅ sem alertas", "✅ no alerts"),
    "drift_n": ("⚠️ {n} feature(s) em drift", "⚠️ {n} feature(s) drifting"),
    "fx_series_note": ("ℹ️ Em {c}, o modelo é ajustado sobre a série já convertida: as projeções embutem a deriva cambial histórica.",
                       "ℹ️ In {c}, the model is fitted on the already-converted series: forecasts embed the historical FX drift."),
    "fx_spot_note": ("ℹ️ O modelo roda em USD e as projeções são convertidas para {c} pela taxa atual ({r}); o histórico usa o câmbio exato de cada dia.",
                     "ℹ️ The model runs in USD and forecasts are converted to {c} at today's rate ({r}); history uses each day's exact FX."),
    # ---- modelo 1
    "m1_title": ("Modelo 1 — distribuição do retorno por horizonte (previsão direta de quantis)",
                 "Model 1 — return distribution by horizon (direct quantile forecasts)"),
    "m1_unavailable": ("Modelo 1 indisponível: {err}", "Model 1 unavailable: {err}"),
    "m1_short": ("histórico insuficiente para a configuração escolhida", "insufficient history for the chosen configuration"),
    "m1_vol_garch": ("Vol. GJR-GARCH-t (1 dia, anual.)", "GJR-GARCH-t vol (1 day, annualized)"),
    "m1_vol_garch_help": ("Volatilidade prevista para o próximo pregão.", "Forecast volatility for the next trading day."),
    "m1_vol_stack": ("Vol. prevista 1 mês (pilha)", "Forecast vol, 1 month (stack)"),
    "m1_vol_stack_help": ("Combinação NNLS de GARCH, HAR-RV e VIX para os próximos 21 pregões.",
                          "NNLS combination of GARCH, HAR-RV and VIX for the next 21 trading days."),
    "m1_vrp": ("Prêmio de risco de variância", "Variance risk premium"),
    "m1_vrp_help": ("VIX menos a volatilidade realizada de 21 dias (pontos de volatilidade).", "VIX minus 21-day realized volatility (vol points)."),
    "m1_age": ("Idade do modelo", "Model age"),
    "m1_age_v": ("{a} pregão(ões) (re-treino em {nxt})", "{a} day(s) (re-train in {nxt})"),
    "m1_age_help": ("Último treino com dados até {d}.", "Last trained with data up to {d}."),
    "col_horizon": ("Horizonte", "Horizon"), "col_median": ("Mediana", "Median"),
    "col_band90": ("Faixa 90% (conformal)", "90% range (conformal)"), "col_band50": ("Faixa 50% (conformal)", "50% range (conformal)"),
    "col_ploss": ("Prob. de perda", "Prob. of loss"), "col_lambda": ("λ (peso do modelo)", "λ (model weight)"),
    "col_skill": ("Habilidade comprovada?", "Skill proven?"),
    "hz_21": ("1 mês", "1 month"), "hz_63": ("3 meses", "3 months"), "hz_126": ("6 meses", "6 months"),
    "hz_252": ("1 ano", "1 year"), "hz_504": ("2 anos", "2 years"),
    "verdict_yes": ("Sim", "Yes"), "verdict_no": ("Não", "No"),
    "verdict_inconclusive": ("Inconclusivo (poucas janelas)", "Inconclusive (few windows)"),
    "m1_table_note": ("λ = peso do ensemble contra o benchmark histórico (0 = o modelo recua ao benchmark por falta de ganho comprovado). "
                      "Faixas calibradas por conformal adaptativo.",
                      "λ = ensemble weight against the historical benchmark (0 = model falls back to the benchmark for lack of proven gain). "
                      "Ranges calibrated by adaptive conformal prediction."),
    "m1_show_bench": ("Mostrar faixa do benchmark histórico", "Show historical benchmark range"),
    "m1_chart_title": ("Modelo 1 — leque de probabilidade a 2 anos (S&P 500 em {ccy})", "Model 1 — 2-year probability fan (S&P 500 in {ccy})"),
    "lg_hist": ("Histórico real", "Actual history"), "lg_median": ("Mediana prevista", "Forecast median"),
    "lg_band90": ("Faixa 90%", "90% range"), "lg_band50": ("Faixa 50%", "50% range"), "lg_p5": ("P5", "P5"),
    "lg_bench": ("Benchmark histórico (P5–P95)", "Historical benchmark (P5–P95)"),
    "m1_ens_title": ("Composição do ensemble (escolhida em validação aninhada)", "Ensemble composition (chosen in nested validation)"),
    "m_garch": ("GJR-GARCH-t", "GJR-GARCH-t"), "m_har": ("HAR-RV(+VIX)", "HAR-RV(+VIX)"), "m_ridge": ("Ridge", "Ridge"),
    "m_qrf": ("Quantile Reg. Forest", "Quantile Reg. Forest"), "m_gbm": ("GBM quantílico", "Quantile GBM"),
    "col_nval": ("Origens (validação)", "Origins (validation)"), "col_neff": ("n efetivo", "effective n"),
    "col_gain": ("Ganho CRPS vs bench.", "CRPS gain vs bench."),
    "m1_ens_note": ("Pesos ∝ exp(−80·CRPS relativo). λ pela regra de 1 erro-padrão: sem ganho acima do ruído, λ = 0. "
                    "α = nível de erro efetivo do ACI (nominal: 10% e 50%), amortecido pelo n efetivo.",
                    "Weights ∝ exp(−80·relative CRPS). λ via the 1-standard-error rule: with no gain above noise, λ = 0. "
                    "α = effective ACI error level (nominal: 10% and 50%), damped by effective n."),
    "m1_method_title": ("Metodologia do Modelo 1", "Model 1 methodology"),
    "m1_method": (
        "- **Previsão direta por horizonte:** modelos separados para o retorno acumulado em 21, 63, 126, 252 e 504 pregões, "
        "sem recursão (sem acúmulo de erro). O leque interpola os horizontes.\n"
        "- **Distribuição inteira:** o alvo são os quantis (P1–P99). Membros: benchmark histórico, GJR-GARCH-t, HAR-RV(+VIX), Ridge, "
        "Quantile Regression Forest e GBM quantílico (pinball loss).\n"
        "- **Volatilidade como alvo principal:** os retornos são padronizados por uma previsão de variância (pilha NNLS de GARCH, HAR e "
        "VIX/prêmio de risco de variância); os modelos condicionais explicam só o que sobra (localização e forma).\n"
        "- **Dados com sinal:** valuation (CAPE, earnings yield − juro 10a, dividend yield, S&P/PIB), macro e crédito (curva, spread high yield, "
        "Fed funds, desemprego, inflação, dólar) com defasagem de publicação, e probabilidade de recessão (probit).\n"
        "- **Ensemble com encolhimento:** pesos e λ escolhidos em validação aninhada; sem ganho comprovado o modelo recua ao benchmark.\n"
        "- **Conformal adaptativo (ACI):** os níveis das bandas se ajustam online à cobertura realizada, amortecidos pelo nº de observações independentes.",
        "- **Direct multi-horizon forecasts:** separate models for the cumulative return over 21, 63, 126, 252 and 504 trading days, "
        "with no recursion (no error accumulation). The fan interpolates the horizons.\n"
        "- **Whole distribution:** the target is the quantiles (P1–P99). Members: historical benchmark, GJR-GARCH-t, HAR-RV(+VIX), Ridge, "
        "Quantile Regression Forest and quantile GBM (pinball loss).\n"
        "- **Volatility as the main target:** returns are standardized by a variance forecast (NNLS stack of GARCH, HAR and "
        "VIX/variance risk premium); conditional models explain only what is left (location and shape).\n"
        "- **Informative data:** valuation (CAPE, earnings yield − 10y yield, dividend yield, S&P/GDP), macro and credit (curve, high-yield spread, "
        "Fed funds, unemployment, inflation, dollar) with publication lags, and recession probability (probit).\n"
        "- **Ensemble with shrinkage:** weights and λ chosen in nested validation; without proven gain the model falls back to the benchmark.\n"
        "- **Adaptive conformal (ACI):** band levels adjust online to realized coverage, damped by the number of independent observations."),
    # ---- modelo 2
    "m2_title": ("Modelo 2 — estrutural (até {y})", "Model 2 — structural (to {y})"),
    "m2_x0_clipped": ("⚠️ O CAPE inicial calculado ({raw}) está fora do intervalo histórico; usei {used} (o limite). Isso costuma indicar dado de "
                      "valuation defasado ou fora de escala: confira o arquivo de Shiller em 'Dados avançados'.",
                      "⚠️ The computed starting CAPE ({raw}) is outside the historical range; I used {used} (the limit). This usually means stale or "
                      "mis-scaled valuation data: check the Shiller file under 'Advanced data'."),
    "m2_diag": ("Diagnóstico: CAPE atual {c} · CAPE de longo prazo {l} · meia-vida da reversão {h} anos · crescimento real do lucro {g} a.a. · payout {p}.",
                "Diagnostics: current CAPE {c} · long-run CAPE {l} · reversion half-life {h} years · real earnings growth {g} p.a. · payout {p}."),
    "m2_implausible": ("⚠️ Atenção: nesta configuração até o cenário otimista (P90) termina abaixo do nível atual em {y}. Isso equivale a mais de 90% de chance "
                       "de perda e normalmente aponta para premissas extremas (reversão total do múltiplo) ou dados de valuation inconsistentes — não é uma "
                       "previsão de mercado. Revise a 'Intensidade da reversão do múltiplo', a amostra dos fundamentos e o CAPE atual ({c}).",
                       "⚠️ Warning: in this setup even the optimistic scenario (P90) ends below the current level in {y}. That is a >90% chance of loss and "
                       "usually points to extreme assumptions (full multiple reversion) or inconsistent valuation data — it is not a market forecast. Review the "
                       "'Multiple-reversion intensity', the fundamentals sample and the current CAPE ({c})."),
    "m2_pessimistic": ("A mediana implica retorno de {g} a.a. nesta configuração: é uma hipótese bastante pessimista, dominada pela reversão do CAPE. "
                       "Use a intensidade da reversão para testar a sensibilidade.",
                       "The median implies {g} p.a. in this setup: a rather pessimistic view, dominated by CAPE reversion. Use the reversion intensity to "
                       "test sensitivity."),
    "m2_unavailable": ("Modelo 2 indisponível: {err}", "Model 2 unavailable: {err}"),
    "sc_opt": ("Cenário Otimista (P90)", "Optimistic Scenario (P90)"), "sc_base": ("Cenário Base (P50)", "Base Scenario (P50)"),
    "sc_pess": ("Cenário Pessimista (P10)", "Pessimistic Scenario (P10)"),
    "real_tag": ("reais", "real"), "real_yes": ("reais (poder de compra de hoje)", "real (today's purchasing power)"),
    "real_no": ("nominais", "nominal"),
    "m2_cape_now": ("CAPE atual", "Current CAPE"), "m2_cape_now_help": ("Preço de hoje / média de 10 anos dos lucros reais.", "Today's price / 10-year average of real earnings."),
    "m2_cape_bar": ("CAPE de longo prazo (posterior)", "Long-run CAPE (posterior)"),
    "m2_cape_bar_help": ("Mediana do posterior; intervalo 5–95%: {lo} a {hi}.", "Posterior median; 5–95% range: {lo} to {hi}."),
    "m2_ploss": ("Prob. de perda real", "Prob. of real loss"),
    "m2_ploss_help": ("Probabilidade de o retorno acumulado, descontada a inflação esperada, ser negativo até a data final.",
                      "Probability that the cumulative return, net of expected inflation, is negative by the end date."),
    "m2_vol": ("Vol. de longo prazo (GARCH)", "Long-run vol (GARCH)"), "m2_vol_help": ("Volatilidade incondicional anualizada do histórico completo.", "Unconditional annualized volatility of the full history."),
    "m2_caption": ("{lab} · {n} simulações · reversão do múltiplo a {rev}% da estimativa bayesiana · valores {real}.",
                   "{lab} · {n} simulations · multiple reversion at {rev}% of the Bayesian estimate · values {real}."),
    "m2_show_stress": ("Mostrar cenários de estresse históricos (2000, 2008, 2020)", "Show historical stress scenarios (2000, 2008, 2020)"),
    "stress_dotcom": ("Estresse: bolha pontocom (2000)", "Stress: dot-com bust (2000)"),
    "stress_gfc": ("Estresse: crise financeira (2008)", "Stress: financial crisis (2008)"),
    "stress_covid": ("Estresse: Covid (2020)", "Stress: Covid (2020)"),
    "m2_chart_title": ("Modelo 2 — cone de probabilidade até {year} ({label}, {ccy})", "Model 2 — probability cone to {year} ({label}, {ccy})"),
    "lg_pess": ("Pessimista (P10)", "Pessimistic (P10)"), "lg_base": ("Base (P50, mediana)", "Base (P50, median)"),
    "lg_opt": ("Otimista (P90)", "Optimistic (P90)"),
    "m2_dec_title": ("Decomposição do retorno esperado (próximos 10 anos, parâmetros médios)", "Expected-return decomposition (next 10 years, mean parameters)"),
    "dec_earn": ("Crescimento do lucro (nominal)", "Earnings growth (nominal)"),
    "dec_earn_help": ("Crescimento real do lucro de 10 anos + inflação esperada + ajuste manual.", "Real 10-year earnings growth + expected inflation + manual adjustment."),
    "dec_mult": ("Variação do múltiplo", "Multiple change"),
    "dec_mult_help": ("Reversão do CAPE rumo à média de longo prazo (CAPE final médio: {c}).", "CAPE reversion toward its long-run mean (average ending CAPE: {c})."),
    "dec_div": ("Dividendos", "Dividends"), "dec_div_help": ("Entram apenas no índice de retorno total (yield = payout / CAPE).", "Only in the total-return index (yield = payout / CAPE)."),
    "dec_total": ("Retorno esperado (log, a.a.)", "Expected return (log, p.a.)"), "dec_total_help": ("Soma dos componentes acima.", "Sum of the components above."),
    "m2_tab_title": ("Trajetória mensal (percentis)", "Monthly path (percentiles)"), "col_date": ("Data", "Date"),
    "m2_post_title": ("Parâmetros bayesianos (posterior)", "Bayesian parameters (posterior)"),
    "col_param": ("Parâmetro", "Parameter"), "col_value": ("Valor [5% ; 95%]", "Value [5% ; 95%]"),
    "p_g": ("Crescimento real do lucro (a.a.)", "Real earnings growth (p.a.)"),
    "p_phi": ("Persistência anual do log-CAPE (φ)", "Annual persistence of log-CAPE (φ)"), "p_ols": ("OLS da amostra", "sample OLS"),
    "p_half": ("Meia-vida da reversão", "Reversion half-life"), "p_cape": ("CAPE de longo prazo", "Long-run CAPE"),
    "p_rho": ("Payout (dividendo / lucro 10a)", "Payout (dividend / 10y earnings)"), "p_n": ("Anos usados (amostra / total)", "Years used (sample / total)"),
    "m2_post_note": ("Posterior normal conjugada: prior = histórico completo; verossimilhança = amostra escolhida (com correção de viés de Kendall). "
                     "Cada simulação sorteia seus próprios parâmetros, então a incerteza do retorno esperado alarga o cone.",
                     "Conjugate normal posterior: prior = full history; likelihood = chosen sample (with Kendall bias correction). "
                     "Each simulation draws its own parameters, so uncertainty about expected return widens the cone."),
    "m2_evid_title": ("Evidência: valuation prevê retorno? (fora da amostra)", "Evidence: does valuation predict returns? (out-of-sample)"),
    "col_years": ("Horizonte (anos)", "Horizon (years)"), "col_pcw": ("p (Clark-West)", "p (Clark-West)"), "col_slope": ("Inclinação (log-CAPE)", "Slope (log-CAPE)"),
    "m2_evid_note": ("Regressão do retorno real futuro sobre o log-CAPE com janela expansiva (dados anuais); R² fora da amostra de Campbell-Thompson contra a média histórica.",
                     "Regression of future real return on log-CAPE with an expanding window (annual data); Campbell-Thompson out-of-sample R² against the historical mean."),
    "m2_stress_title": ("Cenários de estresse (reaplicados ao nível atual)", "Stress scenarios (re-applied to the current level)"),
    "col_event": ("Evento", "Event"), "col_dd": ("Queda máxima", "Max drawdown"), "col_trough": ("Até o fundo", "To trough"), "col_recover": ("Até recuperar", "To recover"),
    "m2_stress_note": ("Trajetórias reais pós-pico (2000-03, 2007-10, 2020-02) aplicadas ao nível de hoje; não são previsões, são testes de resistência.",
                       "Actual post-peak paths (2000-03, 2007-10, 2020-02) applied to today's level; they are stress tests, not forecasts."),
    "m2_method_title": ("Metodologia do Modelo 2", "Model 2 methodology"),
    "m2_method": (
        "- **Decomposição:** retorno do preço = crescimento do lucro de 10 anos + variação do múltiplo (CAPE). Dividendos entram no índice de "
        "retorno total (yield = payout / CAPE).\n"
        "- **Reversão do múltiplo:** o log-CAPE segue um AR(1); a deriva diária é (g + inflação + κ·(x̄ − x))/252 e x evolui com o retorno "
        "simulado (trajetórias que caem ficam mais baratas e passam a ter retorno esperado maior).\n"
        "- **Parâmetros bayesianos:** prior no histórico completo (1871+), verossimilhança na amostra escolhida; cada simulação sorteia (g, κ, x̄).\n"
        "- **Choques realistas:** filtered historical simulation — resíduos padronizados do GJR-GARCH-t reamostrados em blocos de 21 dias "
        "(caudas pesadas, agrupamento de volatilidade e efeito alavancagem).\n"
        "- **Estresse:** trajetórias de 2000, 2008 e 2020 reaplicadas ao nível atual.\n"
        "- **Limitações:** o cone não inclui incerteza dos parâmetros do GARCH nem mudanças estruturais de regime; a reversão do múltiplo é a "
        "hipótese mais influente — use o controle de intensidade para testar.",
        "- **Decomposition:** price return = 10-year earnings growth + multiple change (CAPE). Dividends enter the total-return index "
        "(yield = payout / CAPE).\n"
        "- **Multiple reversion:** log-CAPE follows an AR(1); the daily drift is (g + inflation + κ·(x̄ − x))/252 and x evolves with the simulated "
        "return (paths that fall become cheaper and then have higher expected return).\n"
        "- **Bayesian parameters:** prior on the full history (1871+), likelihood on the chosen sample; each simulation draws (g, κ, x̄).\n"
        "- **Realistic shocks:** filtered historical simulation — standardized GJR-GARCH-t residuals resampled in 21-day blocks "
        "(fat tails, volatility clustering and the leverage effect).\n"
        "- **Stress:** 2000, 2008 and 2020 paths re-applied to the current level.\n"
        "- **Limitations:** the cone ignores GARCH parameter uncertainty and structural regime changes; multiple reversion is the most "
        "influential assumption — use the intensity control to test it."),
    # ---- backtesting
    "bt_title": ("Backtesting histórico (walk-forward)", "Historical backtesting (walk-forward)"),
    "bt_intro": ("Escolha um ano de origem: o modelo (incluindo a seleção aninhada) é treinado **apenas com dados anteriores** ao primeiro pregão "
                 "daquele ano, prevê a distribuição e o resultado é confrontado com o que realmente aconteceu.",
                 "Pick an origin year: the model (including nested selection) is trained **only on data before** that year's first trading day, "
                 "forecasts the distribution, and the result is compared with what actually happened."),
    "bt_noyears": ("Histórico insuficiente para backtests nesta configuração.", "Insufficient history for backtests in this configuration."),
    "bt_year": ("Ano de origem", "Origin year"), "bt_run": ("▶️ Executar backtest", "▶️ Run backtest"),
    "bt_err": ("Backtest indisponível: {err}", "Backtest unavailable: {err}"),
    "bt_partial": ("A janela de 2 anos ainda não terminou: a comparação cobre só o período já ocorrido.", "The 2-year window has not ended: the comparison covers only the period elapsed."),
    "bt_chart_title": ("Backtest — origem em {d} (treino só com dados anteriores)", "Backtest — origin on {d} (trained only on earlier data)"),
    "lg_train": ("Histórico (treino)", "History (training)"), "lg_forecast": ("Previsão do modelo", "Model forecast"),
    "lg_actual": ("O que de fato aconteceu", "What actually happened"), "lg_origin": ("Origem", "Origin"),
    "col_pred_med": ("Mediana prevista", "Predicted median"), "col_real": ("Realizado", "Realized"), "col_in90": ("Dentro da faixa 90%?", "Inside 90% range?"),
    "col_vs_bench": ("CRPS vs bench.", "CRPS vs bench."),
    "bt_note": ("Retornos em log acumulados até cada horizonte. PIT ≈ posição do realizado na distribuição prevista (0,5 = mediana). "
                "'CRPS vs bench.' positivo = o modelo foi melhor que o benchmark histórico naquela janela.",
                "Cumulative log returns to each horizon. PIT ≈ position of the realized value in the forecast distribution (0.5 = median). "
                "Positive 'CRPS vs bench.' = the model beat the historical benchmark in that window."),
    "wf_title": ("Walk-forward em todos os anos de origem", "Walk-forward across all origin years"),
    "wf_caption": ("Reexecuta o backtest para cada ano com janela completa de 2 anos. Pode levar vários minutos na primeira execução.",
                   "Re-runs the backtest for every year with a full 2-year window. It may take several minutes the first time."),
    "wf_run": ("▶️ Rodar walk-forward completo", "▶️ Run full walk-forward"),
    "wf_cov": ("Cobertura da faixa 90% ({t})", "90% range coverage ({t})"), "wf_skill": ("Ganho de CRPS vs bench. ({t})", "CRPS gain vs bench. ({t})"),
    "wf_n": ("Janelas (1 ano)", "Windows (1 year)"), "wf_neff": ("Janelas independentes (2 anos)", "Independent windows (2 years)"),
    "wf_neff_help": ("Janelas de 2 anos com origem anual se sobrepõem pela metade: o n efetivo é ~n/2.", "2-year windows with yearly origins overlap by half: effective n ≈ n/2."),
    "col_year": ("Ano", "Year"),
    "wf_note": ("Janelas sobrepostas: as linhas NÃO são testes independentes. Use a aba de validação para testes com HAC (Newey-West).",
                "Overlapping windows: rows are NOT independent tests. Use the validation tab for HAC (Newey-West) tests."),
    # ---- validação
    "val_title": ("Validação rigorosa (purged walk-forward) contra o benchmark", "Rigorous validation (purged walk-forward) against the benchmark"),
    "val_intro": ("O benchmark é a distribuição histórica (passeio aleatório com deriva). Cada bloco de teste é previsto por modelos treinados só com dados "
                  "anteriores (alvos sobrepostos são purgados) e a seleção de pesos/λ ocorre dentro do treino (validação aninhada).",
                  "The benchmark is the historical distribution (random walk with drift). Each test block is forecast by models trained only on earlier "
                  "data (overlapping targets are purged) and weights/λ are selected inside the training set (nested validation)."),
    "val_run": ("▶️ Executar validação rigorosa", "▶️ Run rigorous validation"),
    "val_hint": ("Clique para rodar ({t} blocos de teste, refit com seleção aninhada em cada um). Leva de dezenas de segundos a alguns minutos.",
                 "Click to run ({t} test blocks, a re-fit with nested selection in each). Takes tens of seconds to a few minutes."),
    "val_err": ("Validação indisponível: {err}", "Validation unavailable: {err}"),
    "col_mse": ("MSE (modelo)", "MSE (model)"), "col_mse_b": ("MSE (benchmark)", "MSE (benchmark)"),
    "col_cov90": ("Cobertura 90% (bruta)", "90% coverage (raw)"), "col_cov90aci": ("Cobertura 90% (ACI)", "90% coverage (ACI)"),
    "col_cov50": ("Cobertura 50%", "50% coverage"), "col_pitks": ("PIT uniforme (p)", "PIT uniform (p)"), "col_verdict": ("Habilidade comprovada?", "Skill proven?"),
    "val_note": ("R²_OS e MSE (erro quadrático médio do retorno em log, média implícita da distribuição): Campbell-Thompson vs. média histórica. CRPSS: ganho de CRPS vs. benchmark. p(CW): Clark-West; p(DM): Diebold-Mariano sobre o CRPS — ambos com "
                 "HAC de Newey-West para janelas sobrepostas. n efetivo ≈ nº de janelas independentes (abaixo de 30 o veredito é inconclusivo).",
                 "R²_OS and MSE (mean squared error of the log return, implied distribution mean): Campbell-Thompson vs. historical mean. CRPSS: CRPS gain vs. benchmark. p(CW): Clark-West; p(DM): Diebold-Mariano on CRPS — both with "
                 "Newey-West HAC for overlapping windows. Effective n ≈ number of independent windows (below 30 the verdict is inconclusive)."),
    "skill_title": ("Ganho de CRPS sobre o benchmark (verde = comprovado)", "CRPS gain over the benchmark (green = proven)"),
    "val_h": ("Horizonte", "Horizon"),
    "pit_title": ("Histograma PIT — {h} pregões (ideal: barras ≈ 10%)", "PIT histogram — {h} trading days (ideal: bars ≈ 10%)"),
    "roll_title": ("Cobertura móvel da faixa 90% (nominal 90%)", "Rolling coverage of the 90% range (nominal 90%)"), "lg_roll_cov": ("Cobertura móvel", "Rolling coverage"),
    "val_members": ("Ganho de CRPS por membro (vs benchmark)", "CRPS gain by member (vs benchmark)"),
    "val_members_note": ("Valores negativos = pior que o benchmark. O ensemble só usa um membro na medida em que ele supera o ruído na validação interna.",
                         "Negative values = worse than the benchmark. The ensemble only uses a member insofar as it beats noise in the inner validation."),
    "val_folds": ("Blocos de teste e λ escolhido em cada refit", "Test blocks and λ chosen at each re-fit"), "col_train_until": ("Treino até", "Trained until"),
    "aud_title": ("Auditoria de features (SHAP / permutação com canários de ruído)", "Feature audit (SHAP / permutation with noise canaries)"),
    "aud_intro": ("Uma floresta aleatória é treinada com as features reais + 5 colunas de ruído (canários); a importância é medida fora da amostra. Feature que não supera "
                  "os canários = ruído. {shap}",
                  "A random forest is trained with the real features + 5 pure-noise columns (canaries); importance is measured out-of-sample. A feature that does not "
                  "beat the canaries is noise. {shap}"),
    "aud_shap_yes": ("SHAP disponível.", "SHAP available."), "aud_shap_no": ("Instale `shap` para ver também os valores SHAP (usando importância por permutação).", "Install `shap` to also see SHAP values (using permutation importance)."),
    "aud_h": ("Horizonte da auditoria", "Audit horizon"), "aud_run": ("▶️ Executar auditoria", "▶️ Run audit"),
    "aud_err": ("Auditoria indisponível: {err}", "Audit unavailable: {err}"),
    "aud_nopower": ("A floresta não tem poder preditivo fora da amostra (R² = {r2}): as importâncias abaixo refletem majoritariamente ruído.",
                    "The forest has no out-of-sample predictive power (R² = {r2}): the importances below mostly reflect noise."),
    "aud_r2": ("R² fora da amostra (RF)", "Out-of-sample R² (RF)"), "aud_ceiling": ("Teto dos canários", "Canary ceiling"),
    "aud_ceiling_help": ("Maior importância entre as colunas de ruído.", "Highest importance among the noise columns."),
    "aud_n_above": ("Features acima do ruído", "Features above noise"),
    "imp_title": ("Importância por permutação fora da amostra (cinza = canários de ruído)", "Out-of-sample permutation importance (grey = noise canaries)"),
    "imp_axis": ("Queda de R² ao embaralhar", "R² drop when shuffled"), "col_feature": ("Feature", "Feature"),
    "drift_title": ("Monitor de drift das features", "Feature drift monitor"),
    "drift_intro": ("PSI da janela recente (126 pregões) vs. todo o treino, convertido em percentil frente aos PSIs de janelas históricas do mesmo tamanho "
                    "(features persistentes geram PSI alto sem drift real).",
                    "PSI of the recent window (126 days) vs. the whole training set, converted to a percentile against the PSIs of same-size historical windows "
                    "(persistent features produce high PSI without real drift)."),
    "psi_title": ("Percentil do PSI atual na história da feature", "Percentile of current PSI in the feature's history"),
    "col_pct": ("Percentil", "Percentile"), "col_status": ("Status", "Status"),
    "dstat_ok": ("normal", "normal"), "dstat_warn": ("atenção", "watch"), "dstat_drift": ("drift", "drift"),
    "drift_na": ("Histórico insuficiente para o monitor de drift.", "Insufficient history for the drift monitor."),
    # ---- track record
    "trk_title": ("Track record ao vivo (o único teste sem viés)", "Live track record (the only unbiased test)"),
    "trk_intro": ("Cada previsão é gravada em `{p}` (data de origem, quantis por horizonte, versão do modelo) e avaliada quando o horizonte vence.",
                  "Each forecast is saved to `{p}` (origin date, quantiles by horizon, model version) and evaluated once its horizon is reached."),
    "trk_logged": ("Previsões registradas", "Forecasts logged"), "trk_resolved": ("Já avaliadas", "Already evaluated"),
    "trk_cov": ("Cobertura ao vivo (90%)", "Live coverage (90%)"), "trk_cov_help": ("Compare com a cobertura fora da amostra da aba de validação.", "Compare with the out-of-sample coverage in the validation tab."),
    "col_origin": ("Origem", "Origin"),
    "trk_pending": ("{n} previsão(ões) aguardando o vencimento do horizonte.", "{n} forecast(s) waiting for their horizon to elapse."),
    "trk_few": ("Menos de 30 previsões avaliadas: tire conclusões com cautela.", "Fewer than 30 evaluated forecasts: draw conclusions with caution."),
    "trk_empty": ("Nenhuma previsão registrada ainda (ative 'Registrar previsões' e modele em USD).", "No forecasts logged yet (enable 'Log forecasts' and model in USD)."),
    "trk_download": ("⬇️ Baixar track record (CSV)", "⬇️ Download track record (CSV)"),
    "trk_upload": ("Importar CSV de track record (mescla sem duplicar)", "Import track-record CSV (merges without duplicates)"),
    "trk_merged": ("{n} registro(s) novo(s) importado(s).", "{n} new record(s) imported."),
    "trk_badfile": ("Arquivo inválido: colunas do track record ausentes.", "Invalid file: track-record columns missing."),
})


# ----------------------------------------------------------------------------
# Wrappers com cache (modelos pesados são reaproveitados entre interações)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_package(P_train, cfg):
    return fit_package(P_train, cfg)


@st.cache_resource(show_spinner=False)
def get_validation(P, cfg):
    return run_validation(P, cfg)


@st.cache_data(show_spinner=False)
def get_drift(P, o0):
    return drift_table(P, o0)


@st.cache_data(show_spinner=False)
def run_long_term(close, ann, shiller, x0, start_year, n0, rev, pi, adj, use_tr, level0, date0):
    fp = fhs_pool(close)
    bp = bayes_params(ann, start_year, n0, rev, N_SIMS)
    rho = payout_ratio(shiller)
    x0_used = float(np.clip(x0, *bp["x_range"]))              # CAPE inicial fora do histórico = dado suspeito
    par = {"bayes": bp, "pi": pi, "adj": adj, "date0": date0, "garch": fp["garch"], "pool": fp["pool"],
           "s2_next": fp["s2_next"], "sim": fp["sim"], "x0": x0_used, "rho": rho, "use_tr": use_tr, "level0": level0}
    phi_h = np.log(2.0) / np.maximum(bp["kappa"], 1e-6)
    return {"sim": simulate_long(par), "dec": decomposition(bp, x0_used, rho, pi, adj), "rho": rho, "x0": x0_used,
            "x0_raw": float(x0), "ann_vol": fp["ann_vol"], "evidence": valuation_evidence(ann),
            "stress": stress_paths(close),
            "post": {"g": (float(bp["g"].mean()), float(np.percentile(bp["g"], 5)), float(np.percentile(bp["g"], 95))),
                     "phi": (float(np.median(bp["phi"])), bp["mean_ols_phi"]),
                     "half": (float(np.median(phi_h)), float(np.percentile(phi_h, 5)), float(np.percentile(phi_h, 95))),
                     "cape_bar": tuple(float(np.exp(np.percentile(bp["xbar"], q))) for q in (50, 5, 95)),
                     "n_sel": bp["n_sel"], "n_all": bp["n_all"]}}


def backtest_origin(P, cfg, year):
    """Treina só com dados anteriores a 1º/jan de `year` e compara a previsão com o realizado."""
    cut = int(np.searchsorted(P.index.values, np.datetime64(pd.Timestamp(year=year, month=1, day=1))))
    pkg = get_package(P.iloc[:cut], cfg)
    pred = predict_package(pkg, P, np.array([cut - 1]))
    logp = np.log(P["close"].to_numpy(float))
    n = len(P)
    Qh = {h: pred[h]["ens"][-1] for h in HORIZONS}
    Qb = {h: pred[h]["bench"][-1] for h in HORIZONS}
    rows = []
    for h in HORIZONS:
        q, s = Qh[h], pkg["sel"][h]
        lo, hi = float(np.interp(s["a90"] / 2, QS, q)), float(np.interp(1 - s["a90"] / 2, QS, q))
        rec = {"h": h, "med": float(q[qi(0.5)]), "lo": lo, "hi": hi, "lam": s["lam"], "resolved": cut - 1 + h <= n - 1}
        if rec["resolved"]:
            y = float(logp[cut - 1 + h] - logp[cut - 1])
            rec.update({"y": y, "pit": float(np.interp(y, q, QS)), "in90": bool(lo <= y <= hi),
                        "crps": float(crps_q(q[None, :], np.array([y]))[0]),
                        "crps_b": float(crps_q(Qb[h][None, :], np.array([y]))[0])})
        rows.append(rec)
    return {"cut": cut, "origin": P.index[cut - 1], "sel": pkg["sel"], "Qh": Qh, "Qb": Qb, "rows": rows,
            "p_last": float(P["close"].iloc[cut - 1]), "realized": P["close"].iloc[cut - 1: cut + HORIZONS[-1]]}



# ----------------------------------------------------------------------------
# Widgets que preservam o valor ao trocar de idioma (o rótulo muda o ID do widget)
# ----------------------------------------------------------------------------
def _cur(name, default):
    return st.session_state.get("_v_" + name, default)


def w_select(ct, name, label, options, default, **kw):
    options = list(options)
    cur = _cur(name, default)
    v = ct.selectbox(label, options, index=options.index(cur) if cur in options else 0, key=f"{name}@{_LANG}", **kw)
    st.session_state["_v_" + name] = v
    return v


def w_radio(ct, name, label, options, default, **kw):
    options = list(options)
    cur = _cur(name, default)
    v = ct.radio(label, options, index=options.index(cur) if cur in options else 0, key=f"{name}@{_LANG}", **kw)
    st.session_state["_v_" + name] = v
    return v


def w_slider(ct, name, label, lo, hi, default, step=None, **kw):
    cur = _cur(name, default)
    cur = min(max(cur, lo), hi)
    args = dict(value=cur, key=f"{name}@{_LANG}", **kw)
    if step is not None:
        args["step"] = step
    v = ct.slider(label, lo, hi, **args)
    st.session_state["_v_" + name] = v
    return v


def w_check(ct, name, label, default, **kw):
    v = ct.checkbox(label, value=_cur(name, default), key=f"{name}@{_LANG}", **kw)
    st.session_state["_v_" + name] = v
    return v


def hz(h):
    return f"{tr('hz_' + str(h))} ({h}d)"


# ----------------------------------------------------------------------------
# Aba 1 — Visão geral e alertas
# ----------------------------------------------------------------------------
def render_overview(C):
    P, disp, ccy = C.P, C.disp, C.ccy
    last, d = float(disp.iloc[-1]), disp.index[-1]
    d1 = last / float(disp.iloc[-2]) - 1
    ytd = last / float(disp[disp.index < pd.Timestamp(year=d.year, month=1, day=1)].iloc[-1]) - 1
    y1 = last / float(disp.asof(d - pd.DateOffset(years=1))) - 1
    vix_now = float(C.data["vix"].dropna().iloc[-1])
    rec = P["rec_prob"].dropna()
    rec_now = float(rec.iloc[-1]) if len(rec) else np.nan
    usd = C.data["spx"]["Close"]
    bull = bool(usd.iloc[-1] > usd.ewm(span=200, adjust=False).mean().iloc[-1])
    hv = vix_now > C.vix_thr
    rkey = "reg_stress" if (hv and not bull) else "reg_late" if (np.isfinite(rec_now) and rec_now >= 0.30) \
        else "reg_vol" if hv else "reg_calm"

    k = st.columns(6)
    k[0].metric(f"S&P 500 ({ccy})", fmoney(last, ccy), f"{fnum(d1 * 100, 2)}% " + tr("kpi_day"))
    k[1].metric(tr("kpi_ytd"), fpct(ytd))
    k[2].metric(tr("kpi_12m"), fpct(y1))
    k[3].metric("VIX", fnum(vix_now, 1), help=tr("kpi_vix_help", t=C.vix_thr))
    k[4].metric(tr("kpi_rec"), fpct(rec_now, 0), help=tr("kpi_rec_help"))
    k[5].metric(tr("kpi_last"), fdate(d))
    st.info(tr("regime_now", r=tr(rkey)))

    st.subheader(tr("val_alert_title"))
    zt = float(P["trend_z"].iloc[-1])
    zc = float(P["cape_z"].iloc[-1]) if np.isfinite(P["cape_z"].iloc[-1]) else np.nan
    zs = [v for v in (zt, zc) if np.isfinite(v)]
    zcomb = float(np.mean(zs))
    g = st.columns(3)
    with g[0]:
        show(fig_gauge(zt, C.z_thr, tr("gauge_trend")))
    with g[1]:
        show(fig_gauge(zc if np.isfinite(zc) else 0.0, C.z_thr, tr("gauge_cape")))
        if not np.isfinite(zc):
            st.caption(tr("cape_na"))
    with g[2]:
        show(fig_gauge(zcomb, C.z_thr, tr("gauge_comb")))
    stt = zscore_status(zcomb, C.z_thr)
    {"over": st.error, "under": st.success, "fair": st.warning}[stt](tr("status_" + stt, z=fnum(zcomb, 2)))
    fit = P["trend_fit"]
    st.caption(tr("trend_caption", g=fpct(float(np.exp(fit.iloc[-1] - fit.iloc[-253]) - 1)),
                  sd=fnum(float(P["trend_sd"].iloc[-1]), 3)))

    c1, c2 = st.columns([3, 1])
    win = w_radio(c1, "win_ov", tr("win"), [3, 5, 10, 0], 5, horizontal=True,
                  format_func=lambda v: tr("win_all") if v == 0 else tr("win_n", n=v))
    log = w_check(c2, "log_ov", tr("logscale"), True)
    show(fig_overview(disp, ccy, win, log))
    show(fig_trend(P, C.z_thr, _ylabel(C.mccy), win, log))
    with st.expander(tr("fwd_title")):
        ft = forward_table(P["trend_z"], P["close"], C.z_thr)
        table(pd.DataFrame({tr("col_bucket"): ft["bucket"].map(lambda b: tr("bucket_" + b)),
                            tr("col_yrs"): ft["yrs"], "n": ft["n"],
                            tr("col_med_ann"): ft["median"].map(fpct), tr("col_pneg"): ft["p_neg"].map(fpct)}))
        st.caption(tr("fwd_caption"))

    st.subheader(tr("sum_title"))
    s = st.columns(4)
    if C.live is not None:
        r = C.fan.iloc[252]
        s[0].metric(tr("sum_m1", d=fdate(C.fan.index[252])), fmoney(r["med"], ccy),
                    f"{fnum((r['med'] / (C.last_close * C.mult) - 1) * 100, 1)}%")
        s[0].caption(tr("band90", lo=fmoney(r["lo90"], ccy), hi=fmoney(r["hi90"], ccy)))
    if C.m2 is not None:
        s[1].metric(tr("sum_m2", y=LONG_END_YEAR), fmoney(C.m2_vals[50][-1], ccy),
                    f"{fnum(C.m2_cagr[50] * 100, 1)}% " + tr("pa"))
        s[1].caption(tr("sum_m2_cap", lo=fmoney(C.m2_vals[10][-1], ccy), hi=fmoney(C.m2_vals[90][-1], ccy)))
    if C.val_summ is not None:
        nyes = int((C.val_summ["verdict"] == "yes").sum())
        s[2].metric(tr("sum_skill"), f"{nyes}/{len(C.val_summ)}", help=tr("sum_skill_help"))
    else:
        s[2].metric(tr("sum_skill"), "—", help=tr("sum_skill_na"))
    dr = C.drift
    nd = int((dr["status"] == "drift").sum()) if len(dr) else 0
    s[3].metric(tr("sum_drift"), tr("drift_ok") if nd == 0 else tr("drift_n", n=nd))
    if C.fx_mode == "series" and C.ccy != "USD":
        st.caption(tr("fx_series_note", c=C.ccy))
    elif C.ccy != "USD":
        st.caption(tr("fx_spot_note", c=C.ccy, r=fnum(C.spot, 4)))


# ----------------------------------------------------------------------------
# Aba 2 — Modelo 1 (distribuição por horizonte)
# ----------------------------------------------------------------------------
def render_model1(C):
    st.subheader(tr("m1_title"))
    if C.live is None:
        st.warning(tr("m1_unavailable", err=C.err1 or tr("m1_short")))
        return
    P, sel, live, fan = C.P, C.pkg["sel"], C.live, C.fan
    ccy, lvl0 = C.ccy, C.last_close * C.mult
    g = C.pkg["bundle"]["garch"]
    s2n = float(garch_s2n(g, np.r_[0.0, np.diff(np.log(P["close"].to_numpy(float)))])[-1])
    k = st.columns(5)
    k[0].metric(tr("m1_vol_garch"), fpct(np.sqrt(s2n / 1e4 * 252.0), 1), help=tr("m1_vol_garch_help"))
    k[1].metric(tr("m1_vol_stack"), fpct(np.sqrt(live[21]["Vs"][-1] / 21 * 252.0), 1), help=tr("m1_vol_stack_help"))
    k[2].metric("VIX", fnum(float(P["vix"].dropna().iloc[-1]), 1))
    k[3].metric(tr("m1_vrp"), fnum(float(P["vrp"].iloc[-1]), 1), help=tr("m1_vrp_help"))
    age = C.n - C.cut
    k[4].metric(tr("m1_age"), tr("m1_age_v", a=age, nxt=C.refit - age), help=tr("m1_age_help", d=fdate(C.pkg["date"])))

    rows = []
    for h in HORIZONS:
        r = fan.iloc[h]
        v = "—"
        if C.val_summ is not None and (C.val_summ["h"] == h).any():
            v = tr("verdict_" + C.val_summ.loc[C.val_summ["h"] == h, "verdict"].iloc[0])
        rows.append({tr("col_horizon"): hz(h), tr("col_median"): f"{fmoney(r['med'], ccy)} ({fpct(r['med'] / lvl0 - 1)})",
                     tr("col_band90"): f"{fmoney(r['lo90'], ccy)} – {fmoney(r['hi90'], ccy)}",
                     tr("col_band50"): f"{fmoney(r['lo50'], ccy)} – {fmoney(r['hi50'], ccy)}",
                     tr("col_ploss"): fpct(cdf_at(C.Qh[h], 0.0), 0), tr("col_lambda"): fnum(sel[h]["lam"], 2),
                     tr("col_skill"): v})
    table(pd.DataFrame(rows))
    st.caption(tr("m1_table_note"))
    a, b = st.columns([3, 2])
    win = w_radio(a, "win_m1", tr("win"), [3, 5, 10, 0], 3, horizontal=True,
                  format_func=lambda v: tr("win_all") if v == 0 else tr("win_n", n=v))
    show_b = w_check(b, "bench_m1", tr("m1_show_bench"), True)
    show(fig_fan(C.disp, fan, C.bfan, ccy, win, show_b))

    with st.expander(tr("m1_ens_title")):
        rows = []
        for h in HORIZONS:
            s = sel[h]
            row = {tr("col_horizon"): hz(h), tr("col_lambda"): fnum(s["lam"], 2)}
            for m in MEMBERS:
                row[tr("m_" + m)] = fpct(s["w"].get(m, 0.0), 0)
            row.update({tr("col_nval"): s["n_val"], tr("col_neff"): fnum(s["n_eff"], 0),
                        "α 90%": fnum(s["a90"], 3), "α 50%": fnum(s["a50"], 3),
                        tr("col_gain"): fpct(1 - s["crps_ens"] / s["crps_bench"]) if s.get("crps_ens") else "—"})
            rows.append(row)
        table(pd.DataFrame(rows))
        st.caption(tr("m1_ens_note"))
    with st.expander(tr("m1_method_title")):
        st.markdown(tr("m1_method"))


# ----------------------------------------------------------------------------
# Aba 3 — Modelo 2 (estrutural até 2040)
# ----------------------------------------------------------------------------
def render_model2(C):
    st.subheader(tr("m2_title", y=LONG_END_YEAR))
    m2 = C.m2
    if m2 is None:
        st.warning(tr("m2_unavailable", err=C.err2 or ""))
        return
    ccy, sim = C.ccy, m2["sim"]
    x = sim["dates"]
    if abs(m2["x0_raw"] - m2["x0"]) > 1e-9:                       # CAPE inicial fora do histórico: foi limitado
        st.warning(tr("m2_x0_clipped", raw=fnum(float(np.exp(m2["x0_raw"])), 1), used=fnum(float(np.exp(m2["x0"])), 1)))
    lab = tr("idx_tr") if C.use_tr else tr("idx_price")
    k = st.columns(3)
    for col, p, key, icon in ((k[0], 90, "sc_opt", "🟢"), (k[1], 50, "sc_base", "🔵"), (k[2], 10, "sc_pess", "🔴")):
        col.metric(f"{icon} {tr(key)} · {x[-1].year}", fmoney(C.m2_vals[p][-1], ccy),
                   f"{fnum(C.m2_cagr[p] * 100, 1)}% " + tr("pa") + (" " + tr("real_tag") if C.real_terms else ""))
    ratio_end = (sim["final"] / sim["level0"]) / (1 + C.infl) ** C.m2_yrs[-1]
    k2 = st.columns(4)
    k2[0].metric(tr("m2_cape_now"), fnum(float(np.exp(m2["x0"])), 1), help=tr("m2_cape_now_help"))
    k2[1].metric(tr("m2_cape_bar"), fnum(m2["post"]["cape_bar"][0], 1),
                 help=tr("m2_cape_bar_help", lo=fnum(m2["post"]["cape_bar"][1], 1), hi=fnum(m2["post"]["cape_bar"][2], 1)))
    k2[2].metric(tr("m2_ploss"), fpct(float((ratio_end < 1.0).mean()), 0), help=tr("m2_ploss_help"))
    k2[3].metric(tr("m2_vol"), fpct(m2["ann_vol"], 1), help=tr("m2_vol_help"))
    st.caption(tr("m2_caption", lab=lab, n=fnum(sim["n"], 0), rev=fnum(C.rev * 100, 0),
                  real=tr("real_yes") if C.real_terms else tr("real_no")))
    po = m2["post"]
    st.caption(tr("m2_diag", c=fnum(float(np.exp(m2["x0"])), 1), l=fnum(po["cape_bar"][0], 1),
                  h=fnum(po["half"][0], 1), g=fpct(po["g"][0], 1), p=fpct(m2["rho"], 0)))
    if float(C.m2_vals[90][-1]) < C.level0_disp:                    # nem o cenário otimista supera o nível atual
        st.error(tr("m2_implausible", y=x[-1].year, c=fnum(float(np.exp(m2["x0"])), 1)))
    elif C.m2_cagr[50] < -0.03:
        st.warning(tr("m2_pessimistic", g=fpct(C.m2_cagr[50], 1)))

    c1, c2 = st.columns(2)
    log = w_check(c1, "log_lt", tr("logscale"), True)
    show_stress = w_check(c2, "stress_lt", tr("m2_show_stress"), True)
    stress = {}
    if show_stress:
        for nm, v in m2["stress"].items():
            rel = v["rel"]
            sx = pd.DatetimeIndex([sim["date0"]]).append(pd.date_range(sim["date0"] + BDAY, periods=len(rel) - 1, freq=BDAY))
            defl = (1 + C.infl) ** (np.arange(len(rel)) / 252.0) if C.real_terms else 1.0
            stress[nm] = (sx, C.level0_disp * rel / defl)
    hist = C.hist_idx
    xp = pd.DatetimeIndex([sim["date0"]]).append(x)                     # ponto de origem: cone parte do nível atual
    pcp = {p: np.r_[C.level0_disp, v] for p, v in C.m2_vals.items()}
    show(fig_long(xp, pcp, hist, ccy, lab, log, stress))

    dd = m2["dec"]
    st.markdown("**" + tr("m2_dec_title") + "**")
    d = st.columns(4)
    d[0].metric(tr("dec_earn"), fpct(dd["earn"], 1), help=tr("dec_earn_help"))
    d[1].metric(tr("dec_mult"), fpct(dd["mult"], 1), help=tr("dec_mult_help", c=fnum(float(np.exp(dd["x_end"])), 1)))
    d[2].metric(tr("dec_div"), fpct(dd["div"], 1) if C.use_tr else "—", help=tr("dec_div_help"))
    d[3].metric(tr("dec_total"), fpct(dd["earn"] + dd["mult"] + (dd["div"] if C.use_tr else 0.0), 1), help=tr("dec_total_help"))

    with st.expander(tr("m2_tab_title")):
        ye = [i for i, d in enumerate(x) if d.month == 12]               # fechamento de cada ano (dezembro)
        table(pd.DataFrame({tr("col_date"): [fdate(x[i]) for i in ye],
                            tr("sc_pess"): [fmoney(C.m2_vals[10][i], ccy) for i in ye],
                            tr("sc_base"): [fmoney(C.m2_vals[50][i], ccy) for i in ye],
                            tr("sc_opt"): [fmoney(C.m2_vals[90][i], ccy) for i in ye]}))
    with st.expander(tr("m2_post_title")):
        po = m2["post"]
        table(pd.DataFrame({
            tr("col_param"): [tr("p_g"), tr("p_phi"), tr("p_half"), tr("p_cape"), tr("p_rho"), tr("p_n")],
            tr("col_value"): [f"{fpct(po['g'][0], 2)} [{fpct(po['g'][1], 2)} ; {fpct(po['g'][2], 2)}]",
                              f"{fnum(po['phi'][0], 3)} ({tr('p_ols')}: {fnum(po['phi'][1], 3)})",
                              f"{fnum(po['half'][0], 1)} {tr('years')} [{fnum(po['half'][1], 1)} ; {'>100' if po['half'][2] > 100 else fnum(po['half'][2], 1)}]",
                              f"{fnum(po['cape_bar'][0], 1)} [{fnum(po['cape_bar'][1], 1)} ; {fnum(po['cape_bar'][2], 1)}]",
                              fpct(m2["rho"], 0), f"{po['n_sel']} / {po['n_all']}"]}))
        st.caption(tr("m2_post_note"))
    with st.expander(tr("m2_evid_title")):
        ev = m2["evidence"]
        if len(ev):
            table(pd.DataFrame({tr("col_years"): ev["H"], "n": ev["n"], "R²_OS": ev["r2os"].map(lambda v: fpct(v, 1)),
                                tr("col_pcw"): ev["p_cw"].map(lambda v: fnum(v, 3)),
                                tr("col_slope"): ev["slope"].map(lambda v: fnum(v, 3))}))
        st.caption(tr("m2_evid_note"))
    with st.expander(tr("m2_stress_title")):
        rows = []
        for nm, v in m2["stress"].items():
            rows.append({tr("col_event"): tr("stress_" + nm), tr("col_dd"): fpct(v["dd"], 0),
                         tr("col_trough"): tr("n_days", n=v["days_to_trough"]),
                         tr("col_recover"): tr("n_days", n=v["days_to_recover"]) if v["days_to_recover"] is not None
                         else tr("not_yet")})
        table(pd.DataFrame(rows))
        st.caption(tr("m2_stress_note"))
    with st.expander(tr("m2_method_title")):
        st.markdown(tr("m2_method"))


# ----------------------------------------------------------------------------
# Aba 4 — Backtesting histórico
# ----------------------------------------------------------------------------
def render_backtest(C):
    st.subheader(tr("bt_title"))
    st.write(tr("bt_intro"))
    if C.live is None:
        st.warning(tr("m1_unavailable", err=C.err1 or tr("m1_short")))
        return
    P, cfg = C.P, C.cfg
    ys = list(range(P.index[cfg["o0"]].year + 8, C.last_date.year))
    if not ys:
        st.warning(tr("bt_noyears"))
        return
    yr = w_select(st, "bt_year", tr("bt_year"), ys, 2018 if 2018 in ys else ys[-1])
    key = (C.ccy, C.fx_mode, C.train_year, C.quality, int(yr))
    if st.button(tr("bt_run")):
        try:
            with st.spinner(tr("spin_bt", y=yr)):
                st.session_state["bt"] = {"key": key, "res": backtest_origin(P, cfg, int(yr))}
        except Exception as e:  # noqa: BLE001
            st.error(tr("bt_err", err=str(e)[:160]))
    bt = st.session_state.get("bt")
    if bt and bt["key"] == key:
        r = bt["res"]
        mult = C.spot if C.fx_mode == "spot" else 1.0
        fan = fan_from_quantiles(r["Qh"], r["sel"], r["p_last"], r["origin"], mult)
        bfan = fan_from_quantiles(r["Qb"], {h: {"a90": 0.10, "a50": 0.50} for h in HORIZONS}, r["p_last"], r["origin"], mult)
        hist = C.disp[C.disp.index <= r["origin"]]
        real = r["realized"] * mult
        if len(real) < HORIZONS[-1] + 1:
            st.warning(tr("bt_partial"))
        show(fig_backtest(hist, fan, real, bfan, C.ccy, r["origin"]))
        rows = []
        for q in r["rows"]:
            row = {tr("col_horizon"): hz(q["h"]), tr("col_pred_med"): fpct(q["med"], 1),
                   tr("col_band90"): f"{fpct(q['lo'], 1)} – {fpct(q['hi'], 1)}", tr("col_lambda"): fnum(q["lam"], 2)}
            if q["resolved"]:
                row.update({tr("col_real"): fpct(q["y"], 1), "PIT": fnum(q["pit"], 2),
                            tr("col_in90"): "✅" if q["in90"] else "❌",
                            tr("col_vs_bench"): fpct(1 - q["crps"] / q["crps_b"], 0) if q["crps_b"] > 0 else "—"})
            else:
                row.update({tr("col_real"): "—", "PIT": "—", tr("col_in90"): "—", tr("col_vs_bench"): "—"})
            rows.append(row)
        table(pd.DataFrame(rows))
        st.caption(tr("bt_note"))

    st.markdown("##### " + tr("wf_title"))
    st.caption(tr("wf_caption"))
    sig = (C.ccy, C.fx_mode, C.train_year, C.quality, int(C.n))
    if st.button(tr("wf_run")):
        rows, full = [], [y for y in ys if y <= C.last_date.year - 2]
        prog = st.progress(0.0)
        for i, y in enumerate(full):
            try:
                b = backtest_origin(P, cfg, int(y))
                rr = {q["h"]: q for q in b["rows"]}
                rows.append({"year": y, "r1": rr[252], "r2": rr[504]})
            except Exception:  # noqa: BLE001
                pass
            prog.progress((i + 1) / len(full))
        st.session_state["wf"] = {"sig": sig, "rows": rows}
    wf = st.session_state.get("wf")
    if wf and wf["sig"] == sig and wf["rows"]:
        recs = []
        for w in wf["rows"]:
            for tag, q in (("1y", w["r1"]), ("2y", w["r2"])):
                if q["resolved"]:
                    recs.append({"y": w["year"], "t": tag, "med": q["med"], "real": q["y"], "in90": q["in90"],
                                 "crps": q["crps"], "crps_b": q["crps_b"], "hit": np.sign(q["med"]) == np.sign(q["y"])})
        if recs:
            d = pd.DataFrame(recs)
            k = st.columns(4)
            for col, t in ((k[0], "1y"), (k[2], "2y")):
                s = d[d["t"] == t]
                if len(s):
                    col.metric(tr("wf_cov", t=t), fpct(float(s["in90"].mean()), 0))
                    col.metric(tr("wf_skill", t=t), fpct(1 - s["crps"].sum() / s["crps_b"].sum(), 1))
            k[1].metric(tr("wf_n"), len(d[d["t"] == "1y"]))
            k[3].metric(tr("wf_neff"), fnum(len(d[d["t"] == "2y"]) / 2.0, 0), help=tr("wf_neff_help"))
            table(pd.DataFrame({tr("col_year"): d["y"], tr("col_horizon"): d["t"], tr("col_pred_med"): d["med"].map(fpct),
                                tr("col_real"): d["real"].map(fpct), tr("col_in90"): d["in90"].map({True: "✅", False: "❌"}),
                                tr("col_vs_bench"): (1 - d["crps"] / d["crps_b"]).map(lambda v: fpct(v, 0))}))
            st.caption(tr("wf_note"))


# ----------------------------------------------------------------------------
# Aba 5 — Validação rigorosa e diagnósticos
# ----------------------------------------------------------------------------
def render_validation(C):
    st.subheader(tr("val_title"))
    st.write(tr("val_intro"))
    if C.live is None:
        st.warning(tr("m1_unavailable", err=C.err1 or tr("m1_short")))
        return
    if st.button(tr("val_run")):
        st.session_state["val_sig"] = C.val_sig
        st.rerun()
    if C.val_err:
        st.error(tr("val_err", err=C.val_err[:160]))
    if C.val_summ is None:
        st.info(tr("val_hint", t=fnum(QUALITY[C.quality]["folds"], 0)))
    else:
        sm, det = C.val_summ, C.val_det
        table(pd.DataFrame({
            tr("col_horizon"): [hz(h) for h in sm["h"]], "n": sm["n"], tr("col_neff"): sm["n_eff"].map(lambda v: fnum(v, 0)),
            "R²_OS": sm["r2os"].map(lambda v: fpct(v, 2)), tr("col_mse"): sm["mse"].map(fsci),
            tr("col_mse_b"): sm["mse_b"].map(fsci), "CRPSS": sm["crpss"].map(lambda v: fpct(v, 2)),
            "p (CW)": sm["p_cw"].map(lambda v: fnum(v, 3)), "p (DM)": sm["p_dm"].map(lambda v: fnum(v, 3)),
            tr("col_cov90"): sm["cov90"].map(lambda v: fpct(v, 0)), tr("col_cov90aci"): sm["cov90_aci"].map(lambda v: fpct(v, 0)),
            tr("col_cov50"): sm["cov50"].map(lambda v: fpct(v, 0)), tr("col_pitks"): sm["pit_ks"].map(lambda v: fnum(v, 2)),
            tr("col_verdict"): sm["verdict"].map(lambda v: tr("verdict_" + v))}))
        st.caption(tr("val_note"))
        a, b = st.columns(2)
        with a:
            show(fig_skill(sm))
        hsel = w_select(b, "val_h", tr("val_h"), list(sm["h"]), 252 if 252 in list(sm["h"]) else int(sm["h"].iloc[-1]),
                        format_func=hz)
        with b:
            show(fig_pit(det[hsel]["pit_hist"], hsel))
        show(fig_roll_cov(det[hsel]["roll_cov"].dropna()))
        with st.expander(tr("val_members")):
            rows = []
            for h in sm["h"]:
                row = {tr("col_horizon"): hz(h)}
                for m in MEMBERS:
                    row[tr("m_" + m)] = fpct(det[h]["members"][m], 2)
                rows.append(row)
            table(pd.DataFrame(rows))
            st.caption(tr("val_members_note"))
        with st.expander(tr("val_folds")):
            rows = []
            for f in C.val["folds"]:
                row = {tr("col_train_until"): fdate(f["date"])}
                for h in HORIZONS:
                    row[f"λ {h}d"] = fnum(f["sel"][h]["lam"], 2)
                rows.append(row)
            table(pd.DataFrame(rows))

    st.markdown("##### " + tr("aud_title"))
    st.caption(tr("aud_intro", shap=tr("aud_shap_yes") if HAS_SHAP else tr("aud_shap_no")))
    ah = w_select(st, "aud_h", tr("aud_h"), list(HORIZONS), 63, format_func=hz)
    if st.button(tr("aud_run")):
        try:
            with st.spinner(tr("spin_aud")):
                st.session_state["audit"] = {"key": (C.ccy, C.fx_mode, C.train_year, int(ah)),
                                             "res": feature_audit(C.P, C.cfg, int(ah))}
        except Exception as e:  # noqa: BLE001
            st.error(tr("aud_err", err=str(e)[:160]))
    au = st.session_state.get("audit")
    if au and au["key"] == (C.ccy, C.fx_mode, C.train_year, int(ah)):
        r = au["res"]
        if r["r2_oos"] <= 0:
            st.warning(tr("aud_nopower", r2=fnum(r["r2_oos"], 3)))
        k = st.columns(3)
        k[0].metric(tr("aud_r2"), fnum(r["r2_oos"], 3))
        k[1].metric(tr("aud_ceiling"), fnum(r["ceiling"], 4), help=tr("aud_ceiling_help"))
        k[2].metric(tr("aud_n_above"), int(r["table"]["above_noise"].sum()))
        show(fig_importance(r["table"]))
        if r["has_shap"]:
            t2 = r["table"].sort_values("shap", ascending=False).head(10)
            table(pd.DataFrame({tr("col_feature"): t2["feature"], "mean |SHAP|": t2["shap"].map(lambda v: fnum(v, 4))}))

    st.markdown("##### " + tr("drift_title"))
    st.caption(tr("drift_intro"))
    dr = C.drift
    if len(dr):
        show(fig_psi(dr))
        table(pd.DataFrame({tr("col_feature"): dr["feature"].head(10), "PSI": dr["psi"].head(10).map(lambda v: fnum(v, 2)),
                            tr("col_pct"): dr["pct"].head(10).map(lambda v: fpct(v, 0)),
                            tr("col_status"): dr["status"].head(10).map(lambda v: tr("dstat_" + v))}))
    else:
        st.write(tr("drift_na"))


# ----------------------------------------------------------------------------
# Aba 6 — Track record ao vivo
# ----------------------------------------------------------------------------
def render_track(C):
    st.subheader(tr("trk_title"))
    st.write(tr("trk_intro", p=str(LOG_PATH.name)))
    log = read_log()
    ev = evaluate_log(log, C.data["spx"]["Close"]) if len(log) else pd.DataFrame()
    k = st.columns(4)
    k[0].metric(tr("trk_logged"), int(len(log)))
    res = ev[ev["resolved"]] if len(ev) else ev
    k[1].metric(tr("trk_resolved"), int(len(res)))
    if len(res):
        k[2].metric(tr("trk_cov"), fpct(float(res["in90"].mean()), 0), help=tr("trk_cov_help"))
        k[3].metric("CRPS", fnum(float(res["crps"].mean()), 4))
    if len(ev):
        pend = ev[~ev["resolved"]]
        view = ev.sort_values("origin", ascending=False).head(60)
        table(pd.DataFrame({tr("col_origin"): view["origin"].map(fdate), tr("col_horizon"): view["h"].map(hz),
                            tr("col_pred_med"): view["median"].map(lambda v: fpct(v, 1)),
                            tr("col_real"): view["realized"].map(lambda v: fpct(v, 1)) if "realized" in view else "—",
                            "PIT": view["pit"].map(lambda v: fnum(v, 2)) if "pit" in view else "—",
                            tr("col_in90"): view["in90"].map(lambda v: "✅" if v is True else "❌" if v is False else "—")
                            if "in90" in view else "—"}))
        st.caption(tr("trk_pending", n=int(len(pend))))
        if len(res) < 30:
            st.info(tr("trk_few"))
    else:
        st.info(tr("trk_empty"))
    st.download_button(tr("trk_download"), data=log.to_csv(index=False).encode("utf-8"), file_name="forecast_log.csv",
                       mime="text/csv")
    up = st.file_uploader(tr("trk_upload"), type="csv")
    if up is not None:
        try:
            new = pd.read_csv(up)
            if all(c in new.columns for c in LOG_COLS):
                st.success(tr("trk_merged", n=merge_log(new)))
            else:
                st.error(tr("trk_badfile"))
        except Exception as e:  # noqa: BLE001
            st.error(tr("trk_badfile") + f" ({str(e)[:80]})")


# ----------------------------------------------------------------------------
# Aplicação
# ----------------------------------------------------------------------------
def main():
    global _LANG
    sb = st.sidebar
    lang = sb.selectbox("Idioma / Language", ["Português", "English"], key="lang")
    _LANG = "pt" if lang == "Português" else "en"
    st.title(tr("title"))
    st.caption(tr("subtitle", d=fdate(START_FULL)))

    try:
        with st.spinner(tr("spin_data")):
            data = load_market_data(START_FULL, date.today().isoformat(), st.session_state.get("shiller_path", ""))
    except Exception as e:  # noqa: BLE001
        st.error(tr("err_data", err=str(e)[:400]))
        if st.button(tr("retry")):
            st.rerun()
        st.stop()

    # ------------------------------ sidebar ------------------------------
    avail = [c for c in CCY if c in data["fx"].columns and data["fx"][c].notna().sum() > 750]
    sb.header(tr("sb_controls"))
    ccy = w_select(sb, "ccy", tr("sb_ccy"), avail, "USD", format_func=lambda c: f"{c} ({SYMBOL[c]})", help=tr("sb_ccy_help"))
    fx_mode = "spot"
    if ccy != "USD":
        fx_mode = w_radio(sb, "fxmode", tr("sb_fxmode"), ["spot", "series"], "spot", format_func=lambda v: tr("fx_" + v),
                          help=tr("sb_fxmode_help"))
    sb.markdown("**" + tr("sb_m1") + "**")
    train_year = w_select(sb, "train_year", tr("sb_start1"), [1990, 2000, 2010], 1990, help=tr("sb_start1_help"))
    quality = w_select(sb, "quality", tr("sb_quality"), ["fast", "standard", "full"], "standard",
                       format_func=lambda v: tr("q_" + v), help=tr("sb_quality_help"))
    refit = w_slider(sb, "refit", tr("sb_refit"), 5, 63, 21, help=tr("sb_refit_help"))
    vix_thr = w_slider(sb, "vix", tr("sb_vix"), 15, 35, 22)
    sb.markdown("**" + tr("sb_m2") + "**")
    use_tr = w_radio(sb, "index", tr("sb_index"), ["price", "tr"], "price", format_func=lambda v: tr("idx_" + v)) == "tr"
    real_terms = w_check(sb, "real", tr("sb_real"), False, help=tr("sb_real_help"))
    infl = w_slider(sb, "infl", tr("sb_infl"), 0.0, 6.0, 2.5, 0.1, help=tr("sb_infl_help")) / 100.0
    start2 = w_select(sb, "fund_start", tr("sb_fund_start"), [1871, 1950, 1990], 1950, help=tr("sb_fund_start_help"))
    n0 = w_slider(sb, "prior", tr("sb_prior"), 10, 100, 40, help=tr("sb_prior_help"))
    rev = w_slider(sb, "rev", tr("sb_rev"), 0, 100, 100, 10, help=tr("sb_rev_help")) / 100.0
    adj = w_slider(sb, "adj", tr("sb_adj"), -3.0, 3.0, 0.0, 0.5, help=tr("sb_adj_help")) / 100.0
    sb.markdown("**" + tr("sb_alert") + "**")
    z_thr = w_slider(sb, "zthr", tr("sb_zthr"), 1.0, 3.0, 1.5, 0.1)
    autolog = w_check(sb, "autolog", tr("sb_autolog"), True, help=tr("sb_autolog_help"))
    with sb.expander(tr("sb_adv")):
        st.text_input(tr("sb_shiller"), key="shiller_path", help=tr("sb_shiller_help"))
    if sb.button(tr("sb_refresh")):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    if data["warnings"]:
        with sb.expander(tr("sb_warn", n=len(data["warnings"]))):
            for key, kw in data["warnings"]:
                st.write("• " + tr(key, **kw))
            if st.button(tr("sb_retry_data")):
                load_market_data.clear()                  # só os dados: os modelos já treinados são mantidos
                st.rerun()

    # ------------------------------ painel + Modelo 1 ------------------------------
    disp = (data["spx"]["Close"] * data["fx"][ccy]).dropna()
    spot = float(data["fx"][ccy].dropna().iloc[-1])
    P, o0 = build_panel(data, fx_mode, ccy, f"{train_year}-01-01")
    n = len(P)
    cfg = dict(QUALITY[quality], seed=SEED, o0=o0, gamma=0.02)
    mccy = "USD" if fx_mode == "spot" else ccy
    pkg = live = err1 = None
    cut = n
    if n - o0 >= 1500:
        cut = n - ((n - o0) % refit)
        try:
            with st.spinner(tr("spin_m1")):
                pkg = get_package(P.iloc[:cut], cfg)
                live = predict_package(pkg, P, np.array([n - 1]))
        except Exception as e:  # noqa: BLE001
            pkg = live = None
            err1 = str(e)
    fan = bfan = Qh = None
    last_close = float(P["close"].iloc[-1])
    mult = spot if fx_mode == "spot" else 1.0
    if live is not None:
        Qh = {h: live[h]["ens"][-1] for h in HORIZONS}
        fan = fan_from_quantiles(Qh, pkg["sel"], last_close, P.index[-1], mult)
        bfan = fan_from_quantiles({h: live[h]["bench"][-1] for h in HORIZONS},
                                  {h: {"a90": 0.10, "a50": 0.50} for h in HORIZONS}, last_close, P.index[-1], mult)
        if autolog and fx_mode == "spot":                               # track record (sempre em USD)
            version = f"v2|{train_year}|{quality}|{pd.Timestamp(pkg['date']):%Y-%m-%d}"
            key = (str(P.index[-1].date()), version)
            if st.session_state.get("logged") != key:
                try:
                    merge_log(make_log_rows(live, pkg["sel"], last_close, P.index[-1], version))
                    st.session_state["logged"] = key
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------ validação (se já solicitada) ------------------------------
    val_sig = (ccy, fx_mode, train_year, quality, int(n))
    val = val_summ = val_det = None
    val_err = ""
    if live is not None and st.session_state.get("val_sig") == val_sig:
        try:
            with st.spinner(tr("spin_val")):
                val = get_validation(P, cfg)
                val_summ, val_det = summarize_validation(val)
        except Exception as e:  # noqa: BLE001
            val_err = str(e)

    # ------------------------------ Modelo 2 ------------------------------
    m2, err2 = None, ""
    last_date = data["spx"].index[-1]
    index_usd = data["tr"] if use_tr else data["spx"]["Close"]
    try:
        with st.spinner(tr("spin_m2")):
            ann = fund_annual(data["shiller"])
            x0 = float(data["ext"]["log_cape"].dropna().iloc[-1])
            m2 = run_long_term(data["spx"]["Close"], ann, data["shiller"], x0, start2, n0, rev, infl, adj, use_tr,
                               float(index_usd.iloc[-1]), last_date)
    except Exception as e:  # noqa: BLE001
        err2 = str(e)
    fx_drift = 0.0
    if fx_mode == "series" and ccy != "USD":
        rate = data["fx"][ccy].dropna()
        fx_drift = float((np.log(rate.iloc[-1]) - np.log(rate.iloc[0])) / ((rate.index[-1] - rate.index[0]).days / 365.25))
    m2_vals = m2_cagr = m2_yrs = None
    level0_disp = float(index_usd.iloc[-1]) * spot
    if m2 is not None:
        sim = m2["sim"]
        m2_yrs = np.array([(dd - sim["date0"]).days / 365.25 for dd in sim["dates"]])
        defl = (1 + infl) ** m2_yrs if real_terms else np.ones_like(m2_yrs)
        fxm = spot * (np.exp(fx_drift * m2_yrs) if fx_mode == "series" else 1.0)
        m2_vals = {p: v * fxm / defl for p, v in sim["pct"].items()}
        m2_cagr = {p: float((m2_vals[p][-1] / level0_disp) ** (1.0 / m2_yrs[-1]) - 1.0) for p in (10, 50, 90)}
    hist_idx = (index_usd * data["fx"][ccy]).dropna()

    drift = get_drift(P, o0) if n - o0 > 800 else pd.DataFrame(columns=["feature", "psi", "pct", "status"])

    C = types.SimpleNamespace(
        data=data, P=P, o0=o0, n=n, cfg=cfg, ccy=ccy, fx_mode=fx_mode, mccy=mccy, spot=spot, disp=disp, train_year=train_year,
        quality=quality, refit=refit, vix_thr=vix_thr, z_thr=z_thr, use_tr=use_tr, real_terms=real_terms, infl=infl,
        rev=rev, pkg=pkg, live=live, err1=err1, cut=cut, fan=fan, bfan=bfan, Qh=Qh, last_close=last_close, mult=mult,
        val=val, val_summ=val_summ, val_det=val_det, val_err=val_err, val_sig=val_sig, m2=m2, err2=err2, m2_vals=m2_vals,
        m2_cagr=m2_cagr, m2_yrs=m2_yrs, level0_disp=level0_disp, hist_idx=hist_idx, last_date=last_date, drift=drift)

    tabs = st.tabs([tr("tab1"), tr("tab2"), tr("tab3", y=LONG_END_YEAR), tr("tab4"), tr("tab5"), tr("tab6")])
    with tabs[0]:
        render_overview(C)
    with tabs[1]:
        render_model1(C)
    with tabs[2]:
        render_model2(C)
    with tabs[3]:
        render_backtest(C)
    with tabs[4]:
        render_validation(C)
    with tabs[5]:
        render_track(C)
    st.divider()
    st.caption(tr("disclaimer"))


if __name__ == "__main__":
    main()
