"""預測模型。

結構：個股 15 日預期報酬 = β × 大盤預期報酬 + 相對強度分數 × 個股波動 +（美股隔夜調整）

* 相對強度分數（alpha）：對全體股票做橫斷面嶺迴歸，用技術因子預測「未來 N 日扣除大盤連動後的
  剩餘報酬」。因子與目標都先做風險中和（拿掉波動與 β 的影響）並以個股波動標準化，所以它回答的是
  「同樣風險下，這檔會比別檔強還是弱」，而不是在賭高波動股會不會整群上漲。
  線性模型的好處是每個因子的貢獻可以直接拆出來，當作判斷依據。
* 大盤：用大盤自身與美股的時間序列因子做強懲罰的嶺迴歸。樣本少、雜訊大，僅供參考。
* 信心值：把相對強度分數對應到滾動回測（walk-forward，樣本外）中「15 日後實際贏過股票池中位數」
  的比例。57 的意思是「過去同等強度的訊號，約 57% 真的贏過一半的股票」，基準是 50，可在復盤頁驗證。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .features import FACTOR_KEYS, zscore_cs


def ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    k = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * len(y) * np.eye(k), X.T @ y)


def rank_ic(pred: np.ndarray, real: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(real)
    if ok.sum() < 20:
        return np.nan
    a, b = pd.Series(pred[ok]).rank().values, pd.Series(real[ok]).rank().values
    return float(np.corrcoef(a, b)[0, 1])


def norm_cdf(z):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(z, dtype=float) / math.sqrt(2)))


def fit_logistic(z: np.ndarray, up: np.ndarray, iters: int = 30) -> tuple[float, float]:
    """兩個參數的邏輯斯迴歸 P = σ(a + b·z)，牛頓法求解。"""
    X = np.column_stack([np.ones_like(z), z])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (up - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


def prob_win(score, slope: float):
    """相對強度分數 → 贏過股票池中位數的機率。"""
    return 1 / (1 + np.exp(-slope * np.asarray(score, dtype=float)))


@dataclass
class Panel:
    """模型需要的所有對齊後資料。"""
    dates: pd.DatetimeIndex
    codes: list[str]
    adj: pd.DataFrame                      # 還原權息收盤價
    taiex: pd.Series
    F: np.ndarray                          # [日期, 股票, 因子] 標準化、風險中和後的因子
    M: pd.DataFrame                        # 大盤因子（原始值）
    beta: pd.DataFrame                     # 個股對大盤 β
    sigma_d: pd.DataFrame                  # 個股日波動
    fwd: dict[int, np.ndarray] = field(default_factory=dict)       # 未來 h 日報酬
    y: dict[int, np.ndarray] = field(default_factory=dict)         # 訓練目標：標準化、風險中和後的剩餘報酬
    mfwd: dict[int, np.ndarray] = field(default_factory=dict)      # 大盤未來 h 日報酬
    risk: np.ndarray | None = None                                 # [日期, 股票, 2] 風險暴露（波動、β）


def neutralize(A: np.ndarray, R: np.ndarray, rescale: bool = False) -> np.ndarray:
    """每天做一次橫斷面迴歸，把 A 中能被風險暴露 R（波動、β）解釋的部分拿掉，只留殘差。

    為什麼需要：高波動／高 β 股票會整群一起強、一起弱（風險偏好的循環），這種循環無法預測，
    卻會讓動能、乖離等因子的方向每幾個月翻轉一次。拿掉之後剩下的才是比較穩定的選股訊號。
    """
    out = np.full_like(A, np.nan)
    for t in range(A.shape[0]):
        ok = np.isfinite(A[t]) & np.isfinite(R[t]).all(axis=1)
        if ok.sum() < 30:
            continue
        X = np.column_stack([np.ones(ok.sum()), R[t][ok]])
        coef, *_ = np.linalg.lstsq(X, A[t][ok], rcond=None)
        res = A[t][ok] - X @ coef
        if rescale:
            res = res / (res.std() + 1e-12)
        out[t, ok] = res
    return out


def build_panel(adj: pd.DataFrame, taiex: pd.Series, raw_factors: dict[str, pd.DataFrame], M: pd.DataFrame) -> Panel:
    ret = adj.pct_change(fill_method=None)
    mret = taiex.pct_change()
    beta = ret.rolling(120, min_periods=60).cov(mret).div(mret.rolling(120, min_periods=60).var(), axis=0)
    beta = (0.7 * beta + 0.3).clip(0.3, 2.2)
    sigma_d = ret.ewm(span=60, min_periods=20).std()
    risk = np.stack([zscore_cs(np.log(sigma_d)).values, zscore_cs(beta).values], axis=2)
    F = np.stack([neutralize(zscore_cs(raw_factors[k]).values, risk, rescale=True) for k in FACTOR_KEYS], axis=2)
    F = np.clip(F, -3, 3)
    p = Panel(adj.index, list(adj.columns), adj, taiex, F, M, beta, sigma_d, risk=risk)
    with np.errstate(invalid="ignore", divide="ignore"):
        for h in C.HORIZONS:
            f = (adj.shift(-h) / adj - 1).values
            m = (taiex.shift(-h) / taiex - 1).values
            # 剩餘報酬 = 個股報酬 − β×大盤報酬，再除以個股 h 日波動；β 與波動都只用當天以前的資料
            y = np.clip((f - beta.values * m[:, None]) / (sigma_d.values * math.sqrt(h)), -4, 4)
            p.fwd[h], p.mfwd[h] = f, m
            p.y[h] = np.clip(neutralize(y, risk), -3, 3)
    return p


def factor_ics(p: Panel, x: pd.DataFrame, rows) -> np.ndarray:
    """任一因子（未放進模型的也可以）經相同標準化與風險中和後，逐日與實際結果的排序相關。"""
    z = np.clip(neutralize(zscore_cs(x).values, p.risk, rescale=True), -3, 3)
    return np.array([rank_ic(z[t], p.y[C.HORIZON][t]) for t in rows])


# ── 相對強度（橫斷面）─────────────────────────────────────────────────────

def fit_alpha(p: Panel, T: int, h: int) -> np.ndarray:
    """用第 T 天當下已知的資料訓練（第 t 天的標籤要到 t+h 才知道，所以只用 t ≤ T−h）。

    技術因子彼此高度相關、訊號又弱，自由估計的權重在樣本外很不穩。所以用很強的懲罰，
    讓權重接近「各因子單獨的預測力」，再用一個係數把整體幅度調回正確的尺度。
    """
    end = T - h
    rows = np.arange(max(0, end - C.TRAIN_WINDOW + 1), end + 1)
    X = p.F[rows].reshape(-1, p.F.shape[2])
    y = p.y[h][rows].reshape(-1)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < 1000:
        raise RuntimeError(f"訓練樣本只有 {len(y)} 筆，無法訓練（可能有因子整欄缺值或歷史資料不足）")
    b = ridge(X, y, C.RIDGE_LAMBDA)
    s = X @ b
    return b * float(s @ y) / float(s @ s + 1e-12)


# ── 大盤（時間序列）───────────────────────────────────────────────────────

@dataclass
class MarketModel:
    cols: list[str]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float

    def parts(self, row: pd.Series) -> np.ndarray:
        z = (row[self.cols].values.astype(float) - self.mean) / self.std
        return np.nan_to_num(np.clip(z, -3, 3)) * self.coef

    def predict(self, row: pd.Series) -> float:
        return float(self.intercept + self.parts(row).sum())


def fit_market(p: Panel, T: int, h: int) -> MarketModel:
    end = T - h
    rows = np.arange(max(0, end - C.TRAIN_WINDOW + 1), end + 1)
    cols = [c for c in p.M.columns if p.M[c].iloc[rows].notna().mean() > 0.8]
    Xr = p.M[cols].iloc[rows].values.astype(float)
    y = p.mfwd[h][rows]
    ok = np.isfinite(Xr).all(axis=1) & np.isfinite(y)
    Xr, y = Xr[ok], y[ok]
    mean, std = Xr.mean(axis=0), Xr.std(axis=0) + 1e-12
    X = np.clip((Xr - mean) / std, -3, 3)
    ybar = y.mean()
    # 重疊樣本的有效樣本數只有約 n/h，所以用很強的懲罰
    coef = ridge(X, y - ybar, 2.0)
    prior = (1 + C.MARKET_DRIFT_PRIOR) ** (h / 252) - 1
    intercept = C.MARKET_DRIFT_BLEND * ybar + (1 - C.MARKET_DRIFT_BLEND) * prior
    return MarketModel(cols, mean, std, coef, float(intercept))


# ── 滾動回測（樣本外）─────────────────────────────────────────────────────

@dataclass
class Backtest:
    idx: np.ndarray                 # 回測涵蓋的日期位置
    alpha: np.ndarray               # [天, 股票] 樣本外相對強度分數
    mkt: np.ndarray                 # [天] 樣本外大盤預測
    mu: np.ndarray                  # [天, 股票] 個股總預期報酬
    slope: float                    # 信心校準斜率
    alpha_sd: float                 # 分數的典型離散度
    stats: dict


def walk_forward(p: Panel) -> Backtest:
    h = C.HORIZON
    n = len(p.dates)
    start = max(C.MIN_HISTORY_DAYS + h, n - C.BACKTEST_DAYS)
    idx = np.arange(start, n)
    alpha = np.full((len(idx), len(p.codes)), np.nan)
    mkt = np.full(len(idx), np.nan)
    b = mm = None
    for i, T in enumerate(idx):
        if i % C.REFIT_EVERY == 0:
            b = fit_alpha(p, T, h)
            mm = fit_market(p, T, h)
        alpha[i] = p.F[T] @ b
        mkt[i] = mm.predict(p.M.iloc[T])
    sig = p.sigma_d.values[idx] * math.sqrt(h)
    mu = p.beta.values[idx] * mkt[:, None] + alpha * sig          # 分數是標準化單位，乘回個股波動才是報酬
    real = p.fwd[h][idx]

    ok = np.isfinite(alpha) & np.isfinite(real)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(ok, real, np.nan), axis=1, keepdims=True)
    beat = real > med
    slope = 0.0
    if ok.sum() > 2000:
        _, slope = fit_logistic(alpha[ok], beat[ok].astype(float))
        slope = max(slope, 0.0)          # 樣本外沒有鑑別力時斜率歸零，信心值全部回到 50
    alpha_sd = float(np.nanstd(alpha[ok])) if ok.any() else 0.05

    pw = prob_win(alpha, slope)
    conf = np.maximum(pw, 1 - pw) * 100
    rel_hit = ((alpha > 0) == beat) & ok
    hit = (np.sign(mu) == np.sign(real)) & ok
    matured = ok.any(axis=1)
    ics = np.array([rank_ic(alpha[i], p.y[h][idx[i]]) for i in range(len(idx))])
    fin = np.isfinite(ics)

    top_ret, bot_ret, uni_ret = [], [], []
    for i in np.where(matured)[0]:
        order = np.argsort(np.where(ok[i], -alpha[i], np.inf))
        nv = int(ok[i].sum())
        top_ret.append(real[i][order[: C.TOP_N]].mean())
        bot_ret.append(real[i][order[nv - C.TOP_N: nv]].mean())
        uni_ret.append(real[i][ok[i]].mean())
    top_ret, bot_ret, uni_ret = map(np.array, (top_ret, bot_ret, uni_ret))
    mreal = p.mfwd[h][idx]
    mok = np.isfinite(mreal)

    buckets = []
    for lo, hi in ((50, 52), (52, 54), (54, 56), (56, 101)):
        m = ok & (conf >= lo) & (conf < hi)
        if m.sum() >= 50:
            buckets.append({"range": f"{lo}–{min(hi, 100)}", "n": int(m.sum()),
                            "hit": round(float(rel_hit[m].mean()) * 100, 1), "avg_conf": round(float(conf[m].mean()), 1)})
    hi_conf = ok & (conf >= C.HICONF)
    half = int(fin.sum()) // 2
    icv = ics[fin]

    def pct(x, nd=1):
        return round(float(x) * 100, nd)

    stats = {
        "days": int(matured.sum()), "samples": int(ok.sum()),
        "from": f"{p.dates[idx[0]]:%Y-%m-%d}",
        "to": f"{p.dates[idx[matured][-1]]:%Y-%m-%d}" if matured.any() else None,
        "rel_hit": pct(rel_hit[ok].mean()) if ok.any() else None,
        "rel_hit_hiconf": pct(rel_hit[hi_conf].mean()) if hi_conf.sum() > 50 else None,
        "hiconf_share": pct(hi_conf.sum() / max(ok.sum(), 1)),
        "hit_rate": pct(hit[ok].mean()) if ok.any() else None,
        "up_share": pct((real[ok] > 0).mean()) if ok.any() else None,
        "ic": round(float(icv.mean()), 4) if len(icv) else None,
        "ic_h1": round(float(icv[:half].mean()), 4) if half else None,
        "ic_h2": round(float(icv[half:].mean()), 4) if half else None,
        "ic_pos_days": pct((icv > 0).mean()) if len(icv) else None,
        "top_ret": pct(top_ret.mean(), 2) if len(top_ret) else None,
        "bot_ret": pct(bot_ret.mean(), 2) if len(bot_ret) else None,
        "uni_ret": pct(uni_ret.mean(), 2) if len(uni_ret) else None,
        "top_win": pct((top_ret > uni_ret).mean()) if len(top_ret) else None,
        "mkt_hit": pct((np.sign(mkt[mok]) == np.sign(mreal[mok])).mean()) if mok.any() else None,
        "mkt_up_share": pct((mreal[mok] > 0).mean()) if mok.any() else None,
        "slope": round(slope, 3), "buckets": buckets,
    }
    return Backtest(idx, alpha, mkt, mu, slope, alpha_sd, stats)


# ── 最新一天的預測 ────────────────────────────────────────────────────────

@dataclass
class Live:
    coef: dict[int, np.ndarray]            # 各天期的因子係數
    market: dict[int, MarketModel]
    mkt_mu: dict[int, float]
    alpha: dict[int, np.ndarray]           # 各天期每檔股票的相對強度分數（標準化單位）
    contrib: np.ndarray                    # [股票, 因子] 15 日分數的因子貢獻
    prev_coef: np.ndarray | None           # 20 個交易日前的係數，用來顯示權重變化


def fit_live(p: Panel) -> Live:
    T = len(p.dates) - 1
    coef, market, mkt_mu, alpha = {}, {}, {}, {}
    for h in C.HORIZONS:
        coef[h] = fit_alpha(p, T, h)
        market[h] = fit_market(p, T, h)
        mkt_mu[h] = market[h].predict(p.M.iloc[T])
        alpha[h] = np.nan_to_num(p.F[T]) @ coef[h]
    contrib = np.nan_to_num(p.F[T]) * coef[C.HORIZON][None, :]
    prev = fit_alpha(p, T - 20, C.HORIZON) if T - 20 - C.HORIZON > C.MIN_HISTORY_DAYS else None
    return Live(coef, market, mkt_mu, alpha, contrib, prev)
