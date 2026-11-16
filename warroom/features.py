"""技術指標。全部用「日期 × 股票」的寬表向量化計算。"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 模型使用的因子：鍵 → (中文名稱, 分類, 簡稱)
FACTORS = {
    "rev1": ("單日漲跌（短線反轉）", "動能", "單日漲跌"),
    "mom5": ("近 5 日漲幅", "動能", "5 日動能"),
    "mom20": ("近 20 日漲幅", "動能", "20 日動能"),
    "mom60": ("近 60 日漲幅", "動能", "60 日動能"),
    "mom120": ("近 120 日漲幅", "動能", "120 日動能"),
    "ma20gap": ("股價與月線乖離", "均線", "月線乖離"),
    "ma60gap": ("股價與季線乖離", "均線", "季線乖離"),
    "matrend": ("月線相對季線（多空排列）", "均線", "均線排列"),
    "rsi": ("RSI(14)", "擺盪指標", "RSI"),
    "macd": ("MACD 柱狀體", "擺盪指標", "MACD"),
    "kd": ("KD 差值 (K−D)", "擺盪指標", "KD"),
    "bbp": ("布林通道位置", "擺盪指標", "布林位置"),
    "volr": ("短期量能變化（5 日 / 20 日均量）", "量能", "短期量能"),
    "turn": ("中期量能變化（20 日 / 120 日均量）", "量能", "中期量能"),
    "amihud": ("流動性（單位成交金額造成的價格變動）", "量能", "流動性"),
    "hi60": ("距 60 日高點", "位置", "60 日高點"),
    "hi250": ("距一年高點", "位置", "一年高點"),
    "intraday": ("近 20 日盤中累計漲幅（收盤相對開盤）", "動能", "盤中動能"),
}
FACTOR_KEYS = list(FACTORS)


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 * up / (up + dn).replace(0, np.nan)).where(up.notna() & dn.notna())      # 全漲 → 100，全跌 → 0


def stochastic(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 9):
    """台灣慣用的 KD(9,3,3)。"""
    lo, hi = low.rolling(n).min(), high.rolling(n).max()
    rsv = (100 * (close - lo) / (hi - lo).replace(0, np.nan)).fillna(50)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    return k, d


def macd_hist(close: pd.DataFrame) -> pd.DataFrame:
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return dif - dif.ewm(span=9, adjust=False).mean()


def compute(adj: pd.DataFrame, open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, volume: pd.DataFrame,
            value: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """價格皆為還原權息後的寬表，value 為成交金額。回傳 {因子: 寬表}（原始值，尚未標準化）。"""
    ret = adj.pct_change(fill_method=None)
    ma20, ma60 = adj.rolling(20).mean(), adj.rolling(60).mean()
    sd20 = adj.rolling(20).std()
    k, d = stochastic(high, low, adj)
    # 資料來源偶爾會有整天成交量為 0 的缺漏，當成缺值並允許視窗內少幾筆，避免一筆缺漏讓因子連續失效 20 天
    volume, value = volume.where(volume > 0), value.where(value > 0)
    v5, v20 = volume.rolling(5, min_periods=3).mean(), volume.rolling(20, min_periods=10).mean()
    f = {
        "rev1": ret,
        "mom5": adj.pct_change(5, fill_method=None),
        "mom20": adj.pct_change(20, fill_method=None),
        "mom60": adj.pct_change(60, fill_method=None),
        "mom120": adj.pct_change(120, fill_method=None),
        "ma20gap": adj / ma20 - 1,
        "ma60gap": adj / ma60 - 1,
        "matrend": ma20 / ma60 - 1,
        "rsi": rsi(adj),
        "macd": macd_hist(adj) / adj,
        "kd": k - d,
        "bbp": (adj - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan),
        "volr": np.log((v5 / v20).clip(lower=0.05)),
        "turn": np.log((v20 / volume.rolling(120, min_periods=60).mean()).clip(lower=0.05)),
        "amihud": np.log((ret.abs() / value).rolling(20, min_periods=10).mean().replace(0, np.nan)),
        "hi60": adj / adj.rolling(60).max() - 1,
        "hi250": adj / adj.rolling(250, min_periods=120).max() - 1,
        "intraday": (adj / open_ - 1).rolling(20).sum(),
    }
    f["_k"], f["_d"] = k, d          # 僅供顯示，不進模型
    f["_ma20"], f["_ma60"] = ma20, ma60
    return f


def zscore_cs(x: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """每天對所有股票做橫斷面標準化：讓模型比較的是「相對其他股票強弱」而非絕對數值。"""
    lo, hi = x.quantile(0.01, axis=1), x.quantile(0.99, axis=1)
    x = x.clip(lower=lo, upper=hi, axis=0)
    z = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)
    return z.clip(-clip, clip)


def market_features(taiex: pd.Series, breadth: pd.Series, us: dict[str, pd.Series]) -> pd.DataFrame:
    """大盤模型用的時間序列因子。us 的序列已對齊到台股交易日（只含當下已知的美股收盤）。"""
    ret = taiex.pct_change()
    t = taiex.to_frame("x")
    f = pd.DataFrame(index=taiex.index)
    f["m_mom5"] = taiex.pct_change(5)
    f["m_mom20"] = taiex.pct_change(20)
    f["m_mom60"] = taiex.pct_change(60)
    f["m_ma60gap"] = taiex / taiex.rolling(60).mean() - 1
    f["m_rsi"] = rsi(t)["x"]
    f["m_vola"] = ret.rolling(20).std()
    f["m_breadth"] = breadth
    if "^SOX" in us:
        f["u_sox20"] = us["^SOX"].pct_change(20)
    if "^GSPC" in us:
        f["u_spx20"] = us["^GSPC"].pct_change(20)
    if "^VIX" in us:
        f["u_vix"] = us["^VIX"]
        f["u_vixchg5"] = us["^VIX"].pct_change(5)
    return f


MARKET_FACTORS = {
    "m_mom5": "大盤近 5 日漲幅", "m_mom20": "大盤近 20 日漲幅", "m_mom60": "大盤近 60 日漲幅",
    "m_ma60gap": "大盤與季線乖離", "m_rsi": "大盤 RSI(14)", "m_vola": "大盤 20 日波動度",
    "m_breadth": "市場寬度（站上月線的個股比例）", "u_sox20": "費半近 20 日漲幅",
    "u_spx20": "S&P 500 近 20 日漲幅", "u_vix": "VIX 水位", "u_vixchg5": "VIX 近 5 日變化",
}
