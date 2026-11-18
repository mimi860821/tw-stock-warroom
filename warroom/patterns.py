"""K 線型態與技術訊號偵測，並用歷史資料統計每種型態「出現後 15 個交易日」的實際表現。

為什麼要統計：教科書說「錘子線看漲」，但在台股、在這幾年、在這個股票池裡是否成立，
要看實際數字。每個訊號都附上出現次數、平均超額報酬與勝率，讓你判斷該不該相信它。
統計只用已經過完 15 天的歷史事件，所以拿來輔助今天的判斷不會偷看未來。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# 鍵 → (名稱, 預期方向 +1 看多 / −1 看空, 說明)
PATTERNS = {
    "long_red": ("長紅 K", 1, "實體大於當日振幅六成、漲幅明顯，多方強勢表態"),
    "long_black": ("長黑 K", -1, "實體大於當日振幅六成、跌幅明顯，空方強勢表態"),
    "hammer": ("錘子線", 1, "下跌後出現長下影線、小實體，代表低檔有買盤承接"),
    "hanging": ("吊人線", -1, "上漲後出現長下影線、小實體，代表高檔賣壓開始出現"),
    "inv_hammer": ("倒鎚線", 1, "下跌後出現長上影線、小實體，多方開始試探反攻"),
    "shooting": ("流星線", -1, "上漲後出現長上影線、小實體，衝高後被賣回"),
    "bull_engulf": ("多頭吞噬", 1, "下跌後一根紅 K 完全包住前一根黑 K 的實體"),
    "bear_engulf": ("空頭吞噬", -1, "上漲後一根黑 K 完全包住前一根紅 K 的實體"),
    "morning_star": ("晨星", 1, "長黑 → 小實體 → 長紅收復一半以上，經典底部反轉"),
    "evening_star": ("夜星", -1, "長紅 → 小實體 → 長黑跌回一半以上，經典頭部反轉"),
    "three_white": ("紅三兵", 1, "連續三根逐步墊高的紅 K，多方穩健推進"),
    "three_black": ("黑三兵", -1, "連續三根逐步走低的黑 K，空方穩健推進"),
    "doji_bottom": ("低檔十字", 1, "跌深後出現十字線，多空拉鋸、可能止跌"),
    "doji_top": ("高檔十字", -1, "漲多後出現十字線，多空拉鋸、可能轉弱"),
    "gap_up": ("向上跳空缺口", 1, "最低價高於前一日最高價，買盤急切"),
    "gap_down": ("向下跳空缺口", -1, "最高價低於前一日最低價，賣壓急切"),
    "vol_breakout": ("帶量突破 20 日高點", 1, "收盤創 20 日新高且成交量達均量 2 倍以上"),
    "vol_breakdown": ("帶量跌破 20 日低點", -1, "收盤創 20 日新低且成交量達均量 2 倍以上"),
    "golden_cross": ("均線黃金交叉", 1, "5 日均線由下往上穿越 20 日均線"),
    "death_cross": ("均線死亡交叉", -1, "5 日均線由上往下穿越 20 日均線"),
    "kd_golden_low": ("KD 低檔黃金交叉", 1, "K 值在 25 以下由下往上穿越 D 值"),
    "kd_dead_high": ("KD 高檔死亡交叉", -1, "K 值在 75 以上由上往下穿越 D 值"),
    "macd_pos": ("MACD 柱翻紅", 1, "MACD 柱狀體由負轉正，動能轉多"),
    "macd_neg": ("MACD 柱翻綠", -1, "MACD 柱狀體由正轉負，動能轉空"),
    "above_ma60": ("站上季線", 1, "收盤由季線下方站回季線之上"),
    "below_ma60": ("跌破季線", -1, "收盤由季線上方跌破季線"),
    "limit_up": ("漲停", 1, "收盤漲幅接近 10% 漲停限制"),
    "limit_down": ("跌停", -1, "收盤跌幅接近 10% 跌停限制"),
}


def detect(o: pd.DataFrame, h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, v: pd.DataFrame,
           raw_close: pd.DataFrame, k: pd.DataFrame, d: pd.DataFrame, macd: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """o/h/l/c 為還原權息後的價格（避免除息造成假缺口），raw_close 用來判斷漲跌停。回傳 {型態: 布林寬表}。"""
    body = c - o
    ab = body.abs()
    rng = (h - l).replace(0, np.nan)
    up_sh = h - np.maximum(o, c)
    lo_sh = np.minimum(o, c) - l
    ret = c.pct_change(fill_method=None)
    sig = ret.rolling(20, min_periods=10).std()
    prior = c.shift(1) / c.shift(6) - 1                       # 型態出現前 5 日的走勢
    avgb = ab.rolling(10, min_periods=5).mean().shift(1)
    v = v.where(v > 0)
    v20 = v.rolling(20, min_periods=10).mean().shift(1)
    small = ab <= 0.35 * rng
    big_up = (body / o >= np.maximum(0.035, 1.5 * sig)) & (ab >= 0.6 * rng)
    big_dn = (-body / o >= np.maximum(0.035, 1.5 * sig)) & (ab >= 0.6 * rng)
    hammer_shape = (lo_sh >= 2 * ab) & (up_sh <= 0.25 * rng) & small
    inv_shape = (up_sh >= 2 * ab) & (lo_sh <= 0.25 * rng) & small
    pb, pab = body.shift(1), ab.shift(1)
    ma5, ma20, ma60 = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    doji = (ab <= 0.1 * rng) & (rng / c >= 0.01)
    raw_ret = raw_close / raw_close.shift(1) - 1

    def up_white(i):    # 第 i 天前的紅 K，實體夠大、收在高檔
        b = body.shift(i)
        return (b > 0) & (ab.shift(i) >= 0.5 * avgb) & (up_sh.shift(i) <= 0.35 * ab.shift(i))

    def dn_black(i):
        b = body.shift(i)
        return (b < 0) & (ab.shift(i) >= 0.5 * avgb) & (lo_sh.shift(i) <= 0.35 * ab.shift(i))

    out = {
        "long_red": big_up,
        "long_black": big_dn,
        "hammer": hammer_shape & (prior <= -0.03),
        "hanging": hammer_shape & (prior >= 0.03),
        "inv_hammer": inv_shape & (prior <= -0.03),
        "shooting": inv_shape & (prior >= 0.03),
        "bull_engulf": (pb < 0) & (body > 0) & (o <= c.shift(1)) & (c >= o.shift(1)) & (ab > pab) & (prior <= -0.02),
        "bear_engulf": (pb > 0) & (body < 0) & (o >= c.shift(1)) & (c <= o.shift(1)) & (ab > pab) & (prior >= 0.02),
        "morning_star": (body.shift(2) < -avgb) & (ab.shift(1) <= 0.5 * avgb) & (body > 0)
                        & (c >= (o.shift(2) + c.shift(2)) / 2) & (c.shift(2) / c.shift(7) - 1 <= -0.03),
        "evening_star": (body.shift(2) > avgb) & (ab.shift(1) <= 0.5 * avgb) & (body < 0)
                        & (c <= (o.shift(2) + c.shift(2)) / 2) & (c.shift(2) / c.shift(7) - 1 >= 0.03),
        "three_white": up_white(0) & up_white(1) & up_white(2) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
                       & (o >= o.shift(1)) & (o.shift(1) >= o.shift(2)),
        "three_black": dn_black(0) & dn_black(1) & dn_black(2) & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
                       & (o <= o.shift(1)) & (o.shift(1) <= o.shift(2)),
        "doji_bottom": doji & (prior <= -0.05),
        "doji_top": doji & (prior >= 0.05),
        "gap_up": l > h.shift(1) * 1.005,
        "gap_down": h < l.shift(1) * 0.995,
        "vol_breakout": (c > h.shift(1).rolling(20).max()) & (v >= 2 * v20),
        "vol_breakdown": (c < l.shift(1).rolling(20).min()) & (v >= 2 * v20),
        "golden_cross": (ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1)),
        "death_cross": (ma5 < ma20) & (ma5.shift(1) >= ma20.shift(1)),
        "kd_golden_low": (k > d) & (k.shift(1) <= d.shift(1)) & (d < 25),
        "kd_dead_high": (k < d) & (k.shift(1) >= d.shift(1)) & (d > 75),
        "macd_pos": (macd > 0) & (macd.shift(1) <= 0),
        "macd_neg": (macd < 0) & (macd.shift(1) >= 0),
        "above_ma60": (c > ma60) & (c.shift(1) <= ma60.shift(1)),
        "below_ma60": (c < ma60) & (c.shift(1) >= ma60.shift(1)),
        "limit_up": raw_ret >= 0.095,
        "limit_down": raw_ret <= -0.095,
    }
    return {kk: vv.fillna(False).astype(bool) for kk, vv in out.items()}


def event_stats(sig: dict[str, pd.DataFrame], fwd: np.ndarray, horizon: int = C.HORIZON) -> dict[str, dict]:
    """每種型態出現後 horizon 日的表現。只用已滿 horizon 天的事件。

    * mean：相對當天股票池「平均」的超額報酬（%）
    * win：贏過當天股票池「中位數」的比例（%），基準 50
    * t：以「日」為單位計算的 t 值，並把有效天數除以 horizon/2 來修正 15 日報酬互相重疊的問題（保守）
    """
    with np.errstate(invalid="ignore"):
        ex = fwd - np.nanmean(fwd, axis=1, keepdims=True)
        beat = fwd > np.nanmedian(fwd, axis=1, keepdims=True)
    fin = np.isfinite(ex)
    out = {}
    for key, df in sig.items():
        m = df.values & fin
        n = int(m.sum())
        if n == 0:
            out[key] = {"n": 0, "mean": None, "win": None, "t": None, "shrunk": 0.0}
            continue
        mean = float(ex[m].mean())
        rows = np.where(m.any(axis=1))[0]
        day_means = np.array([ex[r][m[r]].mean() for r in rows])
        n_eff = max(len(day_means) / (horizon / 2), 1.0)
        sd = float(day_means.std(ddof=1)) if len(day_means) > 1 else np.inf
        t = float(day_means.mean() / (sd / np.sqrt(n_eff))) if sd > 0 and np.isfinite(sd) else 0.0
        out[key] = {"n": n, "days": int(len(rows)), "mean": round(mean * 100, 2),
                    "win": round(float(beat[m].mean()) * 100, 1), "t": round(t, 2),
                    # 只有統計上夠明確的型態才拿來調整預測：|t| ≤ 1 不用，|t| ≥ 3 全額採用
                    "shrunk": mean * float(np.clip((abs(t) - 1) / 2, 0, 1))}
    return out


def verdict(st: dict, direction: int) -> str:
    """一句話：這個型態在台股的歷史表現是否支持教科書的說法，以及是否參與預測（與 shrunk 的門檻一致）。"""
    if not st or st["n"] < 30:
        return "歷史樣本不足 30 次，不參與預測"
    m, t = st["mean"], abs(st.get("t") or 0)
    if t <= 1:
        return "出現後的表現與股票池平均無明顯差異，不參與預測"
    agree = (m > 0) == (direction > 0)
    word = "支持教科書的解讀" if agree else f"與教科書相反（出現後反而{'較強' if m > 0 else '較弱'}）"
    if t >= 3:
        return f"歷史統計明確{word}，全額參與預測"
    if t >= 2:
        return f"歷史統計{word}，大部分參與預測"
    return f"歷史統計傾向{word}，但力道偏弱，小幅參與預測"


def recent(sig: dict[str, pd.DataFrame], code: str, dates: pd.DatetimeIndex, lookback: int = 3) -> list[tuple[str, int]]:
    """最近 lookback 個交易日內出現的型態：[(鍵, 幾天前)]，0 = 今天。"""
    out = []
    for key, df in sig.items():
        col = df[code].values[-lookback:]
        for i, hit in enumerate(col[::-1]):
            if hit:
                out.append((key, i))
                break
    return out
