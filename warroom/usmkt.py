"""前一晚美股對台股的連動分析。

台股第 t 個交易日開盤前，能看到的是「日期早於 t 的最後一個美股收盤」。把這個水位的日變化
稱為「美股隔夜衝擊」，用歷史資料迴歸出台股當日的平均反應，再估每檔股票對這個衝擊的敏感度。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def align_prev(us: pd.Series, tw_dates: pd.DatetimeIndex) -> pd.Series:
    """每個台股交易日 t，對應到日期嚴格早於 t 的最後一筆美股收盤。"""
    pos = np.searchsorted(us.index.values, tw_dates.values, side="left") - 1
    vals = np.where(pos >= 0, us.values[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=tw_dates)


def analyse(us_raw: dict[str, pd.DataFrame], taiex: pd.Series, adj: pd.DataFrame, tw_close: dict[str, float]) -> dict:
    """回傳美股面板資料、對大盤的預估影響，以及每檔股票的調整值。"""
    tw_last = taiex.index[-1]
    panel = []
    for sym, name in C.US_SYMBOLS.items():
        df = us_raw.get(sym)
        if df is None or len(df) < 6:
            continue
        c = df["close"]
        panel.append({"symbol": sym, "name": name, "date": f"{c.index[-1]:%Y-%m-%d}",
                      "close": round(float(c.iloc[-1]), 2),
                      "chg": round(float(c.iloc[-1] / c.iloc[-2] - 1) * 100, 2),
                      "chg5": round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 2),
                      "spark": [round(float(v), 2) for v in c.iloc[-30:]]})

    out = {"panel": panel, "available": False, "impact": 0.0, "stock_adj": {}, "beta_us": {}, "model": None,
           "adr": None, "note": ""}

    # 台積電 ADR 溢價：1 單位 ADR = 5 股普通股
    try:
        adr, fx = us_raw["TSM"]["close"].iloc[-1], us_raw["TWD=X"]["close"].iloc[-1]
        px = tw_close.get("2330")
        if px:
            out["adr"] = {"adr_twd": round(float(adr * fx / 5), 1), "tw_close": px,
                          "premium": round(float(adr * fx / 5 / px - 1) * 100, 2)}
    except (KeyError, IndexError):
        pass

    drivers = [s for s in ("^SOX", "^GSPC") if s in us_raw]
    if not drivers:
        out["note"] = "抓不到美股指數資料，本次未納入美股連動。"
        return out

    imp = pd.DataFrame({s: align_prev(us_raw[s]["close"], taiex.index).pct_change() for s in drivers})
    r = taiex.pct_change()
    df = pd.concat([imp, r.rename("tw")], axis=1).dropna().iloc[-500:]
    X = np.column_stack([np.ones(len(df)), df[drivers].values])
    coef, *_ = np.linalg.lstsq(X, df["tw"].values, rcond=None)
    fitted = X @ coef
    r2 = 1 - ((df["tw"].values - fitted) ** 2).sum() / ((df["tw"].values - df["tw"].mean()) ** 2).sum()
    out["model"] = {"drivers": {C.US_SYMBOLS[s]: round(float(b), 3) for s, b in zip(drivers, coef[1:])},
                    "r2": round(float(r2), 3), "n": len(df)}

    # 個股對「美股帶動的台股變動」的敏感度（近 250 日）
    u = pd.Series(df[drivers].values @ coef[1:], index=df.index).iloc[-250:]
    ret = adj.pct_change(fill_method=None).reindex(u.index)
    beta_us = ret.apply(lambda col: col.cov(u)) / u.var()
    beta_us = (0.7 * beta_us + 0.3).clip(0.0, 3.0).fillna(1.0)
    out["beta_us"] = {c: round(float(v), 2) for c, v in beta_us.items()}

    # 最新一晚的衝擊：只有在美股收盤日期不早於台股最後交易日時才算「已反映前」的新資訊
    nxt = {}
    for s in drivers:
        c = us_raw[s]["close"]
        if c.index[-1] >= tw_last:
            base = c[c.index < tw_last]
            if len(base):
                nxt[s] = float(c.iloc[-1] / base.iloc[-1] - 1)
    if len(nxt) == len(drivers):
        u_next = float(sum(coef[1 + i] * nxt[s] for i, s in enumerate(drivers)))
        out.update(available=True, impact=round(u_next, 5),
                   stock_adj={c: round(float(b * u_next), 5) for c, b in beta_us.items()},
                   moves={C.US_SYMBOLS[s]: round(v * 100, 2) for s, v in nxt.items()})
        parts = "、".join(f"{C.US_SYMBOLS[s]} {v * 100:+.2f}%" for s, v in nxt.items())
        out["note"] = (f"台股上次收盤後，{parts}。依近 {len(df)} 個交易日的歷史關係"
                       f"（解釋力 R²={r2:.2f}），估計對下一個台股交易日的影響約 {u_next * 100:+.2f}%。")
    else:
        out["note"] = ("台股最後一個交易日收盤後，美股尚未有新的收盤價，本次預測未加入美股隔夜調整；"
                       "可先參考下方期貨的即時方向，早上盤前更新時會自動納入。")
    return out
