"""排行榜：全市場（上市＋上櫃的個股與 ETF）的當日行情、法人、融資，加上股票池內的歷史指標。

輸出一張精簡的大表給網頁，排序與篩選都在瀏覽器端做，所以新增排行不必改後端。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 網頁端欄位（順序即陣列位置）
FIELDS = ["c", "n", "m", "i", "k", "px", "chg", "pct", "vol", "val", "turn",
          "f", "t", "d", "fs", "ts", "js", "mc1", "mc5", "sr", "pe", "dy",
          "vr", "r5", "r20", "h52", "l52", "u"]


def _kind(code: str) -> str | None:
    if code.startswith("00"):
        return "etf"
    if len(code) == 4 and code.isdigit():
        return "stock"
    return None                                     # 權證、特別股等不列入


def _streak(mat: np.ndarray) -> np.ndarray:
    """mat: [日期, 股票] 布林。由最後一天往回數連續為真的天數。"""
    out = np.zeros(mat.shape[1], dtype=int)
    alive = np.ones(mat.shape[1], dtype=bool)
    for row in mat[::-1]:
        alive &= row
        out += alive
    return out


def _signed_streak(x: np.ndarray) -> np.ndarray:
    """x: [日期, 股票] 買賣超。正數＝連買天數、負數＝連賣天數。"""
    buy, sell = _streak(np.nan_to_num(x) > 0), _streak(np.nan_to_num(x) < 0)
    return np.where(buy > 0, buy, -sell)


def build(snap: pd.DataFrame, chips: pd.DataFrame, margin: pd.DataFrame, val_today: dict, fin: dict,
          industries: dict, wide: dict, universe: set) -> dict:
    """snap：全市場當日快照（index 為代號）。回傳 {"date", "fields", "rows", "counts"}。"""
    # 整張表對齊到交易所快照的日期（中午執行時快照還是前一天，法人與歷史指標也要用同一天）
    roc = max((d for d in snap["date"].dropna().astype(str) if d.strip()), default=None)
    d_snap = pd.Timestamp(int(roc[:-4]) + 1911, int(roc[-4:-2]), int(roc[-2:])) if roc else None
    if d_snap is not None:
        chips = chips[chips["date"] <= d_snap] if not chips.empty else chips
        margin = margin[margin["date"] <= d_snap] if not margin.empty else margin
        wide = {k: v.loc[:d_snap] for k, v in wide.items()}
    snap = snap[[_kind(c) is not None for c in snap.index]]
    snap = snap[snap["close"].notna() & (snap["close"] > 0)]
    codes = list(snap.index)
    df = pd.DataFrame(index=codes)
    df["n"] = snap["name"]
    df["m"] = np.where(snap["market"] == "TWSE", "上市", "上櫃")
    df["k"] = [_kind(c) for c in codes]
    df["i"] = [industries.get(c, "ETF" if k == "etf" else "其他") for c, k in zip(codes, df["k"])]
    df["px"] = snap["close"]
    df["chg"] = snap["change"]
    prev = snap["close"] - snap["change"]
    df["pct"] = (snap["change"] / prev.where(prev > 0) * 100).round(2)
    df["vol"] = (snap["volume"] / 1000).round(0)                 # 張
    df["val"] = (snap["value"] / 1e8).round(2)                   # 億元

    # 週轉率：成交股數 ÷ 流通股數（股本 ÷ 面額 10 元）
    cap = pd.Series({c: (fin.get(c) or {}).get("capital") for c in codes}, dtype=float) * 1000
    if "capital" in snap:
        cap = cap.fillna(snap["capital"])
    shares = cap / 10
    df["turn"] = (snap["volume"] / shares.where(shares > 0) * 100).round(2)

    # 法人：最新一天的買賣超（張）與連續天數
    for k in ("f", "t", "d", "fs", "ts", "js"):
        df[k] = np.nan
    if not chips.empty:
        last = chips["date"].max()
        today = chips[chips["date"] == last].set_index("code")
        for k, col in (("f", "foreign"), ("t", "trust"), ("d", "dealer")):
            df[k] = (today[col].reindex(codes) / 1000).round(0)
        piv_f = chips.pivot_table(index="date", columns="code", values="foreign").reindex(columns=codes).sort_index()
        piv_t = chips.pivot_table(index="date", columns="code", values="trust").reindex(columns=codes).sort_index()
        df["fs"] = _signed_streak(piv_f.values)
        df["ts"] = _signed_streak(piv_t.values)
        df["js"] = _streak((np.nan_to_num(piv_f.values) > 0) & (np.nan_to_num(piv_t.values) > 0))
        has = df["f"].notna()
        for k in ("fs", "ts", "js"):
            df.loc[~has, k] = np.nan

    # 融資：單日與 5 日增減（張）、券資比
    df["mc1"] = df["mc5"] = df["sr"] = np.nan
    if not margin.empty:
        pm = margin.pivot_table(index="date", columns="code", values="margin").reindex(columns=codes).sort_index()
        ps = margin.pivot_table(index="date", columns="code", values="short").reindex(columns=codes).sort_index()
        if len(pm) >= 2:
            df["mc1"] = pm.iloc[-1] - pm.iloc[-2]
        if len(pm) >= 6:
            df["mc5"] = pm.iloc[-1] - pm.iloc[-6]
        df["sr"] = (ps.iloc[-1] / pm.iloc[-1].where(pm.iloc[-1] > 0) * 100).round(1)

    # 估值（全市場最新）
    df["pe"] = [(val_today.get(c) or (None,))[0] for c in codes]
    df["dy"] = [(val_today.get(c) or (None, None, None))[2] for c in codes]

    # 股票池內才有的歷史指標：量比、5／20 日漲幅、52 週新高新低
    for k in ("vr", "r5", "r20", "h52", "l52"):
        df[k] = np.nan
    adj, vol, hi, lo = wide["adj"], wide["volume"], wide["high"], wide["low"]
    ratio = adj / wide["close"]
    inu = [c for c in codes if c in adj.columns]
    if inu:
        v = vol[inu].where(vol[inu] > 0)
        df.loc[inu, "vr"] = (v.iloc[-1] / v.iloc[-21:-1].mean()).round(2)
        df.loc[inu, "r5"] = ((adj[inu].iloc[-1] / adj[inu].iloc[-6] - 1) * 100).round(2)
        df.loc[inu, "r20"] = ((adj[inu].iloc[-1] / adj[inu].iloc[-21] - 1) * 100).round(2)
        h250 = (hi[inu] * ratio[inu]).iloc[-250:].max()
        l250 = (lo[inu] * ratio[inu]).iloc[-250:].min()
        df.loc[inu, "h52"] = (adj[inu].iloc[-1] >= h250 * 0.999).astype(float)
        df.loc[inu, "l52"] = (adj[inu].iloc[-1] <= l250 * 1.001).astype(float)
    df["u"] = [1 if c in universe else 0 for c in codes]
    df["c"] = codes

    def cell(x):
        if isinstance(x, (float, np.floating)):
            return None if not np.isfinite(x) else (int(x) if float(x).is_integer() and abs(x) < 1e12 else float(x))
        if isinstance(x, np.integer):
            return int(x)
        return x

    rows = [[cell(r[f]) for f in FIELDS] for _, r in df[FIELDS].iterrows()]
    return {"date": f"{d_snap:%Y-%m-%d}" if d_snap is not None else None, "chip_date": f"{chips['date'].max():%Y-%m-%d}" if not chips.empty else None,
            "fields": FIELDS, "rows": rows,
            "counts": {"stock": int((df["k"] == "stock").sum()), "etf": int((df["k"] == "etf").sum()),
                       "universe": int(df["u"].sum())}}
