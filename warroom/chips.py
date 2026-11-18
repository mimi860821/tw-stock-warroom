"""籌碼面：三大法人買賣超、連續買賣超天數、融資融券變化。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _streak(x: np.ndarray) -> int:
    """由最近一天往回數，連續同方向（買超為正、賣超為負）的天數；正數＝連買、負數＝連賣。"""
    if len(x) == 0 or x[-1] == 0:
        return 0
    sign, n = np.sign(x[-1]), 0
    for v in x[::-1]:
        if np.sign(v) != sign:
            break
        n += 1
    return int(sign * n)


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return ((s - s.mean()) / sd).clip(-3, 3).fillna(0.0) if sd and np.isfinite(sd) else pd.Series(0.0, index=s.index)


def scores(chips: pd.DataFrame, margin: pd.DataFrame, volume: pd.DataFrame, codes: list[str]):
    """回傳 (籌碼綜合分數 z, 每檔明細, 法人資料天數, 融資資料天數)。

    分數 = 近 5 日「外資 + 1.5×投信」買賣超占均量的比例（越高越好）
          − MARGIN_WEIGHT × 近 5 日融資增加占均量的比例（散戶融資追價通常是反向指標）
    """
    v20 = volume.iloc[-20:].mean()
    detail: dict[str, dict] = {c: {} for c in codes}
    inst_z = pd.Series(0.0, index=codes)
    n_chip = 0
    if not chips.empty:
        days = sorted(chips["date"].unique())
        n_chip = len(days)
        last5 = chips[chips["date"].isin(days[-5:])].groupby("code")[["foreign", "trust", "dealer"]].sum()
        raw = ((last5["foreign"] + 1.5 * last5["trust"]) / (v20 * min(len(days), 5))).reindex(codes)
        if len(days) >= 3:
            inst_z = _z(raw)
        sub = chips[chips["code"].isin(codes)]
        for code, g in sub.groupby("code"):
            g = g.sort_values("date")
            f, t, d = g["foreign"].values, g["trust"].values, g["dealer"].values
            tail = g.iloc[-10:]
            detail[code].update({
                "days": [[f"{dd:%m-%d}", int(ff / 1000), int(tt / 1000), int(dl / 1000)]
                         for dd, ff, tt, dl in zip(tail["date"], tail["foreign"], tail["trust"], tail["dealer"])],
                "f5": int(last5["foreign"].get(code, 0) / 1000), "t5": int(last5["trust"].get(code, 0) / 1000),
                "d5": int(last5["dealer"].get(code, 0) / 1000),
                "f20": int(f.sum() / 1000), "t20": int(t.sum() / 1000),
                "f_streak": _streak(f), "t_streak": _streak(t),
            })
    mar_z = pd.Series(0.0, index=codes)
    n_mar = 0
    if not margin.empty:
        days = sorted(margin["date"].unique())
        n_mar = len(days)
        piv = margin.pivot_table(index="date", columns="code", values="margin").reindex(columns=codes)
        sht = margin.pivot_table(index="date", columns="code", values="short").reindex(columns=codes)
        if len(days) >= 2:
            k = min(5, len(days) - 1)
            chg = piv.iloc[-1] - piv.iloc[-1 - k]                       # 張
            mar_z = _z(-(chg * 1000) / (v20 * k))
            chg_pct = chg / piv.iloc[-1 - k].replace(0, np.nan) * 100
        else:
            chg, chg_pct = pd.Series(np.nan, index=codes), pd.Series(np.nan, index=codes)
        for c in codes:
            m_now, s_now = piv[c].iloc[-1], sht[c].iloc[-1]
            if not np.isfinite(m_now):
                continue
            detail[c].update({
                "margin": int(m_now), "short": int(s_now) if np.isfinite(s_now) else 0,
                "margin_chg": int(chg[c]) if np.isfinite(chg[c]) else None,
                "margin_chg_pct": round(float(chg_pct[c]), 1) if np.isfinite(chg_pct[c]) else None,
                "short_ratio": round(float(s_now / m_now * 100), 1) if m_now > 0 and np.isfinite(s_now) else None,
                "margin_hist": [[f"{d:%m-%d}", int(a) if np.isfinite(a) else None, int(b) if np.isfinite(b) else None]
                                for d, a, b in zip(piv.index[-10:], piv[c].iloc[-10:], sht[c].iloc[-10:])],
            })
    score = (inst_z + C.MARGIN_WEIGHT * mar_z) / (1 + C.MARGIN_WEIGHT)
    for c in codes:
        detail[c]["inst_z"] = round(float(inst_z[c]), 2)
        detail[c]["margin_z"] = round(float(mar_z[c]), 2)
    return score, {c: d for c, d in detail.items() if d}, n_chip, n_mar


def describe(d: dict, z: float) -> str:
    """籌碼判斷依據的一句話。"""
    bits = []
    if "f5" in d:
        bits.append(f"近 5 日外資 {d['f5']:+,} 張、投信 {d['t5']:+,} 張")
        for k, who in (("f_streak", "外資"), ("t_streak", "投信")):
            s = d.get(k, 0)
            if abs(s) >= 3:
                bits.append(f"{who}連{'買' if s > 0 else '賣'} {abs(s)} 天")
    if d.get("margin_chg") is not None:
        bits.append(f"融資近 5 日 {d['margin_chg']:+,} 張（{d['margin_chg_pct']:+.1f}%）" if d.get("margin_chg_pct") is not None
                    else f"融資近 5 日 {d['margin_chg']:+,} 張")
    if d.get("short_ratio") and d["short_ratio"] >= 20:
        bits.append(f"券資比 {d['short_ratio']:.0f}%（偏高，有軋空可能）")
    bits.append("籌碼綜合排名" + ("偏前" if z > 0 else "偏後"))
    return "；".join(bits)
