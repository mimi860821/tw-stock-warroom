"""把數字翻成人看得懂的話：技術指標解讀、建議等級、判斷依據。"""
from __future__ import annotations

import numpy as np

from . import config as C
from .features import FACTORS

ACTION_TEXT = {
    "strong_buy": "相對強度排在股票池前 5%，且絕對預期報酬為正。可分批布局，並以停損參考價控管風險。",
    "buy": "相對強度排在前段，預期優於多數股票。可少量布局或續抱，等拉回支撐不破再加碼。",
    "neutral": "相對強度居中，或相對判斷與大盤方向互相抵銷，沒有明顯優勢。建議觀望；已持有者續抱並留意支撐價。",
    "sell": "相對強度排在後段，預期落後多數股票。不建議新進場；持有者可趁反彈減碼。",
    "strong_sell": "相對強度排在股票池後 5%，且絕對預期報酬為負。建議避開，持有者反彈至壓力區宜調節。",
}
ACTION_LABEL = {"strong_buy": "強力看多", "buy": "偏多", "neutral": "中立觀望", "sell": "偏空", "strong_sell": "強力看空"}


def action(rank: int, n: int, mu: float) -> str:
    """rank：相對強度排名（1 最強）。回傳建議等級。"""
    top, bottom = rank / n, (n - rank + 1) / n
    if top <= C.ACTION_STRONG_PCT:
        return "strong_buy" if mu > 0 else "buy"
    if top <= C.ACTION_MILD_PCT:
        return "buy" if mu > 0 else "neutral"
    if bottom <= C.ACTION_STRONG_PCT:
        return "strong_sell" if mu < 0 else "sell"
    if bottom <= C.ACTION_MILD_PCT:
        return "sell" if mu < 0 else "neutral"
    return "neutral"


def _fmt(key: str, v: float) -> str:
    if key == "rsi":
        return f"{v:.0f}"
    if key == "kd":
        return f"{v:+.1f}"
    if key == "bbp":
        return f"{v * 100:.0f}%"
    if key in ("volr", "turn"):
        return f"{np.exp(v):.2f} 倍"
    if key == "amihud":
        return "流動性較低" if v > 0 else "流動性較高"
    if key == "macd":
        return f"{v * 100:+.2f}%"
    return f"{v * 100:+.1f}%"


def factor_reasons(raw: dict[str, float], pct: dict[str, float], contrib: dict[str, float], coef: dict[str, float], top: int = 6) -> list[dict]:
    """alpha 的因子拆解。contrib 為對 15 日預期報酬的貢獻（小數）。"""
    out = []
    for k in sorted(contrib, key=lambda x: -abs(contrib[x]))[:top]:
        name, group, short = FACTORS[k]
        c = contrib[k]
        if abs(c) < 0.0003 or not np.isfinite(raw.get(k, np.nan)):
            continue
        rel = f"排除波動與 β 的影響後，高於 {min(pct[k], 99):.0f}% 的個股" if pct[k] >= 50 else f"排除波動與 β 的影響後，低於 {min(100 - pct[k], 99):.0f}% 的個股"
        rule = "此因子越高、後續相對越強" if coef[k] > 0 else "此因子越高、後續相對越弱"
        val = _fmt(k, pct[k] - 50 if k == "amihud" else raw[k])
        out.append({"group": group, "label": name, "short": short, "value": val, "contrib": round(c * 100, 2),
                    "text": f"{rel}；模型近兩年學到「{rule}」"})
    return out


def signals(x: dict) -> list[dict]:
    """傳統技術指標的解讀（給人對照用，與模型拆解分開呈現）。"""
    s = []
    px, ma20, ma60 = x["close"], x["ma20"], x["ma60"]
    if px > ma20 > ma60:
        s.append(("均線", "多頭排列", "bull", "股價站上月線，月線在季線之上，中期趨勢向上"))
    elif px < ma20 < ma60:
        s.append(("均線", "空頭排列", "bear", "股價跌破月線，月線在季線之下，中期趨勢向下"))
    elif px > ma60:
        s.append(("均線", "季線之上", "neutral", "股價仍在季線之上，但月線與股價糾結，短線整理"))
    else:
        s.append(("均線", "季線之下", "bear", "股價在季線之下，反彈容易遇到季線壓力"))
    r = x["rsi"]
    if r >= 70:
        s.append(("RSI(14)", f"{r:.0f}", "bear", "進入超買區，短線過熱、追高風險高"))
    elif r <= 30:
        s.append(("RSI(14)", f"{r:.0f}", "bull", "進入超賣區，短線跌深、有反彈機會"))
    else:
        s.append(("RSI(14)", f"{r:.0f}", "neutral", "介於 30–70 之間，沒有過熱或超賣"))
    k, d = x["k"], x["d"]
    if k > d and k < 80:
        s.append(("KD(9)", f"K {k:.0f} / D {d:.0f}", "bull", "K 值在 D 值之上（黃金交叉後），短線偏多"))
    elif k > d:
        s.append(("KD(9)", f"K {k:.0f} / D {d:.0f}", "neutral", "高檔鈍化，強勢但隨時可能拉回"))
    elif k < 20:
        s.append(("KD(9)", f"K {k:.0f} / D {d:.0f}", "neutral", "低檔區且 K 仍在 D 之下，等待黃金交叉確認止跌"))
    else:
        s.append(("KD(9)", f"K {k:.0f} / D {d:.0f}", "bear", "K 值在 D 值之下（死亡交叉後），短線偏空"))
    m, mp = x["macd"], x["macd_prev"]
    if m > 0:
        s.append(("MACD", "紅柱" + ("放大" if m > mp else "縮小"), "bull" if m > mp else "neutral",
                  "多方動能" + ("持續增強" if m > mp else "開始減弱")))
    else:
        s.append(("MACD", "綠柱" + ("放大" if m < mp else "縮小"), "bear" if m < mp else "neutral",
                  "空方動能" + ("持續增強" if m < mp else "開始減弱")))
    b = x["bbp"]
    if b >= 1:
        s.append(("布林通道", "突破上緣", "neutral", "強勢突破，但乖離偏大，留意拉回"))
    elif b <= 0:
        s.append(("布林通道", "跌破下緣", "neutral", "急跌超出常態區間，留意跌深反彈或趨勢轉空"))
    else:
        s.append(("布林通道", f"通道內 {b * 100:.0f}% 位置", "neutral", "在常態波動區間內（0%=下緣，100%=上緣）"))
    v = x["volr"]
    if v >= 1.5:
        s.append(("量能", f"{v:.2f} 倍", "neutral", "近 5 日均量明顯大於 20 日均量，放量代表多空交戰加劇"))
    elif v <= 0.7:
        s.append(("量能", f"{v:.2f} 倍", "neutral", "近 5 日量縮，觀望氣氛濃"))
    else:
        s.append(("量能", f"{v:.2f} 倍", "neutral", "量能與近一個月平均相當"))
    return [{"name": a, "value": b_, "view": c, "text": d_} for a, b_, c, d_ in s]


def summary_line(act: str, mu: float, conf: float, parts: dict) -> str:
    """一句話結論：最大的加分與扣分來源。"""
    named = {"market": "大盤走勢", "alpha": "個股技術面", "news": "新聞情緒", "chips": "法人籌碼", "us": "美股隔夜"}
    pos = [named[k] for k, v in sorted(parts.items(), key=lambda kv: -kv[1]) if v > 0.002][:2]
    neg = [named[k] for k, v in sorted(parts.items(), key=lambda kv: kv[1]) if v < -0.002][:2]
    bits = [f"未來 15 個交易日預期報酬 {mu * 100:+.1f}%，信心 {conf:.0f}（{'贏過' if act in ('strong_buy', 'buy') or (act == 'neutral' and parts.get('alpha', 0) >= 0) else '落後'}股票池中位數的機率）"]
    if pos:
        bits.append("主要加分：" + "、".join(pos))
    if neg:
        bits.append("主要扣分：" + "、".join(neg))
    return "；".join(bits) + "。"
