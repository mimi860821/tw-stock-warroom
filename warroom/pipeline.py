"""每日主流程：抓資料 → 算指標 → 訓練與預測 → 新聞 / 籌碼 / 美股 → 復盤 → 輸出網站用的 JSON。

執行：python -m warroom.pipeline
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from . import explain, features, model, news as newsmod, review, sources, usmkt

log = logging.getLogger("warroom")


def _clean(o):
    """轉成可寫入 JSON 的型別；NaN / inf 一律變 null。"""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return f"{o:%Y-%m-%d}"
    return o


def _write(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _read(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _pct(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x) * 100, nd)


def _path(close: float, mus: dict[int, float], jump: float, sigma_d: float) -> list[list]:
    """預期路徑：[第幾天, 中位, 下緣, 上緣]。上下緣為 ±1 個標準差（約 68% 的機率落在區間內）。"""
    xs = [0] + list(C.HORIZONS)
    ys = [0.0] + [mus[h] - jump for h in C.HORIZONS]
    out = []
    for t in range(C.HORIZON + 1):
        mid = float(np.interp(t, xs, ys)) + (jump if t >= 1 else 0.0)
        band = sigma_d * math.sqrt(t)
        out.append([t, round(close * (1 + mid), 2), round(close * (1 + mid - band), 2), round(close * (1 + mid + band), 2)])
    return out


def load_prices(now: datetime):
    snap = sources.fetch_snapshot()
    uni = sources.build_universe(snap, _read(C.DATA_DIR / "universe.json", []))
    industries = _read(C.DATA_DIR / "industries.json", {})
    industries.update(sources.fetch_industries())          # 來源偶爾逾時，所以合併進快取而不是每次重抓
    _write(C.DATA_DIR / "industries.json", dict(sorted(industries.items())))
    log.info("股票池 %d 檔，開始抓歷史股價…", len(uni))

    symbols = {c: sources.yahoo_symbol(c, r.market) for c, r in uni.iterrows()}
    symbols["^TWII"] = "^TWII"
    raw = sources.fetch_many(symbols)
    us_raw = sources.fetch_many({s: s for s in C.US_SYMBOLS})
    if "^TWII" not in raw:
        raise RuntimeError("抓不到加權指數 (^TWII)，中止")

    now_us = now.astimezone(C.TZ_US)
    for k in list(raw):
        raw[k] = sources.drop_partial_bar(raw[k], now, 9.0, 13.75)
    for k in list(us_raw):
        if "=" not in k:        # 期貨與匯率幾乎 24 小時交易，保留即時值當參考
            us_raw[k] = sources.drop_partial_bar(us_raw[k], now_us, 9.5, 16.25)

    tw = raw.pop("^TWII")
    dates = tw.index
    asof = dates[-1]
    keep = {}
    for c, df in raw.items():
        df = df.reindex(dates)
        if df["adj"].notna().sum() >= C.MIN_HISTORY_DAYS and np.isfinite(df["adj"].iloc[-1]):
            keep[c] = df
    dropped = sorted(set(uni.index) - set(keep))
    if dropped:
        log.info("略過 %d 檔（歷史不足、停牌或抓不到）：%s", len(dropped), " ".join(dropped[:20]))
    if len(keep) < 50:
        raise RuntimeError(f"只有 {len(keep)} 檔股票有足夠資料，中止（資料來源可能異常）")
    codes = sorted(keep)
    wide = {f: pd.DataFrame({c: keep[c][f] for c in codes}) for f in ("open", "high", "low", "close", "adj", "volume")}
    uni = uni.loc[codes]
    uni["industry"] = [industries.get(c, "ETF" if c.startswith("00") else "其他") for c in codes]
    _write(C.DATA_DIR / "universe.json", codes)
    return uni, wide, tw, us_raw, asof


def chip_scores(chips: pd.DataFrame, volume: pd.DataFrame, codes: list[str]):
    """近 5 日「外資 + 1.5×投信」買賣超占成交量的比例，橫斷面標準化。"""
    if chips.empty:
        return pd.Series(0.0, index=codes), {}, 0
    days = sorted(chips["date"].unique())
    last5 = chips[chips["date"].isin(days[-5:])].groupby("code")[["foreign", "trust", "dealer"]].sum()
    v20 = volume.iloc[-20:].mean()
    raw = ((last5["foreign"] + 1.5 * last5["trust"]) / (v20 * min(len(days), 5))).reindex(codes)
    z = ((raw - raw.mean()) / raw.std()).clip(-3, 3).fillna(0.0) if len(days) >= 3 and raw.std() > 0 else pd.Series(0.0, index=codes)
    detail = {}
    sub = chips[chips["code"].isin(codes) & chips["date"].isin(days[-10:])]
    for code, g in sub.groupby("code"):
        g = g.sort_values("date")
        detail[code] = {
            "days": [[f"{d:%m-%d}", int(f / 1000), int(t / 1000), int(dl / 1000)]
                     for d, f, t, dl in zip(g["date"], g["foreign"], g["trust"], g["dealer"])],
            "f5": int(last5["foreign"].get(code, 0) / 1000), "t5": int(last5["trust"].get(code, 0) / 1000),
            "d5": int(last5["dealer"].get(code, 0) / 1000),
        }
    return z, detail, len(days)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore", category=RuntimeWarning)      # 缺值日的平均／對數警告，計算上已處理
    now = datetime.now(C.TZ_TW)
    uni, wide, tw, us_raw, asof = load_prices(now)
    codes = list(uni.index)
    names = uni["name"].to_dict()
    adj, close = wide["adj"], wide["close"]
    ratio = adj / close
    taiex = tw["close"]
    asof_s = f"{asof:%Y-%m-%d}"
    log.info("資料基準日 %s，%d 檔股票，%d 個交易日", asof_s, len(codes), len(adj))

    # ── 因子與模型 ──
    F = features.compute(adj, wide["open"] * ratio, wide["high"] * ratio, wide["low"] * ratio, wide["volume"],
                         wide["volume"] * close)
    breadth = (adj > F["_ma20"]).sum(axis=1) / adj.notna().sum(axis=1)
    us_aligned = {s: usmkt.align_prev(df["close"], adj.index) for s, df in us_raw.items()}
    M = features.market_features(taiex, breadth, us_aligned)
    panel = model.build_panel(adj, taiex, F, M)
    bt = model.walk_forward(panel)
    live = model.fit_live(panel)
    log.info("回測：相對強弱命中率 %s%%，IC %s（前半 %s / 後半 %s），前 %d 名 %s%% vs 股票池 %s%%", bt.stats["rel_hit"],
             bt.stats["ic"], bt.stats["ic_h1"], bt.stats["ic_h2"], C.TOP_N, bt.stats["top_ret"], bt.stats["uni_ret"])

    # ── 首次執行：用樣本外回測補齊最近幾天的歷史預測，讓復盤一開始就有東西可看 ──
    H = C.HORIZON
    for i in range(max(0, len(bt.idx) - C.HINDCAST_DAYS - 1), len(bt.idx) - 1):
        d = f"{adj.index[bt.idx[i]]:%Y-%m-%d}"
        if (review.PRED_DIR / f"{d}.json").exists():
            continue
        p_win = model.prob_win(bt.alpha[i], bt.slope)
        m_sig = float(taiex.pct_change().ewm(span=60).std().iloc[bt.idx[i]] * math.sqrt(H))
        review.save_prediction(d, {
            "asof": d, "kind": "backtest",
            "market": {"mu": round(float(bt.mkt[i]), 5), "p_up": round(float(model.norm_cdf(bt.mkt[i] / m_sig)), 4)},
            "stocks": {c: [round(float(bt.mu[i, j]), 5), round(float(p_win[j]), 4), 0, 0, 0]
                       for j, c in enumerate(codes) if np.isfinite(bt.mu[i, j])},
        })

    # ── 疊加因子權重（依先前實際預測的復盤結果自動調整）──
    prior_rv = review.evaluate(review.load_predictions(), adj, taiex, names)
    weights = review.adapt_weights(prior_rv["overlay_ic"])
    _write(C.DATA_DIR / "weights.json", {"updated": asof_s, **weights})

    # ── 新聞 ──
    hours = {0: 72, 6: 48}.get(now.weekday(), 24)
    try:
        nz = newsmod.analyse(newsmod.fetch_all(hours, now), names, now)
    except Exception as e:      # 新聞抓不到不該讓整個流程失敗
        log.warning("新聞處理失敗：%s", e)
        nz = {"news": [], "stock_sent": {}, "market_sent": 0.0, "counts": {"total": 0, "pos": 0, "neg": 0}}
    mkt_news_adj = C.MARKET_NEWS_MAX * math.tanh(2 * nz["market_sent"]) * weights["news"]["weight"]

    # ── 籌碼 ──
    try:
        chips = sources.load_chips(list(adj.index[-20:]))
    except Exception as e:
        log.warning("籌碼資料失敗：%s", e)
        chips = pd.DataFrame(columns=["date", "code", "foreign", "trust", "dealer"])
    chip_z, chip_detail, chip_days = chip_scores(chips, wide["volume"], codes)
    flows = sources.fetch_market_flows()

    # ── 美股 ──
    us = usmkt.analyse(us_raw, taiex, adj, {"2330": float(close["2330"].iloc[-1])} if "2330" in close else {})
    edition = "盤前版（已納入前一晚美股）" if us["available"] else "盤後版（美股尚未收盤）"

    # ── 大盤預測 ──
    m_sig_d = float(taiex.pct_change().ewm(span=60).std().iloc[-1])
    mkt_mu = {h: live.mkt_mu[h] + mkt_news_adj * h / H + us["impact"] for h in C.HORIZONS}
    mkt_p = float(model.norm_cdf(mkt_mu[H] / (m_sig_d * math.sqrt(H))))
    mm = live.market[H]
    mkt_parts = [{"label": "長期平均漂移", "contrib": _pct(mm.intercept),
                  "text": f"訓練期間平均與長期年化 {C.MARKET_DRIFT_PRIOR:.0%} 先驗值各半混合"}]
    row = M.iloc[-1]
    for col, c in sorted(zip(mm.cols, mm.parts(row)), key=lambda x: -abs(x[1])):
        if abs(c) >= 0.0005:
            v = row[col]
            val = f"{v:.1f}" if col in ("m_rsi", "u_vix") else f"{v * 100:+.1f}%"
            mkt_parts.append({"label": features.MARKET_FACTORS[col], "contrib": _pct(c), "text": f"目前數值 {val}"})
    if abs(mkt_news_adj) >= 0.0003:
        mkt_parts.append({"label": "新聞情緒", "contrib": _pct(mkt_news_adj),
                          "text": f"市場相關新聞加權情緒 {nz['market_sent']:+.2f}（−1 極負面 ~ +1 極正面）"})
    if us["available"]:
        mkt_parts.append({"label": "美股隔夜", "contrib": _pct(us["impact"]), "text": us["note"]})

    # ── 個股預測 ──
    last = len(adj) - 1
    beta = panel.beta.iloc[-1].fillna(1.0)
    sig_d = panel.sigma_d.iloc[-1]
    raw_last = {k: F[k].iloc[-1] for k in features.FACTOR_KEYS}
    pct_last = {k: pd.Series(panel.F[last, :, i], index=codes).rank(pct=True) * 100 for i, k in enumerate(features.FACTOR_KEYS)}
    macd_prev = F["macd"].iloc[-2]
    atr = (pd.concat([(wide["high"] - wide["low"]).stack(), (wide["high"] - close.shift()).abs().stack(),
                      (wide["low"] - close.shift()).abs().stack()], axis=1).max(axis=1).unstack()).rolling(14).mean().iloc[-1]
    chg1 = adj.pct_change(fill_method=None).iloc[-1]
    coef15 = dict(zip(features.FACTOR_KEYS, live.coef[H]))
    alpha_sd = float(np.nanstd(live.alpha[H])) or bt.alpha_sd

    stocks, pred_store, score = {}, {}, {}
    for j, c in enumerate(codes):
        ns = nz["stock_sent"].get(c)
        sd = float(sig_d[c]) if np.isfinite(sig_d[c]) else 0.03
        sd15 = sd * math.sqrt(H)
        # 疊加因子以「相對強度分數的離散度」為單位，和技術面分數相加後一起排名
        news_z = C.OVERLAY_SCALE["news"] * alpha_sd * math.tanh(2.5 * ns["score"]) * weights["news"]["weight"] if ns else 0.0
        chips_z = C.OVERLAY_SCALE["chips"] * alpha_sd * float(np.clip(chip_z[c] / 2, -1, 1)) * weights["chips"]["weight"]
        news_adj, chips_adj = news_z * sd15, chips_z * sd15
        us_adj = us["stock_adj"].get(c, 0.0)
        b = float(beta[c])
        a_ret = {h: float(live.alpha[h][j]) * sd * math.sqrt(h) for h in C.HORIZONS}
        mus = {h: b * (live.mkt_mu[h] + mkt_news_adj * h / H) + a_ret[h]
               + (news_adj + chips_adj) * h / H + us_adj for h in C.HORIZONS}
        score[c] = float(live.alpha[H][j]) + news_z + chips_z
        p_win = float(model.prob_win(score[c], bt.slope))
        conf = max(p_win, 1 - p_win) * 100
        parts = {"market": b * (live.mkt_mu[H] + mkt_news_adj), "alpha": a_ret[H],
                 "news": news_adj, "chips": chips_adj, "us": us_adj}
        px = float(close[c].iloc[-1])
        x = {"close": px, "ma20": float(F["_ma20"][c].iloc[-1] / ratio[c].iloc[-1]),
             "ma60": float(F["_ma60"][c].iloc[-1] / ratio[c].iloc[-1]), "rsi": float(raw_last["rsi"][c]),
             "k": float(F["_k"][c].iloc[-1]), "d": float(F["_d"][c].iloc[-1]), "macd": float(raw_last["macd"][c]),
             "macd_prev": float(macd_prev[c]), "bbp": float(raw_last["bbp"][c]), "volr": float(np.exp(raw_last["volr"][c]))}
        reasons = [{"group": "大盤", "label": "大盤連動", "value": f"β {b:.2f}", "contrib": _pct(parts["market"]),
                    "text": f"大盤 15 日預期 {(live.mkt_mu[H] + mkt_news_adj) * 100:+.2f}% × 個股對大盤敏感度 β {b:.2f}"}]
        reasons += explain.factor_reasons({k: float(raw_last[k][c]) for k in features.FACTOR_KEYS},
                                          {k: float(pct_last[k][c]) for k in features.FACTOR_KEYS},
                                          dict(zip(features.FACTOR_KEYS, live.contrib[j] * sd15)), coef15)
        cd = chip_detail.get(c)
        if cd and abs(chips_adj) >= 0.0003:
            reasons.append({"group": "籌碼", "label": "法人籌碼", "value": f"外資 {cd['f5']:+,} 張", "contrib": _pct(chips_adj),
                            "text": f"近 5 日外資 {cd['f5']:+,} 張、投信 {cd['t5']:+,} 張、自營商 {cd['d5']:+,} 張；"
                                    f"占成交量比重在股票池中排名{'偏前（買超相對積極）' if chip_z[c] > 0 else '偏後（賣超相對明顯）'}"})
        if ns and abs(news_adj) >= 0.0003:
            reasons.append({"group": "新聞", "label": "新聞情緒", "value": f"{ns['n']} 則", "contrib": _pct(news_adj),
                            "text": f"近 {hours} 小時有 {ns['n']} 則相關新聞，加權情緒 {ns['score']:+.2f}；代表新聞：{ns['items'][0]['title']}"})
        if us["available"] and abs(us_adj) >= 0.0003:
            reasons.append({"group": "美股", "label": "美股隔夜", "value": f"敏感度 {us['beta_us'].get(c, 1):.2f}", "contrib": _pct(us_adj),
                            "text": f"前一晚美股估計帶動台股 {us['impact'] * 100:+.2f}%，本股歷史敏感度 {us['beta_us'].get(c, 1):.2f} 倍"})
        a14 = float(atr[c]) if np.isfinite(atr[c]) else px * 0.03
        tail = slice(-130, None)
        stocks[c] = {
            "code": c, "name": names[c], "market": "上市" if uni.market[c] == "TWSE" else "上櫃", "industry": uni.industry[c],
            "asof": asof_s, "close": px, "chg": _pct(chg1[c]),
            "ohlc": [[f"{d:%Y-%m-%d}", o, cl, lo, hi, int(v) if np.isfinite(v) else 0] for d, o, cl, lo, hi, v in zip(
                close.index[tail], wide["open"][c].iloc[tail].round(2), close[c].iloc[tail].round(2),
                wide["low"][c].iloc[tail].round(2), wide["high"][c].iloc[tail].round(2), wide["volume"][c].iloc[tail])
                if np.isfinite(cl) and np.isfinite(o)],
            "forecast": {"mu": {str(h): _pct(mus[h]) for h in C.HORIZONS}, "p_win": round(p_win * 100, 1), "conf": round(conf, 1),
                         "parts": {k: _pct(v) for k, v in parts.items()}, "sigma15": _pct(sd15),
                         "path": _path(px, mus, us_adj, sd)},
            "levels": {"support": round(float(wide["low"][c].iloc[-20:].min()), 2), "resistance": round(float(wide["high"][c].iloc[-20:].max()), 2),
                       "ma20": round(x["ma20"], 2), "ma60": round(x["ma60"], 2), "stop": round(px - 2 * a14, 2),
                       "target": round(px * (1 + mus[H]), 2), "atr": round(a14, 2)},
            "indicators": {"rsi": round(x["rsi"], 1), "k": round(x["k"], 1), "d": round(x["d"], 1), "beta": round(b, 2),
                           "vol_ann": _pct(sd * math.sqrt(252), 1), "mom5": _pct(raw_last["mom5"][c]),
                           "mom20": _pct(raw_last["mom20"][c]), "mom60": _pct(raw_last["mom60"][c])},
            "signals": explain.signals(x), "reasons": reasons, "chips": cd,
            "news": ns["items"] if ns else [], "news_score": ns["score"] if ns else None,
            "us": {"beta": us["beta_us"].get(c), "adj": _pct(us_adj)},
        }
        pred_store[c] = [round(mus[H], 5), round(p_win, 4), round(news_adj, 5), round(chips_adj, 5), round(us_adj, 5)]
        stocks[c]["_m"] = (mus[H], conf, parts)

    # 依相對強度分數排名，再決定建議等級
    order = sorted(codes, key=lambda c: -score[c])
    for rank, c in enumerate(order, 1):
        mu_, conf_, parts_ = stocks[c].pop("_m")
        act = explain.action(rank, len(order), mu_)
        stocks[c]["rank"], stocks[c]["of"] = rank, len(order)
        stocks[c]["forecast"].update(action=act, label=explain.ACTION_LABEL[act], advice=explain.ACTION_TEXT[act],
                                     summary=explain.summary_line(act, mu_, conf_, parts_))

    def brief(c: str) -> dict:
        s = stocks[c]
        f = s["forecast"]
        return {"code": c, "name": s["name"], "industry": s["industry"], "close": s["close"], "chg": s["chg"],
                "mu": f["mu"][str(H)], "conf": f["conf"], "p_win": f["p_win"], "action": f["action"], "label": f["label"],
                "rank": s["rank"], "why": [r.get("short", r["label"]) + (" ↑" if r["contrib"] > 0 else " ↓") for r in
                                           sorted(s["reasons"][1:], key=lambda r: -abs(r["contrib"]))[:3]],
                "summary": f["summary"]}

    # ── 大波動事件與可能原因 ──
    mkt_chg = float(taiex.iloc[-1] / taiex.iloc[-2] - 1)
    events = []
    if abs(mkt_chg) >= C.BIG_MOVE_MARKET_PCT:
        tops = [n for n in nz["news"] if n["market"]][:4]
        events.append({"type": "market", "name": "加權指數", "chg": _pct(mkt_chg),
                       "cause": "同時段衝擊分數最高的市場新聞如下，最可能是這次大盤" + ("大漲" if mkt_chg > 0 else "重挫") + "的主因。"
                       if tops else "未找到明確對應的市場新聞。",
                       "news": [{"title": n["title"], "url": n["url"], "source": n["source"], "time": n["time"],
                                 "why": newsmod.explain(n)} for n in tops]})
    movers = sorted([c for c in codes if np.isfinite(chg1[c]) and abs(chg1[c]) >= C.BIG_MOVE_PCT], key=lambda c: -abs(chg1[c]))
    for c in movers[:14]:
        s = stocks[c]
        rel = float(chg1[c]) - float(beta[c]) * mkt_chg
        if s["news"]:
            cause = f"找到 {len(s['news'])} 則相關新聞（情緒 {s['news_score']:+.2f}），列於下方。"
        elif abs(mkt_chg) >= C.BIG_MOVE_MARKET_PCT and abs(rel) < 0.03:
            cause = f"沒有個股專屬新聞；跌幅與大盤 × β（{float(beta[c]):.2f}）相當，判斷為跟隨大盤的系統性波動。"
        else:
            cause = "未找到直接相關新聞，可能是族群連動、籌碼或技術面因素。"
        events.append({"type": "stock", "code": c, "name": s["name"], "industry": s["industry"], "chg": s["chg"],
                       "excess": _pct(rel), "cause": cause, "news": s["news"][:3]})

    major = [{**n, "why": newsmod.explain(n), "names": [names[c] for c in n["stocks"][:6]],
              "tone": "利多" if n["sent"] > 0.15 else "利空" if n["sent"] < -0.15 else "中性"} for n in nz["news"][:14]]

    # ── 產業彙整 ──
    sect = pd.DataFrame({"ind": uni["industry"], "chg": chg1, "mu": [stocks[c]["forecast"]["mu"][str(H)] for c in codes]})
    sectors = [{"name": k, "n": int(len(g)), "chg": _pct(g["chg"].mean()), "mu": round(float(g["mu"].mean()), 2)}
               for k, g in sect.groupby("ind") if len(g) >= 3]
    sectors.sort(key=lambda s: -s["mu"])

    # ── 因子診斷（近 30 個已到期交易日的實際 IC，對照模型權重）──
    diag = []
    rng_ = range(last - H - 29, last - H + 1)
    for k_i, k in enumerate(features.FACTOR_KEYS):
        ics = [model.rank_ic(panel.F[t, :, k_i], panel.y[H][t]) for t in rng_]
        ic = float(np.nanmean(ics)) if np.isfinite(ics).any() else 0.0
        cf = float(live.coef[H][k_i])
        flag = "against" if ic * cf < 0 and abs(ic) > 0.03 and abs(cf) > 0.005 else "with" if ic * cf > 0 and abs(ic) > 0.03 else "flat"
        diag.append({"key": k, "name": features.FACTORS[k][0], "group": features.FACTORS[k][1], "coef": round(cf, 4),
                     "prev": round(float(live.prev_coef[k_i]), 4) if live.prev_coef is not None else None,
                     "ic": round(ic, 3), "flag": flag})
    diag_sorted = sorted(diag, key=lambda d: -abs(d["ic"]))

    # ── 存檔今日預測並復盤 ──
    review.save_prediction(asof_s, {
        "asof": asof_s, "kind": "live", "generated_at": now.isoformat(timespec="minutes"), "edition": edition,
        "market": {"mu": round(mkt_mu[H], 5), "p_up": round(mkt_p, 4)},
        "top": order[: C.TOP_N], "stocks": pred_store,
    })
    rv = review.evaluate(review.load_predictions(), adj, taiex, names)
    rv["suggestions"] = review.suggestions(rv, bt.stats, diag_sorted, weights)
    rv["backtest"] = bt.stats
    rv["weights"] = weights
    rv["factors"] = diag

    # ── 輸出 ──
    news_day = now.strftime("%Y-%m-%d")
    _write(C.DATA_DIR / "news" / f"{news_day}.json", {"date": news_day, "market_sent": nz["market_sent"],
                                                       "news": nz["news"][: C.NEWS_KEEP]})
    if (C.SITE_DATA_DIR / "stocks").exists():
        shutil.rmtree(C.SITE_DATA_DIR / "stocks")
    for c in codes:
        _write(C.SITE_DATA_DIR / "stocks" / f"{c}.json", stocks[c])
    _write(C.SITE_DATA_DIR / "index.json", [
        {"c": c, "n": s["name"], "i": s["industry"], "px": s["close"], "chg": s["chg"], "mu": s["forecast"]["mu"][str(H)],
         "conf": s["forecast"]["conf"], "p": s["forecast"]["p_win"], "a": s["forecast"]["action"], "r": s["rank"]}
        for c, s in ((c, stocks[c]) for c in order)])
    _write(C.SITE_DATA_DIR / "review.json", rv)
    tail = slice(-130, None)
    m_mus = {h: mkt_mu[h] for h in C.HORIZONS}
    _write(C.SITE_DATA_DIR / "latest.json", {
        "meta": {"generated_at": now.strftime("%Y-%m-%d %H:%M"), "asof": asof_s, "edition": edition,
                 "next_session": f"{sources.next_trading_day(asof):%Y-%m-%d}", "universe": len(codes), "horizon": H,
                 "news_hours": hours, "chip_days": chip_days, "us_available": us["available"]},
        "market": {
            "close": round(float(taiex.iloc[-1]), 2), "chg": _pct(mkt_chg),
            "ohlc": [[f"{d:%Y-%m-%d}", round(o, 2), round(cl, 2), round(lo, 2), round(hi, 2)] for d, o, cl, lo, hi in
                     zip(tw.index[tail], tw["open"].iloc[tail], tw["close"].iloc[tail], tw["low"].iloc[tail], tw["high"].iloc[tail])],
            "mu": {str(h): _pct(mkt_mu[h]) for h in C.HORIZONS}, "p_up": round(mkt_p * 100, 1),
            "conf": round(max(mkt_p, 1 - mkt_p) * 100, 1),
            "view": "偏多" if mkt_p >= 0.55 else "偏空" if mkt_p <= 0.45 else "中性",
            "path": _path(float(taiex.iloc[-1]), m_mus, us["impact"], m_sig_d),
            "parts": mkt_parts, "breadth": _pct(breadth.iloc[-1], 1),
            "breadth_note": "站上月線的個股比例；高於 60% 代表多數股票偏強，低於 40% 代表普遍偏弱",
            "flows": flows,
            "dist": {k: sum(stocks[c]["forecast"]["action"] == k for c in codes) for k in explain.ACTION_LABEL},
        },
        "us": {k: us[k] for k in ("panel", "available", "impact", "note", "model", "adr") } | {"impact": _pct(us["impact"]), "moves": us.get("moves")},
        "top_bull": [brief(c) for c in order[: C.TOP_N]],
        "top_bear": [brief(c) for c in order[::-1][: C.TOP_N]],
        "events": events, "major_news": major,
        "news": {"market_sent": nz["market_sent"], **nz["counts"], "hours": hours},
        "sectors": sectors,
        "model": {"train_days": C.TRAIN_WINDOW, "factors": diag, "backtest": bt.stats,
                  "calibrated": bt.slope > 0, "weights": weights, "hiconf": C.HICONF},
    })
    log.info("完成：%s %s｜大盤 15 日預期 %+.2f%%（上漲機率 %.0f%%）｜看好第一名 %s %s",
             asof_s, edition, mkt_mu[H] * 100, mkt_p * 100, order[0], names[order[0]])


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logging.getLogger("warroom").exception("執行失敗")
        sys.exit(1)
