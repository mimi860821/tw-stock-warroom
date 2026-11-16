"""復盤驗證：把過去每天存下來的預測，拿來跟之後實際的走勢比對。"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import config as C

PRED_DIR = C.DATA_DIR / "predictions"


def save_prediction(asof: str, payload: dict) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    (PRED_DIR / f"{asof}.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load_predictions(limit: int = C.REVIEW_DAYS) -> list[dict]:
    if not PRED_DIR.exists():
        return []
    files = sorted(PRED_DIR.glob("*.json"))[-limit:]
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _corr(a: np.ndarray, b: np.ndarray, rank: bool = True) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 15 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    if rank:
        a, b = pd.Series(a[ok]).rank().values, pd.Series(b[ok]).rank().values
    else:
        a, b = a[ok], b[ok]
    return float(np.corrcoef(a, b)[0, 1])


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def evaluate(preds: list[dict], adj: pd.DataFrame, taiex: pd.Series, names: dict[str, str]) -> dict:
    """逐日比對。尚未滿 15 個交易日的預測，用「目前為止」的實際走勢先看方向。"""
    H = C.HORIZON
    dates = adj.index
    rows, pool = [], []          # pool: (kind, conf, hit, 已到期?) 供信心校準分析
    overlay_ic = {"news": [], "chips": [], "us": []}
    for pr in preds:
        d = pd.Timestamp(pr["asof"])
        if d not in dates:
            continue
        i = dates.get_loc(d)
        h = min(H, len(dates) - 1 - i)
        row = {"date": pr["asof"], "kind": pr.get("kind", "live"), "h": h, "matured": h >= H,
               "mkt_mu": _r(pr["market"]["mu"] * 100), "mkt_p_up": _r(pr["market"]["p_up"] * 100, 1)}
        if h == 0:
            row["status"] = "等待下一個交易日"
            rows.append(row)
            continue
        codes = [c for c in pr["stocks"] if c in adj.columns]
        arr = np.array([pr["stocks"][c] for c in codes], dtype=float)      # mu, p_win, news, chips, us
        mu, p_up = arr[:, 0], arr[:, 1]                                    # p_up：贏過股票池中位數的機率
        real = (adj[codes].iloc[i + h] / adj[codes].iloc[i] - 1).values
        ok = np.isfinite(real) & np.isfinite(mu)
        conf = np.maximum(p_up, 1 - p_up) * 100
        hit = np.sign(mu) == np.sign(real)                                 # 絕對方向
        rel = (p_up > 0.5) == (real > np.nanmedian(real[ok]))              # 相對強弱：有沒有猜對會贏過中位數
        ex = real - np.nanmean(real)
        order = np.argsort(-np.where(ok, p_up, -np.inf))
        top, bottom = order[: C.TOP_N], np.argsort(np.where(ok, p_up, np.inf))[: C.TOP_N]
        mreal = float(taiex.iloc[i + h] / taiex.iloc[i] - 1)
        hi = ok & (conf >= C.HICONF)
        row.update({
            "status": "已到期" if h >= H else f"進行中（第 {h} 天）",
            "n": int(ok.sum()),
            "hit": _r(hit[ok].mean() * 100, 1),
            "rel_hit": _r(rel[ok].mean() * 100, 1),
            "up_share": _r((real[ok] > 0).mean() * 100, 1),
            "hit_hi": _r(rel[hi].mean() * 100, 1) if hi.sum() >= 5 else None,
            "n_hi": int(hi.sum()),
            "top_ret": _r(np.nanmean(real[top]) * 100),
            "bottom_ret": _r(np.nanmean(real[bottom]) * 100),
            "uni_ret": _r(np.nanmean(real[ok]) * 100),
            "ic": _r(_corr(mu, real), 3),
            "mkt_real": _r(mreal * 100),
            "mkt_hit": bool(np.sign(pr["market"]["mu"]) == np.sign(mreal)),
            "pred_avg": _r(np.nanmean(mu[ok]) * h / H * 100),
            "picks": [{"code": codes[j], "name": names.get(codes[j], codes[j]), "mu": _r(mu[j] * 100),
                       "conf": _r(conf[j], 0), "real": _r(real[j] * 100)} for j in top],
        })
        row["excess"] = _r(row["top_ret"] - row["uni_ret"])
        rows.append(row)
        for c_, h_ in zip(conf[ok], rel[ok]):
            pool.append((row["kind"], c_, h_, h >= H))
        if row["kind"] == "live" and h >= 5 and arr.shape[1] >= 5:
            for k, col in (("news", 2), ("chips", 3), ("us", 4)):
                m = ok & (arr[:, col] != 0)
                ic = _corr(arr[m, col], ex[m]) if m.sum() >= 15 else None
                if ic is not None:
                    overlay_ic[k].append(ic)

    def agg(sel: list[dict]) -> dict | None:
        sel = [r for r in sel if r["h"] > 0]
        if not sel:
            return None
        return {
            "days": len(sel),
            "hit": _r(_mean(r["hit"] for r in sel), 1),
            "rel_hit": _r(_mean(r["rel_hit"] for r in sel), 1),
            "up_share": _r(_mean(r["up_share"] for r in sel), 1),
            "hit_hi": _r(sum(r["hit_hi"] * r["n_hi"] for r in sel if r["hit_hi"] is not None)
                         / max(sum(r["n_hi"] for r in sel if r["hit_hi"] is not None), 1), 1),
            "n_hi": sum(r["n_hi"] for r in sel if r["hit_hi"] is not None),
            "top_ret": _r(_mean(r["top_ret"] for r in sel)),
            "uni_ret": _r(_mean(r["uni_ret"] for r in sel)),
            "excess": _r(_mean(r["excess"] for r in sel)),
            "top_win": _r(np.mean([r["excess"] > 0 for r in sel]) * 100, 1),
            "ic": _r(_mean(r["ic"] for r in sel), 3),
            "mkt_hit": _r(np.mean([r["mkt_hit"] for r in sel]) * 100, 1),
            "bias": _r(_mean(r["pred_avg"] - r["uni_ret"] for r in sel)),
        }

    buckets = []
    if pool:
        conf = np.array([p[1] for p in pool])
        hit = np.array([p[2] for p in pool], dtype=float)
        for lo, hi_ in ((50, 52), (52, 55), (55, 60), (60, 101)):
            m = (conf >= lo) & (conf < hi_)
            if m.sum() >= 30:
                buckets.append({"range": f"{lo}–{min(hi_, 100)}", "n": int(m.sum()),
                                "hit": _r(hit[m].mean() * 100, 1), "avg_conf": _r(conf[m].mean(), 1)})

    live = [r for r in rows if r["kind"] == "live"]
    return {
        "rows": rows[::-1],
        "summary": {
            "matured": agg([r for r in rows if r["matured"]]),
            "recent": agg([r for r in rows if not r["matured"]]),
            "live": agg(live),
            "all": agg(rows),
        },
        "n_live": len(live), "n_backtest": len(rows) - len(live),
        "buckets": buckets,
        "overlay_ic": {k: {"ic": _r(_mean(v), 3), "days": len(v)} for k, v in overlay_ic.items()},
    }


def adapt_weights(overlay_ic: dict) -> dict:
    """依實際預測的復盤結果調整新聞／籌碼疊加的權重。證據不足時維持 1.0。

    權重 = 1 + 10 × 平均IC × min(天數/30, 1)，限制在 ADAPT_BOUNDS 內。每次都從證據重新計算，
    所以一天跑幾次結果都一樣，不會越調越偏。
    """
    out = {}
    for k in ("news", "chips"):
        info = overlay_ic.get(k) or {}
        ic, days = info.get("ic"), info.get("days", 0)
        w, why = 1.0, f"實際預測樣本僅 {days} 天（需 ≥ {C.ADAPT_MIN_DAYS} 天），暫維持預設權重"
        if ic is not None and days >= C.ADAPT_MIN_DAYS:
            w = float(np.clip(1 + 10 * ic * min(days / 30, 1), *C.ADAPT_BOUNDS))
            why = f"近 {days} 天此因子與實際超額報酬的相關性 IC={ic:+.3f}，權重調為 {w:.2f}"
        out[k] = {"weight": round(w, 2), "why": why}
    return out


def suggestions(rv: dict, bt_stats: dict, factor_diag: list[dict], weights: dict) -> list[dict]:
    """把復盤數字翻成可以採取行動的改進建議。"""
    out = []
    S = rv["summary"]
    a = S.get("all")
    if rv["n_live"] < 15:
        out.append({"level": "info", "title": "實際發布的預測樣本還很少",
                    "text": f"目前只有 {rv['n_live']} 天是系統實際發布的預測，其餘 {rv['n_backtest']} 天是用滾動回測補齊"
                            "（只含技術面，不含新聞與籌碼）。15 日預測要 3 週後才能完整驗證，建議至少累積一個月再評斷模型。"})
    if a:
        if a["n_hi"] >= 100 and a["rel_hit"] is not None:
            if a["hit_hi"] - a["rel_hit"] >= 2:
                out.append({"level": "good", "title": "信心值有鑑別力",
                            "text": f"信心 ≥ {C.HICONF} 的訊號相對強弱命中率 {a['hit_hi']}%，高於全部訊號的 {a['rel_hit']}%。"
                                    f"建議只把信心 {C.HICONF} 以上的訊號當作進出依據，其餘當作參考。"})
            elif a["hit_hi"] < a["rel_hit"] - 2:
                out.append({"level": "warn", "title": "高信心訊號近期反而較不準",
                            "text": f"信心 ≥ {C.HICONF} 的訊號命中率 {a['hit_hi']}%，低於整體 {a['rel_hit']}%。通常代表行情風格剛轉換"
                                    "（原本有效的因子失靈）。模型每天會重新訓練與校準，但在命中率回升前建議降低部位。"})
        if a["rel_hit"] is not None:
            lvl = "good" if a["rel_hit"] >= 52 else "warn" if a["rel_hit"] < 49 else "info"
            out.append({"level": lvl, "title": f"相對強弱命中率 {a['rel_hit']}%（基準 50%）",
                        "text": f"判斷「這檔會比股票池中位數強還是弱」的命中率。這個數字不受大盤漲跌影響，是衡量選股能力最乾淨的指標；"
                                f"同期間方向命中率 {a['hit']}%，而個股實際上漲的比例是 {a['up_share']}%（全部猜漲就能拿到的基準）。"})
        if a["excess"] is not None:
            if a["excess"] > 0.3 and a["top_win"] >= 55:
                out.append({"level": "good", "title": "看好名單持續贏過股票池",
                            "text": f"前 {C.TOP_N} 名平均報酬 {a['top_ret']:+.2f}%，股票池平均 {a['uni_ret']:+.2f}%，"
                                    f"超額 {a['excess']:+.2f}%，{a['top_win']}% 的日子勝出。選股排序目前有效。"})
            elif a["excess"] < -0.3:
                out.append({"level": "warn", "title": "看好名單近期落後股票池",
                            "text": f"前 {C.TOP_N} 名平均報酬 {a['top_ret']:+.2f}%，落後股票池平均 {a['uni_ret']:+.2f}%。"
                                    "可檢視下方「近期失靈的因子」，考慮在 config.py 提高 RIDGE_LAMBDA（更保守）"
                                    "或縮短 TRAIN_WINDOW（更快適應新行情）。"})
        if a["mkt_hit"] is not None and a["days"] >= 10:
            lvl = "good" if a["mkt_hit"] >= 58 else "warn" if a["mkt_hit"] < 45 else "info"
            out.append({"level": lvl, "title": "大盤方向預測",
                        "text": f"近 {a['days']} 個預測日的大盤方向命中率 {a['mkt_hit']}%。"
                                + ("大盤方向比個股相對強弱難預測得多，建議把「個股排序」看得比「大盤漲跌」重。"
                                   if a["mkt_hit"] < 58 else "近期大盤方向判斷可靠度尚可。")})
        if a["bias"] is not None and abs(a["bias"]) >= 1.5:
            word = "偏樂觀" if a["bias"] > 0 else "偏保守"
            out.append({"level": "warn", "title": f"預測整體{word}",
                        "text": f"同期間平均預測比實際高出 {a['bias']:+.2f} 個百分點。解讀預期報酬時請自行扣除這個偏差；"
                                "若持續出現，可調整 config.py 的 MARKET_DRIFT_PRIOR。"})
    bad = [f for f in factor_diag if f["flag"] == "against"][:3]
    if bad:
        out.append({"level": "warn", "title": "近期失靈的因子",
                    "text": "；".join(f"「{f['name']}」模型權重 {f['coef']:+.3f}，但近 30 個已到期交易日的實際 IC 為 {f['ic']:+.3f}"
                                     for f in bad)
                            + "。這些因子近期的方向與模型學到的相反，相關訊號請打折看待；每日重訓會逐步修正權重。"})
    good = [f for f in factor_diag if f["flag"] == "with"][:3]
    if good:
        out.append({"level": "good", "title": "近期最有效的因子",
                    "text": "、".join(f"「{f['name']}」(IC {f['ic']:+.3f})" for f in good) + "，方向與模型權重一致。"})
    for k, label in (("news", "新聞情緒"), ("chips", "法人籌碼")):
        w = weights.get(k)
        if w:
            out.append({"level": "info" if w["weight"] == 1.0 else "good" if w["weight"] > 1 else "warn",
                        "title": f"{label}疊加權重：{w['weight']:.2f}", "text": w["why"] + "。"})
    if bt_stats.get("hit_rate") is not None:
        out.append({"level": "info", "title": "長期回測基準",
                    "text": f"滾動回測 {bt_stats['days']} 個交易日（{bt_stats['from']} 起，樣本外）：相對強弱命中率 {bt_stats['rel_hit']}%（基準 50%），"
                            f"排序相關 IC {bt_stats['ic']:+.3f}（前半 {bt_stats['ic_h1']:+.3f}、後半 {bt_stats['ic_h2']:+.3f}）；"
                            f"前 {C.TOP_N} 名平均 {bt_stats['top_ret']:+.2f}% 對股票池 {bt_stats['uni_ret']:+.2f}%，"
                            f"勝率 {bt_stats['top_win']}%。注意：股票池是用「現在」的成交金額選的，回測會有倖存者偏誤，實際績效通常較低。"})
    return out
