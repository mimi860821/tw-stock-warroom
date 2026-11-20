"""核心邏輯的單元測試（不連網）。執行：python -m pytest -q"""
import numpy as np
import pandas as pd

from warroom import explain, features, model, news, usmkt


def _prices(n=320, k=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    mkt = rng.normal(0.0004, 0.01, n)
    rets = mkt[:, None] * rng.uniform(0.6, 1.4, k) + rng.normal(0, 0.015, (n, k))
    adj = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=[f"{1000 + i}" for i in range(k)])
    taiex = pd.Series(20000 * np.exp(np.cumsum(mkt)), index=idx)
    vol = pd.DataFrame(rng.uniform(1e6, 5e6, (n, k)), index=idx, columns=adj.columns)
    return adj, taiex, vol


def test_rsi_bounds_and_direction():
    up = pd.DataFrame({"a": np.linspace(10, 20, 60)})
    down = pd.DataFrame({"a": np.linspace(20, 10, 60)})
    assert features.rsi(up)["a"].iloc[-1] > 95
    assert features.rsi(down)["a"].iloc[-1] < 5


def test_zscore_cs_is_standardised_per_day():
    x = pd.DataFrame(np.random.default_rng(1).normal(5, 3, (10, 50)))
    z = features.zscore_cs(x)
    assert np.allclose(z.mean(axis=1), 0, atol=0.15)
    assert z.abs().max().max() <= 3


def test_neutralize_removes_risk_exposure():
    rng = np.random.default_rng(2)
    risk = rng.normal(size=(5, 200, 2))
    a = 2.0 * risk[:, :, 0] - 1.0 * risk[:, :, 1] + rng.normal(0, 0.1, (5, 200))
    res = model.neutralize(a, risk)
    for t in range(5):
        assert abs(np.corrcoef(res[t], risk[t, :, 0])[0, 1]) < 1e-8
        assert abs(np.corrcoef(res[t], risk[t, :, 1])[0, 1]) < 1e-8


def test_walk_forward_never_uses_future_labels():
    """把最後 15 天的價格改掉，不應影響 15 天以前那些日子的樣本外預測。"""
    adj, taiex, vol = _prices()
    noise = 1 + np.random.default_rng(9).normal(0, 0.005, adj.shape)
    def run(a):
        f = features.compute(a, a.shift(1).bfill() * noise, a * 1.01, a * 0.99, vol, vol * a)
        breadth = (a > f["_ma20"]).sum(axis=1) / a.notna().sum(axis=1)
        p = model.build_panel(a, taiex, f, features.market_features(taiex, breadth, {}))
        return model.walk_forward(p)
    base = run(adj)
    changed = adj.copy()
    changed.iloc[-15:] *= np.linspace(1.0, 1.5, 15)[:, None]
    alt = run(changed)
    n = len(base.idx) - 15
    assert np.allclose(base.alpha[:n], alt.alpha[:n], equal_nan=True)
    assert np.allclose(base.p_win[:n], alt.p_win[:n], equal_nan=True)     # 信心校準也只用當時已知的資料


def test_align_prev_uses_strictly_earlier_us_close():
    us = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-11-11", "2026-11-12", "2026-11-13"]))
    tw = pd.to_datetime(["2026-11-12", "2026-11-13", "2026-11-16"])
    assert usmkt.align_prev(us, tw).tolist() == [1.0, 2.0, 3.0]


def test_us_impact_only_when_us_closed_after_tw():
    adj, taiex, _ = _prices()
    rng = np.random.default_rng(3)
    us_idx = taiex.index
    sox = pd.DataFrame({"close": 5000 * np.exp(np.cumsum(rng.normal(0, 0.015, len(us_idx))))}, index=us_idx)
    spx = pd.DataFrame({"close": 6000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(us_idx))))}, index=us_idx)
    raw = {"^SOX": sox, "^GSPC": spx}
    assert usmkt.analyse(raw, taiex, adj, {})["available"] is True            # 美股最後一筆與台股同日 → 已有新資訊
    assert usmkt.analyse({k: v.iloc[:-1] for k, v in raw.items()}, taiex, adj, {})["available"] is False


def test_sentiment_polarity_and_negation():
    assert news.sentiment("台積電法說報喜 營收創新高")[0] > 0.3
    assert news.sentiment("台股重挫 外資大賣 跌停家數創紀錄")[0] < -0.3
    assert news.sentiment("法人不看好後市")[0] < 0
    assert news.sentiment("營收未下修")[0] > 0


def test_matcher_longest_name_and_code():
    m = news.build_matcher({"2492": "華新科", "1605": "華新", "2330": "台積電", "3661": "世芯-KY"})
    assert set(m("華新科漲停", "")) == {"2492"}
    assert set(m("半導體族群走強", "法人看好台積電（2330）與世芯。")) == {"2330", "3661"}


def test_cluster_merges_duplicates():
    from datetime import datetime, timezone
    t = datetime(2026, 11, 16, tzinfo=timezone.utc)
    items = [{"title": "台股重挫2635點 創史上第二大跌點", "summary": "", "url": "a", "source": "A", "time": t},
             {"title": "台股重挫2635點創史上第二大跌點紀錄", "summary": "更長的摘要", "url": "b", "source": "B", "time": t},
             {"title": "聯發科發表新晶片", "summary": "", "url": "c", "source": "A", "time": t}]
    out = news.cluster(items)
    assert len(out) == 2 and out[0]["dup"] == 2 and out[0]["sources"] == ["A", "B"]


def test_action_grades_by_rank_and_downgrades_on_conflict():
    assert explain.action(1, 100, 0.03) == "strong_buy"
    assert explain.action(1, 100, -0.01) == "buy"            # 排名最前但絕對預期為負 → 降一級
    assert explain.action(15, 100, 0.02) == "buy"
    assert explain.action(50, 100, 0.02) == "neutral"
    assert explain.action(100, 100, -0.03) == "strong_sell"
    assert explain.action(90, 100, 0.01) == "neutral"


# ── 新增模組 ────────────────────────────────────────────────────────────

def test_chip_streaks():
    from warroom import chips
    assert chips._streak(np.array([-5, 3, 4, 1])) == 3
    assert chips._streak(np.array([2, -1, -2])) == -2
    assert chips._streak(np.array([1, 0])) == 0


def test_chip_scores_penalise_margin_buildup():
    from warroom import chips
    days = pd.bdate_range("2026-11-02", periods=6)
    codes = ["1111", "2222", "3333"]
    inst = pd.DataFrame([{"date": d, "code": c, "foreign": 0, "trust": 0, "dealer": 0} for d in days for c in codes])
    mar = pd.DataFrame([{"date": d, "code": c, "margin": 1000 + (i * 200 if c == "1111" else 0), "short": 10}
                        for i, d in enumerate(days) for c in codes])
    vol = pd.DataFrame(1e6, index=days, columns=codes)
    score, detail, _, n_mar = chips.scores(inst, mar, vol, codes)
    assert n_mar == 6 and score["1111"] < score["2222"]          # 融資暴增的股票籌碼分數較差
    assert detail["1111"]["margin_chg"] == 1000


def test_patterns_detect_gap_and_limit_up():
    from warroom import patterns
    idx = pd.bdate_range("2026-01-01", periods=30)
    c = pd.DataFrame({"a": np.linspace(100, 101, 30)}, index=idx)
    o, h, l = c * 0.999, c * 1.005, c * 0.995
    c.iloc[-1], o.iloc[-1], h.iloc[-1], l.iloc[-1] = 111.0, 109.0, 111.0, 108.5      # 跳空漲停
    v = pd.DataFrame({"a": [1e6] * 29 + [5e6]}, index=idx)
    k = d = pd.DataFrame({"a": [50.0] * 30}, index=idx)
    sig = patterns.detect(o, h, l, c, v, c, k, d, c * 0)
    assert sig["gap_up"]["a"].iloc[-1] and sig["limit_up"]["a"].iloc[-1] and sig["vol_breakout"]["a"].iloc[-1]
    assert not sig["gap_down"]["a"].any()
    assert patterns.recent(sig, "a", idx, 3)[0][1] == 0


def test_pattern_stats_shrink_insignificant():
    from warroom import patterns
    rng = np.random.default_rng(5)
    fwd = rng.normal(0, 0.05, (300, 50))
    hit = pd.DataFrame(rng.random((300, 50)) < 0.05)
    st = patterns.event_stats({"x": hit}, fwd)["x"]
    assert st["n"] > 0 and abs(st["t"]) < 3 and abs(st["shrunk"]) <= abs(st["mean"] / 100) + 1e-12


def test_master_scorecards_and_graham_number():
    from warroom import fundamentals as F
    fin = {"fin_type": "ci", "year": 115, "season": 3, "eps_ytd": 6.0, "ni_parent": 300.0, "equity_parent": 1500.0,
           "assets": 3000.0, "liab": 900.0, "cur_assets": 800.0, "cur_liab": 300.0, "capital": 1_000_000.0, "bvps": 30.0,
           "op_income": 380.0, "pretax": 400.0, "gross_margin": 45.0, "op_margin": 25.0, "net_margin": 18.0,
           "rev_cum_yoy": 25.0, "rev_yoy": 30.0}
    m = F.metrics("9999", 60.0, fin, (10.0, 2.0, 5.0), (12.0, 2.0, 4.0), 48.0, "電子零組件")
    assert m["eps_ttm"] == 6.0 and m["roe"] == 26.7 and m["debt"] == 30.0
    assert abs(m["graham_number"] - (22.5 * 6 * 30) ** 0.5) < 0.1
    assert m["eps_g"] == 50.0 and m["peg"] == 0.2 and m["lynch"]["category"] == "快速成長股"
    assert m["buffett"]["score"] == 100
    # 金融業：沒有流動比率與負債比的檢核
    mf = F.metrics("2882", 70.0, {}, (13.0, 1.6, 3.0), None, 0.0, "金融保險")
    assert mf["roe"] == round(1.6 / 13 * 100, 1) and mf["graham"]["items"][1]["pass"] is None


def test_value_factors_treat_loss_as_expensive():
    from warroom import fundamentals as F
    idx = pd.bdate_range("2025-01-01", periods=260)
    pe = pd.DataFrame({"a": 10.0, "b": np.nan, "c": np.nan}, index=idx)
    listed = pd.DataFrame({"a": True, "b": True, "c": False}, index=idx)
    val = {"pe": pe, "pb": pe / 5, "dy": pe * 0 + 3, "listed": listed}
    close = pd.DataFrame(100.0, index=idx, columns=["a", "b", "c"])
    vf = F.value_factors(val, close)
    assert vf["ey"]["a"].iloc[-1] == 0.1 and vf["ey"]["b"].iloc[-1] == 0.0 and np.isnan(vf["ey"]["c"].iloc[-1])


def test_review_handles_old_and_new_prediction_widths():
    from warroom import review
    adj, taiex, _ = _prices(n=60, k=30)
    codes = list(adj.columns)
    preds = [{"asof": f"{adj.index[30]:%Y-%m-%d}", "kind": "live", "market": {"mu": 0.01, "p_up": 0.55},
              "stocks": {c: ([0.01, 0.6, 0, 0, 0] if i % 2 else [0.01, 0.4, 0.001, 0, 0, 0.002, -0.001]) for i, c in enumerate(codes)}}]
    rv = review.evaluate(preds, adj, taiex, {c: c for c in codes})
    assert rv["rows"][0]["h"] == 15 and set(rv["overlay_ic"]) == set(review.OVERLAY_COLS)


def test_rank_bucket_lookup():
    stats = {"rank_buckets": [{"lo": 0, "hi": .01, "label": "前 1%"}, {"lo": .01, "hi": .05, "label": "前 1–5%"},
                              {"lo": .95, "hi": 1.0, "label": "後 5%"}]}
    assert model.rank_bucket(stats, 1, 450)["label"] == "前 1%"
    assert model.rank_bucket(stats, 10, 450)["label"] == "前 1–5%"
    assert model.rank_bucket(stats, 450, 450)["label"] == "後 5%"
    assert model.rank_bucket(stats, 200, 450) is None


def test_market_up_freq_shrinks_toward_half():
    adj, taiex, vol = _prices(n=400, k=30)
    f = features.compute(adj, adj, adj * 1.01, adj * 0.99, vol, vol * adj)
    breadth = (adj > f["_ma20"]).sum(axis=1) / adj.notna().sum(axis=1)
    p = model.build_panel(adj, taiex, f, features.market_features(taiex, breadth, {}))
    freq, k = model.up_freq(p, 399, 15)
    raw = (p.mfwd[15][60:399 - 15 + 1] > 0).mean()
    assert k > 300 and abs(freq - 0.5) < abs(raw - 0.5) + 1e-12
    fr, kr = model.up_freq(p, 399, 15, regime=True)
    assert 0 < kr < k


def test_rankings_streaks_and_kinds():
    from warroom import rankings
    assert rankings._kind("2330") == "stock" and rankings._kind("0050") == "etf" and rankings._kind("00878") == "etf"
    assert rankings._kind("084028") is None and rankings._kind("710001") is None
    m = np.array([[True, True], [False, True], [True, True]])
    assert rankings._streak(m).tolist() == [1, 3]
    x = np.array([[5, -1], [3, -2], [2, -4]])
    assert rankings._signed_streak(x).tolist() == [3, -3]
