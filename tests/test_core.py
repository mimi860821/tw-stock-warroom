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
