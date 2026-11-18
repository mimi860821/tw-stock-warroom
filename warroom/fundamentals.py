"""基本面：本益比／淨值比／殖利率（含歷史）、財報、月營收，以及葛拉漢、巴菲特、彼得林區三套評分。

資料全部來自證交所、櫃買中心的公開資料，免金鑰。

* 估值歷史：約每半個月抓一次全市場的本益比、淨值比、殖利率，存在 data/valuation/。
  有了歷史才能做「樣本外回測」，判斷估值因子在台股到底有沒有用、方向是什麼。
* 財報與月營收：只有最新一期，用來做三位大師的檢核清單（無法回測，所以只以小權重參與排名，
  並由復盤自動調整權重）。
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from .sources import _num, get_json

log = logging.getLogger("warroom")
TWSE = "https://openapi.twse.com.tw/v1"
TPEX = "https://www.tpex.org.tw/openapi/v1"
VAL_DIR = C.DATA_DIR / "valuation"
FUND_DIR = C.DATA_DIR / "fundamentals"

# 林區分類用：景氣循環產業
CYCLICAL = {"水泥", "塑膠", "鋼鐵", "航運", "造紙", "橡膠", "玻璃陶瓷", "化學", "汽車", "油電燃氣", "紡織纖維", "建材營造"}
FINANCIAL = {"金融保險"}


def _roc(s: str) -> datetime | None:
    """民國日期 '1151117' 或 '115/11/17' → datetime。"""
    s = str(s).replace("/", "").strip()
    try:
        return datetime(int(s[:-4]) + 1911, int(s[-4:-2]), int(s[-2:]))
    except (ValueError, IndexError):
        return None


def _gz_write(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0))


def _gz_read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


# ── 估值（本益比、淨值比、殖利率）──────────────────────────────────────────

def _pos(x: float) -> float | None:
    return x if np.isfinite(x) and x > 0 else None


def _fetch_val_twse(day: datetime) -> dict | None:
    js = get_json(f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={day:%Y%m%d}&selectType=ALL&response=json",
                  tries=2, pause=8.0)
    if not js or js.get("stat") != "OK" or not js.get("data"):
        return None
    f = js["fields"]
    try:
        ic, ipe, ipb, idy = f.index("證券代號"), f.index("本益比"), f.index("股價淨值比"), f.index("殖利率(%)")
    except ValueError:
        return None
    return {r[ic].strip(): [_pos(_num(r[ipe])), _pos(_num(r[ipb])), _num(r[idy]) if np.isfinite(_num(r[idy])) else 0.0]
            for r in js["data"]}


def _fetch_val_tpex(day: datetime) -> dict | None:
    js = get_json(f"https://www.tpex.org.tw/www/zh-tw/afterTrading/peQryDate?date={day:%Y}%2F{day:%m}%2F{day:%d}&response=json",
                  tries=2)
    try:
        tb = js["tables"][0]
        if tb.get("date") != f"{day.year - 1911}/{day:%m}/{day:%d}" or not tb["data"]:
            return None
        f = tb["fields"]
        ic, ipe, ipb, idy = f.index("股票代號"), f.index("本益比"), f.index("股價淨值比"), f.index("殖利率(%)")
        return {r[ic].strip(): [_pos(_num(r[ipe])), _pos(_num(r[ipb])), _num(r[idy]) if np.isfinite(_num(r[idy])) else 0.0]
                for r in tb["data"]}
    except (TypeError, KeyError, IndexError, ValueError):
        return None


def fetch_val_today() -> tuple[datetime | None, dict]:
    """OpenAPI 的最新一日估值（沒有頻率限制）。"""
    out, day = {}, None
    for x in get_json(f"{TWSE}/exchangeReport/BWIBBU_ALL") or []:
        day = day or _roc(x.get("Date"))
        out[x["Code"].strip()] = [_pos(_num(x.get("PEratio"))), _pos(_num(x.get("PBratio"))),
                                  _num(x.get("DividendYield")) if np.isfinite(_num(x.get("DividendYield"))) else 0.0]
    tp_day = None
    for x in get_json(f"{TPEX}/tpex_mainboard_peratio_analysis") or []:
        tp_day = tp_day or _roc(x.get("Date"))
        out[x["SecuritiesCompanyCode"].strip()] = [_pos(_num(x.get("PriceEarningRatio"))), _pos(_num(x.get("PriceBookRatio"))),
                                                   _num(x.get("YieldRatio")) if np.isfinite(_num(x.get("YieldRatio"))) else 0.0]
    if day and tp_day and day != tp_day:
        log.warning("上市與上櫃估值日期不同（%s / %s），以較早者為準", day.date(), tp_day.date())
        day = min(day, tp_day)
    return day or tp_day, out


def load_valuation(dates: pd.DatetimeIndex, codes: list[str], max_fetch: int = 45) -> dict[str, pd.DataFrame]:
    """回傳 {pe, pb, dy} 三張「交易日 × 股票」寬表（向前填補，最多沿用 45 個交易日）。

    每一筆都是當天公布、當天已知的數字，所以回測不會偷看未來。
    """
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    day, today = fetch_val_today()
    if day is not None and today:
        p = VAL_DIR / f"{day:%Y%m%d}.json.gz"
        if not p.exists():
            _gz_write(p, today)
    # 取樣日固定為「每月 1 日與 16 日之後的第一個交易日」，不隨執行日期位移，快取才能重複使用
    ds = pd.Series(dates, index=dates)
    anchors = ds.groupby([dates.year, dates.month, dates.day >= 16]).first()
    have = sorted(pd.Timestamp(f.name[:8]) for f in VAL_DIR.glob("*.json.gz"))
    have_idx = pd.DatetimeIndex(have) if have else pd.DatetimeIndex([])

    def covered(d: pd.Timestamp) -> bool:          # 前後 8 天內已有快照就不必再抓
        return len(have_idx) > 0 and bool((abs(have_idx - d) <= pd.Timedelta(days=8)).any())

    wanted = [d for d in anchors.sort_values(ascending=False) if d != dates[-1] and not covered(d)]
    fetched = failed = 0
    for d in wanted:
        p = VAL_DIR / f"{d:%Y%m%d}.json.gz"
        if p.exists() or fetched >= max_fetch:
            continue
        if failed >= 2:                                # 被交易所限流時不要一直重試，下次執行再補
            log.warning("估值歷史連續抓取失敗，本次先停止補抓")
            break
        if fetched:
            time.sleep(6)                              # 證交所對連續請求很敏感
        fetched += 1
        tw, tp = _fetch_val_twse(d), _fetch_val_tpex(d)
        if tw and tp:
            _gz_write(p, {**tp, **tw})
            failed = 0
        else:
            failed += 1
            log.warning("估值歷史 %s 抓取不完整（上市 %s／上櫃 %s），下次再補", d.date(), bool(tw), bool(tp))
    if fetched:
        log.info("估值歷史：補抓 %d 天", fetched)

    snaps = {}
    for p in sorted(VAL_DIR.glob("*.json.gz")):
        d = pd.Timestamp(p.name[:8])
        if dates[0] <= d <= dates[-1] + pd.Timedelta(days=5):
            snaps[d] = _gz_read(p)
    def to_wide(rows: dict) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=codes)
        # 快照日若不是交易日，對齊到之後第一個交易日
        df.index = dates[np.clip(np.searchsorted(dates.values, pd.DatetimeIndex(df.index).values), 0, len(dates) - 1)]
        df = df[~df.index.duplicated(keep="last")]
        return df.reindex(dates).ffill(limit=45)

    out: dict = {}
    for k, i in (("pe", 0), ("pb", 1), ("dy", 2)):
        out[k] = to_wide({d: {c: (v[i] if v[i] is not None else np.nan) for c, v in s.items()} for d, s in snaps.items()}).astype(float)
    # 有出現在估值表、但本益比空白的，是「虧損」而不是缺資料
    out["listed"] = to_wide({d: {c: 1.0 for c in s} for d, s in snaps.items()}).fillna(0.0).astype(bool)
    out["n_snapshots"] = len(snaps)
    return out


def value_factors(val: dict, close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """估值 → 模型因子。盈餘殖利率用 1/本益比（虧損記為 0，也就是最貴），比本益比本身好處理。"""
    listed = val["listed"]
    ey = (1 / val["pe"]).where(val["pe"].notna(), 0.0).where(listed)
    bp = (1 / val["pb"]).where(listed)
    dy = (val["dy"] / 100).where(listed)
    eps = close / val["pe"]                                     # 近四季 EPS（每股）
    eps_prev = eps.shift(250)
    # 盈餘變化 ÷ 股價：比成長率穩定，不會因為去年 EPS 很小而爆量
    dey = ((eps.fillna(0) - eps_prev.fillna(0)) / close).where(listed & listed.shift(250).fillna(False).astype(bool))
    return {"ey": ey, "bp": bp, "dy": dy, "dey": dey}


# ── 財報、月營收（最新一期）─────────────────────────────────────────────────

def _first(rec: dict, *names, contains: tuple[str, ...] = (), exclude: tuple[str, ...] = ()):
    for n in names:
        if n in rec and str(rec[n]).strip() not in ("", "-"):
            return rec[n]
    if contains:
        for k, v in rec.items():
            if all(w in k for w in contains) and not any(w in k for w in exclude) and str(v).strip() not in ("", "-"):
                return v
    return None


def _code(rec: dict) -> str:
    return str(_first(rec, "公司代號", "SecuritiesCompanyCode") or "").strip()


def fetch_financials() -> dict[str, dict]:
    """抓最新一季財報、營益分析、月營收。抓不到時沿用上次存的快取。"""
    cache = FUND_DIR / "latest.json.gz"
    out: dict[str, dict] = {}
    variants = ("ci", "basi", "bd", "fh", "ins", "mim")
    got_any = False
    for v in variants:
        for url in (f"{TWSE}/opendata/t187ap06_L_{v}", f"{TPEX}/mopsfin_t187ap06_O_{v}"):
            for r in get_json(url, tries=2) or []:
                got_any = True
                c = _code(r)
                d = out.setdefault(c, {"fin_type": v})
                d["year"] = int(_num(_first(r, "年度", "Year")) or 0) if _first(r, "年度", "Year") else None
                d["season"] = int(_num(_first(r, "季別", "Season"))) if _first(r, "季別", "Season") else None
                d["eps_ytd"] = _num(_first(r, contains=("基本每股盈餘",)))
                d["ni_parent"] = _num(_first(r, contains=("歸屬於母公司業主",), exclude=("綜合",)))
                d["revenue_ytd"] = _num(_first(r, "營業收入", "收益", "淨收益"))
                d["op_income"] = _num(_first(r, "營業利益（損失）"))
                d["pretax"] = _num(_first(r, "稅前淨利（淨損）", "繼續營業單位稅前淨利（淨損）"))
        for url in (f"{TWSE}/opendata/t187ap07_L_{v}", f"{TPEX}/mopsfin_t187ap07_O_{v}"):
            for r in get_json(url, tries=2) or []:
                c = _code(r)
                d = out.setdefault(c, {"fin_type": v})
                d["assets"] = _num(_first(r, "資產總計", "資產總額"))
                d["liab"] = _num(_first(r, "負債總計", "負債總額"))
                d["cur_assets"] = _num(_first(r, "流動資產"))
                d["cur_liab"] = _num(_first(r, "流動負債"))
                d["equity_parent"] = _num(_first(r, contains=("歸屬於母公司業主", "權益")))
                d["capital"] = _num(_first(r, "股本"))
                d["bvps"] = _num(_first(r, "每股參考淨值"))
    for url, keys in ((f"{TWSE}/opendata/t187ap17_L", ("毛利率(%)(營業毛利)/(營業收入)", "營業利益率(%)(營業利益)/(營業收入)", "稅後純益率(%)(稅後純益)/(營業收入)")),
                      (f"{TPEX}/mopsfin_187ap17_O", ("毛利率", "營業利益率", "稅後純益率"))):
        for r in get_json(url, tries=2) or []:
            d = out.setdefault(_code(r), {})
            d["gross_margin"], d["op_margin"], d["net_margin"] = (_num(r.get(k)) for k in keys)
    for url in (f"{TWSE}/opendata/t187ap05_L", f"{TPEX}/mopsfin_t187ap05_O"):
        for r in get_json(url, tries=3) or []:
            d = out.setdefault(_code(r), {})
            ym = str(r.get("資料年月", ""))
            d["rev_month"] = f"{int(ym[:-2]) + 1911}-{ym[-2:]}" if len(ym) >= 4 else None
            d["rev"] = _num(r.get("營業收入-當月營收"))
            d["rev_mom"] = _num(r.get("營業收入-上月比較增減(%)"))
            d["rev_yoy"] = _num(r.get("營業收入-去年同月增減(%)"))
            d["rev_cum_yoy"] = _num(r.get("累計營業收入-前期比較增減(%)"))
    out.pop("", None)
    if got_any and len(out) > 500:
        _gz_write(cache, out)
        # 月營收逐月存檔，累積之後可以看營收趨勢
        months = {d.get("rev_month") for d in out.values() if d.get("rev_month")}
        for m in months:
            _gz_write(FUND_DIR / "revenue" / f"{m}.json.gz",
                      {c: [d.get("rev"), d.get("rev_yoy"), d.get("rev_cum_yoy")] for c, d in out.items() if d.get("rev_month") == m})
        return out
    if cache.exists():
        log.warning("財報資料抓取失敗，沿用上次快取")
        return _gz_read(cache)
    return out


def revenue_history(code: str, months: int = 13) -> list[list]:
    """從逐月存檔取出某檔股票的月營收序列 [月份, 營收(千元), 年增率%]。"""
    out = []
    for p in sorted((FUND_DIR / "revenue").glob("*.json.gz"))[-months:]:
        v = _gz_read(p).get(code)
        if v:
            out.append([p.name[:7], v[0], v[1]])
    return out


# ── 三位大師的檢核清單 ─────────────────────────────────────────────────────

def _f(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _crit(name: str, rule: str, value: str, ok: bool | None) -> dict:
    return {"name": name, "rule": rule, "value": value, "pass": ok}


def _score(items: list[dict]) -> int | None:
    xs = [i["pass"] for i in items if i["pass"] is not None]
    return round(100 * sum(xs) / len(xs)) if len(xs) >= 3 else None


def metrics(code: str, price: float, fin: dict, val_now: tuple, val_prev: tuple, price_prev: float, industry: str) -> dict:
    """單一股票的基本面指標與三位大師評分。val_* = (本益比, 淨值比, 殖利率%)。"""
    pe, pb, dy = val_now
    fin = fin or {}
    is_fin = industry in FINANCIAL or fin.get("fin_type") in ("basi", "fh", "ins", "bd")
    eps_ttm = price / pe if pe else None
    pe_prev = val_prev[0] if val_prev else None
    eps_prev = price_prev / pe_prev if (pe_prev and price_prev) else None
    # 一年前虧損（本益比空白）但目前有本益比 → 轉虧為盈
    turnaround = bool(pe and val_prev and val_prev[1] is not None and pe_prev is None)
    eps_g = (eps_ttm / eps_prev - 1) * 100 if (eps_ttm and eps_prev and eps_prev > 0) else None
    bvps = fin.get("bvps") if fin.get("bvps") and np.isfinite(fin.get("bvps")) and fin.get("bvps") > 0 else (price / pb if pb else None)
    season = fin.get("season")
    roe = None
    if season and fin.get("ni_parent") is not None and fin.get("equity_parent") and np.isfinite(fin["ni_parent"]) and fin["equity_parent"] > 0:
        roe = fin["ni_parent"] * 4 / season / fin["equity_parent"] * 100
    roe_src = "最新財報年化"
    if roe is None and pe and pb:
        roe, roe_src = pb / pe * 100, "淨值比 ÷ 本益比（近四季）"      # E/B = (P/B)/(P/E)
    debt = fin["liab"] / fin["assets"] * 100 if fin.get("assets") and np.isfinite(fin.get("liab", np.nan)) and fin["assets"] > 0 else None
    cur = fin["cur_assets"] / fin["cur_liab"] if fin.get("cur_liab") and np.isfinite(fin.get("cur_assets", np.nan)) and fin["cur_liab"] > 0 else None
    core = fin["op_income"] / fin["pretax"] * 100 if (fin.get("pretax") and np.isfinite(fin.get("op_income", np.nan)) and fin["pretax"] > 0) else None
    mcap = price * fin["capital"] * 1000 / 10 / 1e8 if fin.get("capital") and np.isfinite(fin["capital"]) else None   # 億元（面額 10 元）
    gm, om, nm = (fin.get(k) for k in ("gross_margin", "op_margin", "net_margin"))
    gm, om, nm = (x if x is not None and np.isfinite(x) else None for x in (gm, om, nm))
    rev_cum, rev_yoy = fin.get("rev_cum_yoy"), fin.get("rev_yoy")
    rev_cum = rev_cum if rev_cum is not None and np.isfinite(rev_cum) else None
    rev_yoy = rev_yoy if rev_yoy is not None and np.isfinite(rev_yoy) else None
    growth = eps_g if eps_g is not None else rev_cum            # 林區用盈餘成長；沒有時以累計營收成長代替
    growth_src = "近四季 EPS 年增" if eps_g is not None else ("累計營收年增（EPS 歷史不足時代用）" if rev_cum is not None else None)
    peg = pe / growth if (pe and growth and growth > 0) else None
    lynch_ratio = (growth + (dy or 0)) / pe if (pe and growth is not None) else None
    gn = math.sqrt(22.5 * eps_ttm * bvps) if (eps_ttm and bvps and eps_ttm > 0) else None
    mos = (gn / price - 1) * 100 if gn else None

    na = None
    graham = [
        _crit("公司規模", "市值 ≥ 100 億元", f"{mcap:,.0f} 億" if mcap else "–", (mcap >= 100) if mcap else na),
        _crit("財務穩健", "流動比率 ≥ 2", f"{cur:.2f}" if cur else ("金融業不適用" if is_fin else "–"), (cur >= 2) if (cur and not is_fin) else na),
        _crit("持續獲利", "近四季 EPS > 0", f"{eps_ttm:.2f} 元" if eps_ttm else ("虧損" if pb else "–"), (eps_ttm is not None and eps_ttm > 0) if pb else na),
        _crit("盈餘成長", "近四季 EPS 高於一年前", f"{eps_g:+.1f}%" if eps_g is not None else "–", (eps_g > 0) if eps_g is not None else na),
        _crit("穩定配息", "有現金股利", f"殖利率 {dy:.2f}%" if dy is not None else "–", (dy > 0) if dy is not None else na),
        _crit("本益比適中", "本益比 ≤ 15", f"{pe:.1f}" if pe else "–", (pe <= 15) if pe else (False if pb else na)),
        _crit("價格合理", "淨值比 ≤ 1.5 或 本益比×淨值比 ≤ 22.5", f"PB {pb:.2f}" + (f"、PE×PB {pe * pb:.1f}" if pe else "") if pb else "–",
              (pb <= 1.5 or (pe is not None and pe * pb <= 22.5)) if pb else na),
    ]
    buffett = [
        _crit("股東權益報酬率", "ROE（年化）≥ 15%", f"{roe:.1f}%" if roe is not None else "–", (roe >= 15) if roe is not None else na),
        _crit("低負債", "負債比 ≤ 50%", f"{debt:.0f}%" if debt is not None else "–", (debt <= 50) if (debt is not None and not is_fin) else na),
        _crit("定價能力（護城河）", "毛利率 ≥ 30%", f"{gm:.1f}%" if gm is not None else "–", (gm >= 30) if gm is not None else na),
        _crit("本業賺錢", "營業利益率 ≥ 15%", f"{om:.1f}%" if om is not None else "–", (om >= 15) if om is not None else na),
        _crit("獲利來自本業", "營業利益占稅前淨利 ≥ 70%", f"{core:.0f}%" if core is not None else "–", (core >= 70) if core is not None else na),
        _crit("盈餘持續成長", "近四季 EPS > 0 且高於一年前", f"{eps_g:+.1f}%" if eps_g is not None else "–",
              (eps_ttm > 0 and eps_g > 0) if (eps_g is not None and eps_ttm) else na),
        _crit("價格合理", "本益比 ≤ 25（盈餘殖利率 ≥ 4%）", f"{pe:.1f}" if pe else "–", (pe <= 25) if pe else (False if pb else na)),
    ]
    lynch = [
        _crit("PEG", "本益比 ÷ 成長率 ≤ 1", f"{peg:.2f}" if peg else "–", (peg <= 1) if peg else (False if (pe and growth is not None) else na)),
        _crit("成長動能", "累計營收年增 ≥ 15%", f"{rev_cum:+.1f}%" if rev_cum is not None else "–", (rev_cum >= 15) if rev_cum is not None else na),
        _crit("林區比率", "(成長率 + 殖利率) ÷ 本益比 ≥ 1.5", f"{lynch_ratio:.2f}" if lynch_ratio is not None else "–",
              (lynch_ratio >= 1.5) if lynch_ratio is not None else na),
        _crit("本益比低於成長率", "本益比 < 成長率", f"PE {pe:.1f} / 成長 {growth:.1f}%" if (pe and growth is not None) else "–",
              (pe < growth) if (pe and growth is not None) else na),
        _crit("財務保守", "負債比 ≤ 50%", f"{debt:.0f}%" if debt is not None else "–", (debt <= 50) if (debt is not None and not is_fin) else na),
        _crit("近期營收", "最新月營收年增 > 0", f"{rev_yoy:+.1f}%" if rev_yoy is not None else "–", (rev_yoy > 0) if rev_yoy is not None else na),
    ]
    if turnaround:
        cat = "轉機股"
    elif industry in CYCLICAL:
        cat = "景氣循環股"
    elif growth is not None and growth >= 20:
        cat = "快速成長股"
    elif growth is not None and growth >= 10:
        cat = "穩定成長股"
    elif pb and pb < 0.8:
        cat = "資產股"
    elif growth is not None:
        cat = "緩慢成長股"
    else:
        cat = None
    cat_note = {
        "快速成長股": "林區最愛的類型：成長率 20% 以上，重點是成長能否持續、本益比是否已透支。",
        "穩定成長股": "大型穩健公司，林區通常在漲 30–50% 後獲利了結，適合當防守部位。",
        "緩慢成長股": "成長慢，主要賺股利；林區認為除非殖利率夠高，否則吸引力有限。",
        "景氣循環股": "獲利隨景氣大起大落；林區提醒：本益比最低時往往是景氣高峰，反而要小心。",
        "轉機股": "從虧損轉為獲利，若轉機成真，報酬可能很高，但風險也最大。",
        "資產股": "股價低於帳面淨值，價值藏在資產裡，需要催化劑才會反映。",
    }.get(cat)
    return {
        "pe": _f(pe, 1), "pb": _f(pb, 2), "dy": _f(dy, 2), "eps_ttm": _f(eps_ttm), "eps_ytd": _f(fin.get("eps_ytd")),
        "eps_g": _f(eps_g, 1), "bvps": _f(bvps), "roe": _f(roe, 1), "roe_src": roe_src if roe is not None else None, "debt": _f(debt, 1), "cur": _f(cur, 2), "core": _f(core, 0),
        "gm": _f(gm, 1), "om": _f(om, 1), "nm": _f(nm, 1), "mcap": _f(mcap, 0), "season": f"{fin['year']}Q{season}" if fin.get("year") and season else None,
        "rev_month": fin.get("rev_month"), "rev_yoy": _f(rev_yoy, 1), "rev_mom": _f(fin.get("rev_mom"), 1), "rev_cum_yoy": _f(rev_cum, 1),
        "peg": _f(peg, 2), "growth": _f(growth, 1), "growth_src": growth_src, "lynch_ratio": _f(lynch_ratio, 2),
        "graham_number": _f(gn, 1), "mos": _f(mos, 1), "is_fin": is_fin,
        "graham_price_ok": bool(graham[5]["pass"] and graham[6]["pass"]),        # 葛拉漢的兩個價格條件都要過
        "graham": {"score": _score(graham), "items": graham},
        "buffett": {"score": _score(buffett), "items": buffett},
        "lynch": {"score": _score(lynch), "items": lynch, "category": cat, "category_note": cat_note},
    }


def composite(m: dict[str, dict]) -> pd.Series:
    """品質＋成長的綜合分數（橫斷面標準化）。估值另外放進技術模型裡回測，這裡不重複計算。"""
    df = pd.DataFrame({c: {"roe": v["roe"], "gm": v["gm"], "debt": -v["debt"] if v["debt"] is not None and not v["is_fin"] else None,
                           "rev": v["rev_cum_yoy"], "epsg": v["eps_g"]} for c, v in m.items()}).T.astype(float)
    z = (df - df.mean()) / df.std()
    z = z.clip(-3, 3)
    quality = z[["roe", "gm", "debt"]].mean(axis=1)
    growth = z[["rev", "epsg"]].mean(axis=1)
    return pd.concat([quality, growth], axis=1).mean(axis=1).fillna(0.0)
