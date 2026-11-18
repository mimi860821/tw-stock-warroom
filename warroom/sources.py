"""行情與籌碼資料來源：證交所、櫃買中心、Yahoo Finance。全部免金鑰。"""
from __future__ import annotations

import gzip
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from . import config as C

log = logging.getLogger("warroom")
_session = requests.Session()
_session.headers.update({"User-Agent": C.USER_AGENT, "Accept": "application/json,text/xml,*/*"})


def http_get(url: str, *, tries: int = 3, timeout: int = 30, pause: float = 2.0) -> requests.Response | None:
    """帶重試的 GET。失敗回傳 None，由呼叫端決定如何降級。"""
    for i in range(tries):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r
            log.warning("GET %s -> HTTP %s", url[:90], r.status_code)
        except requests.RequestException as e:
            log.warning("GET %s 失敗：%s", url[:90], e)
        time.sleep(pause * (i + 1))
    return None


def get_json(url: str, **kw):
    r = http_get(url, **kw)
    if r is None:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def _num(s) -> float:
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, TypeError):
        return float("nan")


# ── 當日行情快照與股票池 ─────────────────────────────────────────────────

def _is_common(code: str) -> bool:
    """只保留 4 碼普通股（排除 ETF、權證、存託憑證、特別股）。"""
    return len(code) == 4 and code.isdigit() and not code.startswith("00") and not code.startswith("91")


def fetch_snapshot() -> pd.DataFrame:
    """上市＋上櫃最新一日行情，用來決定股票池與取得股票名稱。"""
    rows = []
    twse = get_json("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL") or []
    for x in twse:
        rows.append({"code": x["Code"].strip(), "name": x["Name"].strip(), "market": "TWSE",
                     "value": _num(x.get("TradeValue")), "close": _num(x.get("ClosingPrice"))})
    tpex = get_json("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes") or []
    for x in tpex:
        rows.append({"code": x["SecuritiesCompanyCode"].strip(), "name": x["CompanyName"].strip(),
                     "market": "TPEX", "value": _num(x.get("TransactionAmount")), "close": _num(x.get("Close"))})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("證交所與櫃買中心的行情快照都抓不到，無法決定股票池")
    return df.drop_duplicates("code").set_index("code")


def fetch_industries() -> dict[str, str]:
    out: dict[str, str] = {}
    for url, kc, ki in [
        ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "公司代號", "產業別"),
        ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "SecuritiesCompanyCode", "SecuritiesIndustryCode"),
    ]:
        for x in get_json(url, tries=2) or []:
            code = str(x.get(kc, "")).strip()
            ind = C.INDUSTRY.get(str(x.get(ki, "")).strip())
            if code and ind:
                out[code] = ind
    return out


def build_universe(snap: pd.DataFrame, previous: list[str]) -> pd.DataFrame:
    """依成交金額挑選股票池；前一次已在池內的股票有保留優勢，避免名單每天大幅變動。"""
    prev = set(previous)
    picked: list[str] = []
    for market, n in (("TWSE", C.UNIVERSE_TWSE), ("TPEX", C.UNIVERSE_TPEX)):
        sub = snap[(snap.market == market) & snap.index.map(_is_common)].sort_values("value", ascending=False)
        rank = {c: i for i, c in enumerate(sub.index)}
        keep = [c for c in sub.index if c in prev and rank[c] < n * C.UNIVERSE_STICKY][:n]
        fresh = [c for c in sub.index if c not in set(keep)]
        picked += (keep + fresh)[:n]
    uni = snap.loc[picked, ["name", "market"]].copy()
    for code, name in C.EXTRA_SYMBOLS.items():
        if code not in uni.index:
            uni.loc[code] = {"name": snap.name.get(code, name), "market": snap.market.get(code, "TWSE")}
    return uni


# ── Yahoo Finance 歷史股價 ────────────────────────────────────────────────

def yahoo_symbol(code: str, market: str) -> str:
    return f"{code}.TWO" if market == "TPEX" else f"{code}.TW"


def fetch_yahoo(symbol: str, rng: str = C.HISTORY_RANGE) -> pd.DataFrame | None:
    """回傳以交易日為索引的 OHLCV（含還原權息的 adj 收盤價）。"""
    for host in ("query1", "query2"):
        url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{quote(symbol)}"
               f"?range={rng}&interval=1d&events=div%2Csplits")
        js = get_json(url, tries=2, pause=3.0)
        try:
            res = js["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
            off = res["meta"].get("gmtoffset", 0)
        except (TypeError, KeyError, IndexError):
            continue
        idx = pd.to_datetime([datetime.fromtimestamp(t + off, tz=timezone.utc).date() for t in ts])
        df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                           "adj": adj, "volume": q["volume"]}, index=idx, dtype="float64")
        df = df[~df.index.duplicated(keep="last")].dropna(subset=["close", "adj"])
        df = df[df.close > 0]
        if len(df):
            return df
    log.warning("Yahoo 抓不到 %s", symbol)
    return None


def fetch_many(symbols: dict[str, str], workers: int = 6) -> dict[str, pd.DataFrame]:
    """symbols: {自訂鍵: Yahoo 代號}。平行抓取，抓不到的直接略過。"""
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(workers) as ex:
        for key, df in zip(symbols, ex.map(fetch_yahoo, symbols.values())):
            if df is not None:
                out[key] = df
    return out


def drop_partial_bar(df: pd.DataFrame, now: datetime, open_h: float, close_h: float) -> pd.DataFrame:
    """盤中執行時，最後一根 K 棒還沒收完，拿掉以免把盤中價當成收盤價。now 需為該市場當地時間。"""
    if len(df) and df.index[-1].date() == now.date():
        h = now.hour + now.minute / 60
        if open_h <= h < close_h:
            return df.iloc[:-1]
    return df


# ── 三大法人買賣超（籌碼）────────────────────────────────────────────────

def _chips_path(day: datetime):
    return C.DATA_DIR / "chips" / f"{day:%Y%m%d}.json.gz"


def _fetch_chips_twse(day: datetime) -> dict | None:
    js = get_json(f"https://www.twse.com.tw/rwd/zh/fund/T86?date={day:%Y%m%d}&selectType=ALLBUT0999&response=json",
                  tries=2, pause=8.0)
    if not js or js.get("stat") != "OK" or not js.get("data"):
        return None
    f = js["fields"]
    try:
        i_f, i_t, i_d = f.index("外陸資買賣超股數(不含外資自營商)"), f.index("投信買賣超股數"), f.index("自營商買賣超股數")
    except ValueError:
        return None
    return {r[0].strip(): [int(_num(r[i_f])), int(_num(r[i_t])), int(_num(r[i_d]))] for r in js["data"]}


def _fetch_chips_tpex(day: datetime) -> dict | None:
    js = get_json(f"https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?date={day:%Y}%2F{day:%m}%2F{day:%d}"
                  "&type=Daily&sect=EW&response=json", tries=2)
    try:
        tb = js["tables"][0]
        want = f"{day.year - 1911}/{day:%m}/{day:%d}"
        if tb.get("date") != want or not tb["data"]:
            return None
        # 欄位：10=外資合計買賣超、13=投信買賣超、22=自營商合計買賣超
        return {r[0].strip(): [int(_num(r[10])), int(_num(r[13])), int(_num(r[22]))] for r in tb["data"]}
    except (TypeError, KeyError, IndexError):
        return None


def load_chips(trading_days: list[pd.Timestamp], max_fetch: int = 25) -> pd.DataFrame:
    """回傳長表 (date, code, foreign, trust, dealer)。每個交易日抓一次後存入 data/chips 快取。"""
    (C.DATA_DIR / "chips").mkdir(parents=True, exist_ok=True)
    frames, fetched = [], 0
    for day in reversed(trading_days):
        p = _chips_path(day)
        if p.exists():
            data = json.loads(gzip.decompress(p.read_bytes()))
        elif fetched < max_fetch:
            if fetched:
                time.sleep(5)          # 證交所對連續請求很敏感，間隔太短會被擋
            fetched += 1
            tw, tp = _fetch_chips_twse(day), _fetch_chips_tpex(day)
            if tw is None and tp is None:
                continue
            data = {**(tp or {}), **(tw or {})}
            # 兩邊都抓到才寫快取；只抓到一邊的日子下次再補
            if tw is not None and tp is not None:
                p.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0))
        else:
            continue
        df = pd.DataFrame.from_dict(data, orient="index", columns=["foreign", "trust", "dealer"])
        df["date"] = day
        frames.append(df.rename_axis("code").reset_index())
    if not frames:
        return pd.DataFrame(columns=["date", "code", "foreign", "trust", "dealer"])
    return pd.concat(frames, ignore_index=True)


# ── 融資融券 ──────────────────────────────────────────────────────────────

def _fetch_margin_twse(day: datetime) -> dict | None:
    js = get_json(f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={day:%Y%m%d}&selectType=STOCK&response=json",
                  tries=2, pause=8.0)
    try:
        if js.get("stat") != "OK":
            return None
        tb = next(t for t in js["tables"] if t.get("data") and len(t.get("fields", [])) >= 13)
        # 欄位：0 代號、6 融資今日餘額、12 融券今日餘額（單位：張）
        return {r[0].strip(): [int(_num(r[6])), int(_num(r[12]))] for r in tb["data"] if r[0].strip() and r[0].strip() != "合計"}
    except (AttributeError, StopIteration, KeyError, IndexError, TypeError):
        return None


def _fetch_margin_tpex(day: datetime) -> dict | None:
    js = get_json(f"https://www.tpex.org.tw/www/zh-tw/margin/balance?date={day:%Y}%2F{day:%m}%2F{day:%d}&response=json", tries=2)
    try:
        tb = js["tables"][0]
        if tb.get("date") != f"{day.year - 1911}/{day:%m}/{day:%d}" or not tb["data"]:
            return None
        # 欄位：0 代號、6 資餘額、14 券餘額（單位：張）
        return {r[0].strip(): [int(_num(r[6])), int(_num(r[14]))] for r in tb["data"]}
    except (TypeError, KeyError, IndexError):
        return None


def load_margin(trading_days: list[pd.Timestamp], max_fetch: int = 25) -> pd.DataFrame:
    """回傳長表 (date, code, margin, short)，單位：張。每日抓一次後存入 data/margin 快取。"""
    d_ = C.DATA_DIR / "margin"
    d_.mkdir(parents=True, exist_ok=True)
    frames, fetched = [], 0
    for day in reversed(trading_days):
        p = d_ / f"{day:%Y%m%d}.json.gz"
        if p.exists():
            data = json.loads(gzip.decompress(p.read_bytes()))
        elif fetched < max_fetch:
            if fetched:
                time.sleep(5)
            fetched += 1
            tw, tp = _fetch_margin_twse(day), _fetch_margin_tpex(day)
            if tw is None and tp is None:
                continue
            data = {**(tp or {}), **(tw or {})}
            if tw is not None and tp is not None:
                p.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0))
        else:
            continue
        df = pd.DataFrame.from_dict(data, orient="index", columns=["margin", "short"])
        df["date"] = day
        frames.append(df.rename_axis("code").reset_index())
    if not frames:
        return pd.DataFrame(columns=["date", "code", "margin", "short"])
    return pd.concat(frames, ignore_index=True)


# ── 風險提示與公告：注意股、處置股、除權息、內部人申報轉讓、重大訊息 ─────────────

def _roc_date(s) -> str | None:
    s = str(s or "").replace("/", "").strip()
    try:
        return f"{int(s[:-4]) + 1911}-{s[-4:-2]}-{s[-2:]}"
    except (ValueError, IndexError):
        return None


def fetch_alerts() -> dict[str, dict]:
    """每檔股票的風險提示與公告。任何一個來源失敗都只是少一項資訊，不影響其他。"""
    T, P = "https://openapi.twse.com.tw/v1", "https://www.tpex.org.tw/openapi/v1"
    out: dict[str, dict] = {}

    def add(code, key, item):
        code = str(code or "").strip()
        if code:
            out.setdefault(code, {}).setdefault(key, []).append(item)

    for x in get_json(f"{T}/announcement/notice", tries=2) or []:
        add(x.get("Code"), "notice", {"date": _roc_date(x.get("Date")), "text": (x.get("TradingInfoForAttention") or "").strip()})
    for x in get_json(f"{P}/tpex_trading_warning_information", tries=2) or []:
        add(x.get("SecuritiesCompanyCode"), "notice", {"date": _roc_date(x.get("Date")), "text": (x.get("TradingInformation") or "").strip()})
    for x in get_json(f"{T}/announcement/punish", tries=2) or []:
        add(x.get("Code"), "punish", {"date": _roc_date(x.get("Date")), "period": x.get("DispositionPeriod"),
                                      "text": f"{x.get('ReasonsOfDisposition', '')}，{x.get('DispositionMeasures', '')}".strip("，")})
    for x in get_json(f"{P}/tpex_disposal_information", tries=2) or []:
        add(x.get("SecuritiesCompanyCode"), "punish", {"date": _roc_date(x.get("Date")), "period": x.get("DispositionPeriod"),
                                                       "text": (x.get("DispositionReasons") or "").strip()})
    for x in get_json(f"{T}/exchangeReport/TWT48U_ALL", tries=2) or []:
        add(x.get("Code"), "exdiv", {"date": _roc_date(x.get("Date")), "type": f"除{x.get('Exdividend', '')}",
                                     "cash": _num(x.get("CashDividend")) if np.isfinite(_num(x.get("CashDividend"))) else None})
    for x in get_json(f"{P}/tpex_exright_prepost", tries=2) or []:
        add(x.get("SecuritiesCompanyCode"), "exdiv", {"date": _roc_date(x.get("ExRrightsExDividendDate")), "type": x.get("ExRrightsExDividend", ""),
                                                      "cash": _num(x.get("CashDividend")) if np.isfinite(_num(x.get("CashDividend"))) else None})
    for url, kc in ((f"{T}/opendata/t187ap12_L", "公司代號"), (f"{P}/mopsfin_t187ap12_O", "SecuritiesCompanyCode")):
        for x in get_json(url, tries=2) or []:
            add(x.get(kc), "insider", {"who": x.get("申報人身分") or x.get("申請人身分"), "method": x.get("預定轉讓方式及股數-轉讓方式"),
                                       "shares": _num(x.get("預定轉讓總股數-自有持股")), "period": x.get("有效轉讓期間")})
    for url, kc, ks in ((f"{T}/opendata/t187ap04_L", "公司代號", "主旨 "), (f"{P}/mopsfin_t187ap04_O", "SecuritiesCompanyCode", "主旨")):
        for x in get_json(url, tries=2) or []:
            subj = " ".join(str(x.get(ks) or x.get("主旨") or "").split())
            add(x.get(kc), "material", {"date": _roc_date(x.get("發言日期")), "subject": subj[:160]})
    return out


def fetch_market_flows() -> dict | None:
    """集中市場三大法人當日買賣金額合計（億元）。"""
    js = get_json("https://www.twse.com.tw/rwd/zh/fund/BFI82U?type=day&response=json", tries=2, pause=8.0)
    if not js or js.get("stat") != "OK":
        return None
    out = {"date": js.get("date"), "foreign": 0.0, "trust": 0.0, "dealer": 0.0}
    for name, _, _, net in js["data"]:
        v = _num(net) / 1e8
        if name.startswith("外資"):
            out["foreign"] += v
        elif name.startswith("投信"):
            out["trust"] += v
        elif name.startswith("自營商"):
            out["dealer"] += v
    out["total"] = out["foreign"] + out["trust"] + out["dealer"]
    return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out.items()}


def next_trading_day(day: pd.Timestamp) -> pd.Timestamp:
    """下一個平日（不處理國定假日，只用來判斷美股收盤是否已反映）。"""
    d = day + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d
