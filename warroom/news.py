"""新聞蒐集、去重、對應個股、情緒與衝擊評分（規則式，不需要任何金鑰）。"""
from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

from . import config as C
from .sources import get_json, http_get

log = logging.getLogger("warroom")

RSS_FEEDS = [
    ("Yahoo股市", "https://tw.stock.yahoo.com/rss?category=tw-market"),
    ("Yahoo股市", "https://tw.stock.yahoo.com/rss?category=intl-markets"),
    ("經濟日報", "https://money.udn.com/rssfeed/news/1001/5590?ch=money"),
    ("經濟日報", "https://money.udn.com/rssfeed/news/1001/5591?ch=money"),
    ("經濟日報", "https://money.udn.com/rssfeed/news/1001/5588?ch=money"),
    ("中央社", "https://feeds.feedburner.com/rsscna/finance"),
]
GOOGLE_QUERIES = ["台股", "美股 OR 費半 OR 那斯達克", "外資 買超 OR 賣超", "聯準會 OR 關稅 OR 降息 OR 升息", "半導體 OR AI 伺服器"]
CNYES_CATS = [("tw_stock", 3), ("wd_stock", 1), ("us_stock", 1)]

# 情緒字典：詞 → 權重。標題命中加倍。
POS = {
    "漲停": 3, "飆漲": 3, "狂飆": 3, "噴出": 3, "暴漲": 3, "大漲": 2, "勁揚": 2, "創新高": 3, "新高": 2, "創高": 2,
    "利多": 2, "看好": 2, "上修": 2, "調升": 2, "調高": 2, "優於預期": 3, "超乎預期": 3, "報喜": 2, "買超": 2,
    "加碼": 1, "成長": 1, "年增": 1, "月增": 1, "獲利": 1, "接單": 1, "擴產": 1, "受惠": 2, "回升": 1, "反彈": 1,
    "強勢": 2, "突破": 2, "轉盈": 2, "旺": 1, "滿載": 2, "供不應求": 3, "漲價": 2, "搶單": 2, "大單": 2,
    "庫藏股": 1, "護盤": 1, "買進": 1, "升評": 3, "目標價上調": 3, "走高": 1, "收紅": 1, "上漲": 1, "亮眼": 2,
    "復甦": 2, "回溫": 2, "填息": 2, "拉抬": 1, "降息": 1,
}
NEG = {
    "跌停": 3, "暴跌": 3, "崩跌": 3, "重挫": 3, "崩盤": 3, "股災": 3, "血洗": 3, "屠殺": 3, "慘": 2, "大跌": 2,
    "殺盤": 2, "利空": 2, "看壞": 2, "下修": 2, "調降": 2, "不如預期": 3, "低於預期": 3, "賣超": 2, "減碼": 1,
    "衰退": 2, "年減": 1, "月減": 1, "虧損": 2, "轉虧": 3, "砍單": 3, "遞延": 2, "延後": 1, "下滑": 1, "警示": 1,
    "疑慮": 1, "隱憂": 1, "違約": 3, "掏空": 3, "裁罰": 2, "處分": 1, "降評": 3, "目標價下調": 3, "走低": 1,
    "收黑": 1, "下跌": 1, "失守": 2, "跌破": 2, "破底": 2, "賣壓": 2, "拋售": 2, "恐慌": 2, "停工": 2, "罷工": 2,
    "制裁": 2, "禁令": 2, "關稅": 1, "升息": 1, "戰爭": 2, "斷頭": 3, "追繳": 2, "減產": 2, "庫存過高": 2,
    "腰斬": 3, "辭職": 1, "下市": 3, "風暴": 2, "承壓": 1, "拖累": 2, "瓶頸": 1,
}
# 代表事件量級的字眼：命中越多，越可能造成大波動
MAGNITUDE = ["史上", "歷史", "創紀錄", "股災", "崩盤", "暴跌", "暴漲", "重挫", "漲停", "跌停", "熔斷", "關稅", "聯準會",
             "Fed", "升息", "降息", "戰爭", "制裁", "禁令", "財報", "法說", "辭職", "併購", "收購", "下市", "違約",
             "砍單", "地震", "停工", "風暴", "恐慌", "血洗", "護盤", "國安基金", "非農", "CPI", "通膨"]
MARKET_WORDS = ["台股", "大盤", "加權指數", "櫃買", "外資", "三大法人", "美股", "費半", "那斯達克", "道瓊", "標普",
                "聯準會", "Fed", "關稅", "匯率", "新台幣", "亞股"]
NEGATORS = ("不", "未", "無", "免", "非", "難", "止")

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_CODE = re.compile(r"[（(]\s*(\d{4,6})(?:[-.\s]?(?:TW|TWO|tw))?\s*[)）]")
_SPLIT = re.compile(r"[。！？；!?;\n]")


def _clean(s: str | None) -> str:
    s = html.unescape(html.unescape(s or ""))
    return _SPACE.sub(" ", _TAG.sub("", s)).replace("　", " ").strip()


def _parse_rss(source: str, xml: bytes) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return out
    for it in root.iter("item"):
        title = _clean(it.findtext("title"))
        try:
            ts = parsedate_to_datetime(it.findtext("pubDate") or "")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        src = _clean(it.findtext("source")) or source
        if src and title.endswith(f" - {src}"):          # Google News 會把來源接在標題後面
            title = title[: -len(src) - 3].strip()
        summary = _clean(it.findtext("description"))
        if summary.startswith(title[:12]) and len(summary) < len(title) + 40:
            summary = ""                                   # Google News 的摘要只是重複標題
        if title:
            out.append({"title": title, "summary": summary[:400], "url": (it.findtext("link") or "").strip(),
                        "source": src, "time": ts})
    return out


def fetch_all(hours: int, now: datetime) -> list[dict]:
    items: list[dict] = []
    for cat, pages in CNYES_CATS:
        for page in range(1, pages + 1):
            js = get_json(f"https://api.cnyes.com/media/api/v1/newslist/category/{cat}?limit=30&page={page}", tries=2)
            for x in (js or {}).get("items", {}).get("data", []):
                try:
                    ts = datetime.fromtimestamp(int(x["publishAt"]), tz=timezone.utc)
                except (KeyError, ValueError, TypeError):
                    continue
                items.append({"title": _clean(x.get("title")), "summary": _clean(x.get("summary") or x.get("content"))[:400],
                              "url": f"https://news.cnyes.com/news/id/{x.get('newsId')}", "source": "鉅亨網", "time": ts,
                              "tagged": [m.get("code") for m in (x.get("market") or []) if isinstance(m, dict)]})
    for source, url in RSS_FEEDS:
        r = http_get(url, tries=2)
        if r is not None:
            items += _parse_rss(source, r.content)
    for q in GOOGLE_QUERIES:
        r = http_get(f"https://news.google.com/rss/search?q={quote(q + ' when:1d')}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", tries=2)
        if r is not None:
            items += _parse_rss("Google 新聞", r.content)[:40]
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=hours)
    items = [x for x in items if x["title"] and cutoff <= x["time"] <= now.astimezone(timezone.utc) + timedelta(minutes=10)]
    log.info("新聞：抓到 %d 則（近 %d 小時）", len(items), hours)
    return items


# ── 去重 ──────────────────────────────────────────────────────────────────

def _grams(title: str) -> set[str]:
    t = re.sub(r"[^\w]", "", title)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def cluster(items: list[dict], thresh: float = 0.42) -> list[dict]:
    """把報導同一件事的新聞併成一則，並記錄有幾家媒體報導（越多越重要）。"""
    reps: list[dict] = []
    for it in sorted(items, key=lambda x: x["time"]):
        g = _grams(it["title"])
        if not g:
            continue
        for rep in reps:
            inter = len(g & rep["_g"])
            if inter and inter / (len(g) + len(rep["_g"]) - inter) >= thresh:
                rep["sources"].add(it["source"])
                rep["dup"] += 1
                if len(it["summary"]) > len(rep["summary"]):
                    rep["summary"] = it["summary"]
                rep["tagged"] = list(set(rep.get("tagged", [])) | set(it.get("tagged", [])))
                break
        else:
            reps.append({**it, "_g": g, "sources": {it["source"]}, "dup": 1, "tagged": it.get("tagged", [])})
    for rep in reps:
        rep.pop("_g")
        rep["sources"] = sorted(rep["sources"])
    return reps


# ── 情緒 ──────────────────────────────────────────────────────────────────

def _hits(text: str, lex: dict[str, int]) -> tuple[float, float, list[str]]:
    """回傳 (正常命中的分數, 被否定詞反轉的分數, 命中的詞)。"""
    score, flipped, words = 0.0, 0.0, []
    for w in sorted(lex, key=len, reverse=True):       # 先比長詞並遮蔽，避免「創新高」又被「新高」重複計分
        wt = lex[w]
        i = text.find(w)
        while i >= 0:
            if i > 0 and text[i - 1] in NEGATORS:      # 「不看好」「未下修」：算到相反那一邊，力道減半
                flipped += wt * 0.5
            else:
                score += wt
                words.append(w)
            text = text[:i] + "□" * len(w) + text[i + len(w):]
            i = text.find(w, i + len(w))
    return score, flipped, words


def sentiment(title: str, body: str = "") -> tuple[float, list[str], list[str]]:
    """回傳 (−1~+1 的情緒分數, 命中的正面詞, 命中的負面詞)。標題的權重是內文的兩倍。"""
    pt, pt_f, pw1 = _hits(title, POS)
    nt, nt_f, nw1 = _hits(title, NEG)
    pb, pb_f, pw2 = _hits(body, POS)
    nb, nb_f, nw2 = _hits(body, NEG)
    pos = 2 * (pt + nt_f) + pb + nb_f
    neg = 2 * (nt + pt_f) + nb + pb_f
    return (pos - neg) / (pos + neg + 3), sorted(set(pw1 + pw2)), sorted(set(nw1 + nw2))


def build_matcher(names: dict[str, str]):
    """names: {代號: 名稱}。回傳函式：文字 → {代號: [提到該股的句子]}。"""
    alias: list[tuple[str, str]] = []
    for code, name in names.items():
        for a in {name, re.sub(r"(-KY|\*|-創|-DR)$", "", name)}:
            if len(a) >= 2:
                alias.append((a, code))
    alias.sort(key=lambda x: -len(x[0]))

    def match(title: str, body: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for seg_i, seg in enumerate([title] + _SPLIT.split(body)):
            if not seg:
                continue
            work = seg
            for m in _CODE.finditer(seg):
                if m.group(1) in names:
                    found.setdefault(m.group(1), []).append(seg if seg_i else "")
            for a, code in alias:
                if a in work:
                    work = work.replace(a, "□" * len(a))      # 遮蔽已比對到的名稱，避免「華新」誤中「華新科」
                    found.setdefault(code, []).append(seg if seg_i else "")
        return found

    return match


def analyse(items: list[dict], names: dict[str, str], now: datetime) -> dict:
    """輸出：整理後的新聞清單、各股情緒、市場情緒。"""
    match = build_matcher(names)
    news, per_stock = [], {}
    for it in cluster(items):
        title, body = it["title"], it["summary"]
        s, pw, nw = sentiment(title, body)
        hit = match(title, body)
        for c in it.get("tagged") or []:
            if c in names:
                hit.setdefault(c, [])
        if len(hit) > 6:
            # 一次提到很多檔的是盤勢彙整，不代表個股消息；只保留標題點名的股票
            hit = {c: segs for c, segs in hit.items() if "" in segs}
        mag = [w for w in MAGNITUDE if w in title or w in body]
        is_mkt = any(w in title or w in body for w in MARKET_WORDS)
        impact = min(it["dup"], 5) * 1.0 + min(len(mag), 4) * 1.5 + abs(s) * 3 + (1 if is_mkt else 0) \
            + sum(1.5 for w in mag if w in title)
        age_h = max((now.astimezone(timezone.utc) - it["time"]).total_seconds() / 3600, 0)
        rec = {
            "title": title, "summary": body[:220], "url": it["url"], "source": "、".join(it["sources"][:3]),
            "n_sources": len(it["sources"]), "dup": it["dup"], "time": it["time"].astimezone(C.TZ_TW).strftime("%m-%d %H:%M"),
            "sent": round(s, 2), "pos": pw[:6], "neg": nw[:6], "mag": mag[:6], "impact": round(impact, 1),
            "market": is_mkt, "stocks": sorted(hit)[:12],
        }
        news.append(rec)
        w_time = 0.5 ** (age_h / 24)          # 越新的新聞權重越高（半衰期 24 小時）
        for code, segs in hit.items():
            in_title = "" in segs
            local = "。".join(x for x in segs if x)
            ls, _, _ = sentiment(title if in_title else "", local)
            if ls == 0:
                ls = s * 0.3                    # 該股所在句子沒有明確情緒時，只借用整篇情緒的三成
            st = per_stock.setdefault(code, {"sum": 0.0, "w": 0.0, "n": 0, "items": []})
            w = w_time * (1.5 if in_title else 1.0)
            st["sum"] += ls * w
            st["w"] += w
            st["n"] += 1
            st["items"].append({"title": title, "url": it["url"], "source": rec["source"], "time": rec["time"],
                                "sent": round(ls, 2), "impact": rec["impact"]})
    news.sort(key=lambda x: -x["impact"])
    stock_sent = {}
    for code, st in per_stock.items():
        avg = st["sum"] / st["w"] if st["w"] else 0.0
        # 新聞則數越多越可信：用 n/(n+2) 收縮，單一則新聞不會造成滿額調整
        stock_sent[code] = {"score": round(avg * st["n"] / (st["n"] + 2), 3), "n": st["n"],
                            "items": sorted(st["items"], key=lambda x: -x["impact"])[:6]}
    mk = [x for x in news if x["market"]]
    wsum = sum(x["impact"] for x in mk) or 1.0
    market_sent = sum(x["sent"] * x["impact"] for x in mk) / wsum if mk else 0.0
    return {"news": news, "stock_sent": stock_sent, "market_sent": round(market_sent, 3),
            "counts": {"total": len(news), "pos": sum(x["sent"] > 0.15 for x in news),
                       "neg": sum(x["sent"] < -0.15 for x in news)}}


def explain(rec: dict) -> str:
    """把一則新聞的評分依據寫成一句話。"""
    bits = []
    if rec["n_sources"] > 1 or rec["dup"] > 1:
        bits.append(f"{max(rec['n_sources'], 1)} 個來源、共 {rec['dup']} 則報導同一事件")
    if rec["mag"]:
        bits.append("含重大事件字眼「" + "、".join(rec["mag"][:4]) + "」")
    if rec["neg"]:
        bits.append("負面用語「" + "、".join(rec["neg"][:4]) + "」")
    if rec["pos"]:
        bits.append("正面用語「" + "、".join(rec["pos"][:4]) + "」")
    return "；".join(bits) or "無明顯情緒字眼"
