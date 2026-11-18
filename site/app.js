/* 台股戰情室前端：讀取 data/*.json 並繪製。沒有框架、沒有建置步驟。
   電腦版：分頁＋多欄，一個畫面看完，各區塊內部捲動；手機版：區塊依序排列，往下滑。 */
(() => {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const C = { up: css('--up'), dn: css('--dn'), text: css('--text'), text2: css('--text-2'), text3: css('--text-3'),
              line: css('--line'), accent: css('--accent'), s2: css('--series-2') };
  const desktop = () => matchMedia('(min-width: 1100px)').matches;

  // 外部文字（新聞標題、公司名稱、公告）一律跳脫後才放進 HTML
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const safeUrl = (u) => (/^https?:\/\//i.test(u || '') ? u : '#');
  const cls = (v) => (v > 0 ? 'up' : v < 0 ? 'dn' : '');
  const arrow = (v) => (v > 0 ? '▲' : v < 0 ? '▼' : '');
  const pct = (v, nd = 2) => (v == null ? '–' : `${v > 0 ? '+' : ''}${v.toFixed(nd)}%`);
  const signed = (v, nd = 2) => (v == null ? '–' : `<span class="num ${cls(v)}">${arrow(v)} ${pct(v, nd)}</span>`);
  const num = (v, nd = 2) => (v == null || !isFinite(v) ? '–' : v.toLocaleString('zh-TW', { minimumFractionDigits: nd, maximumFractionDigits: nd }));
  const intf = (v) => (v == null ? '–' : `${v > 0 ? '+' : ''}${Math.round(v).toLocaleString()}`);
  const getJSON = (p) => fetch(p, { cache: 'no-cache' }).then((r) => { if (!r.ok) throw new Error(`${p} ${r.status}`); return r.json(); });
  const ACT = { strong_buy: '強力看多', buy: '偏多', neutral: '中立', sell: '偏空', strong_sell: '強力看空' };
  const ALERT = { punish: ['處置股', 'warn'], notice: ['注意股', 'warn'], insider: ['內部人申報轉讓', 'neutral'], exdiv: ['除權息', 'neutral'], material: ['重大訊息', 'neutral'] };

  let INDEX = [], IDX = {}, LATEST = null, REVIEW = null, PSTAT = {};
  const state = { tab: 'dash', code: null, list: 'bull', ind: '', stockTab: 'why', side: 'major', screen: 'model', sort: null, stock: null };
  const rendered = new Set();
  const charts = {};
  const mount = (id) => { charts[id]?.dispose(); const el = document.getElementById(id); charts[id] = echarts.init(el, null, { renderer: 'canvas' }); return charts[id]; };
  addEventListener('resize', () => Object.values(charts).forEach((c) => c.resize()));

  const axisStyle = { axisLine: { lineStyle: { color: C.line } }, axisTick: { show: false }, axisLabel: { color: C.text3, fontSize: 11 },
                      splitLine: { lineStyle: { color: C.line, opacity: 0.5 } } };
  const tipStyle = { backgroundColor: '#0b0d10', borderColor: C.line, textStyle: { color: C.text, fontSize: 12 }, padding: [7, 9] };
  const row = (k, v, color) => `<div style="display:flex;justify-content:space-between;gap:16px"><span style="color:${C.text2}">${color ? `<i style="display:inline-block;width:10px;height:2px;background:${color};vertical-align:middle;margin-right:6px"></i>` : ''}${k}</span><b>${v}</b></div>`;

  /* ── 圖表 ─────────────────────────────────────────────────────────── */
  function priceChart(id, ohlc, path, { candle = false, compact = false } = {}) {
    const hist = ohlc.map((r) => r[0].slice(5));
    const x = hist.concat(path.slice(1).map((p) => `+${p[0]}日`));
    const n = hist.length;
    const pad = (arr) => Array(n - 1).fill(null).concat(arr);
    const mid = pad(path.map((p) => p[1])), lo = pad(path.map((p) => p[2])), hi = pad(path.map((p) => p[3]));
    const closes = ohlc.map((r) => r[2]);
    const ma = (w) => closes.map((_, i) => (i + 1 < w ? null : +(closes.slice(i + 1 - w, i + 1).reduce((a, b) => a + b, 0) / w).toFixed(2)));
    const ma20 = ma(20), ma60 = ma(60);
    const series = [];
    if (candle) {
      series.push({ type: 'candlestick', name: 'K', data: ohlc.map((r) => [r[1], r[2], r[3], r[4]]),
                    itemStyle: { color: C.up, color0: C.dn, borderColor: C.up, borderColor0: C.dn } },
                  { type: 'line', name: 'MA20', data: ma20, showSymbol: false, lineStyle: { width: 1.5, color: C.accent }, itemStyle: { color: C.accent } },
                  { type: 'line', name: 'MA60', data: ma60, showSymbol: false, lineStyle: { width: 1.5, color: C.s2 }, itemStyle: { color: C.s2 } });
    } else {
      series.push({ type: 'line', name: '收盤', data: closes, showSymbol: false, lineStyle: { width: 2, color: C.accent }, itemStyle: { color: C.accent } });
    }
    series.push({ type: 'line', name: 'lo', data: lo, stack: 'band', showSymbol: false, lineStyle: { opacity: 0 }, silent: true },
                { type: 'line', name: 'band', data: hi.map((h, i) => (h == null ? null : +(h - lo[i]).toFixed(2))), stack: 'band', showSymbol: false,
                  lineStyle: { opacity: 0 }, areaStyle: { color: 'rgba(232,235,240,.14)' }, silent: true },
                { type: 'line', name: '預期中位', data: mid, showSymbol: false, lineStyle: { width: 2, type: 'dashed', color: C.text }, itemStyle: { color: C.text } });
    const left = compact ? 44 : 54;
    const grids = [{ left, right: 10, top: 8, bottom: candle ? 88 : (compact ? 20 : 26) }];
    const xAxes = [{ type: 'category', data: x, boundaryGap: candle, ...axisStyle, splitLine: { show: false },
                     axisLabel: { ...axisStyle.axisLabel, show: !candle }, axisPointer: { label: { show: false } } }];
    const yAxes = [{ type: 'value', scale: true, ...axisStyle, splitNumber: compact ? 3 : 5 }];
    if (candle) {
      grids.push({ left, right: 10, height: 46, bottom: 22 });
      xAxes.push({ type: 'category', data: x, gridIndex: 1, boundaryGap: true, ...axisStyle, splitLine: { show: false } });
      yAxes.push({ type: 'value', gridIndex: 1, ...axisStyle, splitNumber: 2, axisLabel: { ...axisStyle.axisLabel, formatter: (v) => (v >= 1e4 ? `${(v / 1e4).toFixed(0)}萬` : v) } });
      series.push({ type: 'bar', name: '成交量', xAxisIndex: 1, yAxisIndex: 1, barMaxWidth: 7,
                    data: ohlc.map((r) => ({ value: Math.round(r[5] / 1000), itemStyle: { color: r[2] >= r[1] ? C.up : C.dn, opacity: 0.7 } })) });
    }
    mount(id).setOption({
      animation: false, grid: grids, xAxis: xAxes, yAxis: yAxes, series, axisPointer: { link: [{ xAxisIndex: 'all' }] },
      tooltip: { trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: C.text3 } }, ...tipStyle,
        formatter: (ps) => {
          const i = ps[0].dataIndex;
          if (i >= n) { const p = path[i - n + 1]; return `<b>第 ${p[0]} 個交易日（預測）</b>${row('預期中位', num(p[1]))}${row('68% 區間', `${num(p[2])} ~ ${num(p[3])}`)}`; }
          const r = ohlc[i];
          if (!candle) return `<b>${r[0]}</b>${row('收盤', num(r[2]), C.accent)}`;
          return `<b>${r[0]}</b>${row('開 / 收', `${num(r[1])} / ${num(r[2])}`)}${row('高 / 低', `${num(r[4])} / ${num(r[3])}`)}`
            + row('MA20', ma20[i] == null ? '–' : num(ma20[i]), C.accent) + row('MA60', ma60[i] == null ? '–' : num(ma60[i]), C.s2)
            + row('成交量', `${Math.round(r[5] / 1000).toLocaleString()} 張`);
        } },
      dataZoom: candle ? [{ type: 'inside', xAxisIndex: [0, 1], start: 40, end: 100 }] : [],
    });
  }

  function barChart(id, labels, values, { base = 0, faded = [], unit = '', details = [] } = {}) {
    mount(id).setOption({
      animation: false, grid: { left: 40, right: 10, top: 10, bottom: 24 },
      xAxis: { type: 'category', data: labels, ...axisStyle, splitLine: { show: false }, axisLine: { show: false } },
      yAxis: { type: 'value', ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: (v) => +(v + base).toFixed(1) } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(255,255,255,.05)' } }, ...tipStyle,
        formatter: (ps) => { const i = ps[0].dataIndex; return `<b>${labels[i]}</b>${row('數值', values[i] == null ? '–' : values[i] + unit)}${details[i] || ''}`; } },
      series: [{ type: 'bar', barMaxWidth: 12,
        data: values.map((v, i) => ({ value: v == null ? null : +(v - base).toFixed(2), itemStyle: { color: v >= base ? C.up : C.dn, opacity: faded[i] ? 0.4 : 1,
                                                                                     borderRadius: v >= base ? [3, 3, 0, 0] : [0, 0, 3, 3] } })),
        markLine: { silent: true, symbol: 'none', lineStyle: { color: C.text3, type: 'solid', width: 1 }, label: { show: false }, data: [{ yAxis: 0 }] } }],
    });
  }

  const spark = (vals, up) => {
    const lo = Math.min(...vals), hi = Math.max(...vals), w = 120, h = 24;
    const pts = vals.map((v, i) => `${(i / (vals.length - 1) * w).toFixed(1)},${(h - 2 - (v - lo) / (hi - lo || 1) * (h - 4)).toFixed(1)}`).join(' ');
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="${up ? C.up : C.dn}" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>`;
  };

  const contribRows = (parts, total) => {
    const max = Math.max(...parts.map((p) => Math.abs(p.contrib || 0)), 0.01);
    const one = (p, extra = '') => {
      const v = p.contrib || 0, w = Math.min(Math.abs(v) / max * 50, 50);
      return `<div class="contrib ${extra}"><div class="t"><b>${esc(p.label)}</b>${p.value ? ` <span class="chip">${esc(p.value)}</span>` : ''}<small>${esc(p.text || '')}</small></div>
        <div class="bar"><i class="${v >= 0 ? 'pos' : 'neg'}" style="width:${w}%"></i></div><div class="n ${cls(v)}">${pct(v)}</div></div>`;
    };
    return parts.map((p) => one(p)).join('') + (total != null ? one({ label: '合計：15 日預期報酬', contrib: total }, 'total') : '');
  };
  const empty = (t) => `<p class="empty">${esc(t)}</p>`;

  /* ── 頁首指標列 ───────────────────────────────────────────────────── */
  function renderTicker(L) {
    const m = L.market, f = m.flows, us = Object.fromEntries(L.us.panel.map((p) => [p.symbol, p]));
    const items = [
      ['加權指數', `${num(m.close)}<small class="${cls(m.chg)}">${pct(m.chg)}</small>`],
      ['15 日預期', `<span class="${cls(m.mu['15'])}">${pct(m.mu['15'])}</span><small>${esc(m.view)}</small>`],
      ['上漲機率', `${m.p_up}%`],
      ['市場寬度', `${m.breadth}%`],
      f && ['外資買賣超', `<span class="${cls(f.foreign)}">${intf(f.foreign)}</span><small>億</small>`],
      f && ['投信買賣超', `<span class="${cls(f.trust)}">${intf(f.trust)}</span><small>億</small>`],
      us['^SOX'] && ['費城半導體', `${num(us['^SOX'].close, 0)}<small class="${cls(us['^SOX'].chg)}">${pct(us['^SOX'].chg)}</small>`],
      us['^GSPC'] && ['S&amp;P 500', `${num(us['^GSPC'].close, 0)}<small class="${cls(us['^GSPC'].chg)}">${pct(us['^GSPC'].chg)}</small>`],
      us['NQ=F'] && ['那指期貨', `${num(us['NQ=F'].close, 0)}<small class="${cls(us['NQ=F'].chg)}">${pct(us['NQ=F'].chg)}</small>`],
      us['^VIX'] && ['VIX', `${num(us['^VIX'].close)}<small class="${cls(-us['^VIX'].chg)}">${pct(us['^VIX'].chg)}</small>`],
      L.us.adr && ['台積電 ADR 溢價', `<span class="${cls(L.us.adr.premium)}">${pct(L.us.adr.premium, 1)}</span>`],
      ['新聞情緒', `<span class="${cls(L.news.market_sent)}">${L.news.market_sent > 0 ? '+' : ''}${L.news.market_sent.toFixed(2)}</span>`],
      ['看多 / 看空', `<span class="up">${m.dist.strong_buy + m.dist.buy}</span> / <span class="dn">${m.dist.strong_sell + m.dist.sell}</span>`],
    ].filter(Boolean);
    $('#ticker').innerHTML = items.map(([k, v]) => `<div class="tk"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  }

  /* ── 名單 ─────────────────────────────────────────────────────────── */
  const masterScore = (x) => { const s = [x.gs, x.bs, x.ls].filter((v) => v != null); return s.length >= 2 ? s.reduce((a, b) => a + b, 0) / s.length : null; };
  const patName = (k) => PSTAT[k]?.name || k;

  function listRows() {
    let rows = INDEX.filter((x) => !state.ind || x.i === state.ind);
    if (state.list === 'bull') rows = rows.slice(0, 40);
    else if (state.list === 'bear') rows = rows.slice().reverse().slice(0, 40);
    else if (state.list === 'master') rows = rows.filter((x) => masterScore(x) != null).sort((a, b) => masterScore(b) - masterScore(a)).slice(0, 40);
    return rows;
  }
  const HINT = { bull: '相對強度排名前段（依信心值排序）', bear: '相對強度排名後段', master: '葛拉漢、巴菲特、林區三套評分的平均，越高越符合大師條件', all: '全部個股，依相對強度排名' };

  function renderList() {
    $('#list-hint').textContent = HINT[state.list];
    const rows = listRows();
    $('#list').innerHTML = rows.length ? rows.map((x) => {
      const tags = [x.i];
      if (state.list === 'master') tags.push(`葛 ${x.gs ?? '–'}・巴 ${x.bs ?? '–'}・林 ${x.ls ?? '–'}`);
      if (x.lc) tags.push(x.lc);
      x.pt.filter((p) => p[1] === 0 && PSTAT[p[0]]?.used).forEach((p) => tags.push(patName(p[0])));
      if (x.ts >= 3) tags.push(`投信連買 ${x.ts}`);
      if (x.fs >= 3) tags.push(`外資連買 ${x.fs}`);
      x.al.filter((a) => a === 'punish' || a === 'notice').forEach((a) => tags.push(ALERT[a][0]));
      return `<div class="li${x.c === state.code ? ' sel' : ''}" data-code="${esc(x.c)}">
        <div class="rk">${x.r}</div>
        <div class="nm"><b>${esc(x.n)}</b><small>${esc(x.c)}</small></div>
        <div class="rt num ${cls(x.mu)}">${pct(x.mu, 1)}</div>
        <div class="sub">${signed(x.chg)} <span class="hint" style="margin:0">${num(x.px)}</span></div>
        <div class="rt2">信心 ${x.conf.toFixed(1)}・${ACT[x.a]}</div>
        ${tags.length ? `<div class="why">${tags.map(esc).join('・')}</div>` : ''}</div>`;
    }).join('') : empty('沒有符合的個股');
  }

  /* ── 個股 ─────────────────────────────────────────────────────────── */
  async function selectStock(code, { focus = false, switchTab = true } = {}) {
    if (!IDX[code]) { $('#stock-head').innerHTML = empty(`「${code}」不在今日的股票池內（共 ${INDEX.length} 檔，以成交金額選出）。`); return; }
    state.code = code;
    $$('.li').forEach((el) => el.classList.toggle('sel', el.dataset.code === code));
    setHash();
    if (switchTab && state.tab !== 'dash') showTab('dash');
    const panel = $('#stock-panel');
    panel.style.opacity = 0.55;
    let s;
    try { s = await getJSON(`data/stocks/${encodeURIComponent(code)}.json`); } catch (e) { panel.style.opacity = 1; $('#stock-head').innerHTML = empty(`讀取 ${code} 失敗`); return; }
    if (state.code !== code) return;
    state.stock = s;
    panel.style.opacity = 1;
    renderStockHead(s);
    priceChart('stock-chart', s.ohlc, s.forecast.path, { candle: true });
    renderStockTab();
    if (focus && !desktop()) panel.scrollIntoView({ behavior: 'smooth' });
  }

  function alertBadges(al) {
    return Object.keys(ALERT).filter((k) => al?.[k]?.length && k !== 'material')
      .map((k) => `<span class="badge ${ALERT[k][1]}">${ALERT[k][0]}${k === 'exdiv' ? ` ${esc((al.exdiv[0].date || '').slice(5))}` : ''}</span>`).join(' ');
  }

  function renderStockHead(s) {
    const f = s.forecast, L = s.levels, I = s.indicators, fu = s.fund || {};
    const bear = f.p_win < 50;
    $('#stock-head').innerHTML = `
      <div class="sh-top"><span class="nm">${esc(s.name)} <small class="hint">${esc(s.code)}・${esc(s.market)}・${esc(s.industry)}</small></span>
        <span class="px">${num(s.close)}</span>${signed(s.chg)}<span class="badge ${f.action}">${esc(f.label)}</span>
        <span class="hint" style="margin:0">排名 ${s.rank} / ${s.of}</span>${alertBadges(s.alerts)}</div>
      <div class="verdict"><b>${esc(f.summary)}</b> <span>${esc(f.advice)}</span></div>
      <div class="kpis">
        <div class="kpi"><div class="k">15 日預期報酬</div><div class="v ${cls(f.mu['15'])}">${pct(f.mu['15'])}</div><div class="s">5日 ${pct(f.mu['5'], 1)}・10日 ${pct(f.mu['10'], 1)}</div></div>
        <div class="kpi"><div class="k">信心（${bear ? '落後' : '贏過'}中位數）</div><div class="v">${f.conf.toFixed(1)}</div><div class="meter split" style="margin-top:4px"><i style="width:${f.p_win}%"></i></div></div>
        <div class="kpi"><div class="k">15 日後預期價</div><div class="v">${num(L.target)}</div><div class="s">${num(f.path[15][2], 1)} ~ ${num(f.path[15][3], 1)}</div></div>
        <div class="kpi"><div class="k">支撐 / 壓力</div><div class="v">${num(L.support, 1)} / ${num(L.resistance, 1)}</div><div class="s">月線 ${num(L.ma20, 1)}・季線 ${num(L.ma60, 1)}</div></div>
        <div class="kpi"><div class="k">停損參考</div><div class="v">${num(L.stop)}</div><div class="s">收盤 − 2×ATR（${(2 * L.atr / s.close * 100).toFixed(1)}%）</div></div>
        <div class="kpi"><div class="k">本益比・殖利率</div><div class="v">${fu.pe != null ? num(fu.pe, 1) : '–'} ・ ${fu.dy != null ? fu.dy.toFixed(2) + '%' : '–'}</div><div class="s">淨值比 ${fu.pb ?? '–'}・ROE ${fu.roe != null ? fu.roe + '%' : '–'}</div></div>
        <div class="kpi"><div class="k">風險</div><div class="v">β ${I.beta}</div><div class="s">年化波動 ${I.vol_ann}%・15 日 ±${f.sigma15}%</div></div>
      </div>`;
  }

  function renderStockTab() {
    const s = state.stock; if (!s) return;
    $$('#stock-seg button').forEach((b) => b.classList.toggle('on', b.dataset.k === state.stockTab));
    const box = $('#stock-tab');
    box.innerHTML = { why: tabWhy, tech: tabTech, fund: tabFund, chips: tabChips, news: tabNews }[state.stockTab](s);
    box.scrollTop = 0;
  }

  function tabWhy(s) {
    const f = s.forecast, L = s.levels;
    const named = { market: '大盤走勢', alpha: '技術面與估值', news: '新聞', chips: '籌碼', fund: '基本面', pattern: 'K 線型態', us: '美股隔夜' };
    const parts = Object.entries(f.parts).filter(([, v]) => v != null && Math.abs(v) >= 0.005);
    return `<div class="cols2"><div>${contribRows(s.reasons)}<p class="hint">列出影響最大的項目；技術面細項僅列前 6 名，所以加總略小於預期報酬。</p></div>
      <div><h4 style="margin-top:0">各來源貢獻（15 日預期報酬）</h4>${contribRows(parts.map(([k, v]) => ({ label: named[k] || k, contrib: v })), f.mu['15'])}
      <h4>操作參考價位</h4><table class="tbl"><tbody>
        <tr><td>收盤</td><td class="r">${num(s.close)}</td></tr>
        <tr><td>15 日後預期中位價</td><td class="r">${num(L.target)}</td></tr>
        <tr><td>68% 可能區間</td><td class="r">${num(f.path[15][2])} ~ ${num(f.path[15][3])}</td></tr>
        <tr><td>近 20 日支撐 / 壓力</td><td class="r">${num(L.support)} / ${num(L.resistance)}</td></tr>
        <tr><td>停損參考（收盤 − 2×ATR）</td><td class="r">${num(L.stop)}</td></tr></tbody></table>
      ${s.alerts?.exdiv?.length ? `<p class="hint">⚠ ${esc(s.alerts.exdiv[0].date)} ${esc(s.alerts.exdiv[0].type)}：除權息當天股價會依配發金額調降，預期價未扣除這部分。</p>` : ''}
      ${!LATEST.meta.us_available ? '<p class="hint">本次為盤後版，尚未納入今晚美股；隔天早上的盤前版會自動更新。</p>' : ''}</div></div>`;
  }

  function tabTech(s) {
    const I = s.indicators;
    const pats = s.patterns.length ? s.patterns.map((p) => `<div class="pat"><div><span class="dot ${p.dir > 0 ? 'bull' : 'bear'}"></span><b>${esc(p.name)}</b>
        <span class="hint" style="margin:0">　${p.ago === 0 ? '今天' : p.ago + ' 個交易日前'}（${esc(p.date)}）</span></div>
        <div class="t">${esc(p.desc)}</div>
        <div class="t">歷史 ${p.n.toLocaleString()} 次：15 日平均超額 <b class="${cls(p.mean)}">${pct(p.mean)}</b>、贏過中位數 ${p.win}%、t 值 ${p.t ?? '–'} → ${esc(p.verdict)}</div></div>`).join('')
      : empty('最近 3 個交易日沒有偵測到 K 線型態或技術訊號');
    return `<div class="cols2"><div><h4 style="margin-top:0">最近 3 日的 K 線型態與技術訊號</h4>${pats}</div>
      <div><h4 style="margin-top:0">傳統技術指標解讀</h4>${s.signals.map((g) => `<div class="sig"><span>${esc(g.name)}</span><span><i class="dot ${g.view}"></i>${esc(g.value)}</span><span class="t">${esc(g.text)}</span></div>`).join('')}
      <p class="hint">近 5 日 ${pct(I.mom5, 1)}・近 20 日 ${pct(I.mom20, 1)}・近 60 日 ${pct(I.mom60, 1)}・RSI ${I.rsi}・K ${I.k} / D ${I.d}</p></div></div>`;
  }

  function masterCard(title, m, extra = '') {
    if (!m) return '';
    const mark = (p) => (p === true ? '<span class="ok">✓</span>' : p === false ? '<span class="no">✗</span>' : '<span class="na">–</span>');
    return `<div class="master"><div class="mh"><b>${title}</b><span class="sc">${m.score ?? '–'}<small> / 100</small></span></div>${extra}
      ${m.items.map((i) => `<div class="crit">${mark(i.pass)}<div>${esc(i.name)}<small>${esc(i.rule)}</small></div><div class="v">${esc(i.value)}</div></div>`).join('')}</div>`;
  }

  function tabFund(s) {
    const f = s.fund;
    if (!f) return empty('沒有基本面資料');
    const kv = [['本益比（近四季）', f.pe], ['股價淨值比', f.pb], ['現金殖利率', f.dy != null ? f.dy + '%' : null], ['近四季 EPS', f.eps_ttm != null ? f.eps_ttm + ' 元' : null],
      ['今年累計 EPS', f.eps_ytd != null ? `${f.eps_ytd} 元（${f.season || ''}）` : null], ['EPS 年增', f.eps_g != null ? pct(f.eps_g, 1) : null],
      ['ROE', f.roe != null ? `${f.roe}%` : null], ['毛利率', f.gm != null ? f.gm + '%' : null], ['營業利益率', f.om != null ? f.om + '%' : null],
      ['稅後淨利率', f.nm != null ? f.nm + '%' : null], ['負債比', f.debt != null ? f.debt + '%' : null], ['流動比率', f.cur],
      ['最新月營收年增', f.rev_yoy != null ? `${pct(f.rev_yoy, 1)}（${f.rev_month || ''}）` : null], ['累計營收年增', f.rev_cum_yoy != null ? pct(f.rev_cum_yoy, 1) : null],
      ['PEG', f.peg], ['葛拉漢合理價', f.graham_number != null ? `${num(f.graham_number, 1)}（安全邊際 ${pct(f.mos, 0)}）` : null],
      ['市值', f.mcap != null ? `${num(f.mcap, 0)} 億` : null], ['每股淨值', f.bvps]];
    return `<div class="tiles" style="grid-template-columns:repeat(auto-fill,minmax(150px,1fr))">${kv.map(([k, v]) => `<div class="tile"><div class="k">${k}</div><div class="v" style="font-size:15px">${v == null ? '–' : esc(v)}</div></div>`).join('')}</div>
      ${f.roe_src && f.roe_src !== '最新財報年化' ? `<p class="hint">ROE 由${esc(f.roe_src)}推算。</p>` : ''}
      <div class="cols3" style="margin-top:10px">
        ${masterCard('葛拉漢・防禦型價值', f.graham, `<p class="hint">重安全邊際：便宜、穩健、持續獲利與配息。</p>`)}
        ${masterCard('巴菲特・優質護城河', f.buffett, `<p class="hint">重品質：高 ROE、高毛利、低負債、本業賺錢、價格合理。</p>`)}
        ${masterCard('彼得林區・合理價成長', f.lynch, `<p class="hint">${f.lynch.category ? `分類：<b>${esc(f.lynch.category)}</b>。${esc(f.lynch.category_note || '')}` : '重成長與 PEG。'}${f.growth_src ? `成長率採用${esc(f.growth_src)}。` : ''}</p>`)}
      </div>
      ${f.rev_hist?.length > 1 ? `<h4>月營收（千元）</h4><table class="tbl"><thead><tr><th>月份</th><th class="r">營收</th><th class="r">年增</th></tr></thead><tbody>${f.rev_hist.slice().reverse().map((r) => `<tr><td>${esc(r[0])}</td><td class="r">${num(r[1], 0)}</td><td class="r ${cls(r[2])}">${pct(r[2], 1)}</td></tr>`).join('')}</tbody></table>` : ''}
      <p class="hint">基本面以「品質＋成長」小權重參與排名（估值另外放進模型回測）。分數只代表符合大師條件的比例，不是買賣建議。</p>`;
  }

  function tabChips(s) {
    const c = s.chips || {};
    const inst = c.days ? `<table class="tbl"><thead><tr><th>日期</th><th class="r">外資</th><th class="r">投信</th><th class="r">自營商</th></tr></thead><tbody>
        ${[...c.days].reverse().map((r) => `<tr><td>${esc(r[0])}</td>${[1, 2, 3].map((i) => `<td class="r ${cls(r[i])}">${intf(r[i])}</td>`).join('')}</tr>`).join('')}
        <tr><td><b>近 5 日</b></td>${['f5', 't5', 'd5'].map((k) => `<td class="r ${cls(c[k])}"><b>${intf(c[k])}</b></td>`).join('')}</tr>
        <tr><td><b>近 20 日</b></td><td class="r ${cls(c.f20)}">${intf(c.f20)}</td><td class="r ${cls(c.t20)}">${intf(c.t20)}</td><td></td></tr></tbody></table>` : empty('尚無法人資料');
    const streak = (v, who) => (v ? `<span class="badge ${v > 0 ? 'bull' : 'bear'}">${who}連${v > 0 ? '買' : '賣'} ${Math.abs(v)} 天</span>` : '');
    const mar = c.margin_hist ? `<table class="tbl"><thead><tr><th>日期</th><th class="r">融資餘額</th><th class="r">融券餘額</th></tr></thead><tbody>
        ${[...c.margin_hist].reverse().map((r) => `<tr><td>${esc(r[0])}</td><td class="r">${r[1] == null ? '–' : r[1].toLocaleString()}</td><td class="r">${r[2] == null ? '–' : r[2].toLocaleString()}</td></tr>`).join('')}</tbody></table>` : empty('尚無融資融券資料');
    return `<div class="cols2"><div><h4 style="margin-top:0">三大法人買賣超（張）</h4><div style="margin-bottom:6px">${streak(c.f_streak, '外資')} ${streak(c.t_streak, '投信')}</div>${inst}</div>
      <div><h4 style="margin-top:0">融資融券（張）</h4>
        <div class="tiles" style="grid-template-columns:repeat(3,minmax(0,1fr));margin-bottom:8px">
          <div class="tile"><div class="k">融資近 5 日</div><div class="v ${cls(c.margin_chg)}" style="font-size:15px">${intf(c.margin_chg)}</div><div class="s">${pct(c.margin_chg_pct, 1)}</div></div>
          <div class="tile"><div class="k">券資比</div><div class="v" style="font-size:15px">${c.short_ratio != null ? c.short_ratio + '%' : '–'}</div><div class="s">≥ 20% 有軋空題材</div></div>
          <div class="tile"><div class="k">籌碼分數</div><div class="v" style="font-size:15px">法人 ${c.inst_z ?? '–'}</div><div class="s">融資（反向）${c.margin_z ?? '–'}</div></div></div>
        ${mar}<p class="hint">融資大增常代表散戶追價，歷史上多為反向指標，所以在籌碼分數中以反向計入。</p></div></div>`;
  }

  function tabNews(s) {
    const al = s.alerts || {};
    const news = s.news.length ? s.news.map((n) => `<div class="ni"><a href="${safeUrl(n.url)}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a>
        <div class="meta">${esc(n.source)}　${esc(n.time)}　情緒 <span class="${cls(n.sent)}">${n.sent > 0 ? '+' : ''}${n.sent.toFixed(2)}</span></div></div>`).join('') : empty('這段期間沒有找到直接提到這檔股票的新聞');
    const items = [];
    (al.punish || []).forEach((x) => items.push(['處置股', `${x.period || ''}　${x.text || ''}`]));
    (al.notice || []).forEach((x) => items.push(['注意股', x.text]));
    (al.exdiv || []).forEach((x) => items.push([x.type || '除權息', `${x.date}${x.cash ? `　現金 ${x.cash} 元` : ''}`]));
    (al.insider || []).forEach((x) => items.push(['內部人申報轉讓', `${x.who || ''}　${x.method || ''}　${x.shares ? Math.round(x.shares / 1000).toLocaleString() + ' 張' : ''}　${x.period || ''}`]));
    (al.material || []).forEach((x) => items.push(['重大訊息', `${x.date || ''}　${x.subject}`]));
    return `<div class="cols2"><div><h4 style="margin-top:0">相關新聞</h4>${news}</div>
      <div><h4 style="margin-top:0">公司公告與風險提示</h4>${items.length ? items.map(([k, v]) => `<div class="ni"><span class="badge neutral">${esc(k)}</span> <span class="why">${esc(v)}</span></div>`).join('') : empty('近期沒有公告或風險提示')}
      ${s.us?.beta != null ? `<p class="hint">美股連動：對「美股帶動的台股變動」敏感度 ${s.us.beta} 倍${LATEST.meta.us_available ? `，本次調整 ${pct(s.us.adj)}` : ''}。</p>` : ''}</div></div>`;
  }

  /* ── 右側：大盤摘要＋新聞 ─────────────────────────────────────────── */
  function renderSideMkt(L) {
    const m = L.market;
    const viewCls = m.view === '偏多' ? 'bull' : m.view === '偏空' ? 'bear' : 'neutral';
    const seg = [['strong_buy', C.up, 1], ['buy', C.up, 0.55], ['neutral', C.text3, 0.5], ['sell', C.dn, 0.55], ['strong_sell', C.dn, 1]];
    $('#side-mkt').innerHTML = `<div class="ph" style="margin:0"><h3>大盤 15 日判斷</h3><span class="badge ${viewCls}">${esc(m.view)}</span></div>
      <div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap"><span class="num ${cls(m.mu['15'])}" style="font-size:22px;font-weight:700">${pct(m.mu['15'])}</span>
        <span class="hint" style="margin:0">上漲機率 ${m.p_up}%・區間 ${num(m.path[15][2], 0)} ~ ${num(m.path[15][3], 0)}</span></div>
      <div id="side-chart" style="height:120px"></div>
      <div class="distbar" style="margin:4px 0">${seg.map(([k, c, o]) => `<i style="flex:${m.dist[k]};background:${c};opacity:${o}" data-tip="${ACT[k]} ${m.dist[k]} 檔"></i>`).join('')}</div>
      <div class="legend"><span>看多 ${m.dist.strong_buy + m.dist.buy}</span><span>中立 ${m.dist.neutral}</span><span>看空 ${m.dist.strong_sell + m.dist.sell}</span><span>寬度 ${m.breadth}%</span></div>`;
    priceChart('side-chart', m.ohlc.slice(-60), m.path, { compact: true });
  }

  function newsItem(n) {
    return `<div class="ni"><div class="hd"><span class="badge ${n.tone === '利多' ? 'bull' : n.tone === '利空' ? 'bear' : 'neutral'}">${esc(n.tone)}</span>
      <a href="${safeUrl(n.url)}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a></div>
      <div class="meta">${esc(n.source)}　${esc(n.time)}　衝擊 ${n.impact}　情緒 ${n.sent > 0 ? '+' : ''}${n.sent.toFixed(2)}</div>
      <div class="why">依據：${esc(n.why)}</div>${n.names.length ? `<div>${n.names.map((x) => `<span class="chip">${esc(x)}</span>`).join('')}</div>` : ''}</div>`;
  }
  function eventItem(e) {
    return `<div class="ni"><div class="hd"><b>${e.code ? `<a href="#dash/${esc(e.code)}" data-code="${esc(e.code)}">${esc(e.name)}</a> <small class="hint">${esc(e.code)}・${esc(e.industry)}</small>` : esc(e.name)}</b>${signed(e.chg)}</div>
      <div class="why">${esc(e.cause)}${e.excess != null ? `（扣除大盤連動後 ${pct(e.excess)}）` : ''}</div>
      ${e.news.length ? `<ul>${e.news.map((n) => `<li><a href="${safeUrl(n.url)}" target="_blank" rel="noopener noreferrer" style="font-weight:400">${esc(n.title)}</a> <span class="meta">${esc(n.source)}</span>${n.why ? `<div class="meta">${esc(n.why)}</div>` : ''}</li>`).join('')}</ul>` : ''}</div>`;
  }
  function alertItem(a) {
    const txt = a.type === 'material' ? a.subject : a.type === 'exdiv' ? `${a.date} ${a.label || ''}${a.cash ? ` 現金 ${a.cash} 元` : ''}`
      : a.type === 'insider' ? `${a.who || ''} ${a.method || ''} ${a.shares ? Math.round(a.shares / 1000).toLocaleString() + ' 張' : ''}` : (a.text || '') + (a.period ? `（${a.period}）` : '');
    return `<div class="ni"><div class="hd"><span class="badge ${ALERT[a.type][1]}">${ALERT[a.type][0]}</span><a href="#dash/${esc(a.code)}" data-code="${esc(a.code)}">${esc(a.name)}</a><small class="hint">${esc(a.code)}</small></div>
      <div class="why">${esc(txt)}</div></div>`;
  }
  function sectorTable(L) {
    const mx = Math.max(...L.sectors.map((s) => Math.abs(s.mu)), 0.5);
    return `<table class="tbl"><thead><tr><th>產業</th><th class="r">檔數</th><th class="r">當日</th><th class="r">15 日預期</th></tr></thead><tbody>
      ${L.sectors.map((s) => `<tr class="click" data-ind="${esc(s.name)}"><td>${esc(s.name)}</td><td class="r">${s.n}</td><td class="r">${signed(s.chg)}</td>
        <td class="r"><span class="num ${cls(s.mu)}">${pct(s.mu)}</span><i class="inbar" style="width:${Math.abs(s.mu) / mx * 36}px;background:${s.mu >= 0 ? C.up : C.dn}"></i></td></tr>`).join('')}</tbody></table>
      <p class="hint">點產業可篩選左側名單。</p>`;
  }
  function renderSide() {
    $$('#side-seg button').forEach((b) => b.classList.toggle('on', b.dataset.k === state.side));
    const L = LATEST, box = $('#side-feed');
    box.innerHTML = state.side === 'major' ? (L.major_news.map(newsItem).join('') || empty('沒有抓到新聞'))
      : state.side === 'events' ? (L.events.map(eventItem).join('') || empty('最近一個交易日沒有達到門檻的大波動'))
      : state.side === 'sectors' ? sectorTable(L)
      : (L.alerts.filter((a) => a.type !== 'material').map(alertItem).join('') || empty('股票池內沒有處置、注意或內部人轉讓'));
    box.scrollTop = 0;
  }

  /* ── 選股雷達 ─────────────────────────────────────────────────────── */
  const today = new Date();
  const days = (d) => (d ? Math.round((new Date(d) - today) / 864e5) : null);
  const hasPat = (x, dir) => x.pt.some((p) => p[1] === 0 && PSTAT[p[0]]?.used && Math.sign(PSTAT[p[0]].mean) === dir);
  const COLS = {
    r: ['排名', (x) => x.r, (x) => x.r], mu: ['15 日預期', (x) => `<span class="num ${cls(x.mu)}">${pct(x.mu)}</span>`, (x) => x.mu],
    conf: ['信心', (x) => x.conf.toFixed(1), (x) => x.conf], a: ['建議', (x) => `<span class="badge ${x.a}">${ACT[x.a]}</span>`, (x) => x.r],
    chg: ['當日', (x) => signed(x.chg), (x) => x.chg], px: ['收盤', (x) => num(x.px), (x) => x.px],
    pe: ['本益比', (x) => (x.pe == null ? '–' : x.pe.toFixed(1)), (x) => x.pe ?? 1e9], pb: ['淨值比', (x) => x.pb ?? '–', (x) => x.pb ?? 1e9],
    dy: ['殖利率', (x) => (x.dy == null ? '–' : x.dy.toFixed(2) + '%'), (x) => x.dy ?? -1], roe: ['ROE', (x) => (x.roe == null ? '–' : x.roe + '%'), (x) => x.roe ?? -1e9],
    gm: ['毛利率', (x) => (x.gm == null ? '–' : x.gm + '%'), (x) => x.gm ?? -1e9], debt: ['負債比', (x) => (x.debt == null ? '–' : x.debt + '%'), (x) => x.debt ?? 1e9],
    rev: ['累計營收年增', (x) => pct(x.rev, 1), (x) => x.rev ?? -1e9], revm: ['月營收年增', (x) => pct(x.revm, 1), (x) => x.revm ?? -1e9],
    epsg: ['EPS 年增', (x) => pct(x.epsg, 1), (x) => x.epsg ?? -1e9], peg: ['PEG', (x) => x.peg ?? '–', (x) => x.peg ?? 1e9],
    mos: ['安全邊際', (x) => pct(x.mos, 0), (x) => x.mos ?? -1e9],
    gs: ['葛拉漢', (x) => x.gs ?? '–', (x) => x.gs ?? -1], bs: ['巴菲特', (x) => x.bs ?? '–', (x) => x.bs ?? -1], ls: ['林區', (x) => x.ls ?? '–', (x) => x.ls ?? -1],
    ms: ['大師平均', (x) => (masterScore(x) == null ? '–' : masterScore(x).toFixed(0)), (x) => masterScore(x) ?? -1],
    lc: ['林區分類', (x) => esc(x.lc || '–'), (x) => x.lc || ''],
    ts: ['投信連買', (x) => (x.ts ? `${x.ts} 天` : '–'), (x) => x.ts ?? 0], fs: ['外資連買', (x) => (x.fs ? `${x.fs} 天` : '–'), (x) => x.fs ?? 0],
    t5: ['投信 5 日', (x) => intf(x.t5), (x) => x.t5 ?? 0], f5: ['外資 5 日', (x) => intf(x.f5), (x) => x.f5 ?? 0],
    mc: ['融資 5 日', (x) => pct(x.mc, 1), (x) => x.mc ?? -1e9], sr: ['券資比', (x) => (x.sr == null ? '–' : x.sr + '%'), (x) => x.sr ?? -1],
    pt: ['今日型態', (x) => x.pt.filter((p) => p[1] === 0).map((p) => `<span class="chip">${esc(patName(p[0]))}</span>`).join('') || '–', (x) => x.pt.length],
    exd: ['除權息日', (x) => esc(x.exd || '–'), (x) => x.exd || '9'], al: ['提示', (x) => x.al.filter((a) => a !== 'material').map((a) => `<span class="badge ${ALERT[a][1]}">${ALERT[a][0]}</span>`).join(' '), (x) => x.al.length],
    vol: ['年化波動', (x) => (x.vol == null ? '–' : x.vol + '%'), (x) => x.vol ?? 1e9],
  };
  const SCREENS = [
    { g: '模型', key: 'model', name: '模型綜合排名', desc: '技術面、估值、籌碼、新聞、基本面與 K 線型態綜合的相對強度排名。', f: () => true, s: 'r', d: 1, cols: ['mu', 'conf', 'a', 'pe', 'roe', 'rev'] },
    { g: '模型', key: 'strong', name: '強力看多（前 5%）', desc: '相對強度前 5% 且絕對預期報酬為正。', f: (x) => x.a === 'strong_buy', s: 'r', d: 1, cols: ['mu', 'conf', 'pt', 'f5', 't5'] },
    { g: '模型', key: 'weak', name: '強力看空（後 5%）', desc: '相對強度後 5% 且絕對預期報酬為負，持有者宜留意。', f: (x) => x.a === 'strong_sell', s: 'r', d: -1, cols: ['mu', 'conf', 'pt', 'f5', 'mc'] },
    { g: '大師', key: 'graham', name: '葛拉漢價值股', desc: '防禦型投資人 7 項檢核（規模、流動比、持續獲利、盈餘成長、配息、本益比 ≤ 15、淨值比 ≤ 1.5 或 PE×PB ≤ 22.5）通過 70% 以上，且兩個價格條件都必須通過；依安全邊際排序。', f: (x) => x.gs >= 70 && x.gv, s: 'mos', d: -1, cols: ['gs', 'pe', 'pb', 'dy', 'mos', 'mu'] },
    { g: '大師', key: 'buffett', name: '巴菲特優質股', desc: 'ROE ≥ 15%、低負債、毛利率 ≥ 30%、營益率 ≥ 15%、本業獲利、盈餘成長、價格合理，通過 70% 以上，且 ROE 必須 ≥ 15%。', f: (x) => x.bs >= 70 && x.roe >= 15, s: 'roe', d: -1, cols: ['bs', 'roe', 'gm', 'debt', 'pe', 'mu'] },
    { g: '大師', key: 'lynch', name: '林區合理價成長股', desc: 'PEG ≤ 1、營收成長、林區比率、本益比低於成長率、低負債，通過 67% 以上且 PEG ≤ 1，依 PEG 由低到高。', f: (x) => x.ls >= 67 && x.peg != null && x.peg <= 1, s: 'peg', d: 1, cols: ['ls', 'lc', 'peg', 'epsg', 'rev', 'pe'] },
    { g: '大師', key: 'masters', name: '三大師綜合', desc: '葛拉漢、巴菲特、林區三套評分平均最高的股票。', f: (x) => masterScore(x) != null, s: 'ms', d: -1, cols: ['ms', 'gs', 'bs', 'ls', 'pe', 'roe'] },
    { g: '基本面', key: 'dividend', name: '高殖利率', desc: '現金殖利率 ≥ 4% 且有獲利（本益比有值）。', f: (x) => x.dy >= 4 && x.pe != null, s: 'dy', d: -1, cols: ['dy', 'pe', 'pb', 'epsg', 'exd'] },
    { g: '基本面', key: 'growth', name: '營收高成長', desc: '今年累計營收年增 ≥ 20%。', f: (x) => x.rev >= 20, s: 'rev', d: -1, cols: ['rev', 'revm', 'epsg', 'pe', 'peg'] },
    { g: '基本面', key: 'cheap', name: '低本益比＋低淨值比', desc: '本益比 ≤ 12 且淨值比 ≤ 1.2，有獲利。', f: (x) => x.pe != null && x.pe <= 12 && x.pb != null && x.pb <= 1.2, s: 'pe', d: 1, cols: ['pe', 'pb', 'dy', 'roe', 'mu'] },
    { g: '籌碼', key: 'trust', name: '投信連續買超', desc: '投信連續買超 3 天以上。', f: (x) => x.ts >= 3, s: 'ts', d: -1, cols: ['ts', 't5', 'fs', 'f5', 'mu'] },
    { g: '籌碼', key: 'foreign', name: '外資連續買超', desc: '外資連續買超 3 天以上。', f: (x) => x.fs >= 3, s: 'fs', d: -1, cols: ['fs', 'f5', 'ts', 't5', 'mu'] },
    { g: '籌碼', key: 'margin', name: '融資大增（散戶追價）', desc: '近 5 日融資餘額增加 10% 以上。歷史上多為反向訊號，宜留意。', f: (x) => x.mc >= 10, s: 'mc', d: -1, cols: ['mc', 'sr', 'f5', 'mu'] },
    { g: '籌碼', key: 'squeeze', name: '高券資比（軋空題材）', desc: '券資比 ≥ 20%。', f: (x) => x.sr >= 20, s: 'sr', d: -1, cols: ['sr', 'mc', 'f5', 'mu'] },
    { g: 'K 線', key: 'pat_bull', name: '今日有效多方型態', desc: '今天出現、且歷史統計顯示出現後表現較好的型態（例如帶量突破、向上跳空、漲停、長紅 K）。', f: (x) => hasPat(x, 1), s: 'r', d: 1, cols: ['pt', 'mu', 'conf', 'chg'] },
    { g: 'K 線', key: 'pat_bear', name: '今日有效空方型態', desc: '今天出現、且歷史統計顯示出現後表現較差的型態。', f: (x) => hasPat(x, -1), s: 'r', d: -1, cols: ['pt', 'mu', 'conf', 'chg'] },
    { g: '風險', key: 'exdiv', name: '30 日內除權息', desc: '除權息當天股價會調降，短線操作要留意。', f: (x) => x.exd && days(x.exd) >= -1 && days(x.exd) <= 30, s: 'exd', d: 1, cols: ['exd', 'dy', 'mu'] },
    { g: '風險', key: 'risk', name: '處置／注意股', desc: '交易所公布的注意股、處置股（價量異常），波動與流動性風險高。', f: (x) => x.al.includes('punish') || x.al.includes('notice'), s: 'r', d: 1, cols: ['al', 'chg', 'vol', 'mu'] },
  ];

  function renderScreens() {
    let g = '';
    $('#screens').innerHTML = SCREENS.map((s) => {
      const n = INDEX.filter(s.f).length;
      const head = s.g !== g ? `<div class="scr-group">${(g = s.g)}</div>` : '';
      return `${head}<button class="scr${s.key === state.screen ? ' on' : ''}" data-screen="${s.key}"><b>${esc(s.name)}</b><small>${n}</small></button>`;
    }).join('');
    renderScreen();
  }
  function renderScreen() {
    const sc = SCREENS.find((s) => s.key === state.screen) || SCREENS[0];
    $$('.scr').forEach((b) => b.classList.toggle('on', b.dataset.screen === sc.key));
    $('#screen-title').textContent = sc.name;
    $('#screen-desc').textContent = sc.desc;
    const sort = state.sort || { k: sc.s, d: sc.d };
    const val = COLS[sort.k][2];
    const rows = INDEX.filter(sc.f).sort((a, b) => { const x = val(a), y = val(b); return (x > y ? 1 : x < y ? -1 : 0) * sort.d; });
    const cols = ['px', 'chg', ...sc.cols.filter((c) => c !== 'chg')];
    $('#screen-tbl').innerHTML = `<thead><tr><th>#</th><th>個股</th>${cols.map((k) => `<th class="sortable r" data-sort="${k}">${COLS[k][0]}${sort.k === k ? (sort.d > 0 ? ' ↑' : ' ↓') : ''}</th>`).join('')}</tr></thead><tbody>`
      + (rows.length ? rows.map((x, i) => `<tr class="click" data-code="${esc(x.c)}"><td class="r">${i + 1}</td><td class="name"><b>${esc(x.n)}</b><small>${esc(x.c)}・${esc(x.i)}</small></td>
        ${cols.map((k) => `<td class="r">${COLS[k][1](x)}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="${cols.length + 2}">${empty('目前沒有符合條件的個股')}</td></tr>`) + '</tbody>';
  }

  /* ── 大盤與美股 ───────────────────────────────────────────────────── */
  function renderMarket() {
    const L = LATEST, m = L.market, u = L.us, f = m.flows;
    priceChart('mkt-chart', m.ohlc, m.path);
    $('#mkt-parts').innerHTML = contribRows(m.parts, m.mu['15']);
    $('#us-note').textContent = u.note;
    $('#us-tiles').innerHTML = u.panel.map((p) => `<div class="tile"><div class="k">${esc(p.name)}</div><div class="v">${num(p.close)}</div>
        <div>${signed(p.chg)}</div>${spark(p.spark, p.chg >= 0)}<div class="s">${p.date.slice(5)}・近 5 日 ${pct(p.chg5, 1)}</div></div>`).join('')
      + (u.adr ? `<div class="tile"><div class="k">台積電 ADR 溢價</div><div class="v ${cls(u.adr.premium)}">${pct(u.adr.premium, 1)}</div><div class="s">ADR 換算 ${num(u.adr.adr_twd, 0)} 元，台股 ${num(u.adr.tw_close, 0)} 元</div></div>` : '');
    if (u.model) {
      const b = Object.entries(u.model.drivers).map(([k, v]) => `${k} 每漲 1%，台股隔日平均 ${v >= 0 ? '+' : ''}${v.toFixed(2)}%`).join('；');
      $('#us-model').textContent = `連動關係（近 ${u.model.n} 個交易日迴歸）：${b}。解釋力 R² = ${u.model.r2}。${u.available ? `本次已把估計影響 ${pct(u.impact)} 加入預測。` : ''}`;
    }
    $('#mkt-health').innerHTML = `<div class="tiles">
        <div class="tile"><div class="k">市場寬度</div><div class="v">${m.breadth}%</div><div class="meter" style="margin-top:4px"><i style="width:${m.breadth}%"></i></div><div class="s">站上月線的個股比例</div></div>
        ${f ? [['外資', f.foreign], ['投信', f.trust], ['自營商', f.dealer], ['三大法人合計', f.total]].map(([k, v]) => `<div class="tile"><div class="k">${k}（億元）</div><div class="v ${cls(v)}">${v > 0 ? '+' : ''}${num(v, 1)}</div><div class="s">集中市場・${esc(f.date)}</div></div>`).join('') : ''}
        <div class="tile"><div class="k">新聞情緒（${L.news.hours} 小時）</div><div class="v ${cls(L.news.market_sent)}">${L.news.market_sent > 0 ? '+' : ''}${L.news.market_sent.toFixed(2)}</div><div class="s">正面 ${L.news.pos}・負面 ${L.news.neg} 則</div></div></div>
      <h3 class="sub-h">產業彙整</h3>${sectorTable(L)}`;
  }

  /* ── 新聞與公告 ───────────────────────────────────────────────────── */
  function renderNews() {
    const L = LATEST;
    $('#news-meta').textContent = `近 ${L.news.hours} 小時共 ${L.news.total} 則（已合併重複報導）。衝擊分數＝報導家數＋重大事件字眼＋情緒強度。`;
    $('#events').innerHTML = L.events.map(eventItem).join('') || empty('最近一個交易日沒有達到門檻的大波動');
    $('#major').innerHTML = L.major_news.map(newsItem).join('') || empty('沒有抓到新聞');
    $('#alerts').innerHTML = L.alerts.map(alertItem).join('') || empty('股票池內沒有公告或風險提示');
  }

  /* ── 復盤 ─────────────────────────────────────────────────────────── */
  function renderReview() {
    const rv = REVIEW, a = rv.summary.all, mt = rv.summary.matured, bt = rv.backtest;
    const tile = (k, v, s) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
    $('#rv-tiles').innerHTML = a ? [
      tile('驗證的預測日數', a.days, `實際發布 ${rv.n_live}・回測補齊 ${rv.n_backtest}`),
      tile('相對強弱命中率', `${a.rel_hit}%`, '基準 50%'),
      tile('前 10 名超額報酬', `<span class="${cls(a.excess)}">${pct(a.excess)}</span>`, `${a.top_win}% 的日子勝出`),
      tile('高信心命中率', a.n_hi >= 30 ? `${a.hit_hi}%` : '–', `信心 ≥ ${LATEST.model.hiconf}・${a.n_hi} 筆`),
      tile('絕對方向命中率', `${a.hit}%`, `同期上漲比例 ${a.up_share}%`),
      tile('大盤方向命中率', `${a.mkt_hit}%`, `${a.days} 個預測日`),
      tile('已滿 15 日', mt ? mt.days : 0, mt ? `超額 ${pct(mt.excess)}・命中 ${mt.rel_hit}%` : '尚無'),
      tile('一年滾動回測 IC', bt.ic, `前 10 名 ${pct(bt.top_ret)} 對 ${pct(bt.uni_ret)}`),
    ].join('') : empty('尚無可驗證的預測');
    const rows = rv.rows.filter((r) => r.h > 0).slice().reverse();
    const labels = rows.map((r) => r.date.slice(5)), faded = rows.map((r) => r.kind !== 'live');
    const det = rows.map((r) => row('狀態', r.status) + row('前 10 名', pct(r.top_ret)) + row('股票池', pct(r.uni_ret)) + row('來源', r.kind === 'live' ? '實際發布' : '回測補齊'));
    barChart('rv-excess', labels, rows.map((r) => r.excess), { faded, unit: ' 百分點', details: det });
    barChart('rv-hit', labels, rows.map((r) => r.rel_hit), { base: 50, faded, unit: '%', details: det });
    $('#rv-sugg').innerHTML = rv.suggestions.map((s) => `<div class="sugg ${s.level}"><b>${esc(s.title)}</b><span>${esc(s.text)}</span></div>`).join('');
    $('#tbl-calib').innerHTML = '<thead><tr><th>信心區間</th><th class="r">樣本數</th><th class="r">平均信心</th><th class="r">實際命中率</th></tr></thead><tbody>'
      + (rv.buckets.length ? rv.buckets.map((b) => `<tr><td>${esc(b.range)}</td><td class="r">${b.n.toLocaleString()}</td><td class="r">${b.avg_conf}</td><td class="r"><b>${b.hit}%</b></td></tr>`).join('') : '<tr><td colspan="4" class="hint">樣本不足</td></tr>') + '</tbody>';
    $('#tbl-rv').innerHTML = '<thead><tr><th>預測日</th><th>狀態</th><th class="r">相對命中</th><th class="r">前 10 名</th><th class="r">股票池</th><th class="r">超額</th><th class="r">大盤預測／實際</th></tr></thead><tbody>'
      + rv.rows.map((r) => (r.h > 0 ? `<tr><td class="nw">${r.date}${r.kind === 'live' ? '' : ' <span class="chip">回測</span>'}</td><td class="nw">${esc(r.status)}</td><td class="r">${r.rel_hit}%</td>
        <td class="r ${cls(r.top_ret)}">${pct(r.top_ret)}</td><td class="r ${cls(r.uni_ret)}">${pct(r.uni_ret)}</td><td class="r ${cls(r.excess)}"><b>${pct(r.excess)}</b></td>
        <td class="r">${pct(r.mkt_mu)} / <span class="${cls(r.mkt_real)}">${pct(r.mkt_real)}</span> ${r.mkt_hit ? '✓' : '✗'}</td></tr>`
        : `<tr><td class="nw">${r.date}</td><td colspan="6" class="hint">${esc(r.status)}</td></tr>`)).join('') + '</tbody>';
  }

  /* ── 模型說明 ─────────────────────────────────────────────────────── */
  function renderMethod() {
    const L = LATEST, m = L.model, bt = m.backtest, fs = m.factors;
    const mx = Math.max(...fs.map((f) => Math.abs(f.coef || 0)));
    const flag = { with: '<span class="chip">近期有效</span>', against: '<span class="chip" style="color:var(--warn)">近期失靈</span>', flat: '', excluded: '' };
    const sgn = (v, nd) => (v == null ? '–' : `${v > 0 ? '+' : ''}${v.toFixed(nd)}`);
    $('#tbl-factors').innerHTML = '<thead><tr><th>因子</th><th>類別</th><th class="r">權重</th><th class="r">近期 IC</th><th></th></tr></thead><tbody>'
      + [...fs].sort((a, b) => (b.coef == null) - (a.coef == null) || Math.abs(b.coef || 0) - Math.abs(a.coef || 0)).map((f) => `<tr><td>${esc(f.name)}</td><td class="hint">${esc(f.group)}</td>
        <td class="r">${f.coef == null ? `<span class="hint">未納入（一年 IC ${sgn(f.ic_year, 3)}）</span>` : `<span class="num">${sgn(f.coef, 4)}</span><i class="inbar" style="width:${Math.abs(f.coef) / mx * 40}px;background:${f.coef >= 0 ? C.up : C.dn}"></i>`}</td>
        <td class="r ${cls(f.ic)}">${sgn(f.ic, 3)}</td><td>${flag[f.flag]}</td></tr>`).join('') + '</tbody>';
    $('#tbl-patterns').innerHTML = '<thead><tr><th>型態</th><th class="r">次數</th><th class="r">15 日超額</th><th class="r">勝率</th><th class="r">t 值</th><th class="r">今日</th></tr></thead><tbody>'
      + L.pattern_stats.slice().sort((a, b) => Math.abs(b.t || 0) - Math.abs(a.t || 0)).map((p) => `<tr title="${esc(p.verdict)}"><td><span class="dot ${p.dir > 0 ? 'bull' : 'bear'}"></span>${esc(p.name)}${p.used ? ' <span class="chip">採用</span>' : ''}<div class="hint" style="margin:0">${esc(p.verdict)}</div></td>
        <td class="r">${p.n.toLocaleString()}</td><td class="r ${cls(p.mean)}">${pct(p.mean)}</td><td class="r">${p.win ?? '–'}%</td><td class="r">${p.t ?? '–'}</td><td class="r">${p.today}</td></tr>`).join('') + '</tbody>';
    const w = m.weights;
    $('#method').innerHTML = `<h3 style="margin-top:0">預測怎麼算</h3>
      <p><b>15 日預期報酬 ＝ β × 大盤預期 ＋ 相對強度 × 個股波動 ＋ 美股隔夜調整</b></p>
      <ul>
        <li><b>相對強度（樣本外回測）</b>：${fs.filter((f) => f.coef != null).length} 個技術因子（動能、均線、RSI、MACD、KD、布林、量能、流動性、距高點）${m.value_factors.length ? '加上估值因子' : ''}。每天以最近 ${m.train_days} 個交易日、${L.meta.universe} 檔股票重新訓練，並先排除波動與 β 的影響。</li>
        <li><b>估值因子的檢驗</b>：本益比、淨值比、殖利率、EPS 成長的歷史（證交所／櫃買中心 ${L.meta.valuation_snapshots} 個快照）${m.value_factors.length ? '已納入模型。' : '經回測加入後樣本外表現沒有變好——近兩年低本益比的股票 15 日內反而較弱，所以沒有放進模型，只用於大師評分與選股雷達；每天仍會檢驗它們的 IC（左表「未納入」）。'}</li>
        <li><b>疊加因子（無完整回測，小權重、依復盤自動調整）</b>：新聞情緒 ×${w.news.weight}、法人與融資籌碼 ×${w.chips.weight}、基本面品質與成長 ×${w.fund?.weight ?? 1}、K 線型態 ×${w.pattern?.weight ?? 1}。</li>
        <li><b>三位大師</b>：葛拉漢（防禦型價值 7 項）、巴菲特（品質護城河 7 項）、林區（合理價成長 6 項＋六種股票分類），依最新財報、月營收與估值逐項檢核。分數顯示在個股與選股雷達，品質與成長部分以小權重進入排名。</li>
        <li><b>K 線型態</b>：28 種型態與技術訊號，每種都以近三年股票池的歷史表現檢驗；統計不明確的型態只顯示、不參與預測。</li>
        <li><b>信心值</b>：回測中同等強度的訊號，15 日後實際贏過股票池中位數的比例，基準 50。</li>
        <li><b>建議等級</b>：相對強度前 5% 強力看多、前 20% 偏多，後 20% 偏空、後 5% 強力看空；與絕對預期方向相反時降一級。</li>
      </ul>
      <h3>一年滾動回測（樣本外）</h3>
      <ul><li>${bt.from} ～ ${bt.to}，${bt.days} 個交易日、${bt.samples.toLocaleString()} 筆預測。</li>
        <li>排序相關 IC ${bt.ic}（前半 ${bt.ic_h1}、後半 ${bt.ic_h2}），${bt.ic_pos_days}% 的日子為正；相對強弱命中率 ${bt.rel_hit}%。</li>
        <li>前 10 名 15 日平均 ${pct(bt.top_ret)}、後 10 名 ${pct(bt.bot_ret)}、股票池 ${pct(bt.uni_ret)}；前 10 名 ${bt.top_win}% 的日子勝出。</li></ul>
      <h3>已知限制</h3>
      <ul><li>股票池以「現在」成交金額選出，回測有倖存者偏誤。</li><li>財報與月營收只有最新一期，大師評分無法回測；新聞情緒為關鍵字規則。</li>
        <li>未計入交易成本、漲跌停無法成交、除權息當天的價格調整。</li></ul>`;
  }

  /* ── 分頁與網址 ───────────────────────────────────────────────────── */
  const VIEW_RENDER = { market: renderMarket, news: renderNews, review: renderReview, method: renderMethod, radar: renderScreens };
  function showTab(tab) {
    state.tab = tab;
    $$('#tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
    $$('.view').forEach((v) => v.classList.toggle('on', v.id === `v-${tab}`));
    if (VIEW_RENDER[tab] && !rendered.has(tab)) { VIEW_RENDER[tab](); rendered.add(tab); }
    requestAnimationFrame(() => Object.values(charts).forEach((c) => c.resize()));
    setHash();
  }
  function setHash() {
    const h = state.tab === 'dash' ? `#dash/${state.code || ''}` : state.tab === 'radar' ? `#radar/${state.screen}` : `#${state.tab}`;
    if (location.hash !== h) history.replaceState(null, '', h);
  }

  /* ── 互動 ─────────────────────────────────────────────────────────── */
  function setupEvents() {
    $('#tabs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) showTab(b.dataset.tab); });
    $('#list-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; state.list = b.dataset.k; $$('#list-seg button').forEach((x) => x.classList.toggle('on', x === b)); renderList(); $('#list').scrollTop = 0; });
    $('#list-ind').addEventListener('change', (e) => { state.ind = e.target.value; renderList(); });
    $('#stock-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { state.stockTab = b.dataset.k; renderStockTab(); } });
    $('#side-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { state.side = b.dataset.k; renderSide(); } });
    $('#screens').addEventListener('click', (e) => { const b = e.target.closest('[data-screen]'); if (b) { state.screen = b.dataset.screen; state.sort = null; renderScreen(); setHash(); } });
    document.addEventListener('click', (e) => {
      const th = e.target.closest('th[data-sort]');
      if (th) { const k = th.dataset.sort, cur = state.sort || { k: (SCREENS.find((s) => s.key === state.screen) || {}).s, d: 1 }; state.sort = { k, d: cur.k === k ? -cur.d : (k === 'r' || k === 'pe' || k === 'pb' || k === 'peg' || k === 'debt' || k === 'exd' ? 1 : -1) }; renderScreen(); return; }
      const ind = e.target.closest('[data-ind]');
      if (ind) { state.ind = ind.dataset.ind; $('#list-ind').value = state.ind; state.list = 'all'; $$('#list-seg button').forEach((x) => x.classList.toggle('on', x.dataset.k === 'all')); renderList(); if (state.tab !== 'dash') showTab('dash'); return; }
      const el = e.target.closest('[data-code]');
      if (el && !e.target.closest('.suggest')) { e.preventDefault(); selectStock(el.dataset.code, { focus: true }); }
    });
    document.addEventListener('keydown', (e) => {
      if (state.tab !== 'dash' || e.target.matches('input, select') || !['ArrowUp', 'ArrowDown'].includes(e.key)) return;
      const rows = listRows(), i = rows.findIndex((x) => x.c === state.code);
      const nx = rows[Math.max(0, Math.min(rows.length - 1, i + (e.key === 'ArrowDown' ? 1 : -1)))];
      if (nx) { e.preventDefault(); selectStock(nx.c); $(`.li[data-code="${CSS.escape(nx.c)}"]`)?.scrollIntoView({ block: 'nearest' }); }
    });
    // 搜尋
    const q = $('#q'), box = $('#suggest');
    let hits = [], cur = -1;
    const draw = () => { box.hidden = !hits.length; box.innerHTML = hits.map((x, i) => `<button type="button" data-pick="${esc(x.c)}" class="${i === cur ? 'on' : ''}"><span><b>${esc(x.c)}</b>　${esc(x.n)}</span><small>${esc(x.i)}・${ACT[x.a]}</small></button>`).join(''); };
    const pick = (code) => { hits = []; draw(); q.value = ''; q.blur(); selectStock(code, { focus: true }); };
    q.addEventListener('input', () => { const v = q.value.trim().toLowerCase(); hits = v ? INDEX.filter((x) => x.c.startsWith(v) || x.n.toLowerCase().includes(v)).slice(0, 8) : []; cur = -1; draw(); });
    q.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') { cur = Math.min(cur + 1, hits.length - 1); draw(); e.preventDefault(); }
      else if (e.key === 'ArrowUp') { cur = Math.max(cur - 1, 0); draw(); e.preventDefault(); }
      else if (e.key === 'Enter') { const h = hits[Math.max(cur, 0)]; if (h) pick(h.c); else if (q.value.trim()) pick(q.value.trim()); }
      else if (e.key === 'Escape') { hits = []; draw(); }
    });
    box.addEventListener('click', (e) => { const b = e.target.closest('[data-pick]'); if (b) pick(b.dataset.pick); });
    document.addEventListener('click', (e) => { if (!e.target.closest('.search')) { hits = []; draw(); } });
    document.addEventListener('keydown', (e) => { if (e.key === '/' && !e.target.matches('input, select')) { e.preventDefault(); q.focus(); } });
    // 提示框
    const tip = $('#tip');
    document.addEventListener('pointermove', (e) => {
      const t = e.target.closest?.('[data-tip]');
      if (!t) { tip.hidden = true; return; }
      tip.textContent = t.dataset.tip; tip.hidden = false;
      tip.style.left = `${Math.min(e.clientX + 12, innerWidth - 160)}px`; tip.style.top = `${e.clientY + 14}px`;
    });
  }

  async function main() {
    try {
      const [latest, index, review] = await Promise.all([getJSON('data/latest.json'), getJSON('data/index.json'), getJSON('data/review.json')]);
      LATEST = latest; INDEX = index; REVIEW = review;
      IDX = Object.fromEntries(index.map((x) => [x.c, x]));
      PSTAT = Object.fromEntries((latest.pattern_stats || []).map((p) => [p.key, p]));
      $('#edition').textContent = latest.meta.edition;
      $('#stamp').innerHTML = `資料日 <b>${latest.meta.asof}</b>・更新 ${latest.meta.generated_at}・下一交易日 ${latest.meta.next_session}`;
      [...new Set(index.map((x) => x.i))].sort().forEach((i) => { const o = document.createElement('option'); o.value = i; o.textContent = i; $('#list-ind').appendChild(o); });
      renderTicker(latest); renderList(); renderSideMkt(latest); renderSide(); setupEvents();
      const [tab, arg] = location.hash.replace('#', '').split('/');
      if (tab === 'radar' && arg) state.screen = arg;
      showTab(['dash', 'radar', 'market', 'news', 'review', 'method'].includes(tab) ? tab : 'dash');
      selectStock(tab === 'dash' && arg && IDX[decodeURIComponent(arg)] ? decodeURIComponent(arg) : latest.top_bull[0].code, { switchTab: false });
    } catch (e) {
      const el = $('#error'); el.hidden = false;
      el.textContent = `資料載入失敗（${e.message}）。若是剛建立的網站，請先到 GitHub Actions 手動執行一次「每日更新」。`;
      $('#stamp').textContent = '';
      console.error(e);
    }
  }
  main();
})();
