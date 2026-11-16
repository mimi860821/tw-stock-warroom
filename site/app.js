/* 台股戰情室前端：讀取 data/*.json 並繪製。沒有框架、沒有建置步驟。 */
(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const C = { up: css('--up'), dn: css('--dn'), text: css('--text'), text2: css('--text-2'), text3: css('--text-3'),
              line: css('--line'), accent: css('--accent'), s2: css('--series-2'), surface: css('--surface') };

  // 新聞標題等外部文字一律跳脫後才放進 HTML
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const safeUrl = (u) => (/^https?:\/\//i.test(u || '') ? u : '#');
  const cls = (v) => (v > 0 ? 'up' : v < 0 ? 'dn' : '');
  const arrow = (v) => (v > 0 ? '▲' : v < 0 ? '▼' : '–');
  const pct = (v, nd = 2) => (v == null ? '–' : `${v > 0 ? '+' : ''}${v.toFixed(nd)}%`);
  const signed = (v, nd = 2) => (v == null ? '–' : `<span class="num ${cls(v)}">${arrow(v)} ${pct(v, nd)}</span>`);
  const num = (v, nd = 2) => (v == null ? '–' : v.toLocaleString('zh-TW', { minimumFractionDigits: nd, maximumFractionDigits: nd }));
  const getJSON = (p) => fetch(p, { cache: 'no-cache' }).then((r) => { if (!r.ok) throw new Error(`${p} ${r.status}`); return r.json(); });

  let INDEX = [], LATEST = null;
  const charts = {};
  const mount = (id) => { charts[id]?.dispose(); charts[id] = echarts.init(document.getElementById(id), null, { renderer: 'canvas' }); return charts[id]; };
  addEventListener('resize', () => Object.values(charts).forEach((c) => c.resize()));

  const axisStyle = { axisLine: { lineStyle: { color: C.line } }, axisTick: { show: false }, axisLabel: { color: C.text3, fontSize: 11 },
                      splitLine: { lineStyle: { color: C.line, opacity: 0.5 } } };
  const tipStyle = { backgroundColor: '#0b0d10', borderColor: C.line, textStyle: { color: C.text, fontSize: 12 }, padding: [8, 10] };
  const row = (k, v, color) => `<div style="display:flex;justify-content:space-between;gap:16px"><span style="color:${C.text2}">${color ? `<i style="display:inline-block;width:10px;height:2px;background:${color};vertical-align:middle;margin-right:6px"></i>` : ''}${k}</span><b>${v}</b></div>`;

  /* ── 價格 + 預期路徑圖（大盤用折線，個股用 K 線）─────────────────────── */
  function priceChart(id, ohlc, path, { candle = false } = {}) {
    const hist = ohlc.map((r) => r[0].slice(5));
    const fut = path.slice(1).map((p) => `+${p[0]}日`);
    const x = hist.concat(fut);
    const n = hist.length;
    const pad = (arr) => Array(n - 1).fill(null).concat(arr);
    const mid = pad(path.map((p) => p[1])), lo = pad(path.map((p) => p[2])), hi = pad(path.map((p) => p[3]));
    const closes = ohlc.map((r) => r[2]);
    const ma = (w) => closes.map((_, i) => (i + 1 < w ? null : +(closes.slice(i + 1 - w, i + 1).reduce((a, b) => a + b, 0) / w).toFixed(2)));
    const ma20 = ma(20), ma60 = ma(60);
    const series = [];
    if (candle) {
      series.push({ type: 'candlestick', name: 'K', data: ohlc.map((r) => [r[1], r[2], r[3], r[4]]),
                    itemStyle: { color: C.up, color0: C.dn, borderColor: C.up, borderColor0: C.dn } });
      series.push({ type: 'line', name: 'MA20', data: ma20, showSymbol: false, lineStyle: { width: 1.5, color: C.accent }, itemStyle: { color: C.accent } });
      series.push({ type: 'line', name: 'MA60', data: ma60, showSymbol: false, lineStyle: { width: 1.5, color: C.s2 }, itemStyle: { color: C.s2 } });
    } else {
      series.push({ type: 'line', name: '收盤', data: closes, showSymbol: false, lineStyle: { width: 2, color: C.accent }, itemStyle: { color: C.accent } });
    }
    series.push({ type: 'line', name: 'lo', data: lo, stack: 'band', showSymbol: false, lineStyle: { opacity: 0 }, silent: true },
                { type: 'line', name: 'band', data: hi.map((h, i) => (h == null ? null : +(h - lo[i]).toFixed(2))), stack: 'band', showSymbol: false,
                  lineStyle: { opacity: 0 }, areaStyle: { color: 'rgba(232,235,240,.14)' }, silent: true },
                { type: 'line', name: '預期中位', data: mid, showSymbol: false, lineStyle: { width: 2, type: 'dashed', color: C.text }, itemStyle: { color: C.text } });
    const grids = [{ left: 56, right: 16, top: 14, bottom: candle ? 96 : 28 }];
    const xAxes = [{ type: 'category', data: x, boundaryGap: candle, ...axisStyle, splitLine: { show: false },
                     axisLabel: { ...axisStyle.axisLabel, show: !candle }, axisPointer: { label: { show: false } } }];
    const yAxes = [{ type: 'value', scale: true, ...axisStyle }];
    if (candle) {
      grids.push({ left: 56, right: 16, height: 54, bottom: 28 });
      xAxes.push({ type: 'category', data: x, gridIndex: 1, boundaryGap: true, ...axisStyle, splitLine: { show: false } });
      yAxes.push({ type: 'value', gridIndex: 1, ...axisStyle, splitNumber: 2, axisLabel: { ...axisStyle.axisLabel, formatter: (v) => (v >= 1e4 ? `${(v / 1e4).toFixed(0)}萬` : v) } });
      series.push({ type: 'bar', name: '成交量', xAxisIndex: 1, yAxisIndex: 1, barMaxWidth: 8,
                    data: ohlc.map((r) => ({ value: Math.round(r[5] / 1000), itemStyle: { color: r[2] >= r[1] ? C.up : C.dn, opacity: 0.7 } })) });
    }
    mount(id).setOption({
      animation: false, grid: grids, xAxis: xAxes, yAxis: yAxes, series,
      axisPointer: { link: [{ xAxisIndex: 'all' }] },
      tooltip: { trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: C.text3 } }, ...tipStyle,
        formatter: (ps) => {
          const i = ps[0].dataIndex;
          if (i >= n) {
            const p = path[i - n + 1];
            return `<b>第 ${p[0]} 個交易日（預測）</b>${row('預期中位', num(p[1]))}${row('68% 區間', `${num(p[2])} ~ ${num(p[3])}`)}`;
          }
          const r = ohlc[i];
          let h = `<b>${r[0]}</b>`;
          if (candle) h += row('開 / 收', `${num(r[1])} / ${num(r[2])}`) + row('高 / 低', `${num(r[4])} / ${num(r[3])}`)
            + row('MA20', ma20[i] == null ? '–' : num(ma20[i]), C.accent) + row('MA60', ma60[i] == null ? '–' : num(ma60[i]), C.s2)
            + row('成交量', `${Math.round(r[5] / 1000).toLocaleString()} 張`);
          else h += row('收盤', num(r[2]), C.accent);
          return h;
        } },
      dataZoom: candle ? [{ type: 'inside', xAxisIndex: [0, 1], start: 35, end: 100 }] : [],
    });
  }

  // base：長條的起點（例如命中率以 50 為基準，高於基準往上、低於基準往下）
  function barChart(id, labels, values, { base = 0, faded = [], unit = '', details = [] } = {}) {
    mount(id).setOption({
      animation: false, grid: { left: 44, right: 12, top: 12, bottom: 26 },
      xAxis: { type: 'category', data: labels, ...axisStyle, splitLine: { show: false }, axisLine: { show: false } },
      yAxis: { type: 'value', ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: (v) => +(v + base).toFixed(1) } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(255,255,255,.05)' } }, ...tipStyle,
        formatter: (ps) => { const i = ps[0].dataIndex; return `<b>${labels[i]}</b>${row('數值', values[i] == null ? '–' : values[i] + unit)}${details[i] || ''}`; } },
      series: [{ type: 'bar', barMaxWidth: 14, itemStyle: { borderRadius: [3, 3, 0, 0] },
        data: values.map((v, i) => ({ value: v == null ? null : +(v - base).toFixed(2), itemStyle: { color: v >= base ? C.up : C.dn, opacity: faded[i] ? 0.4 : 1,
                                                           borderRadius: v >= base ? [3, 3, 0, 0] : [0, 0, 3, 3] } })),
        markLine: { silent: true, symbol: 'none', lineStyle: { color: C.text3, type: 'solid', width: 1 }, label: { show: false }, data: [{ yAxis: 0 }] } }],
    });
  }

  const spark = (vals, up) => {
    const lo = Math.min(...vals), hi = Math.max(...vals), w = 120, h = 26;
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

  /* ── 總覽 ─────────────────────────────────────────────────────────── */
  function renderOverview(d) {
    const m = d.market, meta = d.meta;
    $('#edition').textContent = meta.edition;
    $('#stamp').innerHTML = `資料基準日 <b>${meta.asof}</b>　更新於 ${meta.generated_at}（台北時間）　下一個交易日 ${meta.next_session}`;
    const viewCls = m.view === '偏多' ? 'bull' : m.view === '偏空' ? 'bear' : 'neutral';
    $('#mkt-tiles').innerHTML = `<div class="stack">
      <div class="hero"><div class="k hint">加權指數（${meta.asof} 收盤）</div>
        <div class="hero-row"><span class="v num" style="font-size:34px;font-weight:700">${num(m.close)}</span>${signed(m.chg)}</div></div>
      <div><div class="hero-row"><span class="hint" style="margin:0">未來 15 個交易日判斷</span><span class="badge ${viewCls}">${m.view}</span></div>
        <div class="hero-row"><span class="num ${cls(m.mu['15'])}" style="font-size:26px;font-weight:700">${pct(m.mu['15'])}</span><span class="hint" style="margin:0">預期報酬（5 日 ${pct(m.mu['5'])}、10 日 ${pct(m.mu['10'])}）</span></div></div>
      <div><div class="hero-row" style="justify-content:space-between"><span>上漲機率（粗估）</span><b class="num">${m.p_up}%</b></div>
        <div class="meter split"><i style="width:${m.p_up}%"></i></div>
        <div class="hint" style="margin-top:6px">由預期報酬與近期波動換算，高於 55% 視為偏多、低於 45% 視為偏空。大盤方向的可預測性很低，請搭配右側區間一起看。</div></div>
      <div class="tiles" style="grid-template-columns:1fr 1fr">
        <div class="tile"><div class="k">15 日後預期中位</div><div class="v">${num(m.path[15][1], 0)}</div></div>
        <div class="tile"><div class="k">68% 可能區間</div><div class="v" style="font-size:15px">${num(m.path[15][2], 0)} ~ ${num(m.path[15][3], 0)}</div></div>
      </div></div>`;
    priceChart('mkt-chart', m.ohlc, m.path);
    $('#mkt-parts').innerHTML = contribRows(m.parts, m.mu['15']);
    const dist = m.dist, tot = Object.values(dist).reduce((a, b) => a + b, 0);
    const seg = [['strong_buy', '強力看多', C.up, 1], ['buy', '偏多', C.up, 0.55], ['neutral', '中立', C.text3, 0.5], ['sell', '偏空', C.dn, 0.55], ['strong_sell', '強力看空', C.dn, 1]];
    const f = m.flows;
    $('#mkt-health').innerHTML = `<div class="stack">
      <div><div class="hero-row" style="justify-content:space-between"><span>市場寬度</span><b class="num">${m.breadth}%</b></div>
        <div class="meter"><i style="width:${m.breadth}%"></i></div><div class="hint" style="margin-top:6px">${esc(m.breadth_note)}</div></div>
      <div><div>個股建議分布（共 ${tot} 檔）</div>
        <div class="distbar" style="margin:6px 0">${seg.map(([k, , c, o]) => `<i style="flex:${dist[k]};background:${c};opacity:${o}" data-tip="${dist[k]} 檔"></i>`).join('')}</div>
        <div class="legend">${seg.map(([k, l, c, o]) => `<span><i class="key box" style="background:${c};opacity:${o}"></i>${l} ${dist[k]}</span>`).join('')}</div></div>
      ${f ? `<div><div>三大法人買賣超（集中市場，億元）</div><div class="tiles" style="margin-top:6px">
        ${[['外資', f.foreign], ['投信', f.trust], ['自營商', f.dealer], ['合計', f.total]].map(([k, v]) => `<div class="tile"><div class="k">${k}</div><div class="v ${cls(v)}">${v > 0 ? '+' : ''}${num(v, 1)}</div></div>`).join('')}
        </div><div class="hint" style="margin-top:6px">資料日 ${esc(f.date)}</div></div>` : ''}
      <div><div>新聞情緒（近 ${d.news.hours} 小時，${d.news.total} 則）</div>
        <div class="hero-row"><b class="num ${cls(d.news.market_sent)}" style="font-size:20px">${d.news.market_sent > 0 ? '+' : ''}${d.news.market_sent.toFixed(2)}</b>
        <span class="hint" style="margin:0">−1 極負面 ~ +1 極正面；正面 ${d.news.pos} 則、負面 ${d.news.neg} 則</span></div></div></div>`;
  }

  function renderUS(d) {
    const u = d.us;
    $('#us-note').textContent = u.note;
    $('#us-tiles').innerHTML = u.panel.map((p) => {
      const vix = p.symbol === '^VIX';
      return `<div class="tile"><div class="k">${esc(p.name)}</div><div class="v">${num(p.close)}</div>
        <div>${signed(p.chg)}</div>${spark(p.spark, p.chg >= 0)}<div class="s">${p.date.slice(5)}　近 5 日 ${pct(p.chg5, 1)}${vix ? '　越高越恐慌' : ''}</div></div>`;
    }).join('') + (u.adr ? `<div class="tile"><div class="k">台積電 ADR 溢價</div><div class="v ${cls(u.adr.premium)}">${pct(u.adr.premium, 1)}</div>
      <div class="s">ADR 換算每股 ${num(u.adr.adr_twd, 0)} 元，台股收盤 ${num(u.adr.tw_close, 0)} 元</div></div>` : '');
    if (u.model) {
      const b = Object.entries(u.model.drivers).map(([k, v]) => `${k} 每漲 1%，台股隔日平均 ${v >= 0 ? '+' : ''}${v.toFixed(2)}%`).join('；');
      $('#us-model').textContent = `連動關係（近 ${u.model.n} 個交易日迴歸）：${b}。解釋力 R² = ${u.model.r2}，代表台股單日漲跌約有 ${Math.round(u.model.r2 * 100)}% 可由前一晚美股說明。${u.available ? `本次已把估計影響 ${pct(u.impact)} 加入大盤與個股預測。` : ''}`;
    }
  }

  /* ── 名單 ─────────────────────────────────────────────────────────── */
  function pickTable(id, rows) {
    const max = Math.max(...rows.map((r) => Math.abs(r.mu)), 1);
    $(id).innerHTML = `<thead><tr><th class="hm">#</th><th>個股</th><th class="r hm">收盤</th><th class="r">當日</th><th class="r">15 日預期</th><th class="r">信心</th><th>建議</th></tr></thead><tbody>` +
      rows.map((r, i) => `<tr class="click" data-code="${esc(r.code)}"><td class="r hm">${i + 1}</td>
        <td class="name"><b>${esc(r.name)}</b><small>${esc(r.code)}　${esc(r.industry)}</small><div class="why">${r.why.map(esc).join('、')}</div></td>
        <td class="r hm">${num(r.close)}</td><td class="r">${signed(r.chg)}</td>
        <td class="r"><span class="num ${cls(r.mu)}">${pct(r.mu)}</span><i class="inbar hm" style="width:${Math.abs(r.mu) / max * 40}px;background:${r.mu >= 0 ? C.up : C.dn}"></i></td>
        <td class="r">${r.conf.toFixed(1)}</td><td><span class="badge ${r.action}">${esc(r.label)}</span></td></tr>`).join('') + '</tbody>';
  }

  function renderPicks(d) {
    pickTable('#tbl-bull', d.top_bull);
    pickTable('#tbl-bear', d.top_bear);
    const mx = Math.max(...d.sectors.map((s) => Math.abs(s.mu)), 0.5);
    $('#tbl-sector').innerHTML = '<thead><tr><th>產業</th><th class="r">檔數</th><th class="r">當日平均</th><th class="r">平均 15 日預期</th></tr></thead><tbody>' +
      d.sectors.map((s) => `<tr><td>${esc(s.name)}</td><td class="r">${s.n}</td><td class="r">${signed(s.chg)}</td>
        <td class="r"><span class="num ${cls(s.mu)}">${pct(s.mu)}</span><i class="inbar" style="width:${Math.abs(s.mu) / mx * 60}px;background:${s.mu >= 0 ? C.up : C.dn}"></i></td></tr>`).join('') + '</tbody>';
  }

  /* ── 個股 ─────────────────────────────────────────────────────────── */
  const ACT = { strong_buy: '強力看多', buy: '偏多', neutral: '中立觀望', sell: '偏空', strong_sell: '強力看空' };
  let sortKey = 'r', sortDir = 1;
  function renderAll() {
    const cols = [['r', '排名', 1], ['n', '個股', 0], ['i', '產業', 0], ['px', '收盤', 1], ['chg', '當日', 1], ['mu', '15 日預期', 1], ['conf', '信心', 1], ['a', '建議', 0]];
    const rows = [...INDEX].sort((a, b) => (a[sortKey] > b[sortKey] ? 1 : a[sortKey] < b[sortKey] ? -1 : 0) * sortDir);
    $('#tbl-all').innerHTML = `<thead><tr>${cols.map(([k, l, r]) => `<th class="sortable ${r ? 'r' : ''}" data-sort="${k}">${l}${sortKey === k ? (sortDir > 0 ? ' ↑' : ' ↓') : ''}</th>`).join('')}</tr></thead><tbody>` +
      rows.map((x) => `<tr class="click" data-code="${esc(x.c)}"><td class="r">${x.r}</td><td class="name"><b>${esc(x.n)}</b><small>${esc(x.c)}</small></td><td>${esc(x.i)}</td>
        <td class="r">${num(x.px)}</td><td class="r">${signed(x.chg)}</td><td class="r"><span class="num ${cls(x.mu)}">${pct(x.mu)}</span></td><td class="r">${x.conf.toFixed(1)}</td>
        <td><span class="badge ${x.a}">${ACT[x.a]}</span></td></tr>`).join('') + '</tbody>';
  }

  async function showStock(code, scroll = true) {
    const box = $('#stock');
    if (!INDEX.some((x) => x.c === code)) { box.innerHTML = `<p class="hint">「${esc(code)}」不在今日的股票池內。</p>`; return; }
    box.style.opacity = 0.5;
    let s;
    try { s = await getJSON(`data/stocks/${encodeURIComponent(code)}.json`); } catch (e) { box.style.opacity = 1; box.innerHTML = `<p class="hint">讀取 ${esc(code)} 失敗。</p>`; return; }
    const f = s.forecast, L = s.levels, I = s.indicators;
    const bear = f.p_win < 50;
    box.style.opacity = 1;
    box.innerHTML = `
      <div class="stock-head"><span class="nm">${esc(s.name)} <small class="hint">${esc(s.code)}　${esc(s.market)}・${esc(s.industry)}</small></span>
        <span class="px">${num(s.close)}</span>${signed(s.chg)}<span class="badge ${f.action}">${esc(f.label)}</span>
        <span class="hint" style="margin:0">相對強度排名 ${s.rank} / ${s.of}</span></div>
      <div class="verdict"><b>${esc(f.summary)}</b><p>${esc(f.advice)}${LATEST.meta.us_available ? '' : '（本次為盤後版，尚未納入今晚美股；盤前版會自動更新。）'}</p></div>
      <div class="tiles">
        <div class="tile"><div class="k">15 日預期報酬</div><div class="v ${cls(f.mu['15'])}">${pct(f.mu['15'])}</div><div class="s">5 日 ${pct(f.mu['5'])}　10 日 ${pct(f.mu['10'])}</div></div>
        <div class="tile"><div class="k">信心值（${bear ? '落後' : '贏過'}中位數機率）</div><div class="v">${f.conf.toFixed(1)}</div><div class="meter split" style="margin-top:6px"><i style="width:${f.p_win}%"></i></div></div>
        <div class="tile"><div class="k">15 日後預期價</div><div class="v">${num(L.target)}</div><div class="s">68% 區間 ${num(f.path[15][2])} ~ ${num(f.path[15][3])}</div></div>
        <div class="tile"><div class="k">支撐 / 壓力（近 20 日）</div><div class="v" style="font-size:16px">${num(L.support)} / ${num(L.resistance)}</div><div class="s">月線 ${num(L.ma20)}　季線 ${num(L.ma60)}</div></div>
        <div class="tile"><div class="k">停損參考（收盤 − 2×ATR）</div><div class="v">${num(L.stop)}</div><div class="s">ATR(14) ${num(L.atr)}，約 ${(2 * L.atr / s.close * 100).toFixed(1)}%</div></div>
        <div class="tile"><div class="k">風險</div><div class="v" style="font-size:16px">β ${I.beta}　年化波動 ${I.vol_ann}%</div><div class="s">15 日 ±1 標準差約 ±${f.sigma15}%</div></div>
      </div>
      <div class="card-h" style="margin-top:16px"><h4 style="margin:0">日 K 與未來 15 個交易日預期路徑</h4>
        <div class="legend"><span><i class="key candle-up"></i>收紅</span><span><i class="key candle-dn"></i>收黑</span><span><i class="key line"></i>MA20</span><span><i class="key line c-ma60"></i>MA60</span><span><i class="key dash"></i>預期中位</span><span><i class="key band"></i>68% 區間</span></div></div>
      <div id="stock-chart" class="chart lg"></div>
      <div class="sub">
        <div><h4>判斷依據：預期報酬怎麼算出來的</h4>${contribRows(s.reasons)}
          <p class="hint">僅列出影響最大的項目，加總會略小於上方的預期報酬。</p></div>
        <div><h4>技術指標解讀（傳統看法，供對照）</h4>${s.signals.map((g) => `<div class="sig"><span>${esc(g.name)}</span><span><i class="dot ${g.view}"></i>${esc(g.value)}</span><span class="t">${esc(g.text)}</span></div>`).join('')}
          <p class="hint">近 5 日 ${pct(I.mom5, 1)}　近 20 日 ${pct(I.mom20, 1)}　近 60 日 ${pct(I.mom60, 1)}</p></div>
        <div><h4>三大法人買賣超（張）</h4>${s.chips ? `<div class="tablewrap"><table class="tbl"><thead><tr><th>日期</th><th class="r">外資</th><th class="r">投信</th><th class="r">自營商</th></tr></thead><tbody>
          ${[...s.chips.days].reverse().map((r) => `<tr><td>${esc(r[0])}</td>${[1, 2, 3].map((i) => `<td class="r ${cls(r[i])}">${r[i] > 0 ? '+' : ''}${r[i].toLocaleString()}</td>`).join('')}</tr>`).join('')}
          <tr><td><b>近 5 日合計</b></td>${['f5', 't5', 'd5'].map((k) => `<td class="r ${cls(s.chips[k])}"><b>${s.chips[k] > 0 ? '+' : ''}${s.chips[k].toLocaleString()}</b></td>`).join('')}</tr></tbody></table></div>` : '<p class="hint">尚無籌碼資料。</p>'}</div>
        <div><h4>相關新聞</h4>${s.news.length ? s.news.map((n) => `<div class="news-item"><a href="${safeUrl(n.url)}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a>
          <div class="meta">${esc(n.source)}　${esc(n.time)}　情緒 <span class="${cls(n.sent)}">${n.sent > 0 ? '+' : ''}${n.sent.toFixed(2)}</span></div></div>`).join('') : '<p class="hint">這段期間沒有找到直接提到這檔股票的新聞。</p>'}
          ${s.us.beta != null ? `<p class="hint">美股連動：對「美股帶動的台股變動」敏感度 ${s.us.beta} 倍${LATEST.meta.us_available ? `，本次調整 ${pct(s.us.adj)}` : ''}。</p>` : ''}</div>
      </div>`;
    priceChart('stock-chart', s.ohlc, f.path, { candle: true });
    history.replaceState(null, '', `#stock=${encodeURIComponent(code)}`);
    if (scroll) $('#lookup').scrollIntoView();
  }

  function setupSearch() {
    const q = $('#q'), box = $('#suggest');
    let hits = [], cur = -1;
    const draw = () => { box.hidden = !hits.length; box.innerHTML = hits.map((x, i) => `<button type="button" data-code="${esc(x.c)}" class="${i === cur ? 'on' : ''}"><span><b>${esc(x.c)}</b>　${esc(x.n)}</span><small>${esc(x.i)}　${ACT[x.a]}</small></button>`).join(''); };
    q.addEventListener('input', () => {
      const v = q.value.trim().toLowerCase();
      hits = v ? INDEX.filter((x) => x.c.startsWith(v) || x.n.toLowerCase().includes(v)).slice(0, 8) : [];
      cur = -1; draw();
    });
    q.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') { cur = Math.min(cur + 1, hits.length - 1); draw(); e.preventDefault(); }
      else if (e.key === 'ArrowUp') { cur = Math.max(cur - 1, 0); draw(); e.preventDefault(); }
      else if (e.key === 'Enter') { const h = hits[Math.max(cur, 0)]; if (h) pick(h.c); else if (q.value.trim()) showStock(q.value.trim()); }
      else if (e.key === 'Escape') { hits = []; draw(); }
    });
    const pick = (code) => { hits = []; draw(); q.value = ''; showStock(code); };
    box.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) pick(b.dataset.code); });
    document.addEventListener('click', (e) => {
      const tr = e.target.closest('tr[data-code]');
      if (tr) showStock(tr.dataset.code);
      const th = e.target.closest('th[data-sort]');
      if (th) { const k = th.dataset.sort; sortDir = sortKey === k ? -sortDir : (k === 'r' || k === 'n' || k === 'i' ? 1 : -1); sortKey = k; renderAll(); }
      if (!e.target.closest('.search')) { hits = []; draw(); }
    });
  }

  /* ── 新聞 ─────────────────────────────────────────────────────────── */
  function renderNews(d) {
    $('#news-meta').textContent = `近 ${d.news.hours} 小時共 ${d.news.total} 則（已合併重複報導）。衝擊分數＝報導家數＋重大事件字眼＋情緒強度；每則都附上評分依據。`;
    $('#events').innerHTML = d.events.length ? d.events.map((e) => `<div class="news-item"><div class="hd">
        <b>${esc(e.name)}${e.code ? ` <small class="hint">${esc(e.code)}　${esc(e.industry)}</small>` : ''}</b>${signed(e.chg)}
        ${e.code ? `<a href="#stock=${esc(e.code)}" data-jump="${esc(e.code)}" style="font-weight:400;color:var(--accent);font-size:13px">看分析</a>` : ''}</div>
        <div class="why">${esc(e.cause)}${e.excess != null ? `（扣除大盤連動後 ${pct(e.excess)}）` : ''}</div>
        ${e.news.length ? `<ul>${e.news.map((n) => `<li><a href="${safeUrl(n.url)}" target="_blank" rel="noopener noreferrer" style="font-weight:400">${esc(n.title)}</a> <span class="meta">${esc(n.source)}　${esc(n.time)}</span>${n.why ? `<div class="meta">${esc(n.why)}</div>` : ''}</li>`).join('')}</ul>` : ''}</div>`).join('')
      : '<p class="hint">最近一個交易日沒有達到門檻的大波動。</p>';
    $('#major').innerHTML = d.major_news.length ? d.major_news.map((n) => `<div class="news-item"><div class="hd">
        <span class="badge ${n.tone === '利多' ? 'bull' : n.tone === '利空' ? 'bear' : 'neutral'}">${esc(n.tone)}</span>
        <a href="${safeUrl(n.url)}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a></div>
        <div class="meta">${esc(n.source)}　${esc(n.time)}　衝擊分數 ${n.impact}　情緒 ${n.sent > 0 ? '+' : ''}${n.sent.toFixed(2)}</div>
        <div class="why">判斷依據：${esc(n.why)}</div>
        ${n.names.length ? `<div>${n.names.map((x) => `<span class="chip">${esc(x)}</span>`).join('')}</div>` : ''}</div>`).join('')
      : '<p class="hint">這段期間沒有抓到新聞。</p>';
  }

  /* ── 復盤 ─────────────────────────────────────────────────────────── */
  function renderReview(rv, model) {
    const a = rv.summary.all, mt = rv.summary.matured, bt = rv.backtest;
    const tile = (k, v, s) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
    $('#rv-tiles').innerHTML = a ? [
      tile('驗證的預測日數', a.days, `實際發布 ${rv.n_live} 天、回測補齊 ${rv.n_backtest} 天`),
      tile('相對強弱命中率', `${a.rel_hit}%`, '猜對會贏過／落後中位數，基準 50%'),
      tile('看好前 10 名超額報酬', `<span class="${cls(a.excess)}">${pct(a.excess)}</span>`, `前 10 名 ${pct(a.top_ret)}，股票池 ${pct(a.uni_ret)}；${a.top_win}% 的日子勝出`),
      tile('高信心訊號命中率', a.n_hi >= 30 ? `${a.hit_hi}%` : '–', `信心 ≥ ${model.hiconf}，共 ${a.n_hi} 筆`),
      tile('絕對方向命中率', `${a.hit}%`, `同期個股上漲比例 ${a.up_share}%`),
      tile('大盤方向命中率', `${a.mkt_hit}%`, `${a.days} 個預測日`),
      tile('已滿 15 日的預測', mt ? mt.days : 0, mt ? `超額 ${pct(mt.excess)}，相對命中 ${mt.rel_hit}%` : '尚無'),
      tile('一年滾動回測', `IC ${bt.ic}`, `${bt.days} 天；前 10 名 ${pct(bt.top_ret)} 對 ${pct(bt.uni_ret)}`),
    ].join('') : '<p class="hint">尚無可驗證的預測。</p>';
    const rows = rv.rows.filter((r) => r.h > 0).slice().reverse();
    const labels = rows.map((r) => r.date.slice(5)), faded = rows.map((r) => r.kind !== 'live');
    const det = rows.map((r) => row('狀態', r.status) + row('前 10 名', pct(r.top_ret)) + row('股票池', pct(r.uni_ret)) + row('來源', r.kind === 'live' ? '實際發布' : '回測補齊'));
    barChart('rv-excess', labels, rows.map((r) => r.excess), { faded, unit: ' 百分點', details: det });
    barChart('rv-hit', labels, rows.map((r) => r.rel_hit), { base: 50, faded, unit: '%', details: det });
    $('#rv-sugg').innerHTML = rv.suggestions.map((s) => `<div class="sugg ${s.level}"><b>${esc(s.title)}</b><span>${esc(s.text)}</span></div>`).join('');
    $('#tbl-calib').innerHTML = '<thead><tr><th>信心區間</th><th class="r">樣本數</th><th class="r">平均信心</th><th class="r">實際命中率</th></tr></thead><tbody>' +
      (rv.buckets.length ? rv.buckets.map((b) => `<tr><td>${esc(b.range)}</td><td class="r">${b.n.toLocaleString()}</td><td class="r">${b.avg_conf}</td><td class="r"><b>${b.hit}%</b></td></tr>`).join('') : '<tr><td colspan="4" class="hint">樣本不足</td></tr>') + '</tbody>';
    $('#tbl-rv').innerHTML = '<thead><tr><th>預測日</th><th>狀態</th><th class="r">相對命中</th><th class="r">前 10 名</th><th class="r">股票池</th><th class="r">超額</th><th class="r">大盤預測</th><th class="r">大盤實際</th></tr></thead><tbody>' +
      rv.rows.map((r) => r.h > 0 ? `<tr><td class="nw">${r.date}${r.kind === 'live' ? '' : ' <span class="chip">回測</span>'}</td><td class="nw">${esc(r.status)}</td><td class="r">${r.rel_hit}%</td>
        <td class="r ${cls(r.top_ret)}">${pct(r.top_ret)}</td><td class="r ${cls(r.uni_ret)}">${pct(r.uni_ret)}</td><td class="r ${cls(r.excess)}"><b>${pct(r.excess)}</b></td>
        <td class="r">${pct(r.mkt_mu)}</td><td class="r ${cls(r.mkt_real)}">${pct(r.mkt_real)} ${r.mkt_hit ? '✓' : '✗'}</td></tr>`
        : `<tr><td class="nw">${r.date}</td><td colspan="7" class="hint">${esc(r.status)}</td></tr>`).join('') + '</tbody>';
  }

  function renderModel(d) {
    const m = d.model, bt = m.backtest, fs = m.factors;
    const mx = Math.max(...fs.map((f) => Math.abs(f.coef)));
    const flag = { with: '<span class="chip">近期有效</span>', against: '<span class="chip" style="color:var(--warn)">近期失靈</span>', flat: '' };
    $('#tbl-factors').innerHTML = '<thead><tr><th>因子</th><th>類別</th><th class="r">模型權重</th><th class="r">20 日前</th><th class="r">近期 IC</th><th></th></tr></thead><tbody>' +
      [...fs].sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef)).map((f) => `<tr><td>${esc(f.name)}</td><td class="hint">${esc(f.group)}</td>
        <td class="r"><span class="num">${f.coef > 0 ? '+' : ''}${f.coef.toFixed(4)}</span><i class="inbar" style="width:${Math.abs(f.coef) / mx * 50}px;background:${f.coef >= 0 ? C.up : C.dn}"></i></td>
        <td class="r hint">${f.prev == null ? '–' : (f.prev > 0 ? '+' : '') + f.prev.toFixed(4)}</td><td class="r ${cls(f.ic)}">${f.ic > 0 ? '+' : ''}${f.ic.toFixed(3)}</td><td>${flag[f.flag]}</td></tr>`).join('') + '</tbody>';
    $('#method').innerHTML = `<h3>預測怎麼算</h3>
      <p><b>個股 15 日預期報酬 ＝ β × 大盤預期報酬 ＋ 相對強度 × 個股波動 ＋ 美股隔夜調整</b></p>
      <ul>
        <li><b>相對強度</b>：用左表的技術因子，每天以最近 ${m.train_days} 個交易日、全部 ${d.meta.universe} 檔股票重新訓練。因子與結果都先排除波動與 β 的影響，回答的是「同樣風險下，誰比較強」。再疊加新聞情緒與法人籌碼（權重較小，並依復盤結果自動調整）。</li>
        <li><b>大盤</b>：用大盤動能、市場寬度、費半、S&amp;P 500、VIX 等估計。大盤方向很難預測，請把它當作背景，而不是進出依據。</li>
        <li><b>美股隔夜</b>：盤前版會把前一晚美股漲跌依歷史連動關係換算後加入。</li>
        <li><b>信心值</b>：回測中同等強度的訊號，15 日後實際贏過股票池中位數的比例。基準是 50，目前模型的信心值大多落在 50–55 之間——這是技術面訊號的真實水準，不是 90 分那種把握。</li>
        <li><b>建議等級</b>：相對強度前 5% 為強力看多、前 20% 偏多，後 20% 偏空、後 5% 強力看空；若與絕對預期報酬方向相反則降一級。</li>
      </ul>
      <h3 style="margin-top:14px">一年滾動回測（樣本外）</h3>
      <ul>
        <li>${bt.from} ～ ${bt.to}，共 ${bt.days} 個交易日、${bt.samples.toLocaleString()} 筆預測；每次只用當下已知的資料訓練。</li>
        <li>排序相關 IC ${bt.ic}（前半 ${bt.ic_h1}、後半 ${bt.ic_h2}），${bt.ic_pos_days}% 的日子為正。</li>
        <li>看好前 10 名 15 日平均報酬 ${pct(bt.top_ret)}、看淡後 10 名 ${pct(bt.bot_ret)}、股票池平均 ${pct(bt.uni_ret)}；前 10 名在 ${bt.top_win}% 的日子贏過平均。</li>
        <li>相對強弱命中率 ${bt.rel_hit}%（基準 50%）。效果主要來自少數大漲的股票，而不是高勝率。</li>
      </ul>
      <h3 style="margin-top:14px">已知限制</h3>
      <ul>
        <li>股票池是用「現在」成交金額最大的股票選的，回測因此偏樂觀（倖存者偏誤），實際績效通常較低。</li>
        <li>新聞情緒是關鍵字規則判斷，會誤判反諷與複雜語意；新聞與籌碼沒有足夠歷史可回測，只能靠之後的復盤驗證。</li>
        <li>未計入交易成本、漲跌停買不到／賣不掉、除權息與國定假日等因素。</li>
      </ul>`;
  }

  /* ── 小提示框（用於長條分段）─────────────────────────────────────── */
  const tip = $('#tip');
  document.addEventListener('pointermove', (e) => {
    const t = e.target.closest?.('[data-tip]');
    if (!t) { tip.hidden = true; return; }
    tip.textContent = t.dataset.tip; tip.hidden = false;
    tip.style.left = `${Math.min(e.clientX + 12, innerWidth - 160)}px`; tip.style.top = `${e.clientY + 14}px`;
  });
  document.addEventListener('click', (e) => { const j = e.target.closest('[data-jump]'); if (j) { e.preventDefault(); showStock(j.dataset.jump); } });

  async function main() {
    try {
      const [latest, index, review] = await Promise.all([getJSON('data/latest.json'), getJSON('data/index.json'), getJSON('data/review.json')]);
      LATEST = latest; INDEX = index;
      renderOverview(latest); renderUS(latest); renderPicks(latest); renderNews(latest); renderReview(review, latest.model); renderModel(latest);
      renderAll(); setupSearch();
      $('#universe-note').textContent = `可查詢今日股票池內的 ${latest.meta.universe} 檔（上市、上櫃成交金額最大的個股與主要 ETF）。`;
      const m = location.hash.match(/^#stock=(.+)$/);
      showStock(m ? decodeURIComponent(m[1]) : latest.top_bull[0].code, !!m);
    } catch (e) {
      const el = $('#error'); el.hidden = false;
      el.textContent = `資料載入失敗（${e.message}）。若是剛建立的網站，請先到 GitHub Actions 手動執行一次「每日更新」。`;
      $('#stamp').textContent = '';
    }
  }
  main();
})();
