/* ─────────────────────────────────────────────────────────────
   MigraineSense – Dashboard JS v2.0
   Fixes: chart lag, keyboard risk real-time, merged risk graph,
          helmet therapy form + BT connection management
───────────────────────────────────────────────────────────── */
'use strict';

// ═══════════════════════════════════════════════════════════════
//  SOCKET.IO CONNECTION
// ═══════════════════════════════════════════════════════════════
const socket = io({ transports: ['websocket'], upgrade: false });

socket.on('connect',      ()    => setConnection(true));
socket.on('disconnect',   ()    => setConnection(false));
socket.on('risk_update',  data  => scheduleUpdate(data));
socket.on('bt_response',  data  => handleBtResponse(data));
socket.on('bt_ports',     ports => populatePorts(ports));

// ═══════════════════════════════════════════════════════════════
//  RAF-BATCHED UPDATE QUEUE  (eliminates chart/DOM lag)
// ═══════════════════════════════════════════════════════════════
let _pendingData  = null;
let _rafScheduled = false;

function scheduleUpdate(data) {
  _pendingData = data;                        // always keep latest
  if (!_rafScheduled) {
    _rafScheduled = true;
    requestAnimationFrame(() => {
      _rafScheduled = false;
      if (_pendingData) { updateDashboard(_pendingData); _pendingData = null; }
    });
  }
}

// ═══════════════════════════════════════════════════════════════
//  CLOCK
// ═══════════════════════════════════════════════════════════════
function updateClock() {
  const el = document.getElementById('clockDisplay');
  if (el) el.textContent = new Date().toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════════════════════════════════════════════
//  CONNECTION INDICATOR
// ═══════════════════════════════════════════════════════════════
function setConnection(ok) {
  const dot   = document.getElementById('connectionDot');
  const label = document.getElementById('connectionLabel');
  if (dot)   dot.className   = 'status-dot ' + (ok ? 'connected' : 'disconnected');
  if (label) label.textContent = ok ? 'Live' : 'Reconnecting…';
}

// ═══════════════════════════════════════════════════════════════
//  TAB SWITCHING
// ═══════════════════════════════════════════════════════════════
function switchTab(name, btn) {
  document.querySelectorAll('.page-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  const tab = document.getElementById('tab-' + name);
  if (tab) tab.classList.add('active');
  if (btn) btn.classList.add('active');
  // Resize charts when switching to history tab
  if (name === 'history') {
    setTimeout(() => { if (historyChart) historyChart.resize(); }, 60);
    loadFullHistory();
  }
}
window.switchTab = switchTab;

// ═══════════════════════════════════════════════════════════════
//  COMBINED RISK TIMELINE CHART
//  Both Blended Risk + Composite Risk on ONE chart, no animation lag
// ═══════════════════════════════════════════════════════════════
const MAX_POINTS    = 60;
const tlLabels      = [];
const tlBlended     = [];
const tlComposite   = [];

const timelineChart = new Chart(
  document.getElementById('timelineChart').getContext('2d'), {
    type: 'line',
    data: {
      labels: tlLabels,
      datasets: [
        {
          label:           'Blended Risk',
          data:            tlBlended,
          borderColor:     '#4f8ef7',
          backgroundColor: 'rgba(79,142,247,0.08)',
          borderWidth:     2.5,
          pointRadius:     0,
          pointHoverRadius:4,
          fill:            true,
          tension:         0.35,
          order:           1,
        },
        {
          label:           'Composite Risk',
          data:            tlComposite,
          borderColor:     '#06d6d6',
          backgroundColor: 'rgba(6,214,214,0.04)',
          borderWidth:     2,
          borderDash:      [5, 4],
          pointRadius:     0,
          pointHoverRadius:4,
          fill:            true,
          tension:         0.35,
          order:           2,
        },
      ],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           false,          // ← KEY FIX: disable all animation
      interaction:         { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: false,                  // legend handled in HTML
        },
        tooltip: {
          backgroundColor: '#111827',
          borderColor:     '#1e2d4a',
          borderWidth:     1,
          titleColor:      '#e8eef8',
          bodyColor:       '#7a8aaa',
          padding:         10,
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#4a5a7a', maxTicksLimit: 8, font: { size: 10 }, maxRotation: 0 },
          grid:  { color: 'rgba(255,255,255,0.03)' },
          border:{ color: 'rgba(255,255,255,0.06)' },
        },
        y: {
          min:   0,
          max:   100,
          ticks: { color: '#4a5a7a', stepSize: 25, font: { size: 10 },
                   callback: v => v + '%' },
          grid:  { color: 'rgba(255,255,255,0.05)' },
          border:{ color: 'rgba(255,255,255,0.06)' },
        },
      },
    },
  }
);

// ═══════════════════════════════════════════════════════════════
//  HISTORY CHART (Tab 3)
// ═══════════════════════════════════════════════════════════════
const historyChart = new Chart(
  document.getElementById('historyChart').getContext('2d'), {
    type: 'line',
    data: {
      labels:   [],
      datasets: [{
        label:           'Blended Risk',
        data:            [],
        borderColor:     '#4f8ef7',
        backgroundColor: 'rgba(79,142,247,0.07)',
        borderWidth:     2,
        pointRadius:     0,
        fill:            true,
        tension:         0.3,
      },{
        label:           'Composite Risk',
        data:            [],
        borderColor:     '#06d6d6',
        backgroundColor: 'rgba(6,214,214,0.04)',
        borderWidth:     1.5,
        borderDash:      [4,4],
        pointRadius:     0,
        fill:            false,
        tension:         0.3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { labels: { color:'#7a8aaa', font:{ size:11 } } },
        tooltip: {
          backgroundColor:'#111827', borderColor:'#1e2d4a', borderWidth:1,
          titleColor:'#e8eef8', bodyColor:'#7a8aaa',
        },
      },
      scales: {
        x: { ticks:{ color:'#4a5a7a', maxTicksLimit:10, font:{size:10}, maxRotation:0 }, grid:{ color:'rgba(255,255,255,0.03)' } },
        y: { min:0, max:100, ticks:{ color:'#4a5a7a', stepSize:25, font:{size:10}, callback: v => v+'%' }, grid:{ color:'rgba(255,255,255,0.04)' } },
      },
    },
  }
);

// ═══════════════════════════════════════════════════════════════
//  GAUGE
// ═══════════════════════════════════════════════════════════════
const gaugeCtx     = document.getElementById('gaugeCanvas').getContext('2d');
let   gaugeCurrent = 0;
let   gaugeTarget  = 0;
let   gaugeRaf     = null;

function drawGauge(score) {
  const W = 200, H = 200, cx = W/2, cy = H/2, R = 82;
  gaugeCtx.clearRect(0, 0, W, H);

  const startAngle = Math.PI * 0.75;
  const endAngle   = Math.PI * 2.25;
  const fillAngle  = startAngle + (Math.max(0, Math.min(100, score)) / 100) * (endAngle - startAngle);

  // Track
  gaugeCtx.beginPath();
  gaugeCtx.arc(cx, cy, R, startAngle, endAngle);
  gaugeCtx.strokeStyle = '#1a2035';
  gaugeCtx.lineWidth   = 16;
  gaugeCtx.lineCap     = 'round';
  gaugeCtx.stroke();

  // Zone ticks
  [[25,'#10d982'],[50,'#f5c518'],[75,'#f97316'],[100,'#ef4444']].forEach(([v, c]) => {
    const a = startAngle + (v/100)*(endAngle-startAngle);
    gaugeCtx.beginPath();
    gaugeCtx.arc(cx, cy, R, a - 0.012, a + 0.012);
    gaugeCtx.strokeStyle = c;
    gaugeCtx.lineWidth   = 20;
    gaugeCtx.stroke();
  });

  // Fill
  if (score > 0) {
    const color = scoreColor(score);
    const grad  = gaugeCtx.createLinearGradient(0, cy, W, cy);
    grad.addColorStop(0, color + 'aa');
    grad.addColorStop(1, color);
    gaugeCtx.beginPath();
    gaugeCtx.arc(cx, cy, R, startAngle, fillAngle);
    gaugeCtx.strokeStyle = grad;
    gaugeCtx.lineWidth   = 16;
    gaugeCtx.lineCap     = 'round';
    gaugeCtx.stroke();
  }

  // Update glow
  const glowEl = document.getElementById('gaugeGlow');
  if (glowEl) glowEl.style.background =
    `radial-gradient(circle, ${scoreColor(score)}22 0%, transparent 70%)`;
}

function animateGauge(target) {
  if (gaugeRaf) cancelAnimationFrame(gaugeRaf);
  const start = gaugeCurrent;
  const delta = target - start;
  const dur   = 500;
  const t0    = performance.now();
  function step(now) {
    const p    = Math.min((now - t0) / dur, 1);
    const ease = p < 0.5 ? 2*p*p : -1+(4-2*p)*p;
    gaugeCurrent = start + delta * ease;
    drawGauge(gaugeCurrent);
    const el = document.getElementById('gaugeScore');
    if (el) {
      el.textContent  = Math.round(gaugeCurrent);
      el.style.color  = scoreColor(gaugeCurrent);
    }
    if (p < 1) gaugeRaf = requestAnimationFrame(step);
    else gaugeCurrent = target;
  }
  gaugeRaf = requestAnimationFrame(step);
}

function scoreColor(s) {
  if (s >= 75) return '#ef4444';
  if (s >= 55) return '#f97316';
  if (s >= 30) return '#f5c518';
  return '#10d982';
}

// ═══════════════════════════════════════════════════════════════
//  MAIN UPDATE HANDLER
// ═══════════════════════════════════════════════════════════════
function updateDashboard(d) {
  const score  = +(d.blended_risk   || 0);
  const comp   = +(d.composite_risk || 0);
  const kbRisk = +(d.keyboard_risk  || 0);   // ← explicit read
  const msRisk = +(d.mouse_risk     || 0);
  const wcRisk = +(d.webcam_risk    || 0);
  const sysRisk= +(d.system_risk    || 0);
  const level  = d.risk_level       || 'LOW';
  const eta    = d.eta_minutes;
  const trend  = +(d.trend          || 0);
  const f      = d.features         || {};

  // ── Gauge ──────────────────────────────────────────────────
  animateGauge(score);

  // ── Composite display ──────────────────────────────────────
  const compEl = document.getElementById('compositeVal');
  if (compEl) compEl.textContent = comp.toFixed(1) + '%';

  // ── Risk badge ─────────────────────────────────────────────
  const badge = document.getElementById('riskBadge');
  if (badge) { badge.textContent = level; badge.className = 'risk-badge ' + level; }

  // ── ETA ────────────────────────────────────────────────────
  const etaEl = document.getElementById('etaValue');
  if (etaEl) etaEl.textContent = etaString(eta, level);

  // ── Trend ──────────────────────────────────────────────────
  const trendEl   = document.getElementById('trendValue');
  const trendIcon = document.getElementById('trendIcon');
  if (trendEl) {
    if (trend > 0.5) {
      trendEl.textContent = `Worsening (+${trend.toFixed(1)}/min)`;
      trendEl.style.color = '#f97316';
      if (trendIcon) trendIcon.textContent = '⬆️';
    } else if (trend < -0.5) {
      trendEl.textContent = `Improving (${trend.toFixed(1)}/min)`;
      trendEl.style.color = '#10d982';
      if (trendIcon) trendIcon.textContent = '⬇️';
    } else {
      trendEl.textContent = 'Stable';
      trendEl.style.color = '#7a8aaa';
      if (trendIcon) trendIcon.textContent = '→';
    }
  }

  // ── ML badge ────────────────────────────────────────────────
  const mlBadge = document.getElementById('mlBadge');
  if (mlBadge) {
    mlBadge.textContent = d.model_trained ? 'Active' : 'Learning…';
    mlBadge.className   = 'ml-badge' + (d.model_trained ? ' ready' : '');
  }
  const mlScore = document.getElementById('mlScore');
  if (mlScore) mlScore.textContent = (d.ml_anomaly_pct || 0).toFixed(1);

  // ── Domain bars — all 4 updated every tick ──────────────────
  setBar('kb',  kbRisk);   // ← keyboard now uses direct extracted value
  setBar('ms',  msRisk);
  setBar('wc',  wcRisk);
  setBar('sys', sysRisk);

  // ── Domain hints ────────────────────────────────────────────
  setText('kbHints',  kbHint(f));
  setText('msHints',  msHint(f));
  setText('wcHints',  wcHint(f));
  setText('sysHints', sysHint(f));

  // ── Timeline chart — BOTH datasets, no animation ────────────
  const nowLabel = new Date().toLocaleTimeString();
  if (tlLabels.length >= MAX_POINTS) {
    tlLabels.shift();
    tlBlended.shift();
    tlComposite.shift();
  }
  tlLabels.push(nowLabel);
  tlBlended.push(score);
  tlComposite.push(comp);
  timelineChart.update('none');       // ← 'none' = instant, no animation lag

  // ── Live metrics ─────────────────────────────────────────────
  setText('m_kpm',    fv(f.typing_speed_kpm,    1) + ' kpm');
  setText('m_err',    fv(f.error_rate*100,       1) + ' %');
  setText('m_pause',  fv(f.pause_count,          0));
  setText('m_cv',     fv(f.rhythm_cv,            3));
  setText('m_mspd',   fv(f.mouse_avg_speed_px,   1) + ' px/s');
  setText('m_jit',    fv(f.mouse_jitter,         3));
  setText('m_clk',    fv(f.click_rate_per_min,   1) + ' /min');
  setText('m_midle',  fv(f.mouse_idle_sec,       0) + ' s');
  setText('m_bpm',    fv(f.blink_rate_bpm,       1) + ' bpm');
  setText('m_ear',    fv(f.mean_ear,             3));
  setText('m_tilt',   fv(f.head_tilt_deg,        1) + '°');
  setText('m_prox',   fv(f.face_proximity_px,    0) + ' px');
  setText('m_cpu',    fv(f.avg_cpu_pct,          1) + ' %');
  setText('m_mem',    fv(f.avg_mem_pct,          1) + ' %');
  setText('m_bright', fv(f.avg_brightness,       0) + ' %');
  setText('m_sw',     fv(f.app_switch_rate,      2) + ' /min');

  // ── Alert banner ─────────────────────────────────────────────
  if (level !== 'LOW') showAlertBanner(level, score, eta);
  else                 dismissAlert();

  // ── Footer ───────────────────────────────────────────────────
  setText('lastUpdate', 'Last update: ' + new Date().toLocaleTimeString());

  // ── Alert history (throttled) ─────────────────────────────
  throttleAlerts();
}

// ═══════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = (val === undefined || val === null) ? '—' : val;
}

function setBar(prefix, value) {
  const bar = document.getElementById(prefix + 'Bar');
  const pct = document.getElementById(prefix + 'Pct');
  const v   = Math.max(0, Math.min(100, +value || 0));
  if (bar) {
    bar.style.width = v + '%';
    bar.classList.remove('high', 'critical');
    if (v >= 75) bar.classList.add('critical');
    else if (v >= 55) bar.classList.add('high');
  }
  if (pct) pct.textContent = v.toFixed(0) + '%';
}

function fv(v, decimals = 1) {
  if (v === undefined || v === null || isNaN(+v)) return '—';
  return Number(v).toFixed(decimals);
}

function etaString(eta, level) {
  if (!eta || level === 'LOW') return 'No risk detected ✅';
  if (eta >= 120) return `~${Math.round(eta/60)} hours`;
  return `~${eta} minutes`;
}

function kbHint(f) {
  const h = [];
  if (+f.typing_speed_kpm < 20) h.push('slow typing');
  if (+f.error_rate > 0.10)      h.push('high errors');
  if (+f.pause_count > 8)        h.push('many pauses');
  if (+f.rhythm_cv > 1.0)        h.push('irregular rhythm');
  return h.length ? '⚠ ' + h.join(' · ') : '✓ Normal';
}
function msHint(f) {
  const h = [];
  if (+f.mouse_jitter > 0.4)     h.push('hand tremor');
  if (+f.mouse_idle_sec > 120)   h.push('long idle');
  if (+f.mouse_efficiency < 0.4) h.push('erratic path');
  return h.length ? '⚠ ' + h.join(' · ') : '✓ Normal';
}
function wcHint(f) {
  const h = [];
  if (+f.blink_rate_bpm < 8)           h.push('rare blinking');
  if (+f.mean_ear < 0.20)              h.push('squinting');
  if (Math.abs(+f.head_tilt_deg) > 15) h.push('head tilt');
  return h.length ? '⚠ ' + h.join(' · ') : '✓ Normal';
}
function sysHint(f) {
  const h = [];
  if (+f.avg_brightness > 85)    h.push('bright screen');
  if (+f.avg_cpu_pct > 80)       h.push('high CPU');
  if (f.is_late_night)           h.push('late night');
  if (+f.app_switch_rate > 10)   h.push('overloaded');
  return h.length ? '⚠ ' + h.join(' · ') : '✓ Normal';
}

// ═══════════════════════════════════════════════════════════════
//  ALERT BANNER
// ═══════════════════════════════════════════════════════════════
function showAlertBanner(level, score, eta) {
  const banner = document.getElementById('alertBanner');
  if (!banner) return;
  const icons = { MODERATE:'🟡', HIGH:'🟠', CRITICAL:'🔴' };
  setText('alertIcon', icons[level] || '⚠️');
  setText('alertText', `${level} RISK (${score.toFixed(0)}/100) — ${etaString(eta, level)} — Please take a break!`);
  banner.className = 'alert-banner ' + level;
}
function dismissAlert() {
  const b = document.getElementById('alertBanner');
  if (b) b.className = 'alert-banner hidden';
}
window.dismissAlert = dismissAlert;

// ═══════════════════════════════════════════════════════════════
//  ALERT HISTORY (throttled — only fetch when count changes)
// ═══════════════════════════════════════════════════════════════
let _lastAlertCount = 0;
let _alertThrottle  = 0;

function throttleAlerts() {
  const now = Date.now();
  if (now - _alertThrottle < 5000) return;   // max once per 5 s
  _alertThrottle = now;
  loadAlertHistory();
}

async function loadAlertHistory() {
  try {
    const res    = await fetch('/api/alerts');
    const alerts = await res.json();
    if (alerts.length === _lastAlertCount) return;
    _lastAlertCount = alerts.length;
    renderAlerts('alertsList',     alerts, 10);
    renderAlerts('fullAlertsList', alerts, 100);
  } catch(e) {}
}

function renderAlerts(listId, alerts, limit) {
  const list = document.getElementById(listId);
  if (!list) return;
  if (!alerts.length) {
    list.innerHTML = '<div class="no-alerts">No alerts yet — you\'re doing great! 🎉</div>';
    return;
  }
  list.innerHTML = alerts.slice(0, limit).map(a => {
    const t = new Date(a.timestamp * 1000).toLocaleTimeString();
    return `<div class="alert-item ${a.level}">
      <div>
        <div class="alert-item-msg">${a.message}</div>
        <div class="alert-item-time">${t} · Score: ${(a.risk_score||0).toFixed(0)}</div>
      </div>
    </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════════════
//  HISTORY TAB — load full data
// ═══════════════════════════════════════════════════════════════
async function loadFullHistory() {
  try {
    const res  = await fetch('/api/history');
    const rows = await res.json();
    if (!rows.length) return;
    const labels  = rows.map(r => new Date(r.timestamp*1000).toLocaleTimeString());
    const blended = rows.map(r => r.blended_risk   || 0);
    const comp    = rows.map(r => r.composite_risk || 0);
    historyChart.data.labels            = labels;
    historyChart.data.datasets[0].data  = blended;
    historyChart.data.datasets[1].data  = comp;
    historyChart.update('none');
    renderAlerts('fullAlertsList', rows, 100);
  } catch(e) {}
}

// ═══════════════════════════════════════════════════════════════
//  TIPS ROTATOR
// ═══════════════════════════════════════════════════════════════
let tipIdx = 0;
const tips = () => document.querySelectorAll('.tip-item');

function showTip(i) {
  const all = tips();
  all.forEach(t => t.classList.remove('active'));
  tipIdx = ((i % all.length) + all.length) % all.length;
  all[tipIdx].classList.add('active');
  setText('tipCounter', `${tipIdx+1} / ${all.length}`);
}
window.nextTip = () => showTip(tipIdx + 1);
window.prevTip = () => showTip(tipIdx - 1);
setInterval(() => showTip(tipIdx + 1), 9000);


// ═══════════════════════════════════════════════════════════════
//  ─────────────────────────────────────────────────────────────
//  HELMET THERAPY SECTION
//  ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

// Protocol library (mirrors therapy_protocol_library.json)
const PROTOCOLS = [
  { id:'PROTO-001', type:'tension',               phase:'headache',  minPain:1, maxPain:4, profile:'gentle',    label:'Tension Headache — Mild',                    duration:20 },
  { id:'PROTO-002', type:'tension',               phase:'headache',  minPain:5, maxPain:10,profile:'moderate',  label:'Tension Headache — Severe',                  duration:30 },
  { id:'PROTO-003', type:'migraine_with_aura',    phase:'aura',      minPain:1, maxPain:5, profile:'gentle',    label:'Migraine With Aura — Aura Phase',            duration:15 },
  { id:'PROTO-004', type:'migraine_with_aura',    phase:'headache',  minPain:6, maxPain:10,profile:'intensive', label:'Migraine With Aura — Peak Headache',         duration:35 },
  { id:'PROTO-005', type:'cluster',               phase:'headache',  minPain:7, maxPain:10,profile:'intensive', label:'Cluster Headache — Intensive',               duration:30 },
  { id:'PROTO-006', type:'migraine_without_aura', phase:'headache',  minPain:3, maxPain:6, profile:'moderate',  label:'Migraine Without Aura — Moderate',           duration:25 },
  { id:'PROTO-007', type:'vestibular',            phase:'headache',  minPain:1, maxPain:10,profile:'gentle',    label:'Vestibular Migraine — Gentle Stabilisation', duration:20 },
  { id:'PROTO-008', type:'hemiplegic',            phase:'headache',  minPain:1, maxPain:10,profile:'gentle',    label:'Hemiplegic Migraine — Ultra Gentle',         duration:15 },
  { id:'PROTO-009', type:'any',                   phase:'prodrome',  minPain:1, maxPain:4, profile:'gentle',    label:'Prodrome Preventive — Early Intervention',   duration:15 },
  { id:'PROTO-010', type:'any',                   phase:'postdrome', minPain:1, maxPain:3, profile:'gentle',    label:'Postdrome Recovery — Gentle Relaxation',     duration:20 },
];

// Protocol zone configurations
const ZONE_CONFIGS = {
  'PROTO-004': {
    zones: [
      { id:'frontal',        tech:'SUSTAINED_PRESSURE', intensity:7, on:40000, off:10000, dur:2100 },
      { id:'parietal',       tech:'PETRISSAGE',         intensity:6, on:30000, off:10000, dur:2100 },
      { id:'occipital',      tech:'TRIGGER_POINT',      intensity:8, on:60000, off:20000, dur:1600 },
      { id:'right_temporal', tech:'RHYTHMIC_TAPPING',   intensity:5, on:20000, off:10000, dur:1200 },
      { id:'left_temporal',  tech:'RHYTHMIC_TAPPING',   intensity:5, on:20000, off:10000, dur:1200 },
    ],
    pads: [
      { id:'frontal_pad',   mode:'COLD',        dur:2100 },
      { id:'occipital_pad', mode:'ALTERNATING', dur:1600, heatDur:120, coldDur:90, repeats:8 },
    ],
  },
  'PROTO-001': {
    zones: [
      { id:'frontal',        tech:'EFFLEURAGE',         intensity:3, on:20000, off:10000, dur:1200 },
      { id:'parietal',       tech:'CIRCULAR',           intensity:3, on:20000, off:10000, dur:1200 },
      { id:'occipital',      tech:'SUSTAINED_PRESSURE', intensity:3, on:30000, off:10000, dur:900  },
      { id:'right_temporal', tech:'EFFLEURAGE',         intensity:2, on:15000, off:5000,  dur:600  },
      { id:'left_temporal',  tech:'EFFLEURAGE',         intensity:2, on:15000, off:5000,  dur:600  },
    ],
    pads: [
      { id:'frontal_pad',   mode:'HEAT', dur:1200 },
      { id:'occipital_pad', mode:'HEAT', dur:900  },
    ],
  },
};
// Default config for protocols not explicitly listed
function getDefaultConfig(proto) {
  return {
    zones: [
      { id:'frontal',        tech:'SUSTAINED_PRESSURE', intensity:5, on:30000, off:10000, dur:proto.duration*60 },
      { id:'parietal',       tech:'CIRCULAR',           intensity:4, on:20000, off:10000, dur:proto.duration*60 },
      { id:'occipital',      tech:'TRIGGER_POINT',      intensity:5, on:40000, off:15000, dur:proto.duration*60 },
      { id:'right_temporal', tech:'RHYTHMIC_TAPPING',   intensity:3, on:15000, off:5000,  dur:proto.duration*60 },
      { id:'left_temporal',  tech:'RHYTHMIC_TAPPING',   intensity:3, on:15000, off:5000,  dur:proto.duration*60 },
    ],
    pads: [
      { id:'frontal_pad',   mode:'COLD', dur:proto.duration*60 },
      { id:'occipital_pad', mode:'HEAT', dur:proto.duration*60 },
    ],
  };
}

let selectedProto  = null;
let btConnected    = false;
let sessionLoaded  = false;
let sessionRunning = false;
let sessionPaused  = false;
let sessionStartTs = 0;
let sessionTotalMs = 0;
let timerInterval  = null;
let currentSessionId = '';

// ── Pain score badge sync ────────────────────────────────────
const painSlider = document.getElementById('f_painScore');
const painBadge  = document.getElementById('painBadge');
if (painSlider) {
  painSlider.addEventListener('input', () => {
    if (painBadge) painBadge.textContent = painSlider.value;
    autoSelectProtocol();
  });
}

// ── Auto-select protocol from form values ───────────────────
function autoSelectProtocol() {
  const type  = document.getElementById('f_migraineType')?.value || '';
  const phase = document.getElementById('f_phase')?.value       || '';
  const pain  = +(document.getElementById('f_painScore')?.value  || 5);

  let match = PROTOCOLS.find(p =>
    (p.type === type || p.type === 'any') &&
    p.phase === phase &&
    pain >= p.minPain && pain <= p.maxPain
  );
  if (!match) match = PROTOCOLS.find(p => p.type === type) || PROTOCOLS[0];

  selectedProto = match;

  setText('protobadge', '🔬 ' + match.id);
  setText('protoLabel',  match.label);

  const profEl = document.getElementById('f_therapyProfile');
  if (profEl) profEl.value = match.profile;
  const durEl = document.getElementById('f_duration');
  if (durEl) durEl.value = match.duration;
}
window.autoSelectProtocol = autoSelectProtocol;

// ═══════════════════════════════════════════════════════════════
//  BLUETOOTH — PORT SCAN
// ═══════════════════════════════════════════════════════════════
function scanPorts() {
  btLog('sys', 'Scanning serial ports…');
  socket.emit('bt_scan_ports');
  fetch('/api/helmet/ports').then(r => r.json()).then(d => {
    populatePorts(d.ports || []);
  }).catch(() => btLog('err', 'Port scan failed'));
}
window.scanPorts = scanPorts;

function populatePorts(ports) {
  const sel = document.getElementById('btPortSelect');
  if (!sel) return;
  sel.innerHTML = ports.length
    ? ports.map(p => `<option value="${p}">${p}</option>`).join('')
    : '<option value="">No ports found</option>';
  if (ports.length) btLog('sys', `Found ${ports.length} port(s): ${ports.join(', ')}`);
}

// ═══════════════════════════════════════════════════════════════
//  BLUETOOTH — CONNECT / DISCONNECT
// ═══════════════════════════════════════════════════════════════
function btConnect() {
  const port = document.getElementById('btPortSelect')?.value;
  const baud = document.getElementById('btBaud')?.value || '9600';
  if (!port) { btLog('err', 'Select a COM port first'); return; }

  btSetStatus('connecting', 'Connecting…', `${port} @ ${baud} baud`);
  btLog('tx', `Connecting to ${port} @ ${baud} baud…`);

  fetch('/api/helmet/connect', {
    method:  'POST',
    headers: { 'Content-Type':'application/json' },
    body:    JSON.stringify({ port, baud: +baud }),
  })
  .then(r => r.json())
  .then(d => {
    if (d.ok) {
      btConnected = true;
      btSetStatus('connected', 'Connected', `${port} — HC-05 ready`);
      btLog('rx', 'Connection established ✓');
      btLog('tx', 'Sending CMD_HANDSHAKE…');
      btSendCmd({ cmd:'CMD_HANDSHAKE', host_id:'MIGRAINE_APP_V1', timestamp: new Date().toISOString() });
      document.getElementById('btConnectBtn').disabled    = true;
      document.getElementById('btDisconnectBtn').disabled = false;
      document.getElementById('loadSessionBtn').disabled  = false;
    } else {
      btSetStatus('disconnected', 'Connection Failed', d.error || 'Unknown error');
      btLog('err', 'Connection failed: ' + (d.error || 'Unknown'));
    }
  })
  .catch(e => {
    btSetStatus('disconnected', 'Error', e.message);
    btLog('err', 'Error: ' + e.message);
  });
}
window.btConnect = btConnect;

function btDisconnect() {
  fetch('/api/helmet/disconnect', { method:'POST' })
    .then(r => r.json())
    .then(() => {
      btConnected = false;
      btSetStatus('disconnected', 'Disconnected', 'Click Connect to reconnect');
      btLog('sys', 'Disconnected from device');
      document.getElementById('btConnectBtn').disabled    = false;
      document.getElementById('btDisconnectBtn').disabled = true;
      document.getElementById('loadSessionBtn').disabled  = true;
      resetSessionUI();
    });
}
window.btDisconnect = btDisconnect;

function btSetStatus(state, text, sub) {
  const dot  = document.getElementById('btDot');
  const txt  = document.getElementById('btStatusText');
  const sub_ = document.getElementById('btStatusSub');
  if (dot)  dot.className       = 'bt-dot ' + state;
  if (txt)  txt.textContent     = text;
  if (sub_) sub_.textContent    = sub || '';
}

// ═══════════════════════════════════════════════════════════════
//  BLUETOOTH — SEND COMMAND
// ═══════════════════════════════════════════════════════════════
function btSendCmd(obj) {
  const json = JSON.stringify(obj);
  btLog('tx', json.substring(0, 100) + (json.length > 100 ? '…' : ''));
  fetch('/api/helmet/send', {
    method:  'POST',
    headers: { 'Content-Type':'application/json' },
    body:    JSON.stringify({ payload: json }),
  })
  .then(r => r.json())
  .then(d => {
    if (d.response) btLog('rx', d.response.substring(0, 120));
  })
  .catch(e => btLog('err', e.message));
}

function handleBtResponse(data) {
  btLog('rx', JSON.stringify(data).substring(0, 160));
  if (data.status === 'COMPLETED') {
    sessionRunning = false;
    stopTimer();
    updateSessionState('FINISHED');
    btLog('sys', 'Session completed ✓');
  }
}

// ═══════════════════════════════════════════════════════════════
//  HELMET — LOAD SESSION
// ═══════════════════════════════════════════════════════════════
function loadTherapySession() {
  if (!btConnected) { btLog('err', 'Not connected to device'); return; }
  if (!selectedProto) { autoSelectProtocol(); }

  const patId   = document.getElementById('f_patientId')?.value   || 'PAT-00001';
  const profile = document.getElementById('f_therapyProfile')?.value || 'moderate';
  const durMin  = +(document.getElementById('f_duration')?.value || 30);

  currentSessionId = 'SESSION-' + Date.now();
  sessionTotalMs   = durMin * 60 * 1000;

  // Step 1: Load session
  btSendCmd({
    cmd: 'CMD_LOAD_SESSION',
    session_id: currentSessionId,
    patient_id: patId,
    therapy_profile: profile,
    total_duration_min: durMin,
  });

  // Step 2: Configure all 5 actuators
  const cfg = ZONE_CONFIGS[selectedProto.id] || getDefaultConfig(selectedProto);

  const zoneMap = {
    frontal:        'frontal_lobe',
    parietal:       'parietal_lobe',
    occipital:      'occipital_lobe',
    right_temporal: 'right_temporal',
    left_temporal:  'left_temporal',
  };

  cfg.zones.forEach(z => {
    setTimeout(() => {
      btSendCmd({
        cmd:              'CMD_ACTUATOR_SET',
        zone:             zoneMap[z.id] || z.id,
        enabled:          true,
        technique:        z.tech,
        intensity_level:  z.intensity,
        pressure_kpa:     z.intensity * 3,
        frequency_hz:     z.tech.includes('TAPPING') ? 1.2 : (z.tech.includes('VIBRATION') ? 2.0 : 0),
        on_seconds:       Math.round(z.on / 1000),
        off_seconds:      Math.round(z.off / 1000),
        duration_seconds: z.dur,
        ramp_up_seconds:  5,
        ramp_down_seconds:5,
      });
    }, 400);
  });

  // Step 3: Configure 2 temperature pads
  cfg.pads.forEach((p, i) => {
    setTimeout(() => {
      btSendCmd({
        cmd:                   'CMD_TEMP_PAD_SET',
        pad:                   p.id,
        enabled:               true,
        mode:                  p.mode,
        target_temp_celsius:   p.mode === 'COLD' ? 15.0 : (p.mode === 'HEAT' ? 38.5 : null),
        duration_seconds:      p.dur,
        ramp_rate_celsius_per_min: 2.0,
        heat_duration_seconds: p.heatDur  || 0,
        cold_duration_seconds: p.coldDur  || 0,
        repeat_count:          p.repeats  || 0,
      });
    }, 500);
  });

  sessionLoaded = true;
  updateSessionState('LOADED');
  updateZoneUI(cfg);
  updatePadUI(cfg);
  document.getElementById('startBtn').disabled = false;
  btLog('sys', `Session ${currentSessionId} loaded — ${selectedProto.id}: ${selectedProto.label}`);
}
window.loadTherapySession = loadTherapySession;

// ═══════════════════════════════════════════════════════════════
//  HELMET — SESSION CONTROLS
// ═══════════════════════════════════════════════════════════════
function startTherapy() {
  if (!sessionLoaded) return;
  btSendCmd({ cmd:'CMD_START_THERAPY', session_id: currentSessionId });
  sessionRunning = true;
  sessionPaused  = false;
  sessionStartTs = Date.now();
  startTimer();
  updateSessionState('RUNNING');
  document.getElementById('startBtn').disabled  = true;
  document.getElementById('pauseBtn').disabled  = false;
  document.getElementById('stopBtn').disabled   = false;
  document.getElementById('estopBtn').disabled  = false;
  btLog('sys', 'Therapy started ▶');
}
window.startTherapy = startTherapy;

function pauseTherapy() {
  if (!sessionRunning) return;
  btSendCmd({ cmd:'CMD_PAUSE_THERAPY', session_id: currentSessionId });
  sessionPaused = true;
  stopTimer();
  updateSessionState('PAUSED');
  document.getElementById('pauseBtn').disabled  = true;
  document.getElementById('resumeBtn').disabled = false;
  btLog('sys', 'Session paused ⏸');
}
window.pauseTherapy = pauseTherapy;

function resumeTherapy() {
  if (!sessionPaused) return;
  btSendCmd({ cmd:'CMD_RESUME_THERAPY', session_id: currentSessionId });
  sessionPaused = false;
  startTimer();
  updateSessionState('RUNNING');
  document.getElementById('pauseBtn').disabled  = false;
  document.getElementById('resumeBtn').disabled = true;
  btLog('sys', 'Session resumed ⏵');
}
window.resumeTherapy = resumeTherapy;

function stopTherapy() {
  btSendCmd({ cmd:'CMD_STOP_THERAPY', session_id: currentSessionId, reason:'USER_STOP' });
  sessionRunning = false;
  sessionPaused  = false;
  sessionLoaded  = false;
  stopTimer();
  updateSessionState('FINISHED');
  resetSessionControls();
  btLog('sys', 'Session stopped ⏹');
}
window.stopTherapy = stopTherapy;

function emergencyStop() {
  btSendCmd({ cmd:'CMD_EMERGENCY_STOP', code:'ESTOP::0xFF' });
  sessionRunning = false;
  stopTimer();
  updateSessionState('IDLE');
  resetSessionControls();
  allZonesOff();
  btLog('err', '⛔ EMERGENCY STOP sent — all zones off');
}
window.emergencyStop = emergencyStop;

function requestStatus() {
  btSendCmd({ cmd:'CMD_STATUS_REQUEST' });
}
window.requestStatus = requestStatus;

// ═══════════════════════════════════════════════════════════════
//  SESSION TIMER
// ═══════════════════════════════════════════════════════════════
let timerOffset = 0;  // accumulated ms when paused

function startTimer() {
  if (sessionStartTs === 0) sessionStartTs = Date.now();
  stopTimer();
  timerInterval = setInterval(tickTimer, 500);
}
function stopTimer() {
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
  if (sessionRunning && !sessionPaused) {
    timerOffset = Date.now() - sessionStartTs;
  }
}
function tickTimer() {
  const elapsedMs = timerOffset + (Date.now() - sessionStartTs);
  const elSec     = Math.floor(elapsedMs / 1000);
  const totSec    = Math.floor(sessionTotalMs / 1000);
  const pct       = Math.min(100, (elapsedMs / sessionTotalMs) * 100);

  setText('timerDisplay', fmtTime(elSec));
  setText('timerTotal',   fmtTime(totSec));

  const prog = document.getElementById('sessionProgress');
  if (prog) prog.style.width = pct.toFixed(1) + '%';

  if (elapsedMs >= sessionTotalMs) {
    stopTimer();
    sessionRunning = false;
    updateSessionState('FINISHED');
    resetSessionControls();
    btLog('sys', 'Session duration complete ✓');
  }
}
function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

// ═══════════════════════════════════════════════════════════════
//  SESSION UI HELPERS
// ═══════════════════════════════════════════════════════════════
function updateSessionState(state) {
  const badge = document.getElementById('sessionStateBadge');
  if (badge) { badge.textContent = state; badge.className = 'session-state-badge ' + state; }
}

function updateZoneUI(cfg) {
  const zoneMap = {
    frontal:        'frontal',
    parietal:       'parietal',
    occipital:      'occipital',
    right_temporal: 'right',
    left_temporal:  'left',
  };
  cfg.zones.forEach(z => {
    const key    = zoneMap[z.id] || z.id;
    const card   = document.getElementById('zone_' + (z.id === 'right_temporal' ? 'right_temporal' : (z.id === 'left_temporal' ? 'left_temporal' : z.id)));
    const techEl = document.getElementById('zt_' + key);
    const pipsEl = document.getElementById('zp_' + key);

    if (card) card.className = 'zone-card ' + (z.intensity > 0 ? 'active' : 'inactive');
    if (techEl) techEl.textContent = z.tech.replace(/_/g,' ');
    if (pipsEl) {
      const pips = pipsEl.querySelectorAll('.zone-pip');
      pips.forEach((p, i) => p.classList.toggle('on', i < z.intensity));
    }
  });
}

function updatePadUI(cfg) {
  const padModeIcon  = { HEAT:'🔥', COLD:'❄️', ALTERNATING:'🔄', OFF:'⚫' };
  cfg.pads.forEach(p => {
    const key   = p.id === 'frontal_pad' ? 'frontal' : 'occipital';
    const card  = document.getElementById('pad_' + key);
    const icon  = document.getElementById('padicon_' + key);
    const label = document.getElementById('padlabel_' + key);
    if (card) {
      card.className = 'pad-card ' +
        (p.mode === 'HEAT' ? 'heat' : p.mode === 'COLD' ? 'cold' : p.mode === 'ALTERNATING' ? 'heat' : 'off');
    }
    if (icon)  icon.textContent  = padModeIcon[p.mode] || '⚫';
    if (label) label.textContent = p.mode;
  });
}

function allZonesOff() {
  ['frontal','parietal','occipital','right_temporal','left_temporal'].forEach(z => {
    const card = document.getElementById('zone_' + z);
    if (card) card.className = 'zone-card inactive';
  });
  ['frontal','occipital'].forEach(k => {
    const card  = document.getElementById('pad_' + k);
    const icon  = document.getElementById('padicon_' + k);
    const label = document.getElementById('padlabel_' + k);
    if (card)  card.className    = 'pad-card off';
    if (icon)  icon.textContent  = '⚫';
    if (label) label.textContent = 'OFF';
  });
}

function resetSessionControls() {
  ['startBtn','pauseBtn','resumeBtn','stopBtn','estopBtn'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = true;
  });
  const lb = document.getElementById('loadSessionBtn');
  if (lb && btConnected) lb.disabled = false;
}

function resetSessionUI() {
  resetSessionControls();
  updateSessionState('IDLE');
  stopTimer();
  setText('timerDisplay', '00:00');
  const prog = document.getElementById('sessionProgress');
  if (prog) prog.style.width = '0%';
  allZonesOff();
  sessionLoaded  = false;
  sessionRunning = false;
  sessionPaused  = false;
  timerOffset    = 0;
  sessionStartTs = 0;
}

// ── BT log helper ─────────────────────────────────────────────
function btLog(type, msg) {
  const log = document.getElementById('btLog');
  if (!log) return;
  const ts   = new Date().toLocaleTimeString();
  const entry = document.createElement('div');
  entry.className = 'bt-log-entry';
  entry.innerHTML = `<span class="ts">[${ts}]</span> <span class="${type}">${escHtml(msg)}</span>`;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
  // keep max 120 entries
  while (log.children.length > 120) log.removeChild(log.firstChild);
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ═══════════════════════════════════════════════════════════════
//  ─────────────────────────────────────────────────────────────
//  HELMET THERAPY — 3-STEP WIZARD
//  ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

let _wizardStep       = 1;          // current step (1/2/3)
let _selectedPort     = '';         // port chosen in step 1
let _btReady          = false;      // true once handshake OK

// ── Navigate to a step ──────────────────────────────────────
function wizardGoStep(n) {
  // guard: can't skip forward past what's allowed
  if (n === 2 && !_btReady) return;
  if (n === 3 && !sessionLoaded) return;

  _wizardStep = n;

  // panels
  [1, 2, 3].forEach(i => {
    const p = document.getElementById('wpanel' + i);
    if (p) p.classList.toggle('active', i === n);
  });

  // step tracker circles
  [1, 2, 3].forEach(i => {
    const s = document.getElementById('wstep' + i);
    if (!s) return;
    s.classList.remove('active', 'done');
    if (i < n)  s.classList.add('done');
    if (i === n) s.classList.add('active');
  });

  // connector lines between steps
  const lines = document.querySelectorAll('.wstep-line');
  lines.forEach((l, idx) => l.classList.toggle('done', idx + 1 < n));
}
window.wizardGoStep = wizardGoStep;

// ── Check if laptop Bluetooth is on via Web BT API ──────────
function checkBluetoothState() {
  const banner   = document.getElementById('btOffBanner');
  const okBanner = document.getElementById('btOkBanner');
  if (!banner) return;
  // navigator.bluetooth is only available in Chrome/Edge over HTTPS
  // On other browsers we skip the check and assume BT is on
  if (!navigator.bluetooth) {
    banner.classList.add('hidden');
    return;
  }
  navigator.bluetooth.getAvailability()
    .then(available => {
      banner.classList.toggle('hidden', available);
      if (!available) {
        if (okBanner) okBanner.classList.add('hidden');
      }
    })
    .catch(() => banner.classList.add('hidden'));

  // Also listen for BT state changes
  if (navigator.bluetooth.onavailabilitychanged !== undefined) {
    navigator.bluetooth.addEventListener('availabilitychanged', e => {
      banner.classList.toggle('hidden', e.value);
    });
  }
}

// ── Scan ports and populate wizard device list ───────────────
function wizardScanPorts() {
  const list  = document.getElementById('wdeviceList');
  const empty = document.getElementById('wdeviceEmpty');
  const btn   = document.getElementById('btConnectBtn');
  if (list) list.innerHTML = '<div class="wdevice-empty">🔍 Scanning…</div>';
  btLog('sys', 'Scanning serial ports…');

  fetch('/api/helmet/ports')
    .then(r => r.json())
    .then(d => {
      const ports = d.ports || [];
      if (!list) return;
      if (!ports.length) {
        list.innerHTML = '<div class="wdevice-empty">No COM ports found. Is HC-05 plugged in via USB-serial adapter?</div>';
        if (btn) btn.disabled = true;
        btLog('err', 'No ports found');
        return;
      }
      list.innerHTML = '';
      ports.forEach(p => {
        const item = document.createElement('div');
        item.className = 'wdevice-item';
        item.dataset.port = p;
        item.innerHTML = `
          <span class="wdevice-icon">📡</span>
          <div>
            <div class="wdevice-name">${p}</div>
            <div class="wdevice-sub">HC-05 Bluetooth Serial</div>
          </div>
          <span class="wdevice-check">✅</span>`;
        item.addEventListener('click', () => selectWizardDevice(p));
        list.appendChild(item);
      });
      btLog('sys', `Found ${ports.length} port(s): ${ports.join(', ')}`);
    })
    .catch(e => {
      if (list) list.innerHTML = '<div class="wdevice-empty">Scan failed — is the server running?</div>';
      btLog('err', 'Scan error: ' + e.message);
    });
}
window.wizardScanPorts = wizardScanPorts;

// ── Select a device from the list ───────────────────────────
function selectWizardDevice(port) {
  _selectedPort = port;
  document.querySelectorAll('.wdevice-item').forEach(el => {
    el.classList.toggle('selected', el.dataset.port === port);
  });
  const btn = document.getElementById('btConnectBtn');
  if (btn) btn.disabled = false;
  btLog('sys', 'Selected: ' + port);
}

// ── Connect to selected device ───────────────────────────────
function wizardConnect() {
  if (!_selectedPort) { btLog('err', 'Select a device first'); return; }
  const baud   = document.getElementById('btBaud')?.value || '9600';
  const btn    = document.getElementById('btConnectBtn');
  const discBtn= document.getElementById('btDisconnectBtn');
  const w1next = document.getElementById('w1nextBtn');

  if (btn) { btn.disabled = true; btn.textContent = '⏳ Connecting…'; }
  btSetWizardStatus('connecting', 'Connecting…', `${_selectedPort} @ ${baud} baud`);
  btLog('tx', `Connecting to ${_selectedPort} @ ${baud}…`);

  fetch('/api/helmet/connect', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ port: _selectedPort, baud: +baud }),
  })
  .then(r => r.json())
  .then(d => {
    if (d.ok) {
      btConnected = true;
      _btReady    = false;   // wait for handshake
      btSetWizardStatus('connecting', 'Handshaking…', 'Waiting for device…');
      btLog('rx', 'Serial open ✓  Sending CMD_HANDSHAKE…');

      // Send handshake
      fetch('/api/helmet/send', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          payload: JSON.stringify({
            cmd: 'CMD_HANDSHAKE',
            host_id: 'MIGRAINE_APP_V1',
            timestamp: new Date().toISOString(),
          })
        }),
      })
      .then(r => r.json())
      .then(hr => {
        _btReady = true;
        btConnected = true;

        // Show ok banner
        const offB = document.getElementById('btOffBanner');
        const okB  = document.getElementById('btOkBanner');
        const okSub= document.getElementById('btOkSub');
        if (offB) offB.classList.add('hidden');
        if (okB)  okB.classList.remove('hidden');
        if (okSub) okSub.textContent = `${_selectedPort} connected · HC-05 ready`;

        btSetWizardStatus('connected', 'Connected', `${_selectedPort} — HC-05 ready`);
        btLog('rx', hr.response || 'ACK received');

        // Update connect/disconnect buttons
        if (btn)    { btn.style.display    = 'none'; }
        if (discBtn){ discBtn.style.display = 'inline-flex'; }

        // Highlight all device items as connected
        document.querySelectorAll('.wdevice-item.selected').forEach(el => {
          el.style.borderColor = 'var(--low)';
          el.style.boxShadow   = '0 0 12px var(--low-glow)';
        });

        // Enable next button
        if (w1next) w1next.disabled = false;
        btLog('sys', '✓ Device connected — click Continue to set up therapy');
      })
      .catch(e => {
        btLog('err', 'Handshake failed: ' + e.message);
        btSetWizardStatus('disconnected', 'Handshake Failed', e.message);
        if (btn) { btn.disabled = false; btn.textContent = '⚡ Connect'; }
      });
    } else {
      btSetWizardStatus('disconnected', 'Failed', d.error || 'Unknown error');
      btLog('err', 'Connect failed: ' + (d.error || 'Unknown'));
      if (btn) { btn.disabled = false; btn.textContent = '⚡ Connect'; }
    }
  })
  .catch(e => {
    btSetWizardStatus('disconnected', 'Error', e.message);
    btLog('err', 'Error: ' + e.message);
    if (btn) { btn.disabled = false; btn.textContent = '⚡ Connect'; }
  });
}
window.wizardConnect = wizardConnect;

// ── Disconnect ───────────────────────────────────────────────
function wizardDisconnect() {
  fetch('/api/helmet/disconnect', { method: 'POST' })
    .then(() => {
      btConnected = false;
      _btReady    = false;
      btSetWizardStatus('disconnected', 'Disconnected', 'Click Connect to reconnect');
      btLog('sys', 'Disconnected');

      const offB = document.getElementById('btOffBanner');
      const okB  = document.getElementById('btOkBanner');
      if (offB) offB.classList.remove('hidden');
      if (okB)  okB.classList.add('hidden');

      const btn    = document.getElementById('btConnectBtn');
      const discBtn= document.getElementById('btDisconnectBtn');
      const w1next = document.getElementById('w1nextBtn');
      if (btn)    { btn.style.display    = 'inline-flex'; btn.disabled = false; btn.textContent = '⚡ Connect'; }
      if (discBtn){ discBtn.style.display = 'none'; }
      if (w1next) w1next.disabled = true;

      // Go back to step 1
      wizardGoStep(1);
      resetSessionUI();
    });
}
window.wizardDisconnect = wizardDisconnect;

// ── BT status helper (wizard version — no old bt-dot element) ─
function btSetWizardStatus(state, text, sub) {
  // Update old bt-dot if present
  const dot  = document.getElementById('btDot');
  const txt  = document.getElementById('btStatusText');
  const sub_ = document.getElementById('btStatusSub');
  if (dot)  dot.className       = 'bt-dot ' + state;
  if (txt)  txt.textContent     = text;
  if (sub_) sub_.textContent    = sub || '';

  // Update navbar connection dot
  const navDot   = document.getElementById('connectionDot');
  const navLabel = document.getElementById('connectionLabel');
  // only update if BT state, not socket state
  // keep socket status separate — only override label when BT connects
  if (state === 'connected') {
    btLog('sys', `Status: ${text}`);
  }
}

// ── Confirm load — step 2 → step 3 ──────────────────────────
function wizardConfirmLoad() {
  if (!btConnected) { btLog('err', 'Not connected'); return; }
  if (!selectedProto) autoSelectProtocol();

  const btn = document.getElementById('w2nextBtn');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Loading…'; }

  loadTherapySession();   // existing function — sends all CMD packets

  // Populate step-3 summary
  const durMin = +(document.getElementById('f_duration')?.value || 30);
  setText('sum_proto',     selectedProto ? selectedProto.id    : '—');
  setText('sum_intensity', document.getElementById('f_therapyProfile')?.value || '—');
  setText('sum_duration',  durMin + ' min');
  setText('sum_pain',      document.getElementById('f_painScore')?.value + ' / 10' || '—');

  // small delay to let BT packets send, then advance
  setTimeout(() => {
    if (btn) { btn.disabled = false; btn.textContent = 'Load to Device & Continue →'; }
    wizardGoStep(3);

    // Enable start button
    const startB = document.getElementById('startBtn');
    if (startB) startB.disabled = false;

    // Enable estop
    const estopB = document.getElementById('estopBtn');
    if (estopB) estopB.disabled = false;
  }, 1800);
}
window.wizardConfirmLoad = wizardConfirmLoad;

// ── Duration pill helper ─────────────────────────────────────
function setDuration(min, pillEl) {
  const input = document.getElementById('f_duration');
  if (input) input.value = min;
  document.querySelectorAll('.wpill').forEach(p => p.classList.remove('active'));
  if (pillEl) pillEl.classList.add('active');
  autoSelectProtocol();
}
window.setDuration = setDuration;

// Override startTherapy to also update wizard UI
const _origStart = startTherapy;
window.startTherapy = function() {
  _origStart();
  const startB  = document.getElementById('startBtn');
  const inSess  = document.getElementById('inSessionControls');
  if (startB) startB.disabled = true;
  if (inSess) inSess.classList.remove('hidden');
  updateSessionState('RUNNING');
};

// Override stopTherapy to also update wizard UI
const _origStop = stopTherapy;
window.stopTherapy = function() {
  _origStop();
  const startB  = document.getElementById('startBtn');
  const inSess  = document.getElementById('inSessionControls');
  if (startB) { startB.disabled = false; }
  if (inSess) inSess.classList.add('hidden');
  updateSessionState('FINISHED');
};

// Override pauseTherapy to swap pause/resume buttons
const _origPause = pauseTherapy;
window.pauseTherapy = function() {
  _origPause();
  const pauseB  = document.getElementById('pauseBtn');
  const resumeB = document.getElementById('resumeBtn');
  if (pauseB)  pauseB.style.display  = 'none';
  if (resumeB) resumeB.style.display = 'inline-flex';
};

// Override resumeTherapy to swap back
const _origResume = resumeTherapy;
window.resumeTherapy = function() {
  _origResume();
  const pauseB  = document.getElementById('pauseBtn');
  const resumeB = document.getElementById('resumeBtn');
  if (pauseB)  pauseB.style.display  = 'inline-flex';
  if (resumeB) resumeB.style.display = 'none';
};

// ── BT response handler — update wizard when session completes
const _origBtResponse = handleBtResponse;
function handleBtResponse(data) {
  _origBtResponse(data);
  if (data.status === 'COMPLETED' || data.status === 'FINISHED') {
    const startB = document.getElementById('startBtn');
    const inSess = document.getElementById('inSessionControls');
    if (startB) startB.disabled = false;
    if (inSess) inSess.classList.add('hidden');
    updateSessionState('FINISHED');
    btLog('sys', '🎉 Session completed!');
  }
}

// ── Init wizard on page load ─────────────────────────────────
function initWizard() {
  wizardGoStep(1);        // always start at step 1
  checkBluetoothState();  // show BT-off warning if needed

  // Hide disconnect button initially
  const discBtn = document.getElementById('btDisconnectBtn');
  if (discBtn) discBtn.style.display = 'none';

  // Disable next buttons by default
  const w1next = document.getElementById('w1nextBtn');
  if (w1next) w1next.disabled = true;
}

// ═══════════════════════════════════════════════════════════════
//  INITIAL LOAD
// ═══════════════════════════════════════════════════════════════
(async () => {
  // Dashboard init
  drawGauge(0);
  autoSelectProtocol();

  try {
    const res  = await fetch('/api/current');
    const data = await res.json();
    if (data && data.timestamp) updateDashboard(data);
  } catch(e) {}

  loadAlertHistory();

  // Helmet wizard init — runs after DOM is guaranteed ready
  initWizard();
})();