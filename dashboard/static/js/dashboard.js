/* ─────────────────────────────────────────────────────────────
   MigraineSense – Dashboard JavaScript
   Real-time WebSocket updates + Chart.js gauge & timeline
───────────────────────────────────────────────────────────── */

'use strict';

// ── SocketIO connection ───────────────────────────────────────
const socket = io();

socket.on('connect',    () => setConnection(true));
socket.on('disconnect', () => setConnection(false));
socket.on('risk_update', data => updateDashboard(data));

// ── Clock ─────────────────────────────────────────────────────
function updateClock() {
  document.getElementById('clockDisplay').textContent =
    new Date().toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

// ── Connection indicator ───��──────────────────────────────────
function setConnection(ok) {
  const dot   = document.getElementById('connectionDot');
  const label = document.getElementById('connectionLabel');
  dot.className   = 'status-dot ' + (ok ? 'connected' : 'disconnected');
  label.textContent = ok ? 'Live' : 'Reconnecting…';
}

// ── Timeline Chart ────────────────────────────────────────────
const MAX_POINTS = 60;
const timelineLabels = [];
const timelineData   = [];

const timelineChart = new Chart(
  document.getElementById('timelineChart').getContext('2d'),
  {
    type: 'line',
    data: {
      labels:   timelineLabels,
      datasets: [{
        label:           'Blended Risk',
        data:            timelineData,
        borderColor:     '#6366f1',
        backgroundColor: 'rgba(99,102,241,0.12)',
        borderWidth:     2.5,
        pointRadius:     0,
        fill:            true,
        tension:         0.4,
      }, {
        label:           'Composite Risk',
        data:            [],
        borderColor:     '#22d3ee',
        backgroundColor: 'transparent',
        borderWidth:     1.5,
        borderDash:      [4, 4],
        pointRadius:     0,
        fill:            false,
        tension:         0.4,
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#8892a4', font: { size: 11 } } },
        tooltip: {
          backgroundColor: '#1a1d27',
          borderColor:     '#2e3248',
          borderWidth:     1,
          titleColor:      '#e2e8f0',
          bodyColor:       '#8892a4',
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#8892a4', maxTicksLimit: 8, font: { size: 10 } },
          grid:  { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          min:   0,
          max:   100,
          ticks: { color: '#8892a4', stepSize: 25, font: { size: 10 } },
          grid:  { color: 'rgba(255,255,255,0.06)' },
        },
      },
      animation: { duration: 400 },
    },
  }
);

// ── Gauge (arc drawn on canvas) ───────────────────────────────
const gaugeCtx    = document.getElementById('gaugeCanvas').getContext('2d');
let   gaugeCurrent = 0;
let   gaugeTarget  = 0;
let   gaugeAnim    = null;

function drawGauge(score) {
  const W = 220, H = 220, cx = W / 2, cy = H / 2, R = 90;
  gaugeCtx.clearRect(0, 0, W, H);

  const startAngle = Math.PI * 0.75;
  const endAngle   = Math.PI * 2.25;
  const fillAngle  = startAngle + (score / 100) * (endAngle - startAngle);

  // Track
  gaugeCtx.beginPath();
  gaugeCtx.arc(cx, cy, R, startAngle, endAngle);
  gaugeCtx.strokeStyle = '#22263a';
  gaugeCtx.lineWidth   = 18;
  gaugeCtx.lineCap     = 'round';
  gaugeCtx.stroke();

  // Fill
  const color = scoreColor(score);
  const grad  = gaugeCtx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0,   colorFor(score, 0.8));
  grad.addColorStop(1,   color);
  gaugeCtx.beginPath();
  gaugeCtx.arc(cx, cy, R, startAngle, fillAngle);
  gaugeCtx.strokeStyle = grad;
  gaugeCtx.lineWidth   = 18;
  gaugeCtx.lineCap     = 'round';
  gaugeCtx.stroke();

  // Zone marks
  [[30, '#22c55e'], [55, '#eab308'], [75, '#f97316'], [100, '#ef4444']].forEach(([v, c]) => {
    const a = startAngle + (v / 100) * (endAngle - startAngle);
    gaugeCtx.beginPath();
    gaugeCtx.arc(cx, cy, R, a - 0.01, a + 0.01);
    gaugeCtx.strokeStyle = c;
    gaugeCtx.lineWidth   = 22;
    gaugeCtx.stroke();
  });
}

function animateGauge(target) {
  if (gaugeAnim) cancelAnimationFrame(gaugeAnim);
  const start   = gaugeCurrent;
  const delta   = target - start;
  const dur     = 600;
  const t0      = performance.now();
  function step(now) {
    const p = Math.min((now - t0) / dur, 1);
    const ease = p < 0.5 ? 2 * p * p : -1 + (4 - 2 * p) * p;
    gaugeCurrent = start + delta * ease;
    drawGauge(gaugeCurrent);
    document.getElementById('gaugeScore').textContent = Math.round(gaugeCurrent);
    if (p < 1) gaugeAnim = requestAnimationFrame(step);
    else gaugeCurrent = target;
  }
  gaugeAnim = requestAnimationFrame(step);
}

function scoreColor(s) {
  if (s >= 75) return '#ef4444';
  if (s >= 55) return '#f97316';
  if (s >= 30) return '#eab308';
  return '#22c55e';
}
function colorFor(s, alpha = 1) {
  const c = scoreColor(s);
  return c;
}

// ── Main update handler ───────────────────────────────────────
function updateDashboard(d) {
  const score   = d.blended_risk    || 0;
  const comp    = d.composite_risk  || 0;
  const level   = d.risk_level      || 'LOW';
  const eta     = d.eta_minutes;
  const trend   = d.trend           || 0;
  const f       = d.features        || {};

  // Gauge
  animateGauge(score);

  // Gauge score color
  document.getElementById('gaugeScore').style.color = scoreColor(score);

  // Risk badge
  const badge = document.getElementById('riskBadge');
  badge.textContent = level;
  badge.className   = 'risk-badge ' + level;

  // ETA
  document.getElementById('etaValue').textContent = etaString(eta, level);

  // Trend
  const trendEl   = document.getElementById('trendValue');
  const trendIcon = document.getElementById('trendIcon');
  if (trend > 0.5) {
    trendEl.textContent   = `↑ Worsening (+${trend.toFixed(1)}/min)`;
    trendEl.style.color   = '#f97316';
    trendIcon.textContent = '⬆️';
  } else if (trend < -0.5) {
    trendEl.textContent   = `↓ Improving (${trend.toFixed(1)}/min)`;
    trendEl.style.color   = '#22c55e';
    trendIcon.textContent = '⬇️';
  } else {
    trendEl.textContent   = '→ Stable';
    trendEl.style.color   = '#8892a4';
    trendIcon.textContent = '→';
  }

  // ML badge
  const mlBadge = document.getElementById('mlBadge');
  if (d.model_trained) { mlBadge.textContent = 'Active'; mlBadge.className = 'ml-badge ready'; }
  else                  { mlBadge.textContent = 'Learning…'; mlBadge.className = 'ml-badge'; }
  document.getElementById('mlScore').textContent = (d.ml_anomaly_pct || 0).toFixed(1);

  // Domain bars
  setBar('kb',  d.keyboard_risk || 0);
  setBar('ms',  d.mouse_risk    || 0);
  setBar('wc',  d.webcam_risk   || 0);
  setBar('sys', d.system_risk   || 0);

  // Domain hints
  document.getElementById('kbHints').textContent  = kbHint(f);
  document.getElementById('msHints').textContent  = msHint(f);
  document.getElementById('wcHints').textContent  = wcHint(f);
  document.getElementById('sysHints').textContent = sysHint(f);

  // Timeline
  const now = new Date().toLocaleTimeString();
  if (timelineLabels.length >= MAX_POINTS) {
    timelineLabels.shift();
    timelineData.shift();
    timelineChart.data.datasets[1].data.shift();
  }
  timelineLabels.push(now);
  timelineData.push(score);
  timelineChart.data.datasets[1].data.push(comp);
  timelineChart.update('none');

  // Live metrics
  setMetric('m_kpm',   fv(f.typing_speed_kpm,     1), ' kpm');
  setMetric('m_err',   fv(f.error_rate * 100,      1), ' %');
  setMetric('m_pause', fv(f.pause_count,            0), '');
  setMetric('m_cv',    fv(f.rhythm_cv,              3), '');
  setMetric('m_mspd',  fv(f.mouse_avg_speed_px,     1), ' px/s');
  setMetric('m_jit',   fv(f.mouse_jitter,           3), '');
  setMetric('m_clk',   fv(f.click_rate_per_min,     1), ' /min');
  setMetric('m_midle', fv(f.mouse_idle_sec,         0), ' s');
  setMetric('m_bpm',   fv(f.blink_rate_bpm,         1), ' bpm');
  setMetric('m_ear',   fv(f.mean_ear,               3), '');
  setMetric('m_tilt',  fv(f.head_tilt_deg,          1), '°');
  setMetric('m_prox',  fv(f.face_proximity_px,      0), ' px');
  setMetric('m_cpu',   fv(f.avg_cpu_pct,            1), ' %');
  setMetric('m_mem',   fv(f.avg_mem_pct,            1), ' %');
  setMetric('m_bright',fv(f.avg_brightness,         0), ' %');
  setMetric('m_sw',    fv(f.app_switch_rate,        2), ' /min');

  // Alert banner
  if (level !== 'LOW') showAlertBanner(level, score, eta);
  else                 dismissAlert();

  // Footer
  document.getElementById('lastUpdate').textContent =
    'Last update: ' + new Date().toLocaleTimeString();

  // Load alert history async
  loadAlertHistory();
}

// ── Domain bar helper ─────────────────────────────────────────
function setBar(prefix, value) {
  const bar = document.getElementById(prefix + 'Bar');
  const pct = document.getElementById(prefix + 'Pct');
  bar.style.width = value + '%';
  pct.textContent = value.toFixed(0) + '%';
  bar.classList.remove('high', 'critical');
  if (value >= 75) bar.classList.add('critical');
  else if (value >= 55) bar.classList.add('high');
}

// ── Metric helper ─────────────────────────────────────────────
function setMetric(id, val, suffix) {
  const el = document.getElementById(id);
  if (el) el.textContent = val !== '—' ? val + suffix : '—';
}
function fv(v, decimals = 1) {
  if (v === undefined || v === null || isNaN(v)) return '—';
  return Number(v).toFixed(decimals);
}

// ── ETA string ────────────────────────────────────────────────
function etaString(eta, level) {
  if (!eta || level === 'LOW') return 'No risk detected ✅';
  if (eta >= 120) return `~${Math.round(eta / 60)} hours`;
  return `~${eta} minutes`;
}

// ── Hint strings ─────────────────────────────────────────────
function kbHint(f) {
  const hints = [];
  if (f.typing_speed_kpm < 20)  hints.push('slow typing');
  if (f.error_rate > 0.10)       hints.push('high errors');
  if (f.pause_count > 8)         hints.push('many pauses');
  if (f.rhythm_cv > 1.0)         hints.push('irregular rhythm');
  return hints.length ? '⚠ ' + hints.join(' · ') : '✓ Normal';
}
function msHint(f) {
  const hints = [];
  if (f.mouse_jitter > 0.4)        hints.push('hand tremor');
  if (f.mouse_idle_sec > 120)      hints.push('long idle');
  if (f.mouse_efficiency < 0.4)    hints.push('erratic path');
  return hints.length ? '⚠ ' + hints.join(' · ') : '✓ Normal';
}
function wcHint(f) {
  const hints = [];
  if (f.blink_rate_bpm < 8)        hints.push('rare blinking');
  if (f.mean_ear < 0.20)           hints.push('squinting');
  if (Math.abs(f.head_tilt_deg) > 15) hints.push('head tilt');
  return hints.length ? '⚠ ' + hints.join(' · ') : '✓ Normal';
}
function sysHint(f) {
  const hints = [];
  if (f.avg_brightness > 85)       hints.push('bright screen');
  if (f.avg_cpu_pct > 80)          hints.push('high CPU');
  if (f.is_late_night)             hints.push('late night');
  if (f.app_switch_rate > 10)      hints.push('overloaded');
  return hints.length ? '⚠ ' + hints.join(' · ') : '✓ Normal';
}

// ── Alert banner ──────────────────────────────────────────────
function showAlertBanner(level, score, eta) {
  const banner = document.getElementById('alertBanner');
  const icons  = { MODERATE: '🟡', HIGH: '🟠', CRITICAL: '🔴' };
  document.getElementById('alertIcon').textContent = icons[level] || '⚠️';
  document.getElementById('alertText').textContent =
    `${level} RISK (${score.toFixed(0)}/100) — ${etaString(eta, level)} — Please take a break!`;
  banner.className = 'alert-banner ' + level;
}
function dismissAlert() {
  document.getElementById('alertBanner').className = 'alert-banner hidden';
}
window.dismissAlert = dismissAlert;

// ── Alert history ─────────────────────────────────────────────
let _lastAlertCount = 0;
async function loadAlertHistory() {
  try {
    const res   = await fetch('/api/alerts');
    const alerts = await res.json();
    if (alerts.length === _lastAlertCount) return;
    _lastAlertCount = alerts.length;

    const list = document.getElementById('alertsList');
    if (!alerts.length) {
      list.innerHTML = '<div class="no-alerts">No alerts yet — you\'re doing great! 🎉</div>';
      return;
    }
    list.innerHTML = alerts.slice(0, 20).map(a => {
      const t = new Date(a.timestamp * 1000).toLocaleTimeString();
      return `<div class="alert-item ${a.level}">
        <div>
          <div class="alert-item-msg">${a.message}</div>
          <div class="alert-item-time">${t} · Score: ${(a.risk_score||0).toFixed(0)}</div>
        </div>
      </div>`;
    }).join('');
  } catch(e) {}
}

// ── Tips rotator ──────────────────────────────────────────────
let tipIdx = 0;
const tips = document.querySelectorAll('.tip-item');

function showTip(i) {
  tips.forEach(t => t.classList.remove('active'));
  tipIdx = ((i % tips.length) + tips.length) % tips.length;
  tips[tipIdx].classList.add('active');
  document.getElementById('tipCounter').textContent = `${tipIdx + 1} / ${tips.length}`;
}
window.nextTip = () => showTip(tipIdx + 1);
window.prevTip = () => showTip(tipIdx - 1);

setInterval(() => showTip(tipIdx + 1), 8000);

// ── Initial load ──────────────────────────────────────────────
(async () => {
  try {
    const res  = await fetch('/api/current');
    const data = await res.json();
    if (data.timestamp) updateDashboard(data);
  } catch(e) {}
  drawGauge(0);
  loadAlertHistory();
})();
