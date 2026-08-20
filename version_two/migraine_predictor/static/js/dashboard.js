/**
 * Migraine Predictor - Dashboard JavaScript
 * Handles real-time updates, WebSocket communication, and UI interactions
 */

// Global state
let socket = null;
let isMonitoring = false;
let updateInterval = null;

// Desktop notification state
let lastNotifiedRiskScore = 0;   // score at the time of the last desktop alert
let notificationsEnabled   = true; // mirrors the Settings toggle
let notificationPermission = 'default'; // 'granted' | 'denied' | 'default'

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeSocket();
    initializeEventListeners();
    checkInitialStatus();
    initTherapySuggestion();
    requestDesktopNotificationPermission();
    loadNotificationSettings();

    // Monitoring always starts automatically on the server.
    // Begin polling immediately so the dashboard is never blank.
    isMonitoring = true;
    updateMonitoringUI();
    startPolling();

    // Answer option buttons
    document.querySelectorAll('.therapy-option').forEach(btn => {
        btn.addEventListener('click', function() {
            const step  = parseInt(this.dataset.step);
            const value = this.dataset.value;
            handleTherapyAnswer(step, value, this);
        });
    });

    // Back button
    const backBtn = document.getElementById('therapyBackBtn');
    if (backBtn) backBtn.addEventListener('click', therapyGoBack);

    // Restart button
    const restartBtn = document.getElementById('therapyRestartBtn');
    if (restartBtn) restartBtn.addEventListener('click', therapyRestart);

    // "Suggest Me a Therapy" button — refresh risk score before opening
    const suggestBtn = document.getElementById('suggestTherapyBtn');
    if (suggestBtn) {
        suggestBtn.addEventListener('click', () => {
            fetch('/api/predict')
                .then(r => r.json())
                .then(d => { therapyRiskScore = Math.round((d.prediction?.risk_score || 0) * 100); })
                .catch(() => {});
            therapyRestart();
        });
    }
});

// ============================================================
// Desktop Notification System
// ============================================================

/**
 * Show or hide the permission / blocked banners based on current state.
 */
function updateNotifBannerVisibility() {
    const permBanner    = document.getElementById('notifPermissionBanner');
    const blockedBanner = document.getElementById('notifBlockedBanner');
    const dismissed     = sessionStorage.getItem('notifBannerDismissed') === '1';

    if (!('Notification' in window)) return; // browser doesn't support it — hide both

    if (Notification.permission === 'granted') {
        // Already granted — hide both banners
        if (permBanner)    permBanner.classList.add('d-none');
        if (blockedBanner) blockedBanner.classList.add('d-none');
    } else if (Notification.permission === 'denied') {
        // User blocked notifications — show the blocked banner
        if (permBanner)    permBanner.classList.add('d-none');
        if (blockedBanner && !dismissed) blockedBanner.classList.remove('d-none');
    } else {
        // 'default' — not asked yet, show the enable prompt
        if (blockedBanner) blockedBanner.classList.add('d-none');
        if (permBanner && !dismissed)    permBanner.classList.remove('d-none');
    }
}

/**
 * Called by the "Enable Alerts" button in the permission banner.
 */
function askNotificationPermission() {
    if (!('Notification' in window)) return;
    Notification.requestPermission().then(permission => {
        notificationPermission = permission;
        updateNotifBannerVisibility();
        if (permission === 'granted') {
            sendDesktopNotification(
                '\uD83E\uDDE0 Migraine Predictor Active',
                'Desktop alerts are ON. You will be notified when your risk rises sharply.',
                'low'
            );
        }
    });
}

/**
 * Called by the "Dismiss" button in the permission banner.
 * Hides it for the rest of this browser session.
 */
function dismissNotifBanner() {
    sessionStorage.setItem('notifBannerDismissed', '1');
    const b = document.getElementById('notifPermissionBanner');
    if (b) b.classList.add('d-none');
}

/**
 * Request browser permission for desktop notifications on page load.
 */
function requestDesktopNotificationPermission() {
    if (!('Notification' in window)) {
        console.warn('This browser does not support desktop notifications.');
        return;
    }

    notificationPermission = Notification.permission;

    // Show / hide the appropriate banner immediately
    updateNotifBannerVisibility();

    // If already granted, nothing more to do
    if (Notification.permission === 'granted') return;

    // If not yet asked (permission === 'default'), the banner will prompt the user.
    // Do NOT auto-call requestPermission() here — browsers require a user gesture.
}

/**
 * Load the notification enabled/disabled preference from localStorage
 * (written by the Settings page) and watch for cross-tab changes.
 */
function loadNotificationSettings() {
    try {
        const saved = JSON.parse(localStorage.getItem('migraineSettings') || '{}');
        if (typeof saved.enable_notifications === 'boolean') {
            notificationsEnabled = saved.enable_notifications;
        }
    } catch (e) { /* ignore */ }

    // Re-read whenever Settings page saves a change (works across tabs too)
    window.addEventListener('storage', (e) => {
        if (e.key !== 'migraineSettings') return;
        try {
            const updated = JSON.parse(e.newValue || '{}');
            if (typeof updated.enable_notifications === 'boolean') {
                notificationsEnabled = updated.enable_notifications;
                console.log('Desktop notifications', notificationsEnabled ? 'enabled' : 'disabled');
            }
        } catch (_) { /* ignore */ }
    });
}

/**
 * Fire a native desktop (OS-level) notification.
 * @param {string} title   - Notification title
 * @param {string} body    - Notification body text
 * @param {string} level   - 'low' | 'moderate' | 'high' | 'critical'
 */
function sendDesktopNotification(title, body, level = 'high') {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;
    if (!notificationsEnabled) return;

    const icons = {
        low:      '/static/img/icon-low.png',
        moderate: '/static/img/icon-moderate.png',
        high:     '/static/img/icon-high.png',
        critical: '/static/img/icon-critical.png'
    };

    const n = new Notification(title, {
        body,
        icon:    icons[level] || icons['high'],
        badge:   '/static/img/badge.png',
        tag:     'migraine-risk-alert',   // replaces previous notification (no spam)
        requireInteraction: level === 'critical'  // stays visible until dismissed for critical
    });

    // Auto-close non-critical notifications after 8 seconds
    if (level !== 'critical') {
        setTimeout(() => n.close(), 8000);
    }

    // Clicking the notification focuses the dashboard tab
    n.onclick = () => { window.focus(); n.close(); };
}

/**
 * Called on every data_update. Fires a desktop notification when the
 * risk score rises by 30 percentage points or more since the last alert.
 * Also fires immediately when level crosses into 'high' or 'critical'.
 *
 * @param {object} prediction - prediction object from the server
 */
function checkAndFireRiskNotification(prediction) {
    if (!notificationsEnabled) return;
    if (!('Notification' in window) || Notification.permission !== 'granted') return;

    const currentScore = prediction.risk_score || 0;    // 0.0 – 1.0
    const level        = prediction.risk_level  || 'low';
    const riseDelta    = currentScore - lastNotifiedRiskScore; // how much it rose since last alert

    // Trigger conditions:
    //  1. Rose by >= 0.30 (30 percentage points) since last notification
    //  2. OR just crossed into 'high' (>= 0.6) and we haven't alerted for this band yet
    //  3. OR just crossed into 'critical' (>= 0.8)
    const crossedHigh     = currentScore >= 0.6 && lastNotifiedRiskScore < 0.6;
    const crossedCritical = currentScore >= 0.8 && lastNotifiedRiskScore < 0.8;
    const bigJump         = riseDelta >= 0.30;

    if (!bigJump && !crossedHigh && !crossedCritical) return;

    // Build message
    const pct = Math.round(currentScore * 100);
    const prevPct = Math.round(lastNotifiedRiskScore * 100);

    let title, body;

    if (crossedCritical || level === 'critical') {
        title = '🚨 Critical Migraine Risk!';
        body  = `Risk score is now ${pct}%. A migraine is highly likely — stop what you are doing and put on the helmet immediately.`;
    } else if (crossedHigh || level === 'high') {
        title = '⚠️ High Migraine Risk Detected';
        body  = `Risk score jumped to ${pct}%. Take a break and consider starting helmet therapy.`;
    } else {
        title = '📈 Migraine Risk Rising';
        body  = `Risk score rose from ${prevPct}% to ${pct}% (+${Math.round(riseDelta * 100)}%). Keep an eye on your symptoms.`;
    }

    sendDesktopNotification(title, body, level);

    // Update baseline so we measure the NEXT rise from this point
    lastNotifiedRiskScore = currentScore;
}

/**
 * Initialize WebSocket connection
 */
function initializeSocket() {
    // Guard: if the Socket.IO client library failed to load, fall back to
    // pure REST polling (startPolling already handles this via fetchUpdate).
    if (typeof io === 'undefined') {
        console.warn('Socket.IO client not available — using REST polling fallback.');
        return;
    }

    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to server');
        updateConnectionStatus(true);
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
    });
    
    socket.on('status', function(data) {
        isMonitoring = data.monitoring;
        updateMonitoringUI();
        // If server confirms monitoring is active, make sure polling is running
        if (isMonitoring && !updateInterval) {
            startPolling();
        }
    });
    
    socket.on('data_update', function(data) {
        updateDashboard(data);
    });
    
    socket.on('error', function(error) {
        console.error('Socket error:', error);
        showNotification('Connection error', 'danger');
    });
}

/**
 * Initialize event listeners
 */
function initializeEventListeners() {
    // Toggle monitoring button
    const toggleBtn = document.getElementById('toggleMonitoring');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleMonitoring);
    }
    
    // Report migraine form
    const submitReportBtn = document.getElementById('submitReport');
    if (submitReportBtn) {
        submitReportBtn.addEventListener('click', submitMigraineReport);
    }
    
    // Intensity slider
    const intensitySlider = document.getElementById('migraineIntensity');
    if (intensitySlider) {
        intensitySlider.addEventListener('input', function() {
            document.getElementById('intensityValue').textContent = this.value;
        });
    }
}

/**
 * Check initial monitoring status
 */
async function checkInitialStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        isMonitoring = data.monitoring_active;
        updateMonitoringUI();
        updateMonitorStatuses(data.monitors);

        // Always start polling — monitoring is auto-started on the server
        if (!updateInterval) {
            startPolling();
        }
    } catch (error) {
        console.error('Error checking status:', error);
        // Even if status check fails, keep polling
        if (!updateInterval) startPolling();
    }
}

/**
 * Toggle monitoring on/off
 */
async function toggleMonitoring() {
    const endpoint = isMonitoring ? '/api/stop' : '/api/start';
    
    try {
        const response = await fetch(endpoint, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            isMonitoring = !isMonitoring;
            updateMonitoringUI();
            
            if (isMonitoring) {
                startPolling();
                showNotification('Monitoring started', 'success');
            } else {
                stopPolling();
                showNotification('Monitoring stopped', 'info');
            }
        }
    } catch (error) {
        console.error('Error toggling monitoring:', error);
        showNotification('Failed to toggle monitoring', 'danger');
    }
}

/**
 * Update monitoring UI state
 */
function updateMonitoringUI() {
    const toggleBtn  = document.getElementById('toggleMonitoring');
    const statusBadge = document.getElementById('statusBadge');

    if (isMonitoring) {
        if (toggleBtn) {
            toggleBtn.innerHTML = '<i class="bi bi-stop-fill me-1"></i> Stop Monitoring';
            toggleBtn.classList.remove('btn-light');
            toggleBtn.classList.add('btn-danger');
        }
        if (statusBadge) {
            statusBadge.className = 'live-badge';
            statusBadge.innerHTML = '<span class="live-dot"></span> Active';
        }
    } else {
        if (toggleBtn) {
            toggleBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i> Start Monitoring';
            toggleBtn.classList.remove('btn-danger');
            toggleBtn.classList.add('btn-light');
        }
        if (statusBadge) {
            statusBadge.className = 'badge status-inactive px-3 py-2';
            statusBadge.innerHTML = '<i class="bi bi-circle-fill me-1" style="font-size:0.55rem;"></i> Inactive';
        }
    }
}

/**
 * Start polling for updates
 */
function startPolling() {
    if (updateInterval) return;
    
    updateInterval = setInterval(requestUpdate, 3000);
    requestUpdate(); // Initial update
}

/**
 * Stop polling for updates
 */
function stopPolling() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

/**
 * Request data update via WebSocket
 */
function requestUpdate() {
    if (socket && socket.connected) {
        socket.emit('request_update');
    } else {
        // Fallback to REST API
        fetchUpdate();
    }
}

/**
 * Fetch update via REST API
 */
async function fetchUpdate() {
    try {
        const response = await fetch('/api/predict');
        const data = await response.json();
        updateDashboard({
            features: data.features,
            prediction: data.prediction,
            timestamp: data.timestamp
        });
    } catch (error) {
        console.error('Error fetching update:', error);
    }
}

/**
 * Update dashboard with new data
 */
function updateDashboard(data) {
    if (!data) return;
    
    const { features, prediction } = data;
    
    // Update risk score
    updateRiskScore(prediction);
    
    // Update monitor cards
    updateMonitorCards(features);
    
    // Update contributing factors
    updateContributingFactors(prediction.contributing_factors);
    
    // Update recommendations
    if (prediction.risk_score > 0.3) {
        fetchRecommendations();
    } else {
        updateRecommendations([]);
    }
    
    // Show in-page alert banner
    checkRiskAlert(prediction);

    // Fire desktop OS notification if risk jumped by >= 30%
    checkAndFireRiskNotification(prediction);
}

/**
 * Update risk score display
 */
function updateRiskScore(prediction) {
    const riskCard = document.getElementById('riskCard');
    const riskScoreValue = document.getElementById('riskScoreValue');
    const riskLevel = document.getElementById('riskLevel');
    const riskTrend = document.getElementById('riskTrend');
    
    const score = Math.round(prediction.risk_score * 100);
    riskScoreValue.textContent = `${score}%`;
    
    // Update risk level text
    const levelText = {
        'low': 'Low Risk',
        'moderate': 'Moderate Risk',
        'high': 'High Risk',
        'critical': 'Critical Risk'
    };
    riskLevel.textContent = levelText[prediction.risk_level] || 'Unknown';
    
    // Update card color
    riskCard.className = 'card risk-card risk-' + prediction.risk_level;
    
    // Update trend
    const trendIcons = {
        'increasing': '<i class="bi bi-arrow-up text-danger"></i> Increasing',
        'decreasing': '<i class="bi bi-arrow-down text-success"></i> Decreasing',
        'stable': '<i class="bi bi-arrow-right"></i> Stable'
    };
    riskTrend.innerHTML = trendIcons[prediction.trend] || trendIcons['stable'];
    
    // Show migraine type if predicted
    if (prediction.migraine_type) {
        const typeNames = {
            'tension': 'Tension Headache',
            'migraine_with_aura': 'Migraine with Aura',
            'migraine_without_aura': 'Migraine without Aura',
            'cluster': 'Cluster Headache'
        };
        riskLevel.textContent += ` - ${typeNames[prediction.migraine_type] || prediction.migraine_type}`;
    }
}

/**
 * Update monitor cards with feature data
 */
function updateMonitorCards(features) {
    // Keyboard
    if (features.keyboard) {
        const kb = features.keyboard;
        document.getElementById('typingSpeed').textContent = 
            kb.typing_speed ? `${Math.round(kb.typing_speed)} kpm` : '-- kpm';
        document.getElementById('errorRate').textContent = 
            kb.error_rate !== undefined ? `${Math.round(kb.error_rate * 100)}%` : '--%';
        document.getElementById('keyboardFatigue').style.width = 
            `${Math.round((kb.keyboard_fatigue_score || 0) * 100)}%`;
        updateStatusBadge('keyboardStatus', kb.activity_level > 0);
    }
    
    // Mouse
    if (features.mouse) {
        const ms = features.mouse;
        document.getElementById('clickRate').textContent = 
            ms.click_rate !== undefined ? `${Math.round(ms.click_rate)} /min` : '-- /min';
        document.getElementById('idleTime').textContent = 
            ms.current_idle_duration !== undefined ? `${Math.round(ms.current_idle_duration)} sec` : '-- sec';
        document.getElementById('mouseFatigue').style.width = 
            `${Math.round((ms.mouse_fatigue_score || 0) * 100)}%`;
        updateStatusBadge('mouseStatus', ms.movement_activity > 0);
    }
    
    // Webcam
    if (features.webcam) {
        const wc = features.webcam;
        document.getElementById('blinkRate').textContent = 
            wc.blink_rate !== undefined ? `${Math.round(wc.blink_rate)} /min` : '-- /min';
        document.getElementById('eyeStrain').textContent = 
            wc.eye_strain_score !== undefined ? `${Math.round(wc.eye_strain_score * 100)}%` : '--%';
        document.getElementById('webcamFatigue').style.width = 
            `${Math.round((wc.webcam_fatigue_score || 0) * 100)}%`;
        updateStatusBadge('webcamStatus', wc.frames_processed > 0);
    }
    
    // System
    if (features.system) {
        const sys = features.system;
        document.getElementById('sessionTime').textContent = 
            sys.session_duration_minutes !== undefined ? `${Math.round(sys.session_duration_minutes)} min` : '-- min';
        document.getElementById('cpuUsage').textContent = 
            sys.avg_cpu !== undefined ? `${Math.round(sys.avg_cpu)}%` : '--%';
        document.getElementById('systemFatigue').style.width = 
            `${Math.round((sys.system_fatigue_score || 0) * 100)}%`;
        updateStatusBadge('systemStatus', true);
    }
}

/**
 * Update status badge for a monitor
 */
function updateStatusBadge(elementId, isActive) {
    const badge = document.getElementById(elementId);
    if (badge) {
        if (isActive) {
            badge.className = 'badge status-active';
            badge.textContent = 'Active';
        } else {
            badge.className = 'badge status-inactive';
            badge.textContent = 'Inactive';
        }
    }
}

/**
 * Update monitor statuses from status API
 */
function updateMonitorStatuses(monitors) {
    if (!monitors) return;
    
    updateStatusBadge('keyboardStatus', monitors.keyboard);
    updateStatusBadge('mouseStatus', monitors.mouse);
    updateStatusBadge('webcamStatus', monitors.webcam);
    updateStatusBadge('systemStatus', monitors.system);
}

/**
 * Update contributing factors display
 */
function updateContributingFactors(factors) {
    const container = document.getElementById('contributingFactors');
    
    if (!factors || factors.length === 0) {
        container.innerHTML = '<p class="text-muted text-center">No significant factors detected</p>';
        return;
    }
    
    const html = factors.map(factor => `
        <div class="factor-item">
            <div class="factor-icon ${factor.impact}">
                <i class="bi bi-exclamation-triangle"></i>
            </div>
            <div class="factor-content">
                <div class="factor-title">${escapeHtml(factor.factor)}</div>
                <p class="factor-description">${escapeHtml(factor.description)}</p>
            </div>
        </div>
    `).join('');
    
    container.innerHTML = html;
}

/**
 * Fetch therapy recommendations
 */
async function fetchRecommendations() {
    try {
        const response = await fetch('/api/therapy_recommendation');
        const data = await response.json();
        updateRecommendations(data.recommendations || []);
    } catch (error) {
        console.error('Error fetching recommendations:', error);
    }
}

/**
 * Update recommendations display
 */
function updateRecommendations(recommendations) {
    const container = document.getElementById('recommendations');

    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p class="text-muted text-center">Continue working normally</p>';
        return;
    }

    const icons = {
        'rest_recommendation': 'bi-pause-circle',
        'hydration_reminder': 'bi-droplet',
        'cold_therapy': 'bi-snow',
        'warm_therapy': 'bi-thermometer-half',
        'pressure_therapy': 'bi-hand-index',
        'dim_lights': 'bi-brightness-low'
    };

    const durationLabels = {
        'rest_recommendation': '30 min',
        'hydration_reminder':  '5 min',
        'cold_therapy':        '20 min',
        'warm_therapy':        '15 min',
        'pressure_therapy':    '15 min',
        'dim_lights':          '20 min'
    };

    const html = recommendations.slice(0, 4).map(rec => {
        const durMin  = rec.duration_minutes || null;
        const durLabel = durMin ? `${durMin} min` : (durationLabels[rec.therapy] || null);
        const durBadge = durLabel
            ? `<span class="rec-duration-badge"><i class="bi bi-clock"></i> ${durLabel}</span>`
            : '';
        return `
        <div class="recommendation-item priority-${rec.priority || 'medium'}">
            <div class="recommendation-icon">
                <i class="bi ${icons[rec.therapy] || 'bi-lightbulb'}"></i>
            </div>
            <div class="recommendation-content">
                <div class="recommendation-title-row">
                    <span class="recommendation-title">${escapeHtml(rec.therapy.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))}</span>
                    ${durBadge}
                </div>
                <p class="recommendation-description">${escapeHtml(rec.description)}</p>
            </div>
        </div>`;
    }).join('');

    container.innerHTML = html;
}

/**
 * Check and show risk alert if needed
 */
function checkRiskAlert(prediction) {
    const alert = document.getElementById('riskAlert');
    const alertTitle = document.getElementById('alertTitle');
    const alertMessage = document.getElementById('alertMessage');
    
    if (prediction.risk_score >= 0.6) {
        alert.classList.remove('d-none', 'alert-warning');
        alert.classList.add('alert-danger');
        alertTitle.textContent = prediction.risk_level === 'critical' ? 
            '⚠️ Critical Risk Level' : '⚠️ High Risk Detected';
        alertMessage.textContent = 'Consider taking a break immediately. A migraine may be approaching.';
        alert.classList.add('show');
    } else if (prediction.risk_score >= 0.4) {
        alert.classList.remove('d-none', 'alert-danger');
        alert.classList.add('alert-warning');
        alertTitle.textContent = '⚡ Elevated Risk';
        alertMessage.textContent = 'Consider reducing screen time and taking short breaks.';
        alert.classList.add('show');
    } else {
        alert.classList.add('d-none');
        alert.classList.remove('show');
    }
}

/**
 * Submit migraine report
 */
async function submitMigraineReport() {
    const type = document.getElementById('migraineType').value;
    const intensity = document.getElementById('migraineIntensity').value;
    const notes = document.getElementById('migraineNotes').value;
    
    try {
        const response = await fetch('/api/report_migraine', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: type,
                intensity: parseInt(intensity),
                notes: notes
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('Migraine reported successfully. Thank you!', 'success');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('reportModal'));
            modal.hide();
            
            // Reset form
            document.getElementById('reportForm').reset();
            document.getElementById('intensityValue').textContent = '5';
        } else {
            showNotification('Failed to report migraine', 'danger');
        }
    } catch (error) {
        console.error('Error reporting migraine:', error);
        showNotification('Error submitting report', 'danger');
    }
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(connected) {
    const statusBadge = document.getElementById('statusBadge');
    if (!connected && statusBadge) {
        statusBadge.innerHTML = '<i class="bi bi-circle-fill text-danger"></i> Disconnected';
    }
}

/**
 * Show notification toast
 */
function showNotification(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type}" role="alert">
            <div class="d-flex">
                <div class="toast-body">${escapeHtml(message)}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: 3000 });
    toast.show();
    
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


// ============================================================
// Therapy Suggestion Feature
// ============================================================

let therapyAnswers = {};
let therapyCurrentStep = 1;
let therapyRiskScore = 0;
const THERAPY_TOTAL_STEPS = 4;

/**
 * Called on page load — fetches last prediction and auto-shows
 * a toast prompt if risk score is above 30%.
 */
function initTherapySuggestion() {
    setTimeout(() => {
        fetch('/api/predict')
            .then(r => r.json())
            .then(data => {
                const score = data.prediction?.risk_score || 0;
                therapyRiskScore = Math.round(score * 100);
                if (therapyRiskScore > 30) {
                    showTherapyToast(therapyRiskScore);
                }
            })
            .catch(() => {});
    }, 2500);  // slight delay to let the page fully settle
}

/**
 * Shows a non-intrusive toast at the bottom-right corner
 * prompting the user to get a therapy suggestion.
 */
function showTherapyToast(riskPct) {
    // Remove any existing toast first
    dismissTherapyToast();

    const levelColor = riskPct >= 70 ? 'danger' : riskPct >= 50 ? 'warning' : 'info';
    const levelLabel = riskPct >= 70 ? 'Critical' : riskPct >= 50 ? 'High' : 'Moderate';

    const toastEl = document.createElement('div');
    toastEl.id = 'therapyToast';
    toastEl.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9999;min-width:320px;max-width:380px;';
    toastEl.innerHTML = `
        <div class="card border-${levelColor} shadow">
            <div class="card-header bg-${levelColor} text-white d-flex justify-content-between align-items-center py-2">
                <span><i class="bi bi-heart-pulse"></i> <strong>Risk at ${riskPct}% — ${levelLabel}</strong></span>
                <button type="button" class="btn-close btn-close-white btn-sm" onclick="dismissTherapyToast()"></button>
            </div>
            <div class="card-body py-3">
                <p class="mb-2 small">Your migraine risk is elevated. Would you like a personalised therapy suggestion?</p>
                <div class="d-flex gap-2">
                    <button class="btn btn-primary btn-sm flex-fill" onclick="openTherapyFromToast()">
                        <i class="bi bi-heart-pulse"></i> Yes, suggest therapy
                    </button>
                    <button class="btn btn-outline-secondary btn-sm" onclick="dismissTherapyToast()">
                        Dismiss
                    </button>
                </div>
            </div>
        </div>`;
    document.body.appendChild(toastEl);
}

function dismissTherapyToast() {
    const t = document.getElementById('therapyToast');
    if (t) t.remove();
}

function openTherapyFromToast() {
    dismissTherapyToast();
    therapyRestart();
    const modalEl = document.getElementById('therapySuggestionModal');
    if (modalEl) {
        const modal = bootstrap.Modal.getOrCreateInstance(modalEl);
        modal.show();
    }
}

/**
 * Resets the modal back to Step 1.
 */
function therapyRestart() {
    therapyAnswers = {};
    therapyCurrentStep = 1;

    // Hide all steps and result
    for (let i = 1; i <= THERAPY_TOTAL_STEPS; i++) {
        const el = document.getElementById(`therapyStep${i}`);
        if (el) el.classList.add('d-none');
    }
    const result = document.getElementById('therapyResult');
    if (result) { result.classList.add('d-none'); result.innerHTML = ''; }

    // Show step 1
    const step1 = document.getElementById('therapyStep1');
    if (step1) step1.classList.remove('d-none');

    // Reset progress bar
    updateTherapyProgress(1);
    document.getElementById('therapyProgress')?.classList.remove('d-none');

    // Reset footer buttons
    const backBtn = document.getElementById('therapyBackBtn');
    const restartBtn = document.getElementById('therapyRestartBtn');
    if (backBtn) backBtn.style.display = 'none';
    if (restartBtn) restartBtn.style.display = 'none';

    // Clear all option highlights
    document.querySelectorAll('.therapy-option').forEach(b => {
        b.classList.remove('btn-primary', 'active');
        b.classList.add('btn-outline-secondary');
    });
}

function updateTherapyProgress(step) {
    const stepNames = ['Pain Location', 'Pain Type', 'Intensity', 'Sensitivity'];
    const pct = Math.round((step / THERAPY_TOTAL_STEPS) * 100);
    const bar   = document.getElementById('therapyProgressBar');
    const label = document.getElementById('therapyStepLabel');
    const desc  = document.getElementById('therapyStepDesc');
    if (bar)   bar.style.width = pct + '%';
    if (label) label.textContent = `Step ${step} of ${THERAPY_TOTAL_STEPS}`;
    if (desc)  desc.textContent  = stepNames[step - 1] || '';
}

function therapyGoBack() {
    if (therapyCurrentStep <= 1) return;

    // Hide current step / result
    const curStep = document.getElementById(`therapyStep${therapyCurrentStep}`);
    if (curStep) curStep.classList.add('d-none');
    const result = document.getElementById('therapyResult');
    if (result) { result.classList.add('d-none'); result.innerHTML = ''; }

    therapyCurrentStep--;
    updateTherapyProgress(therapyCurrentStep);

    const prevStep = document.getElementById(`therapyStep${therapyCurrentStep}`);
    if (prevStep) prevStep.classList.remove('d-none');

    document.getElementById('therapyProgress')?.classList.remove('d-none');
    document.getElementById('therapyBackBtn').style.display    = therapyCurrentStep > 1 ? 'inline-block' : 'none';
    document.getElementById('therapyRestartBtn').style.display = 'none';
}

/**
 * Called when user clicks any answer button.
 */
function handleTherapyAnswer(step, value, clickedBtn) {
    // Map step number → answer key
    const keyMap = { 1: 'pain_location', 2: 'pain_type', 3: 'pain_intensity', 4: 'sensitivity' };
    therapyAnswers[keyMap[step]] = value;

    // Highlight selected option
    document.querySelectorAll(`.therapy-option[data-step="${step}"]`).forEach(b => {
        b.classList.remove('btn-primary', 'active');
        b.classList.add('btn-outline-secondary');
    });
    if (clickedBtn) {
        clickedBtn.classList.remove('btn-outline-secondary');
        clickedBtn.classList.add('btn-primary', 'active');
    }

    // Brief visual pause then advance
    setTimeout(() => {
        if (step < THERAPY_TOTAL_STEPS) {
            document.getElementById(`therapyStep${step}`).classList.add('d-none');
            therapyCurrentStep = step + 1;
            updateTherapyProgress(therapyCurrentStep);
            document.getElementById(`therapyStep${therapyCurrentStep}`).classList.remove('d-none');
            document.getElementById('therapyBackBtn').style.display = 'inline-block';
        } else {
            // Last question answered — submit
            document.getElementById(`therapyStep${step}`).classList.add('d-none');
            submitTherapySurvey();
        }
    }, 280);
}

/**
 * POSTs answers to the backend and shows a loading spinner.
 */
function submitTherapySurvey() {
    document.getElementById('therapyProgress')?.classList.add('d-none');
    const backBtn    = document.getElementById('therapyBackBtn');
    const restartBtn = document.getElementById('therapyRestartBtn');
    if (backBtn)    backBtn.style.display    = 'none';
    if (restartBtn) restartBtn.style.display = 'inline-block';

    const result = document.getElementById('therapyResult');
    result.classList.remove('d-none');
    result.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status" style="width:3rem;height:3rem;"></div>
            <p class="mt-3 text-muted">Analysing your responses and building your therapy plan…</p>
        </div>`;

    // Use latest live risk score if available
    const scoreToSend = therapyRiskScore || 0;

    fetch('/api/therapy_suggestion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ risk_score_pct: scoreToSend, answers: therapyAnswers })
    })
    .then(r => r.json())
    .then(data => renderTherapyResult(data))
    .catch(() => {
        result.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i>
                Could not fetch suggestions. Please check your connection and try again.
            </div>`;
    });
}

/**
 * Renders the final therapy recommendation with full helmet device clarity.
 */
function renderTherapyResult(data) {
    const result = document.getElementById('therapyResult');

    // ── No therapy needed ─────────────────────────────────────────────────
    if (data.no_therapy_needed) {
        result.innerHTML = `
            <div class="text-center py-4">
                <div style="font-size:4rem;">😊</div>
                <h4 class="text-success fw-bold mt-2">You're Looking Good!</h4>
                <p class="text-muted mx-3 mt-2">${escapeHtml(data.message)}</p>
            </div>`;
        return;
    }

    // ── Decoder maps ──────────────────────────────────────────────────────
    const ZONE = {
        'F': { label: 'Frontal Lobe',   area: 'Forehead & temples',          svgPos: 'front' },
        'B': { label: 'Occipital Lobe', area: 'Base of skull & upper neck',  svgPos: 'back'  }
    };
    const TEMP = {
        'C': { label: 'Cold Therapy', icon: '❄️', color: '#2980b9', bg: '#eaf4fb',
               purpose: 'Constricts dilated blood vessels to reduce throbbing pain' },
        'H': { label: 'Hot Therapy',  icon: '🔥', color: '#c0392b', bg: '#fdf0ef',
               purpose: 'Relaxes muscle tension and improves circulation' }
    };
    const ILEVEL = {
        'L': { label: 'Low',    pct: '33%',  color: '#27ae60', kpa: '~5 – 10 kPa'  },
        'M': { label: 'Medium', pct: '66%',  color: '#f39c12', kpa: '~10 – 20 kPa' },
        'H': { label: 'High',   pct: '100%', color: '#e74c3c', kpa: '~20 – 35 kPa' }
    };
    const PTYPE = {
        'LK': { label: 'Lift & Knead',   icon: '👐',
                device:    'Scalp-kneading pneumatic bladders',
                mechanism: 'Rhythmically inflate and deflate to lift and knead the scalp, boosting lymphatic flow' },
        'AC': { label: 'Acupressure',    icon: '🤲',
                device:    'Targeted acupressure point nodes',
                mechanism: 'Press precisely on meridian points (LI4, P6, GB20) to interrupt pain signals' },
        'SP': { label: 'Scalp Pressure', icon: '👆',
                device:    'Full-scalp distributed pressure bladders',
                mechanism: 'Apply even compression across the scalp to reduce intracranial pressure sensation' }
    };
    const PLEVEL = {
        'L': { label: 'Low Pressure',    pct: '33%',  color: '#27ae60', kpa: '~5 – 10 kPa'  },
        'M': { label: 'Medium Pressure', pct: '66%',  color: '#f39c12', kpa: '~10 – 20 kPa' },
        'H': { label: 'High Pressure',   pct: '100%', color: '#e74c3c', kpa: '~20 – 35 kPa' }
    };

    // ── Parse codes ───────────────────────────────────────────────────────
    const rawHeadCodes  = (data.head_therapy_code || '').split('+').map(s => s.trim()).filter(Boolean);
    const pressureCode  = (data.pressure_therapy_code || '').trim();
    const pressureSteps = (data.pressure_therapy || {}).steps || [];

    const parsedHead = rawHeadCodes.map((code, i) => ({
        raw:       code,
        zone:      ZONE[code[0]]   || { label: code[0], area: 'Head zone', svgPos: '' },
        temp:      TEMP[code[1]]   || { label: 'Thermal', icon: '🌡️', color: '#888', bg: '#f5f5f5', purpose: '' },
        intensity: ILEVEL[code[2]] || ILEVEL['M'],
        steps:     (data.head_therapy || [])[i]?.steps || []
    }));

    const ptypeKey  = pressureCode.length >= 3 ? pressureCode.slice(0, 2) : pressureCode.slice(0, 2);
    const plevelKey = pressureCode.slice(-1);
    const ptype     = PTYPE[ptypeKey]   || null;
    const plevel    = PLEVEL[plevelKey] || null;

    // ── Derived helpers ───────────────────────────────────────────────────
    const frontDev = parsedHead.find(c => c.zone.svgPos === 'front');
    const backDev  = parsedHead.find(c => c.zone.svgPos === 'back');
    const sens     = therapyAnswers.sensitivity || 'no';
    const badgeColor = { Low:'success', Moderate:'warning', High:'danger', Critical:'dark' }[data.risk_level] || 'secondary';

    // ── Sensitivity alert ─────────────────────────────────────────────────
    const sensAlert = (sens === 'very' || sens === 'somewhat') ? `
        <div class="hth-sense-alert">
            <span>💡</span>
            <div><strong>Light & Sound Sensitivity Detected</strong><br>
            <small>Move to a dark, quiet room before fitting the helmet. Dim your screen or close your laptop lid.</small></div>
        </div>` : '';

    // ── Helmet SVG map ────────────────────────────────────────────────────
    const fColor = frontDev ? frontDev.temp.color : '#ccc';
    const bColor = backDev  ? backDev.temp.color  : '#ccc';
    const pColor = plevel   ? plevel.color         : '#ccc';

    const helmetMap = `
    <div class="hth-map-wrap">
        <div class="hth-map-title">🪖 Helmet Activation Map</div>
        <div class="hth-map-body">
            <svg viewBox="0 0 240 280" class="hth-svg">
                <!-- head outline -->
                <ellipse cx="120" cy="140" rx="85" ry="115" fill="#f4f9ff" stroke="#c5ddf5" stroke-width="2"/>

                <!-- front zone -->
                <ellipse cx="120" cy="52" rx="60" ry="38"
                    fill="${fColor}" fill-opacity="${frontDev ? '.2' : '.06'}"
                    stroke="${fColor}" stroke-width="${frontDev ? '2.5' : '1'}"/>
                <text x="120" y="46" text-anchor="middle" font-size="20">${frontDev ? frontDev.temp.icon : '○'}</text>
                <text x="120" y="65" text-anchor="middle" font-size="8.5" font-weight="700"
                      fill="${fColor}" font-family="Inter,sans-serif">FRONTAL LOBE</text>

                <!-- back zone -->
                <ellipse cx="120" cy="228" rx="60" ry="38"
                    fill="${bColor}" fill-opacity="${backDev ? '.2' : '.06'}"
                    stroke="${bColor}" stroke-width="${backDev ? '2.5' : '1'}"/>
                <text x="120" y="222" text-anchor="middle" font-size="20">${backDev ? backDev.temp.icon : '○'}</text>
                <text x="120" y="242" text-anchor="middle" font-size="8.5" font-weight="700"
                      fill="${bColor}" font-family="Inter,sans-serif">OCCIPITAL LOBE</text>

                <!-- pressure nodes -->
                ${ptype ? `
                <circle cx="38"  cy="115" r="8" fill="${pColor}" fill-opacity=".75"/>
                <circle cx="38"  cy="155" r="8" fill="${pColor}" fill-opacity=".75"/>
                <circle cx="202" cy="115" r="8" fill="${pColor}" fill-opacity=".75"/>
                <circle cx="202" cy="155" r="8" fill="${pColor}" fill-opacity=".75"/>
                <circle cx="120" cy="30"  r="6" fill="${pColor}" fill-opacity=".55"/>
                <circle cx="120" cy="250" r="6" fill="${pColor}" fill-opacity=".55"/>
                ` : `
                <circle cx="38"  cy="115" r="8" fill="#ddd"/>
                <circle cx="38"  cy="155" r="8" fill="#ddd"/>
                <circle cx="202" cy="115" r="8" fill="#ddd"/>
                <circle cx="202" cy="155" r="8" fill="#ddd"/>
                `}

                <!-- centre label -->
                <text x="120" y="133" text-anchor="middle" font-size="9" fill="#8faabf" font-family="Inter,sans-serif">PRESSURE</text>
                <text x="120" y="148" text-anchor="middle" font-size="9" fill="#8faabf" font-family="Inter,sans-serif">NODES</text>
            </svg>

            <!-- legend chips -->
            <div class="hth-legend">
                ${parsedHead.map(c => `
                <div class="hth-chip" style="border-color:${c.temp.color};background:${c.temp.bg}">
                    <span class="hth-chip-code" style="background:${c.temp.color}">${c.raw}</span>
                    <div class="hth-chip-info">
                        <span class="hth-chip-zone">${c.zone.label}</span>
                        <span class="hth-chip-type" style="color:${c.temp.color}">${c.temp.icon} ${c.temp.label}</span>
                    </div>
                </div>`).join('')}
                ${ptype && plevel ? `
                <div class="hth-chip" style="border-color:${plevel.color};background:#f5fff8">
                    <span class="hth-chip-code" style="background:${plevel.color}">${pressureCode}</span>
                    <div class="hth-chip-info">
                        <span class="hth-chip-zone">${ptype.label}</span>
                        <span class="hth-chip-type" style="color:${plevel.color}">${ptype.icon} ${plevel.label}</span>
                    </div>
                </div>` : ''}
            </div>
        </div>
    </div>`;

    // ── Temperature device cards ──────────────────────────────────────────
    const tempCards = parsedHead.map(c => `
        <div class="hth-device-card" style="border-left-color:${c.temp.color};background:${c.temp.bg}">

            <!-- header -->
            <div class="hth-device-header">
                <div class="hth-device-badge" style="background:${c.temp.color}">${c.temp.icon}</div>
                <div>
                    <div class="hth-device-code">${c.raw}</div>
                    <div class="hth-device-subtitle">${c.zone.label} &nbsp;·&nbsp; ${c.temp.label} &nbsp;·&nbsp; ${c.intensity.label} Intensity</div>
                </div>
            </div>

            <!-- code pill breakdown -->
            <div class="hth-code-row">
                <div class="hth-pill" style="border-color:${c.temp.color}">
                    <span class="hth-pill-letter" style="background:${c.temp.color}">${c.raw[0]}</span>
                    <span class="hth-pill-text">${c.zone.label}</span>
                </div>
                <div class="hth-pill" style="border-color:${c.temp.color}">
                    <span class="hth-pill-letter" style="background:${c.temp.color}">${c.raw[1]}</span>
                    <span class="hth-pill-text">${c.temp.label.split(' ')[0]}</span>
                </div>
                <div class="hth-pill" style="border-color:${c.temp.color}">
                    <span class="hth-pill-letter" style="background:${c.temp.color}">${c.raw[2]}</span>
                    <span class="hth-pill-text">${c.intensity.label} Intensity</span>
                </div>
            </div>

            <!-- device info rows -->
            <div class="hth-info-row"><i class="bi bi-geo-alt-fill" style="color:${c.temp.color}"></i>
                <span><strong>Brain Region:</strong> ${c.zone.area}</span></div>
            <div class="hth-info-row"><i class="bi bi-cpu-fill" style="color:${c.temp.color}"></i>
                <span><strong>Helmet Device:</strong> ${c.zone.label} thermal pad — ${c.temp.label.toLowerCase()}</span></div>
            <div class="hth-info-row"><i class="bi bi-lightbulb-fill" style="color:${c.temp.color}"></i>
                <span><strong>Purpose:</strong> ${c.temp.purpose}</span></div>

            <!-- intensity bar -->
            <div class="hth-meter">
                <span class="hth-meter-label">Intensity</span>
                <div class="hth-meter-track"><div class="hth-meter-fill" style="width:${c.intensity.pct};background:${c.intensity.color}"></div></div>
                <span class="hth-meter-val" style="color:${c.intensity.color}">${c.intensity.label}</span>
            </div>

            <!-- instructions -->
            ${c.steps.length ? `
            <div class="hth-steps">
                <div class="hth-steps-label">📋 Step-by-step Instructions</div>
                <ol class="mb-0 ps-3">
                    ${c.steps.map(s => `<li class="small mb-1">${escapeHtml(s)}</li>`).join('')}
                </ol>
            </div>` : ''}
        </div>`).join('');

    // ── Pressure device card ──────────────────────────────────────────────
    const pressureCard = (ptype && plevel) ? `
        <div class="hth-device-card" style="border-left-color:${plevel.color};background:#f5fff8">

            <!-- header -->
            <div class="hth-device-header">
                <div class="hth-device-badge" style="background:${plevel.color}">${ptype.icon}</div>
                <div>
                    <div class="hth-device-code">${pressureCode}</div>
                    <div class="hth-device-subtitle">${ptype.label} &nbsp;·&nbsp; ${plevel.label}</div>
                </div>
            </div>

            <!-- code pill breakdown -->
            <div class="hth-code-row">
                <div class="hth-pill" style="border-color:${plevel.color}">
                    <span class="hth-pill-letter" style="background:${plevel.color}">${pressureCode.slice(0,2)}</span>
                    <span class="hth-pill-text">${ptype.label}</span>
                </div>
                <div class="hth-pill" style="border-color:${plevel.color}">
                    <span class="hth-pill-letter" style="background:${plevel.color}">${pressureCode.slice(-1)}</span>
                    <span class="hth-pill-text">${plevel.label.split(' ')[0]} Pressure</span>
                </div>
            </div>

            <!-- device info rows -->
            <div class="hth-info-row"><i class="bi bi-cpu-fill" style="color:${plevel.color}"></i>
                <span><strong>Helmet Device:</strong> ${ptype.device}</span></div>
            <div class="hth-info-row"><i class="bi bi-gear-fill" style="color:${plevel.color}"></i>
                <span><strong>Mechanism:</strong> ${ptype.mechanism}</span></div>
            <div class="hth-info-row"><i class="bi bi-speedometer2" style="color:${plevel.color}"></i>
                <span><strong>Operating Pressure:</strong> ${plevel.kpa} — ${plevel.label}</span></div>

            <!-- pressure bar -->
            <div class="hth-meter">
                <span class="hth-meter-label">Pressure</span>
                <div class="hth-meter-track"><div class="hth-meter-fill" style="width:${plevel.pct};background:${plevel.color}"></div></div>
                <span class="hth-meter-val" style="color:${plevel.color}">${plevel.label}</span>
            </div>

            <!-- instructions -->
            ${pressureSteps.length ? `
            <div class="hth-steps">
                <div class="hth-steps-label">📋 Step-by-step Instructions</div>
                <ol class="mb-0 ps-3">
                    ${pressureSteps.map(s => `<li class="small mb-1">${escapeHtml(s)}</li>`).join('')}
                </ol>
            </div>` : ''}
        </div>` : '';

    // ── General tips ──────────────────────────────────────────────────────
    const tips = [
        ['💧', 'Drink a full glass of water before fitting the helmet — dehydration worsens migraine pain.'],
        ['🛋️', 'Sit in a reclined position or lie down with your neck fully supported throughout the session.'],
        ['🌬️', 'Breathe slowly: inhale for 4 counts, hold for 2, exhale for 6. This calms the nervous system.'],
        ['⏱️', 'A full therapy session typically lasts 15 – 25 minutes depending on the intensity setting.']
    ];
    if (sens === 'very' || sens === 'somewhat')
        tips.push(['🔇', 'Mute all device notifications and ask those nearby to keep noise to a minimum.']);

    const generalTips = `
        <div class="hth-tips">
            <div class="hth-tips-title">💡 Before You Start — General Guidance</div>
            ${tips.map(([icon, text]) => `
            <div class="hth-tip-row">
                <span class="hth-tip-icon">${icon}</span>
                <span class="small">${text}</span>
            </div>`).join('')}
        </div>`;

    // ── Session duration banner ───────────────────────────────────────────
    const durMin = data.session_duration_minutes;
    const durMap = { 20: { label: '20 Minutes', icon: '🟢', color: '#27ae60', bg: '#eafaf1', hint: 'Light session — suitable for mild discomfort' },
                     30: { label: '30 Minutes', icon: '🟡', color: '#f39c12', bg: '#fef9ec', hint: 'Standard session — recommended for moderate pain' },
                     45: { label: '45 Minutes', icon: '🟠', color: '#e67e22', bg: '#fdf2e9', hint: 'Extended session — for high-intensity relief' },
                     60: { label: '60 Minutes', icon: '🔴', color: '#e74c3c', bg: '#fdf0ef', hint: 'Full session — critical pain management protocol' } };
    const durInfo = durMap[durMin] || null;
    const durationBanner = durInfo ? `
        <div class="therapy-duration-banner" style="background:${durInfo.bg};border-color:${durInfo.color}">
            <div class="therapy-duration-main">
                <span class="therapy-duration-icon">${durInfo.icon}</span>
                <div>
                    <div class="therapy-duration-label">Recommended Session Duration</div>
                    <div class="therapy-duration-value" style="color:${durInfo.color}">${durInfo.label}</div>
                </div>
            </div>
            <div class="therapy-duration-hint">${durInfo.hint}</div>
            <div class="therapy-duration-slots">
                ${[20,30,45,60].map(m => `
                <div class="therapy-duration-slot ${m === durMin ? 'active' : ''}" style="${m === durMin ? `background:${durInfo.color};color:#fff;border-color:${durInfo.color}` : ''}">${m} min</div>
                `).join('')}
            </div>
        </div>` : '';

    // ── Assemble ──────────────────────────────────────────────────────────
    result.innerHTML = `
        <!-- Risk badge -->
        <div class="text-center mb-3">
            <span class="badge bg-${badgeColor} fs-6 px-4 py-2 rounded-pill shadow-sm">
                ⚠️ ${escapeHtml(data.risk_level)} Risk &nbsp;|&nbsp; ${data.risk_score_pct}%
            </span>
        </div>

        <!-- Session duration banner -->
        ${durationBanner}

        ${sensAlert}
        ${helmetMap}

        <!-- Temperature devices -->
        <div class="hth-section-header">
            <span class="hth-section-icon">🌡️</span>
            <div>
                <div class="hth-section-title">Temperature Therapy Devices</div>
                <div class="hth-section-sub">Thermal pads in the helmet shell — auto-activated at the prescribed intensity</div>
            </div>
        </div>
        ${tempCards}

        <!-- Pressure device -->
        ${pressureCard ? `
        <div class="hth-section-header mt-3">
            <span class="hth-section-icon">💆</span>
            <div>
                <div class="hth-section-title">Pressure Therapy Device</div>
                <div class="hth-section-sub">Internal pneumatic tubes — inflate/deflate on a timed cycle inside the helmet</div>
            </div>
        </div>
        ${pressureCard}` : ''}

        <!-- Tips -->
        <div class="mt-3">${generalTips}</div>
    `;
}
