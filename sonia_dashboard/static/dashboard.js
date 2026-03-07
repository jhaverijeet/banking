/* ═══════════════════════════════════════════════════════════════
   SONIA Rate Dashboard – Client-side JavaScript
   ═══════════════════════════════════════════════════════════════ */

// ─── Constants ──────────────────────────────────────────────────
const TENOR_COLORS = {
    1: '#6366f1', // indigo
    2: '#8b5cf6', // violet
    3: '#a855f7', // purple
    4: '#d946ef', // fuchsia
    5: '#ec4899', // pink
    6: '#f43f5e', // rose
    7: '#f97316', // orange
};

const TENOR_LABELS = {
    1: '1 Year',
    2: '2 Year',
    3: '3 Year',
    4: '4 Year',
    5: '5 Year',
    6: '6 Year',
    7: '7 Year',
};

// ─── State ──────────────────────────────────────────────────────
let state = {
    data: [],
    summary: null,
    dateRange: { min_date: null, max_date: null },
    selectedTenors: [1, 2, 3, 4, 5, 6, 7],
    startDate: null,
    endDate: null,
    showPoints: false,
};

let mainChart = null;
let termChart = null;

// ─── API Calls ──────────────────────────────────────────────────
async function fetchJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
    return resp.json();
}

async function fetchSummary() {
    state.summary = await fetchJSON('/api/summary');
}

async function fetchDateRange() {
    const dr = await fetchJSON('/api/date-range');
    state.dateRange = dr;
}

async function fetchRates() {
    const params = new URLSearchParams();
    if (state.startDate) params.set('start', state.startDate);
    if (state.endDate) params.set('end', state.endDate);
    params.set('tenors', state.selectedTenors.join(','));

    state.data = await fetchJSON(`/api/rates?${params.toString()}`);
}

// ─── UI Updates ─────────────────────────────────────────────────

function updateHeaderMeta() {
    const s = state.summary;
    if (!s) return;
    document.getElementById('total-rows').textContent = s.total_rows.toLocaleString();
    document.getElementById('latest-date').textContent = formatDate(s.latest_date);
}

function formatDate(dateStr) {
    if (!dateStr) return '--';
    const d = new Date(dateStr + 'T00:00:00');
    return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
}

function formatRate(val) {
    if (val == null || isNaN(val)) return '--';
    return val.toFixed(2);
}

function updateRateCards() {
    const container = document.getElementById('rate-cards');
    const s = state.summary;
    if (!s || !s.latest_rates) return;

    // Find second-to-last data point for change calc
    const prevRow = state.data.length >= 2 ? state.data[state.data.length - 2] : null;

    let html = '';
    for (let t = 1; t <= 7; t++) {
        const col = `tenor_${t}y`;
        const val = s.latest_rates[col];
        const prevVal = prevRow ? prevRow[col] : null;
        const change = (val != null && prevVal != null) ? val - prevVal : null;
        const changeClass = change > 0 ? 'positive' : change < 0 ? 'negative' : 'neutral';
        const changeSign = change > 0 ? '+' : '';
        const arrow = change > 0 ? '&#9650;' : change < 0 ? '&#9660;' : '&#8226;';

        html += `
            <div class="rate-card" style="--card-color: ${TENOR_COLORS[t]}">
                <div class="tenor-label">${t}Y Swap</div>
                <div class="rate-value">${formatRate(val)}<span class="rate-unit">%</span></div>
                ${change != null ? `
                    <div class="rate-change ${changeClass}">
                        ${arrow} ${changeSign}${(change * 100).toFixed(1)}bp
                    </div>
                ` : ''}
            </div>
        `;
    }
    container.innerHTML = html;
}

// ─── Main Chart ─────────────────────────────────────────────────

function buildMainChart() {
    const ctx = document.getElementById('main-chart').getContext('2d');

    if (mainChart) mainChart.destroy();

    const datasets = state.selectedTenors.map(t => {
        const col = `tenor_${t}y`;
        const color = TENOR_COLORS[t];
        return {
            label: TENOR_LABELS[t],
            data: state.data.map(row => ({
                x: row.date,
                y: row[col],
            })).filter(d => d.y != null),
            borderColor: color,
            backgroundColor: hexToRGBA(color, 0.08),
            borderWidth: 1.8,
            pointRadius: state.showPoints ? 2 : 0,
            pointHoverRadius: 5,
            pointBackgroundColor: color,
            pointBorderColor: 'transparent',
            tension: 0.3,
        };
    });

    mainChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: '#94a3b8',
                        font: { family: "'Inter', sans-serif", size: 11, weight: 500 },
                        usePointStyle: true,
                        pointStyle: 'line',
                        padding: 16,
                        boxWidth: 20,
                    },
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    borderColor: 'rgba(99, 102, 241, 0.3)',
                    borderWidth: 1,
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    titleFont: { family: "'Inter', sans-serif", size: 12, weight: 600 },
                    bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        title(items) {
                            if (!items.length) return '';
                            const d = new Date(items[0].raw.x + 'T00:00:00');
                            return d.toLocaleDateString('en-GB', {
                                weekday: 'short',
                                day: '2-digit',
                                month: 'short',
                                year: 'numeric',
                            });
                        },
                        label(item) {
                            return `  ${item.dataset.label}: ${item.raw.y.toFixed(4)}%`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        parser: 'yyyy-MM-dd',
                        tooltipFormat: 'dd MMM yyyy',
                        unit: calculateTimeUnit(),
                        displayFormats: {
                            day: 'dd MMM',
                            week: 'dd MMM',
                            month: 'MMM yyyy',
                            quarter: 'MMM yyyy',
                            year: 'yyyy',
                        },
                    },
                    grid: {
                        color: 'rgba(99, 102, 241, 0.06)',
                        drawBorder: false,
                    },
                    ticks: {
                        color: '#64748b',
                        font: { family: "'Inter', sans-serif", size: 10 },
                        maxTicksLimit: 12,
                    },
                },
                y: {
                    grid: {
                        color: 'rgba(99, 102, 241, 0.06)',
                        drawBorder: false,
                    },
                    ticks: {
                        color: '#64748b',
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                        callback: val => val.toFixed(1) + '%',
                    },
                    title: {
                        display: true,
                        text: 'Rate (%)',
                        color: '#64748b',
                        font: { family: "'Inter', sans-serif", size: 11 },
                    },
                },
            },
            animation: {
                duration: 600,
                easing: 'easeOutQuart',
            },
        },
    });
}

function calculateTimeUnit() {
    if (!state.data.length) return 'month';
    const first = new Date(state.data[0].date);
    const last = new Date(state.data[state.data.length - 1].date);
    const diffDays = (last - first) / (1000 * 60 * 60 * 24);
    if (diffDays <= 60) return 'day';
    if (diffDays <= 180) return 'week';
    if (diffDays <= 730) return 'month';
    return 'quarter';
}

// ─── Term Structure Chart ───────────────────────────────────────

function buildTermChart() {
    const ctx = document.getElementById('term-chart').getContext('2d');
    if (termChart) termChart.destroy();

    const latest = state.data.length ? state.data[state.data.length - 1] : null;
    if (!latest) return;

    // Update subtitle
    document.getElementById('stats-current-date').textContent =
        'As of ' + formatDate(latest.date);

    const labels = state.selectedTenors.map(t => `${t}Y`);
    const values = state.selectedTenors.map(t => latest[`tenor_${t}y`]);
    const colors = state.selectedTenors.map(t => TENOR_COLORS[t]);

    // Gradient background
    const gradient = ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
    gradient.addColorStop(0, hexToRGBA(colors[0], 0.2));
    gradient.addColorStop(1, hexToRGBA(colors[colors.length - 1], 0.2));

    termChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data: values,
                borderColor: '#a855f7',
                borderWidth: 2,
                backgroundColor: gradient,
                fill: true,
                pointBackgroundColor: colors,
                pointBorderColor: 'transparent',
                pointRadius: 4,
                pointHoverRadius: 6,
                tension: 0.3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    borderColor: 'rgba(99, 102, 241, 0.3)',
                    borderWidth: 1,
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
                    padding: 10,
                    cornerRadius: 8,
                    callbacks: {
                        label: (item) => `  ${item.raw.toFixed(4)}%`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        color: '#94a3b8',
                        font: { family: "'JetBrains Mono', monospace", size: 11, weight: 600 },
                    },
                },
                y: {
                    grid: { color: 'rgba(99, 102, 241, 0.06)', drawBorder: false },
                    ticks: {
                        color: '#64748b',
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                        callback: v => v.toFixed(1) + '%',
                    },
                },
            },
            animation: {
                duration: 500,
                easing: 'easeOutQuart',
            },
        },
    });
}

// ─── Stats Table ────────────────────────────────────────────────

function buildStatsTable() {
    const tbody = document.getElementById('stats-tbody');
    if (!state.data.length) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#64748b">No data</td></tr>';
        return;
    }

    const first = state.data[0];
    const last = state.data[state.data.length - 1];

    let html = '';
    for (const t of state.selectedTenors) {
        const col = `tenor_${t}y`;
        const values = state.data.map(r => r[col]).filter(v => v != null);
        if (!values.length) continue;

        const current = last[col];
        const min = Math.min(...values);
        const max = Math.max(...values);
        const avg = values.reduce((a, b) => a + b, 0) / values.length;
        const firstVal = first[col];
        const change = (current != null && firstVal != null) ? current - firstVal : null;
        const changeBps = change != null ? (change * 100).toFixed(0) : '--';
        const changeClass = change > 0 ? 'change-positive' : change < 0 ? 'change-negative' : '';
        const changeSign = change > 0 ? '+' : '';

        html += `
            <tr>
                <td class="tenor-col" style="color:${TENOR_COLORS[t]}">${t}Y</td>
                <td>${current != null ? current.toFixed(2) + '%' : '--'}</td>
                <td>${min.toFixed(2)}%</td>
                <td>${max.toFixed(2)}%</td>
                <td>${avg.toFixed(2)}%</td>
                <td class="${changeClass}">${changeSign}${changeBps}bp</td>
            </tr>
        `;
    }
    tbody.innerHTML = html;
}

// ─── Helpers ────────────────────────────────────────────────────

function hexToRGBA(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function subtractMonths(dateStr, months) {
    if (!dateStr) return null;
    // Parse YYYY-MM-DD manually to avoid timezone/browser issues
    const parts = dateStr.split('-');
    const year = parseInt(parts[0], 10);
    const month = parseInt(parts[1], 10) - 1;  // 0-indexed
    const day = parseInt(parts[2], 10);
    const d = new Date(year, month, day);
    d.setMonth(d.getMonth() - months);
    // Format back to YYYY-MM-DD
    const yy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    return `${yy}-${mm}-${dd}`;
}

function getStartForRange(range, maxDate) {
    switch (range) {
        case '1M': return subtractMonths(maxDate, 1);
        case '3M': return subtractMonths(maxDate, 3);
        case '6M': return subtractMonths(maxDate, 6);
        case '1Y': return subtractMonths(maxDate, 12);
        case '2Y': return subtractMonths(maxDate, 24);
        case '5Y': return subtractMonths(maxDate, 60);
        case 'ALL': return null;
        default: return null;
    }
}

// ─── Refresh Pipeline ───────────────────────────────────────────

async function refresh() {
    await fetchRates();
    buildMainChart();
    buildTermChart();
    buildStatsTable();
    updateRateCards();
}

// ─── Event Handlers ─────────────────────────────────────────────

function setupEventHandlers() {
    // Quick range buttons
    document.querySelectorAll('.quick-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            document.querySelectorAll('.quick-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const range = btn.dataset.range;
            const maxDate = state.dateRange.max_date;
            state.startDate = getStartForRange(range, maxDate);
            state.endDate = maxDate;

            // Sync date inputs
            document.getElementById('start-date').value = state.startDate || state.dateRange.min_date;
            document.getElementById('end-date').value = state.endDate;

            await refresh();
        });
    });

    // Date inputs
    const startInput = document.getElementById('start-date');
    const endInput = document.getElementById('end-date');

    startInput.addEventListener('change', async () => {
        state.startDate = startInput.value || null;
        document.querySelectorAll('.quick-btn').forEach(b => b.classList.remove('active'));
        await refresh();
    });

    endInput.addEventListener('change', async () => {
        state.endDate = endInput.value || null;
        document.querySelectorAll('.quick-btn').forEach(b => b.classList.remove('active'));
        await refresh();
    });

    // Tenor toggles
    document.querySelectorAll('.tenor-toggle input').forEach(cb => {
        cb.addEventListener('change', async () => {
            state.selectedTenors = Array.from(
                document.querySelectorAll('.tenor-toggle input:checked')
            ).map(el => parseInt(el.value)).sort();

            if (state.selectedTenors.length === 0) {
                // Don't allow deselecting all
                cb.checked = true;
                state.selectedTenors = [parseInt(cb.value)];
            }

            await refresh();
        });
    });



    // Toggle points
    document.getElementById('btn-toggle-points').addEventListener('click', async () => {
        state.showPoints = !state.showPoints;
        document.getElementById('btn-toggle-points').classList.toggle('active', state.showPoints);
        await refresh();
    });
}

// ─── Initialization ─────────────────────────────────────────────

async function init() {
    const overlay = document.getElementById('loading-overlay');

    try {
        // Parallel initial data loading
        await Promise.all([fetchSummary(), fetchDateRange()]);

        updateHeaderMeta();

        // Set default date range (5 years)
        const maxDate = state.dateRange.max_date;
        const minDate = state.dateRange.min_date;

        if (maxDate) {
            state.startDate = getStartForRange('5Y', maxDate);
            state.endDate = maxDate;

            document.getElementById('start-date').value = state.startDate || minDate;
            document.getElementById('end-date').value = state.endDate;
            if (minDate) document.getElementById('start-date').min = minDate;
            document.getElementById('end-date').max = maxDate;
        }

        await refresh();
        setupEventHandlers();

    } catch (err) {
        console.error('Initialization failed:', err);
        overlay.querySelector('p').textContent = `Error: ${err.message}`;
        return;
    }

    // Fade out loading
    setTimeout(() => overlay.classList.add('hidden'), 300);
}

// Launch
document.addEventListener('DOMContentLoaded', init);
