/**
 * Chrome Dino AI dashboard: live training and routing from the /ws stream
 * (see src/dashboard/ws_handler.py). No libraries: charts are plain SVG.
 */

const RANDOM_LENGTH = 65; // steps a random policy survives (docs, smoke runs)
const CURVE_WINDOW = 20; // moving average for the learning curve
const LIMITS = { episodes: 3000, training: 200, performance: 500, routing: 100, decisions: 300 };
const COLORS = { dqn: '#C6F24E', ppo: '#4CC3F0' };
const EXTRA_COLORS = ['#E8A15A', '#B48CF2', '#F2C94E', '#7CF2A8'];
// live player: jump lime, duck cyan (Laya Player design), hold grey
const ACTION_COLORS = { jump: '#C6F24E', duck: '#4CC3F0', hold: '#8A9A95' };
const TRACE_LENGTH = 100;
const STALE_SECONDS = 10; // the player stopped if its last decision is older

const state = {
    status: {}, latest: { agents: {}, routing: null },
    episodes: [], training: [], performance: [], routing: [], decisions: [],
    updated: null,
};
let retries = 0;
let renderQueued = false;

// ---- helpers ------------------------------------------------------------

const $ = (id) => document.getElementById(id);

function esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[c]);
}

function color(agent) {
    if (COLORS[agent]) return COLORS[agent];
    const others = Object.keys(state.status).filter((a) => !COLORS[a]);
    return EXTRA_COLORS[Math.max(0, others.indexOf(agent)) % EXTRA_COLORS.length];
}

/** SQLite timestamps are UTC ("2026-09-28 01:58:27") */
function localTime(ts) {
    return ts ? new Date(ts.replace(' ', 'T') + 'Z').toLocaleTimeString() : '–';
}

function fmt(value, digits = 0) {
    return value == null ? '–' : Number(value).toLocaleString(undefined, {
        minimumFractionDigits: digits, maximumFractionDigits: digits,
    });
}

function movingAverage(values, window) {
    const out = [];
    let sum = 0;
    values.forEach((v, i) => {
        sum += v;
        if (i >= window) sum -= values[i - window];
        out.push(sum / Math.min(i + 1, window));
    });
    return out;
}

function byAgent(rows) {
    const groups = {};
    rows.forEach((r) => { (groups[r.agent] = groups[r.agent] || []).push(r); });
    return groups;
}

/**
 * Line chart into an <svg viewBox="0 0 W H">: `series` is
 * [{color, values}], each value plotted at its index; `baseline` a dashed
 * red line (e.g. random play).
 */
function lineChart(svg, series, { baseline = null, digits = 0 } = {}) {
    const [, , W, H] = svg.getAttribute('viewBox').split(' ').map(Number);
    const pad = { l: 44, r: 8, t: 8, b: 18 };
    const all = series.flatMap((s) => s.values).concat(baseline == null ? [] : [baseline]);
    if (!all.length) {
        svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" fill="#8A9A95" font-size="12" text-anchor="middle">no data yet</text>`;
        return;
    }
    const yMax = Math.max(...all) * 1.1 || 1;
    const n = Math.max(...series.map((s) => s.values.length), 2);
    const x = (i) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
    const y = (v) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
    let out = '';
    for (let k = 0; k <= 4; k++) {
        const v = (yMax / 4) * k;
        out += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(v)}" y2="${y(v)}" stroke="#1E2826"/>`;
        out += `<text x="${pad.l - 6}" y="${y(v) + 4}" fill="#8A9A95" font-size="10" text-anchor="end">${fmt(v, digits)}</text>`;
    }
    if (baseline != null) {
        out += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(baseline)}" y2="${y(baseline)}" stroke="#F0605A" stroke-dasharray="5 4"/>`;
    }
    series.forEach((s) => {
        if (!s.values.length) return;
        const points = s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
        out += `<polyline points="${points}" fill="none" stroke="${s.color}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
    });
    svg.innerHTML = out;
}

function legend(el, agents, extra = '') {
    el.innerHTML = agents.map((a) =>
        `<span><i style="background:${color(a)}"></i>${esc(a)}</span>`).join('') + extra;
}

// ---- rendering ------------------------------------------------------------

function renderHeader() {
    const status = Object.values(state.status);
    const training = Object.entries(state.status).filter(([, s]) => s.state === 'training').map(([a]) => a);
    const perf = training.map((a) => (state.latest.agents[a] || {}).performance).filter(Boolean);
    $('statEpisodes').textContent = fmt(status.reduce((sum, s) => sum + s.episodes, 0));
    $('statSpeed').textContent = perf.length ? fmt(perf.reduce((s, p) => s + p.steps_per_s, 0), 1) : '–';
    $('statMemory').textContent = perf.length
        ? fmt(perf.reduce((s, p) => s + p.memory_mb, 0) / 1024, 1) + ' GB' : '–';
    $('statUpdated').textContent = state.updated ? state.updated.toLocaleTimeString() : '–';
}

function renderAgents() {
    $('agents').innerHTML = Object.entries(state.status).map(([agent, s]) => {
        const latest = state.latest.agents[agent] || {};
        const length = latest.mean_length;
        const delta = length == null ? null : (length / RANDOM_LENGTH - 1) * 100;
        const perf = latest.performance || {};
        const best = s.best_checkpoint;
        return `<article class="agent" style="border-top: 2px solid ${color(agent)}">
            <div class="top">
                <span class="name" style="color:${color(agent)}">${esc(agent.toUpperCase())}</span>
                <span class="chip ${s.state}">${s.state.toUpperCase()}</span>
            </div>
            <div class="big">${fmt(length)} <small>steps survived</small>
                ${delta == null ? '' : `<small class="delta ${delta >= 0 ? 'up' : 'down'}">${delta >= 0 ? '+' : ''}${fmt(delta)}% vs random</small>`}
            </div>
            <dl class="kv">
                <dt>mean reward</dt><dd>${fmt(latest.mean_reward, 1)}</dd>
                <dt>episodes</dt><dd>${fmt(s.episodes)}</dd>
                <dt>best checkpoint</dt><dd>${best ? `step ${fmt(best.step)} · ${fmt(best.reward, 1)}` : '–'}</dd>
                <dt>resume from</dt><dd>${s.latest_step == null ? '–' : 'step ' + fmt(s.latest_step)}</dd>
                <dt>speed</dt><dd>${perf.steps_per_s == null ? '–' : fmt(perf.steps_per_s, 1) + ' steps/s'}</dd>
                <dt>last episode</dt><dd>${localTime(s.last_episode_at)}</dd>
            </dl>
        </article>`;
    }).join('') || '<div class="empty">No agents yet.</div>';
}

function renderCharts() {
    const agents = Object.keys(state.status);
    const episodes = byAgent(state.episodes);
    lineChart($('curve'), agents.map((a) => ({
        color: color(a),
        values: movingAverage((episodes[a] || []).map((r) => r.length), CURVE_WINDOW),
    })), { baseline: RANDOM_LENGTH });
    legend($('curveLegend'), agents,
        '<span><i style="background:#F0605A"></i>random ≈ 65</span>');

    const perf = byAgent(state.performance.filter((r) => r.agent));
    const perfAgents = agents.filter((a) => perf[a]);
    lineChart($('speed'), perfAgents.map((a) => ({
        color: color(a), values: perf[a].map((r) => r.steps_per_s),
    })), { digits: 1 });
    lineChart($('memory'), perfAgents.map((a) => ({
        color: color(a), values: perf[a].map((r) => r.memory_mb),
    })));
    legend($('perfLegend'), perfAgents);
}

function renderRouter() {
    const last = state.latest.routing;
    if (!last) {
        $('router').innerHTML = '<div class="empty">No routing decisions yet. <code>dino-ai play</code> logs one at each episode start and difficulty change.</div>';
        $('routerTime').textContent = '';
        return;
    }
    $('routerTime').textContent = localTime(last.timestamp);
    $('router').innerHTML = `
        <div class="action" style="color:${color(last.agent)}">${esc(last.agent.toUpperCase())}</div>
        <div class="meta">conf ${fmt(last.confidence * 100)}% · difficulty ${esc(last.difficulty || '–')}</div>
        <div class="note"><span class="chip">${esc((last.source || '–').toUpperCase())}</span>
        <span>${last.source === 'laya' ? 'Laya chose this agent' : last.source === 'explore' ? 'trying an agent with few results at this difficulty' : 'the agent with the higher recent score'}</span></div>`;
}

function renderDecisions() {
    const rows = state.routing.slice().reverse();
    if (!rows.length) {
        $('decisions').innerHTML = '<div class="empty">The decision stream fills while <code>dino-ai play</code> runs.</div>';
        return;
    }
    $('decisions').innerHTML =
        '<div class="row headrow"><span>TIME</span><span>AGENT</span><span>DIFFICULTY</span><span>CONFIDENCE</span><span>SOURCE</span></div>' +
        rows.map((r, i) => `<div class="row${i === 0 ? ' first' : ''}">
            <span>${localTime(r.timestamp)}</span>
            <span style="color:${color(r.agent)}">${esc(r.agent)}</span>
            <span>${esc(r.difficulty || '–')}</span>
            <span>${fmt(r.confidence * 100)}%</span>
            <span>${esc(r.source || '–')}</span>
        </div>`).join('');
}

/** Seconds since a SQLite UTC timestamp */
function ageSeconds(ts) {
    return (Date.now() - new Date(ts.replace(' ', 'T') + 'Z').getTime()) / 1000;
}

/** Trace chart: one line per action between `min` and `max`, optional cut */
function traceChart(svg, series, { min, max, cut = null }) {
    const [, , W, H] = svg.getAttribute('viewBox').split(' ').map(Number);
    const pad = { l: 44, r: 8, t: 8, b: 8 };
    const span = max - min || 1;
    const n = Math.max(...series.map((s) => s.values.length), 2);
    const x = (i) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
    const y = (v) => H - pad.b - ((v - min) / span) * (H - pad.t - pad.b);
    let out = '';
    [min, (min + max) / 2, max].forEach((v) => {
        out += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(v)}" y2="${y(v)}" stroke="#1E2826"/>`;
        out += `<text x="${pad.l - 6}" y="${y(v) + 4}" fill="#8A9A95" font-size="10" text-anchor="end">${fmt(v, 2)}</text>`;
    });
    if (cut != null) {
        out += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(cut)}" y2="${y(cut)}" stroke="#F0605A" stroke-dasharray="5 4"/>`;
    }
    series.forEach((s) => {
        const points = s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
        out += `<polyline points="${points}" fill="none" stroke="${s.color}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
    });
    svg.innerHTML = out;
}

function renderLive() {
    const all = state.decisions;
    const section = document.querySelector('.live');
    if (!all.length) {
        $('liveAgent').textContent = '–';
        $('liveMeta').textContent = '';
        $('liveInput').innerHTML = '<span class="hint">Nothing is playing. Start a player to see what it reads, how it scores each action and what it presses:<br><code>dino-ai play --player laya --show</code><br><code>python -m src.training.experiment dqn-a2-eps-crop --watch</code></span>';
        ['liveBars', 'liveOutput', 'liveStream', 'liveLegend'].forEach((id) => { $(id).innerHTML = ''; });
        $('liveTrace').innerHTML = '';
        return;
    }
    const last = all[all.length - 1];
    const rows = all.filter((r) => r.agent === last.agent).slice(-TRACE_LENGTH);
    const labels = Object.keys(last.scores);
    const isQ = last.kind === 'q_value';
    const stale = ageSeconds(last.timestamp) > STALE_SECONDS;
    section.classList.toggle('stale', stale);

    $('liveAgent').innerHTML = `<span style="color:${color(last.agent)}">${esc(last.agent.toUpperCase())}</span>`;
    $('liveMeta').textContent = `${stale ? 'STOPPED · ' : ''}game ${last.game} · ${fmt(last.t, 1)}s · speed ${fmt(last.speed, 1)}`;

    $('liveInputTitle').textContent = isQ ? 'SITUATION' : 'INPUT · WHAT IT READS';
    $('liveInput').innerHTML = esc(last.input || '–') + (isQ
        ? '<span class="hint">The agent sees the last 4 frames as pixels, not this sentence; it is the game state at that moment, for reference.</span>'
        : '');

    // 02: probabilities against the cut, or Q-values relative to each other
    const values = labels.map((l) => last.scores[l]);
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    $('liveBarsTitle').textContent = isQ ? 'Q-VALUES · EXPECTED RETURN PER ACTION' : 'DETECTORS · P(YES)';
    $('liveBars').innerHTML = labels.map((l) => {
        const v = last.scores[l];
        const width = isQ ? 15 + 85 * ((v - lo) / ((hi - lo) || 1)) : v * 100;
        return `<div class="bar${l === last.answer ? ' chosen' : ''}">
            <span class="name">${esc(l.toUpperCase())}</span>
            <span class="track"><span class="fill" style="width:${width.toFixed(1)}%;background:${ACTION_COLORS[l] || '#E8A15A'}"></span>
            ${last.cut != null ? `<span class="cut" style="left:${last.cut * 100}%"></span>` : ''}</span>
            <span class="num">${fmt(v, isQ ? 2 : 3)}</span>
        </div>`;
    }).join('') + `<div class="note">${isQ
        ? 'the highest value is chosen (greedy)'
        : `decision cut ${fmt(last.cut, 2)}: yes at or above it`}</div>`;

    $('liveOutput').innerHTML = `
        <div class="action" style="color:${ACTION_COLORS[last.answer] || '#EAF2EF'}">${esc(last.answer.toUpperCase())}</div>
        <div class="meta">${last.exec !== last.answer ? `pressed <span class="chip">${esc(last.exec.toUpperCase())}</span> ` : 'pressed as answered · '}${fmt(last.ms, 0)}ms to decide</div>
        <div class="meta">${last.obstacle ? `${esc(last.obstacle)} at ${fmt(last.distance)}px` : 'nothing ahead'}</div>`;

    // 03: each action's score over the last decisions
    const series = labels.map((l) => ({ color: ACTION_COLORS[l] || '#E8A15A', values: rows.map((r) => r.scores[l]) }));
    const traced = series.flatMap((s) => s.values);
    traceChart($('liveTrace'), series, isQ
        ? { min: Math.min(...traced), max: Math.max(...traced) }
        : { min: 0, max: 1, cut: last.cut });
    $('liveLegend').innerHTML = labels.map((l) =>
        `<span><i style="background:${ACTION_COLORS[l] || '#E8A15A'}"></i>${esc(l)}</span>`).join('') +
        (isQ ? '' : '<span><i style="background:#F0605A"></i>cut</span>');

    // 05: newest first
    $('liveStream').innerHTML =
        '<div class="row headrow"><span>T</span><span>ANSWER</span><span>PRESSED</span><span>SCORES</span><span>OBSTACLE</span></div>' +
        rows.slice(-30).reverse().map((r, i) => `<div class="row${i === 0 ? ' first' : ''}">
            <span>${fmt(r.t, 2)}s</span>
            <span style="color:${ACTION_COLORS[r.answer] || '#EAF2EF'}">${esc(r.answer)}</span>
            <span>${esc(r.exec)}</span>
            <span>${labels.map((l) => `${esc(l)} ${fmt(r.scores[l], isQ ? 2 : 2)}`).join(' · ')}</span>
            <span>${r.obstacle ? `${esc(r.obstacle)} ${fmt(r.distance)}px` : '–'}</span>
        </div>`).join('');
}

function render() {
    renderQueued = false;
    renderLive();
    renderHeader();
    renderAgents();
    renderCharts();
    renderRouter();
    renderDecisions();
}

function queueRender() {
    if (!renderQueued) {
        renderQueued = true;
        requestAnimationFrame(render);
    }
}

// ---- connection -----------------------------------------------------------

function setLink(mode, text) {
    $('link').className = 'link ' + mode;
    $('linkText').textContent = text;
}

function handle(message) {
    if (message.type === 'snapshot') {
        Object.assign(state, message.history);
    } else if (message.type === 'rows') {
        state[message.table] = state[message.table].concat(message.rows).slice(-LIMITS[message.table]);
    }
    if (message.status) {
        state.status = message.status.agents;
        state.latest = message.latest;
    }
    state.updated = new Date();
    queueRender();
}

function connect() {
    setLink('', 'CONNECTING');
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.onopen = () => { retries = 0; setLink('live', 'LIVE'); };
    ws.onmessage = (event) => handle(JSON.parse(event.data));
    ws.onclose = () => {
        // back off 0.5s, 1s, 2s ... up to 8s; the server sends a fresh snapshot
        const delay = Math.min(8000, 500 * 2 ** retries++);
        setLink('reconnecting', `RECONNECTING IN ${Math.round(delay / 1000) || 1}s`);
        setTimeout(connect, delay);
    };
}

connect();
