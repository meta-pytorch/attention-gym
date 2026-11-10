// schedule_html: CTA x slot grid, timeline and work-space map of one static schedule.
const D = JSON.parse(document.getElementById("data").textContent);
const N = D.label.length;
const CTAS = D.rows.length;
const ctaOf = new Int32Array(N);
const slotOf = new Int32Array(N);
D.rows.forEach((row, c) => row.forEach((i, s) => { ctaOf[i] = c; slotOf[i] = s; }));
const SLOTS = D.rows.reduce((m, r) => Math.max(m, r.length), 1);
const critical = new Set(D.critical);
const state = { colorBy: D.colorBy, critical: false, measured: false, hover: -1 };
const tip = document.getElementById("tip");
const views = [];

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function rgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
const mix = (a, b, t) => a.map((x, k) => Math.round(x + (b[k] - x) * t));
const str = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
const RAMP = ["#2f5d73", "#5f7f67", "#c9a35a"].map(rgb);
let palette = {};
function refreshPalette() {
  palette = { paper: rgb(css("--paper")), ink: css("--ink"),
    muted: css("--muted"), rule: css("--rule"), amber: css("--amber") };
  palette.kinds = D.colors.map(rgb);
}
function ramp(t) {
  const x = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
  const k = Math.min(RAMP.length - 2, Math.floor(x));
  return mix(RAMP[k], RAMP[k + 1], x - k);
}
function color(i) {
  if (state.colorBy === "slot") return str(ramp(SLOTS > 1 ? slotOf[i] / (SLOTS - 1) : 0));
  if (state.colorBy === "cta") return `hsl(${(ctaOf[i] * 137.508) % 360}, 38%, 52%)`;
  return str(mix(palette.kinds[D.kind[i]], palette.paper, 0.6 * D.shade[i]));
}

function span(i) {
  if (state.measured) {
    const m = D.measured[i];
    return m ? { cta: m[0], start: m[1], end: m[2] } : null;
  }
  return { cta: ctaOf[i], start: D.start[i], end: D.end[i] };
}
function spanRows() {
  const rows = Array.from({ length: CTAS }, () => []);
  for (let i = 0; i < N; i++) {
    const s = span(i);
    if (s) rows[s.cta].push(i);
  }
  rows.forEach((r) => r.sort((a, b) => span(a).start - span(b).start));
  return rows;
}

function canvas(view, w, h) {
  const dpr = window.devicePixelRatio || 1;
  const c = view.canvas;
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
  c.style.width = `${w}px`;
  c.style.height = `${h}px`;
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "10px " + css("--mono");
  return ctx;
}
function addView(parent, title) {
  const view = document.createElement("div");
  view.className = "view";
  view.innerHTML = `<div class="figure"><p class="figlabel">${title}</p><canvas></canvas></div>`;
  parent.appendChild(view);
  const v = { el: view, canvas: view.querySelector("canvas") };
  v.canvas.addEventListener("mousemove", (ev) => {
    const r = v.canvas.getBoundingClientRect();
    hover(v.hit(ev.clientX - r.left, ev.clientY - r.top), ev);
  });
  v.canvas.addEventListener("mouseleave", () => hover(-1));
  views.push(v);
  return v;
}
const rowH = () => Math.max(3, Math.min(16, Math.floor(620 / CTAS)));
const LABEL_W = 52;

function rowLabels(ctx, top, rh) {
  const every = Math.max(1, Math.ceil(12 / rh));
  ctx.fillStyle = palette.muted;
  ctx.textBaseline = "middle";
  for (let c = 0; c < CTAS; c += every) ctx.fillText(`cta ${c}`, 2, top + c * rh + rh / 2);
}
function highlightRow(ctx, top, rh, w, cta) {
  ctx.strokeStyle = palette.amber;
  ctx.lineWidth = 1;
  ctx.strokeRect(LABEL_W - 2, top + cta * rh - 0.5, w - LABEL_W + 1, rh + 1);
}
function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(x));
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= x) || p * 10;
}

function timeline(parent) {
  const v = addView(parent, "timeline · one row per CTA · predicted start → end");
  v.draw = () => {
    const w = parent.clientWidth - 26;
    const rh = rowH();
    const top = 6;
    const h = top + CTAS * rh + 28;
    const ctx = canvas(v, w, h);
    v.el.querySelector(".figlabel").textContent = "timeline · one row per CTA · " +
      (state.measured ? "measured start → end (capture)" : "predicted start → end");
    v.rows = spanRows();
    let tmax = 0;
    v.rows.forEach((r) => r.forEach((i) => { tmax = Math.max(tmax, span(i).end); }));
    tmax = tmax || 1;
    const x = (t) => LABEL_W + (t / tmax) * (w - LABEL_W - 6);
    v.top = top; v.rh = rh; v.tmax = tmax; v.w = w;
    const dim = state.critical && critical.size;
    v.rows.forEach((r, c) => r.forEach((i) => {
      const s = span(i);
      ctx.globalAlpha = dim && !critical.has(i) ? 0.22 : 1;
      ctx.fillStyle = color(i);
      ctx.fillRect(x(s.start), top + c * rh, Math.max(1, x(s.end) - x(s.start) - 0.4), rh - (rh > 4));
    }));
    ctx.globalAlpha = 1;
    rowLabels(ctx, top, rh);
    const step = niceStep(tmax / 8);
    ctx.fillStyle = palette.muted;
    ctx.strokeStyle = palette.rule;
    ctx.textBaseline = "top";
    for (let t = 0; x(t) < w - 110; t += step) {
      ctx.beginPath();
      ctx.moveTo(x(t), top + CTAS * rh + 2);
      ctx.lineTo(x(t), top + CTAS * rh + 6);
      ctx.stroke();
      ctx.fillText(`${+t.toFixed(3)}`, x(t) - 4, top + CTAS * rh + 8);
    }
    ctx.fillText(`time (${D.unit}) →`, w - 80, top + CTAS * rh + 8);
    if (dim) {
      const path = D.critical.filter((i) => span(i));
      ctx.strokeStyle = palette.amber;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      path.forEach((i, k) => {
        const s = span(i);
        const y = top + s.cta * rh + rh / 2;
        if (k) ctx.lineTo(x(s.start), y); else ctx.moveTo(x(s.start), y);
        ctx.lineTo(x(s.end), y);
      });
      ctx.stroke();
    }
    if (state.hover >= 0 && span(state.hover)) {
      const i = state.hover;
      const s = span(i);
      highlightRow(ctx, top, rh, w, s.cta);
      for (const d of (D.deps && D.deps[i]) || []) {
        const p = span(d);
        if (p) arrow(ctx, x(p.end), top + p.cta * rh + rh / 2, x(s.start), top + s.cta * rh + rh / 2);
      }
      ctx.strokeStyle = palette.ink;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x(s.start) - 1, top + s.cta * rh - 1, Math.max(2, x(s.end) - x(s.start)) + 2, rh + 1);
    }
  };
  v.hit = (px, py) => {
    const c = Math.floor((py - v.top) / v.rh);
    if (c < 0 || c >= CTAS || px < LABEL_W) return -1;
    const t = ((px - LABEL_W) / (v.w - LABEL_W - 6)) * v.tmax;
    const r = v.rows[c];
    let lo = 0;
    let hi = r.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const s = span(r[mid]);
      if (t < s.start) hi = mid - 1;
      else if (t > s.end) lo = mid + 1;
      else return r[mid];
    }
    return -1;
  };
}
function arrow(ctx, x0, y0, x1, y1) {
  ctx.strokeStyle = palette.amber;
  ctx.fillStyle = palette.amber;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  const mx = (x0 + x1) / 2;
  ctx.bezierCurveTo(mx, y0, mx, y1, x1, y1);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - 6, y1 - 3);
  ctx.lineTo(x1 - 6, y1 + 3);
  ctx.fill();
}

function grid(parent) {
  const v = addView(parent, "cta × slot · slot = position in the CTA's list");
  v.draw = () => {
    const w = parent.clientWidth - 26;
    const rh = rowH();
    const cw = Math.max(1, Math.min(40, (w - LABEL_W - 4) / SLOTS));
    const top = 6;
    const ctx = canvas(v, Math.min(w, LABEL_W + cw * SLOTS + 6), top + CTAS * rh + 24);
    Object.assign(v, { rh, cw, top });
    D.rows.forEach((row, c) => row.forEach((i, s) => {
      ctx.fillStyle = color(i);
      ctx.fillRect(LABEL_W + s * cw, top + c * rh, Math.max(1, cw - (cw > 4)), rh - (rh > 4));
    }));
    rowLabels(ctx, top, rh);
    ctx.fillStyle = palette.muted;
    ctx.textBaseline = "top";
    const every = Math.max(1, Math.ceil(28 / cw));
    for (let s = 0; s < SLOTS; s += every) ctx.fillText(`${s}`, LABEL_W + s * cw, top + CTAS * rh + 6);
    if (state.hover >= 0) {
      const i = state.hover;
      highlightRow(ctx, top, rh, LABEL_W + cw * SLOTS + 2, ctaOf[i]);
      ctx.strokeStyle = palette.ink;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(LABEL_W + slotOf[i] * cw - 1, top + ctaOf[i] * rh - 1, cw + 1, rh + 1);
    }
  };
  v.hit = (px, py) => {
    const c = Math.floor((py - v.top) / v.rh);
    const s = Math.floor((px - LABEL_W) / v.cw);
    return c >= 0 && c < CTAS && s >= 0 && s < D.rows[c].length ? D.rows[c][s] : -1;
  };
}

function map(parent) {
  const v = addView(parent, "work space · row ↓, column →");
  const R = 1 + D.coord.reduce((m, c) => Math.max(m, c[0]), 0);
  const C = 1 + D.coord.reduce((m, c) => Math.max(m, c[1]), 0);
  const at = new Int32Array(R * C).fill(-1);
  D.coord.forEach(([r, c], i) => { at[r * C + c] = i; });
  v.draw = () => {
    const w = parent.clientWidth - 26;
    const cell = Math.max(2, Math.min(18, (w - 4) / C, 640 / R));
    const ctx = canvas(v, cell * C + 4, cell * R + 4);
    v.cell = cell;
    const hc = state.hover >= 0 ? ctaOf[state.hover] : -1;
    for (let i = 0; i < N; i++) {
      const [r, c] = D.coord[i];
      ctx.globalAlpha = hc >= 0 && ctaOf[i] !== hc ? 0.35 : 1;
      ctx.fillStyle = color(i);
      ctx.fillRect(2 + c * cell, 2 + r * cell, cell - (cell > 4), cell - (cell > 4));
    }
    ctx.globalAlpha = 1;
    if (hc >= 0) {
      ctx.strokeStyle = palette.ink;
      ctx.lineWidth = 1;
      D.rows[hc].forEach((i) => {
        const [r, c] = D.coord[i];
        ctx.strokeRect(2 + c * cell + 0.5, 2 + r * cell + 0.5, cell - 1, cell - 1);
      });
    }
  };
  v.hit = (px, py) => {
    const c = Math.floor((px - 2) / v.cell);
    const r = Math.floor((py - 2) / v.cell);
    return r >= 0 && r < R && c >= 0 && c < C ? at[r * C + c] : -1;
  };
}

function escapeText(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
function hover(i, ev) {
  if (i === state.hover) {
    if (i >= 0 && ev) place(ev);
    return;
  }
  state.hover = i;
  draw();
  if (i < 0) { tip.hidden = true; return; }
  const [head, ...rest] = D.label[i].split("\n");
  const lines = [...rest, `cta ${ctaOf[i]} · slot ${slotOf[i]}`];
  if (D.start) lines.push(`predicted ${D.start[i].toFixed(2)} → ${D.end[i].toFixed(2)} ${D.unit}`);
  const m = D.measured && D.measured[i];
  if (m) lines.push(`measured ${m[1].toFixed(2)} → ${m[2].toFixed(2)} ${D.unit} on cta ${m[0]}`);
  if (critical.has(i)) lines.push("on the predicted critical path");
  tip.innerHTML = `<b>${escapeText(head)}</b>${lines.map(escapeText).join("\n")}`;
  tip.hidden = false;
  place(ev);
}
function place(ev) {
  const left = Math.min(ev.pageX + 14, window.scrollX + document.documentElement.clientWidth - tip.offsetWidth - 8);
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${ev.pageY + 16}px`;
}
function draw() { views.forEach((v) => v.draw()); }

function legend() {
  const el = document.getElementById("legend");
  if (state.colorBy === "kind") {
    el.innerHTML = D.kinds.map((k, j) => `<span><i class="sw" style="background:${D.colors[j]}"></i>${escapeText(k || "item")}</span>`).join("");
  } else if (state.colorBy === "slot") {
    const stops = [0, 0.5, 1].map((t) => str(ramp(t))).join(",");
    el.innerHTML = `<span>slot 0 <i class="sw" style="width:90px;background:linear-gradient(90deg,${stops})"></i> slot ${SLOTS - 1}</span>`;
  } else {
    el.innerHTML = "<span>one hue per CTA</span>";
  }
}
function readout() {
  const counts = D.rows.map((r) => r.length);
  const full = counts.filter((c) => c === SLOTS).length;
  const parts = [`${CTAS} CTAs`, `${N} items`, `${counts.reduce((m, c) => Math.min(m, c), SLOTS)}–${SLOTS} per CTA`,
    `last slot: ${full}/${CTAS} CTAs busy`];
  if (D.start) parts.push(`predicted makespan ${D.end.reduce((m, e) => Math.max(m, e), 0).toFixed(1)} ${D.unit}`);
  if (D.critical.length) parts.push(`critical path ${D.critical.length} items`);
  if (D.measured) {
    const end = Object.values(D.measured).reduce((m, s) => Math.max(m, s[2]), 0);
    parts.push(`measured span ${end.toFixed(1)} ${D.unit}`);
  }
  document.getElementById("readout").textContent = parts.join(" · ");
}

function setup() {
  refreshPalette();
  const root = document.getElementById("views");
  if (D.start) timeline(root);
  const holder = D.coord ? Object.assign(document.createElement("div"), { className: "pair" }) : root;
  if (D.coord) root.appendChild(holder);
  grid(holder);
  if (D.coord) map(holder);
  document.querySelectorAll("#color-by button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.v === state.colorBy));
    b.addEventListener("click", () => {
      state.colorBy = b.dataset.v;
      document.querySelectorAll("#color-by button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      legend();
      draw();
    });
  });
  const toggle = (key, available) => {
    const b = document.getElementById(key);
    b.hidden = !available;
    b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => {
      state[key] = !state[key];
      b.setAttribute("aria-pressed", String(state[key]));
      draw();
    });
  };
  toggle("critical", Boolean(D.start && D.critical.length));
  toggle("measured", Boolean(D.start && D.measured));
  legend();
  readout();
  draw();
}
setup();
window.addEventListener("resize", draw);
document.addEventListener("themechange", () => { refreshPalette(); draw(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") hover(-1); });
