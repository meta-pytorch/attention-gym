// Linked views of one static schedule. Item layers are cached separately from hover overlays.
const D = JSON.parse(document.getElementById("data").textContent);
const N = D.label.length;
const CTAS = D.rows.length;
const ctaOf = new Int32Array(N);
const slotOf = new Int32Array(N);
D.rows.forEach((row, c) => row.forEach((i, s) => { ctaOf[i] = c; slotOf[i] = s; }));
const SLOTS = D.rows.reduce((m, r) => Math.max(m, r.length), 1);
const critical = new Set(D.critical);
const state = { colorBy: D.colorBy, critical: false, timing: D.measured ? "both" : "predicted",
  focus: new Set(), hover: -1, window: null, drag: null };
const tip = document.getElementById("tip");
const views = [];
const LABEL_W = 62;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function rgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
const mix = (a, b, t) => a.map((x, k) => Math.round(x + (b[k] - x) * t));
const str = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
const RAMP = ["#2f5d73", "#5f7f67", "#c9a35a"].map(rgb);
let palette = {};
let revision = 0;
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
  return str(mix(palette.kinds[D.kind[i]], palette.paper, 0.6 * D.shade[i]));
}
function span(i, mode) {
  if (mode === "measured") {
    const m = D.measured[i];
    return m ? { cta: m[0], start: m[1], end: m[2] } : null;
  }
  return { cta: ctaOf[i], start: D.start[i], end: D.end[i] };
}
const modes = () => state.timing === "both" ? ["predicted", "measured"] : [state.timing];
const timedRows = {};
let timeMin = 0, timeMax = 1;
if (D.start) {
  for (const mode of D.measured ? ["predicted", "measured"] : ["predicted"]) {
    const rows = Array.from({ length: CTAS }, () => []);
    for (let i = 0; i < N; i++) {
      const s = span(i, mode);
      if (s) { rows[s.cta].push(i); timeMin = Math.min(timeMin, s.start); timeMax = Math.max(timeMax, s.end); }
    }
    rows.forEach((r) => r.sort((a, b) => span(a, mode).start - span(b, mode).start));
    timedRows[mode] = rows;
  }
}
const timeWindow = () => state.window || [timeMin, timeMax];
const shownCTAs = () => Array.from({ length: CTAS }, (_, c) => c).filter((c) => !state.focus.size || state.focus.has(c));
const rowH = (count) => Math.max(3, Math.min(state.focus.size ? 48 : 16, Math.floor(620 / Math.max(1, count))));
function sizeCanvas(c, w, h) {
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
  c.style.width = `${w}px`;
  c.style.height = `${h}px`;
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = "10px " + css("--mono");
  return ctx;
}
function addView(parent, title, name) {
  const el = document.createElement("div");
  el.className = "view";
  el.innerHTML = `<div class="figure"><p class="figlabel">${title}</p><canvas aria-label="${title}"></canvas></div>`;
  parent.appendChild(el);
  const v = { el, canvas: el.querySelector("canvas"), cache: document.createElement("canvas"), name };
  v.canvas.dataset.view = name;
  v.canvas.addEventListener("pointermove", (ev) => {
    const r = v.canvas.getBoundingClientRect();
    if (state.drag && state.drag.view === v) {
      state.drag.end = Math.max(LABEL_W, Math.min(v.w - 6, ev.clientX - r.left));
      draw();
    } else hover(v.hit(ev.clientX - r.left, ev.clientY - r.top), ev);
  });
  v.canvas.addEventListener("pointerleave", () => { if (!state.drag) hover(-1); });
  views.push(v);
  return v;
}
function draw() {
  for (const v of views) {
    const key = `${revision}:${v.el.clientWidth}:${window.devicePixelRatio}`;
    if (v.key !== key) {
      v.base();
      sizeCanvas(v.canvas, v.w, v.h);
      v.key = key;
    }
    const ctx = v.canvas.getContext("2d");
    ctx.clearRect(0, 0, v.w, v.h);
    ctx.drawImage(v.cache, 0, 0, v.w, v.h);
    v.overlay(ctx);
  }
}
function changed() { revision++; readout(); draw(); }
function focusCTA(cta, shift) {
  if (cta === undefined) return;
  if (shift) {
    if (state.focus.has(cta)) state.focus.delete(cta); else state.focus.add(cta);
  } else if (state.focus.size === 1 && state.focus.has(cta)) state.focus.clear();
  else state.focus = new Set([cta]);
  hover(-1);
  changed();
}
function resetView() {
  state.window = null; state.focus.clear(); state.drag = null;
  hover(-1); changed();
}
function rowLabels(ctx, v) {
  const every = Math.max(1, Math.ceil(12 / v.rh));
  ctx.fillStyle = palette.muted;
  ctx.textBaseline = "middle";
  // With two timing lanes per CTA, tall rows name each lane: both lanes are the same work list.
  const lanes = v.lh && v.lh >= 9 && v.lh < v.rh ? modes() : null;
  v.ctas.forEach((c, row) => {
    if (lanes) {
      lanes.forEach((mode, lane) => ctx.fillText(`${c} ${mode === "predicted" ? "pred" : "meas"}`,
        2, v.top + row * v.rh + lane * v.lh + v.lh / 2));
    } else if (row % every === 0) ctx.fillText(`CTA ${c}`, 2, v.top + row * v.rh + v.rh / 2);
  });
}
function rowIndex(v, cta) { return v.ctas.indexOf(cta); }
function highlightRow(ctx, v, cta) {
  const row = rowIndex(v, cta);
  if (row < 0) return;
  ctx.strokeStyle = palette.amber;
  ctx.lineWidth = 1;
  ctx.strokeRect(LABEL_W - 2, v.top + row * v.rh - 0.5, v.w - LABEL_W + 1, v.rh + 1);
}
function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(x));
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= x) || p * 10;
}
// Per-CTA spans do not overlap. The predecessor of the first start >= t is the only
// earlier item that can intersect the window. Sorting and indexing happen once at load.
function firstVisible(row, mode, t) {
  let lo = 0, hi = row.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (span(row[mid], mode).start < t) lo = mid + 1; else hi = mid;
  }
  return Math.max(0, lo - 1);
}
function setWindow(lo, hi) {
  const width = Math.min(timeMax - timeMin, Math.max((timeMax - timeMin) * 1e-6, hi - lo));
  lo = Math.max(timeMin, Math.min(timeMax - width, lo));
  state.window = width >= timeMax - timeMin ? null : [lo, lo + width];
  hover(-1); changed();
}
function timeline(parent) {
  const v = addView(parent, "Timeline", "timeline");
  v.x = (t) => LABEL_W + (t - v.lo) / (v.hi - v.lo) * (v.w - LABEL_W - 6);
  v.time = (x) => v.lo + (x - LABEL_W) / (v.w - LABEL_W - 6) * (v.hi - v.lo);
  v.y = (s, lane) => v.top + rowIndex(v, s.cta) * v.rh + lane * v.lh;
  v.base = () => {
    v.ctas = shownCTAs(); v.rh = rowH(v.ctas.length) * (state.timing === "both" ? 2 : 1);
    v.lh = v.rh / modes().length; v.top = 6;
    v.w = Math.max(180, v.el.clientWidth - 26); v.h = v.top + v.ctas.length * v.rh + 28;
    [v.lo, v.hi] = timeWindow();
    const ctx = sizeCanvas(v.cache, v.w, v.h);
    v.el.querySelector(".figlabel").textContent = "Timeline · " + (state.timing === "both"
      ? "top lane predicted, bottom lane measured" : state.timing);
    ctx.save(); ctx.beginPath(); ctx.rect(LABEL_W, v.top, v.w - LABEL_W - 6, v.h - v.top - 28); ctx.clip();
    modes().forEach((mode, lane) => v.ctas.forEach((c) => {
      const row = timedRows[mode][c];
      for (let j = firstVisible(row, mode, v.lo); j < row.length; j++) {
        const i = row[j], s = span(i, mode);
        if (s.start > v.hi) break;
        if (s.end < v.lo) continue;
        ctx.globalAlpha = state.critical && !critical.has(i) ? 0.22 : 1;
        ctx.fillStyle = color(i);
        ctx.fillRect(v.x(s.start), v.y(s, lane), Math.max(1, v.x(s.end) - v.x(s.start) - 0.4), v.lh - (v.lh > 4));
      }
    }));
    ctx.globalAlpha = 1;
    if (state.critical && state.timing !== "measured") {
      // Outline each critical item; an arrow runs from each one's end to the next one's start.
      const shown = (s) => rowIndex(v, s.cta) >= 0 && s.end >= v.lo && s.start <= v.hi;
      let prev = null;
      D.critical.forEach((i) => {
        const s = span(i, "predicted");
        if (shown(s)) {
          ctx.strokeStyle = palette.ink; ctx.lineWidth = 1.5;
          ctx.strokeRect(v.x(s.start), v.y(s, 0), Math.max(2, v.x(s.end) - v.x(s.start)), v.lh);
          if (prev && shown(prev)) {
            arrow(ctx, v.x(prev.end), v.y(prev, 0) + v.lh / 2, v.x(s.start), v.y(s, 0) + v.lh / 2);
          }
        }
        prev = s;
      });
    }
    ctx.restore(); rowLabels(ctx, v);
    const step = niceStep((v.hi - v.lo) / 8);
    ctx.fillStyle = palette.muted; ctx.strokeStyle = palette.rule; ctx.textBaseline = "top";
    for (let t = Math.ceil(v.lo / step) * step; t <= v.hi && v.x(t) < v.w - 105; t += step) {
      ctx.beginPath(); ctx.moveTo(v.x(t), v.h - 26); ctx.lineTo(v.x(t), v.h - 22); ctx.stroke();
      ctx.fillText(`${+t.toFixed(6)}`, v.x(t) - 4, v.h - 20);
    }
    ctx.fillText(`time (${D.unit}) →`, v.w - 90, v.h - 20);
  };
  v.overlay = (ctx) => {
    if (state.hover >= 0) {
      modes().forEach((mode, lane) => {
        const i = state.hover, s = span(i, mode);
        if (!s || rowIndex(v, s.cta) < 0) return;
        highlightRow(ctx, v, s.cta);
        ctx.save(); ctx.beginPath(); ctx.rect(LABEL_W, v.top, v.w - LABEL_W - 6, v.h - v.top - 28); ctx.clip();
        for (const d of (D.deps && D.deps[i]) || []) {
          const p = span(d, mode);
          if (p && rowIndex(v, p.cta) >= 0) arrow(ctx, v.x(p.end), v.y(p, lane) + v.lh / 2, v.x(s.start), v.y(s, lane) + v.lh / 2);
        }
        ctx.strokeStyle = palette.ink; ctx.lineWidth = 1.5;
        ctx.strokeRect(v.x(s.start) - 1, v.y(s, lane) - 1, Math.max(2, v.x(s.end) - v.x(s.start)) + 2, v.lh + 1);
        ctx.restore();
      });
    }
    if (state.drag && state.drag.view === v) {
      const { start, end } = state.drag;
      ctx.fillStyle = palette.amber; ctx.globalAlpha = 0.2;
      ctx.fillRect(Math.min(start, end), v.top, Math.abs(end - start), v.h - v.top - 28);
      ctx.globalAlpha = 1;
    }
  };
  v.ctaAt = (y) => v.ctas[Math.floor((y - v.top) / v.rh)];
  v.hit = (px, py) => {
    const c = v.ctaAt(py);
    if (c === undefined || px < LABEL_W || px > v.w - 6) return -1;
    const lane = Math.floor(((py - v.top) % v.rh) / v.lh);
    const mode = modes()[lane], t = v.time(px), row = timedRows[mode][c];
    const j = firstVisible(row, mode, t);
    for (let k = j; k < Math.min(j + 2, row.length); k++) {
      const s = span(row[k], mode);
      if (s.start <= t && t <= s.end) return row[k];
    }
    return -1;
  };
  v.canvas.title = "Drag horizontally to zoom; double-click to reset. Ctrl/pinch-wheel zooms at the cursor. Shift-wheel or horizontal scroll pans. Click a CTA to focus; Shift-click selects several.";
  v.canvas.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    const r = v.canvas.getBoundingClientRect(), px = ev.clientX - r.left;
    state.drag = { view: v, start: Math.max(LABEL_W, Math.min(v.w - 6, px)), end: Math.max(LABEL_W, Math.min(v.w - 6, px)), cta: v.ctaAt(ev.clientY - r.top), label: px < LABEL_W };
    v.canvas.setPointerCapture(ev.pointerId); hover(-1);
  });
  v.canvas.addEventListener("pointerup", (ev) => {
    const drag = state.drag;
    if (!drag || drag.view !== v) return;
    state.drag = null;
    v.canvas.releasePointerCapture(ev.pointerId);
    if (!drag.label && Math.abs(drag.end - drag.start) > 5) setWindow(v.time(Math.min(drag.start, drag.end)), v.time(Math.max(drag.start, drag.end)));
    else focusCTA(drag.cta, ev.shiftKey);
    draw();
  });
  v.canvas.addEventListener("pointercancel", () => { state.drag = null; draw(); });
  v.canvas.addEventListener("dblclick", resetView);
  v.canvas.addEventListener("wheel", (ev) => {
    if (!ev.ctrlKey && !ev.shiftKey && Math.abs(ev.deltaX) <= Math.abs(ev.deltaY)) return;
    ev.preventDefault();
    const factor = ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? v.w : 1;
    if (ev.ctrlKey) {
      const r = v.canvas.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, (ev.clientX - r.left - LABEL_W) / (v.w - LABEL_W - 6)));
      const anchor = v.lo + fraction * (v.hi - v.lo);
      const width = Math.min(timeMax - timeMin, Math.max((timeMax - timeMin) * 1e-6, (v.hi - v.lo) * Math.exp(ev.deltaY * factor * 0.002)));
      setWindow(anchor - fraction * width, anchor + (1 - fraction) * width);
    } else {
      const delta = (ev.deltaX || ev.deltaY) * factor / (v.w - LABEL_W - 6) * (v.hi - v.lo);
      setWindow(v.lo + delta, v.hi + delta);
    }
  }, { passive: false });
}
// The critical-path panel. Consecutive path items of one kind on one CTA form a segment (a CTA
// working through its queue); a new segment starts where an item waits for another CTA's output.
function criticalPanel() {
  const el = document.getElementById("path");
  el.hidden = !state.critical;
  if (!state.critical || el.dataset.built) return;
  el.dataset.built = "1";
  const fmt = (t) => (D.unit === "us" && t >= 1000 ? (t / 1000).toFixed(2) + " ms" : t.toFixed(1) + " " + D.unit);
  const segs = [];
  D.critical.forEach((i) => {
    const last = segs[segs.length - 1];
    if (last && D.kind[last.items[0]] === D.kind[i] && ctaOf[last.items[0]] === ctaOf[i]) last.items.push(i);
    else segs.push({ items: [i] });
  });
  const total = D.end[D.critical[D.critical.length - 1]];
  const busy = new Map();
  let waiting = 0;
  // A segment's run time sums its items; gaps before and inside it are waiting.
  segs.forEach((s, k) => {
    const first = s.items[0], last = s.items[s.items.length - 1];
    s.start = D.start[first]; s.end = D.end[last];
    s.busy = s.items.reduce((t, i) => t + D.end[i] - D.start[i], 0);
    s.wait = (k ? Math.max(0, s.start - segs[k - 1].end) : s.start) + (s.end - s.start - s.busy);
    waiting += s.wait;
    busy.set(D.kind[first], (busy.get(D.kind[first]) || 0) + s.busy);
  });
  // Built with DOM nodes and textContent, never HTML strings: labels come from page data.
  const node = (tag, props = {}, ...children) => {
    const n = document.createElement(tag);
    Object.assign(n, props);
    n.append(...children);
    return n;
  };
  const swatch = (kind) => {
    const i = node("i", { className: "sw" });
    i.style.background = D.colors[kind];
    return i;
  };
  const bar = node("div", { className: "bar" });
  const key = node("div", { className: "legend" });
  for (const [kind, t] of busy) {
    const part = node("span", { title: D.kinds[kind] + " " + fmt(t) });
    part.style.flex = String(t);
    part.style.background = D.colors[kind];
    bar.append(part);
    key.append(node("span", {}, swatch(kind), D.kinds[kind] + " " + Math.round((100 * t) / total) + "%"));
  }
  const waitPart = node("span", { className: "wait", title: "waiting " + fmt(waiting) });
  waitPart.style.flex = String(waiting);
  bar.append(waitPart);
  key.append(node("span", {}, node("i", { className: "sw wait" }),
    "waiting " + Math.round((100 * waiting) / total) + "%"));
  const body = node("tbody");
  segs.forEach((s, k) => {
    const first = s.items[0];
    const items = node("td", {}, swatch(D.kind[first]), D.label[first].split("\n")[0]);
    if (s.items.length > 1) items.append(" ", node("span", { className: "more" }, "+" + (s.items.length - 1) + " more"));
    const cells = [String(k + 1), null, String(ctaOf[first]), fmt(s.start), fmt(s.busy), s.wait > 0 ? fmt(s.wait) : "—"];
    const tr = node("tr", {}, ...cells.map((c) => (c === null ? items : node("td", {}, c))));
    tr.addEventListener("click", () => {
      const pad = Math.max(s.end - s.start, 1) * 0.5;
      setWindow(s.start - s.wait - pad, s.end + pad);
      body.querySelectorAll("tr.on").forEach((o) => o.classList.remove("on"));
      tr.classList.add("on");
    });
    body.append(tr);
  });
  const head = node("tr", {}, ...["#", "items", "CTA", "starts", "runs for", "waited"].map((h) => node("th", {}, h)));
  el.replaceChildren(
    node("p", {}, node("b", {}, "Predicted critical path"),
      " · " + fmt(total) + " · " + D.critical.length + " items in " + segs.length + " segments. " +
      "A segment is one CTA running items back to back; between segments the path waits for " +
      "another CTA's result. Click a row to zoom to it."),
    bar, key,
    node("div", { className: "table" }, node("table", {}, node("thead", {}, head), body)));
}
function arrow(ctx, x0, y0, x1, y1) {
  ctx.strokeStyle = palette.amber; ctx.fillStyle = palette.amber; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(x0, y0);
  const mx = (x0 + x1) / 2;
  ctx.bezierCurveTo(mx, y0, mx, y1, x1, y1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 6, y1 - 3); ctx.lineTo(x1 - 6, y1 + 3); ctx.fill();
}
function grid(parent) {
  const v = addView(parent, "Work order per CTA · columns are order, not time", "grid");
  v.base = () => {
    v.ctas = shownCTAs(); v.rh = rowH(v.ctas.length); v.top = 6;
    v.cw = Math.max(0.1, Math.min(40, (v.el.clientWidth - 26 - LABEL_W - 6) / SLOTS));
    v.w = LABEL_W + v.cw * SLOTS + 6; v.h = v.top + v.ctas.length * v.rh + 24;
    const ctx = sizeCanvas(v.cache, v.w, v.h);
    v.ctas.forEach((c, row) => D.rows[c].forEach((i, s) => {
      ctx.fillStyle = color(i);
      ctx.fillRect(LABEL_W + s * v.cw, v.top + row * v.rh, Math.max(0.5, v.cw - (v.cw > 4)), v.rh - (v.rh > 4));
    }));
    rowLabels(ctx, v); ctx.fillStyle = palette.muted; ctx.textBaseline = "top";
    const every = Math.max(1, Math.ceil(36 / v.cw));
    for (let s = 0; s < SLOTS; s += every) ctx.fillText(`${s}`, LABEL_W + s * v.cw, v.h - 18);
  };
  v.overlay = (ctx) => {
    const i = state.hover;
    if (i < 0 || rowIndex(v, ctaOf[i]) < 0) return;
    highlightRow(ctx, v, ctaOf[i]); ctx.strokeStyle = palette.ink; ctx.lineWidth = 1.5;
    ctx.strokeRect(LABEL_W + slotOf[i] * v.cw - 1, v.top + rowIndex(v, ctaOf[i]) * v.rh - 1, v.cw + 1, v.rh + 1);
  };
  v.ctaAt = (y) => v.ctas[Math.floor((y - v.top) / v.rh)];
  v.hit = (px, py) => {
    const c = v.ctaAt(py), s = Math.floor((px - LABEL_W) / v.cw);
    return c !== undefined && s >= 0 && s < D.rows[c].length ? D.rows[c][s] : -1;
  };
  v.canvas.title = "Click a CTA row to show it alone; click it again to show all. Shift-click adds or removes a CTA. Slot numbers give execution order, not duration.";
  v.canvas.addEventListener("click", (ev) => focusCTA(v.ctaAt(ev.clientY - v.canvas.getBoundingClientRect().top), ev.shiftKey));
}
function map(parent) {
  const v = addView(parent, "Work space · row ↓, column →", "map");
  const R = 1 + D.coord.reduce((m, c) => Math.max(m, c[0]), 0);
  const C = 1 + D.coord.reduce((m, c) => Math.max(m, c[1]), 0);
  const at = new Int32Array(R * C).fill(-1);
  D.coord.forEach(([r, c], i) => { at[r * C + c] = i; });
  v.base = () => {
    v.cell = Math.max(2, Math.min(18, (v.el.clientWidth - 30) / C, 640 / R));
    v.w = v.cell * C + 4; v.h = v.cell * R + 4;
    const ctx = sizeCanvas(v.cache, v.w, v.h);
    for (let i = 0; i < N; i++) {
      const [r, c] = D.coord[i];
      ctx.globalAlpha = state.focus.size && !state.focus.has(ctaOf[i]) ? 0.2 : 1;
      ctx.fillStyle = color(i);
      ctx.fillRect(2 + c * v.cell, 2 + r * v.cell, v.cell - (v.cell > 4), v.cell - (v.cell > 4));
    }
    ctx.globalAlpha = 1;
  };
  v.overlay = (ctx) => {
    if (state.hover < 0) return;
    ctx.strokeStyle = palette.ink; ctx.lineWidth = 1;
    D.rows[ctaOf[state.hover]].forEach((i) => {
      const [r, c] = D.coord[i];
      ctx.strokeRect(2 + c * v.cell + 0.5, 2 + r * v.cell + 0.5, v.cell - 1, v.cell - 1);
    });
  };
  v.hit = (px, py) => {
    const c = Math.floor((px - 2) / v.cell), r = Math.floor((py - 2) / v.cell);
    return r >= 0 && r < R && c >= 0 && c < C ? at[r * C + c] : -1;
  };
  v.canvas.title = "Hover an item to outline its CTA's work in all views. Colour follows the selected legend; dim items belong to unfocused CTAs.";
}
const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
function escapeText(s) {
  return String(s).replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]);
}
const signed = (n) => `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;
function hover(i, ev) {
  if (i === state.hover) { if (i >= 0 && ev) place(ev); return; }
  state.hover = i; draw();
  if (i < 0) { tip.hidden = true; return; }
  const [head, ...rest] = D.label[i].split("\n");
  const lines = [...rest, `CTA ${ctaOf[i]} · slot ${slotOf[i]} (execution order)`];
  if (D.start) lines.push(`Predicted (model): ${D.start[i].toFixed(2)} → ${D.end[i].toFixed(2)} ${D.unit}; duration ${(D.end[i] - D.start[i]).toFixed(2)} ${D.unit}`);
  const m = D.measured && D.measured[i];
  if (m) {
    lines.push(`Measured (real launch): ${m[1].toFixed(2)} → ${m[2].toFixed(2)} ${D.unit}; duration ${(m[2] - m[1]).toFixed(2)} ${D.unit} · CTA ${m[0]}`);
    lines.push(`Measured − predicted: start shift ${signed(m[1] - D.start[i])} ${D.unit}; duration delta ${signed((m[2] - m[1]) - (D.end[i] - D.start[i]))} ${D.unit}`);
  } else if (D.measured) lines.push("Measured: no captured span for this item");
  if (critical.has(i)) lines.push("On the predicted critical path (model only)");
  const title = document.createElement("b");
  title.textContent = head;
  tip.replaceChildren(title, lines.join("\n"));
  tip.hidden = false; if (ev) place(ev);
}
function place(ev) {
  const left = Math.min(ev.pageX + 14, window.scrollX + document.documentElement.clientWidth - tip.offsetWidth - 8);
  tip.style.left = `${Math.max(8, left)}px`; tip.style.top = `${ev.pageY + 16}px`;
}
function legend() {
  const el = document.getElementById("legend");
  if (state.colorBy === "kind") el.innerHTML = D.kinds.map((k, j) => `<span><i class="sw" style="background:${escapeText(D.colors[j])}"></i>${escapeText(k || "work item")}</span>`).join("");
  else if (state.colorBy === "slot") {
    const stops = [0, 0.5, 1].map((t) => str(ramp(t))).join(",");
    el.innerHTML = `<span>first <i class="sw" style="width:90px;background:linear-gradient(90deg,${stops})"></i> last in each CTA's list</span>`;
  }
}
function readout() {
  const selector = document.getElementById("add-cta");
  selector.hidden = !state.focus.size;
  selector.innerHTML = '<option value="">Add CTA…</option>' + Array.from({ length: CTAS }, (_, c) =>
    `<option value="${c}"${state.focus.has(c) ? " disabled" : ""}>CTA ${c}</option>`).join("");
  const fmt = (t) => (D.unit === "us" && t >= 1000 ? `${(t / 1000).toFixed(2)} ms` : `${t.toFixed(1)} ${D.unit}`);
  const parts = [`${CTAS} CTAs`, `${N.toLocaleString()} items`];
  if (D.start) parts.push(`predicted ${fmt(D.end.reduce((m, e) => Math.max(m, e), 0))}`);
  if (D.measured) parts.push(`measured ${fmt(Object.values(D.measured).reduce((m, s) => Math.max(m, s[2]), 0))}`);
  const view = [];
  if (state.focus.size) view.push(`CTA ${[...state.focus].sort((a, b) => a - b).join(", ")}`);
  const [lo, hi] = D.start ? timeWindow() : [0, 0];
  const zoomed = Boolean(D.start && state.window);
  if (zoomed) view.push(`${fmt(lo)} – ${fmt(hi)}`);
  document.getElementById("readout").textContent =
    parts.join(" · ") + (view.length ? `   ·   showing ${view.join(", ")}` : "");
  document.getElementById("reset-view").hidden = !state.focus.size && !zoomed;
}
function setup() {
  refreshPalette();
  const root = document.getElementById("views");
  if (D.start) timeline(root);
  const holder = D.coord ? Object.assign(document.createElement("div"), { className: "pair" }) : root;
  if (D.coord) root.appendChild(holder);
  grid(holder); if (D.coord) map(holder);
  for (const [id, key] of [["color-by", "colorBy"], ["timing", "timing"]]) {
    document.querySelectorAll(`#${id} button`).forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.v === state[key]));
      b.addEventListener("click", () => {
        state[key] = b.dataset.v;
        document.querySelectorAll(`#${id} button`).forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
        hover(-1); legend(); changed();
      });
    });
  }
  document.getElementById("timing-controls").hidden = !D.measured;
  const b = document.getElementById("critical");
  b.hidden = !D.start || !D.critical.length; b.setAttribute("aria-pressed", "false");
  b.addEventListener("click", () => {
    state.critical = !state.critical; b.setAttribute("aria-pressed", String(state.critical));
    criticalPanel(); changed();
  });
  document.getElementById("reset-view").addEventListener("click", resetView);
  document.getElementById("add-cta").addEventListener("change", (ev) => {
    if (ev.target.value !== "") focusCTA(Number(ev.target.value), true);
  });
  legend(); readout(); draw();
}
setup();
window.addEventListener("resize", draw);
document.addEventListener("themechange", () => { refreshPalette(); changed(); });
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") {
    if (state.drag) { state.drag = null; draw(); }
    else if (state.hover >= 0) hover(-1);
    else resetView();
  }
});
