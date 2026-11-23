// Renders the relations describe_html.py computed. Hover lights a node's neighbours; click,
// Enter or the table selects it into the inspector; Escape or an empty-space click clears.
const details = JSON.parse(document.getElementById("data").textContent);
const svg = document.getElementById("graph");
const tip = document.getElementById("tip");
const wires = [...svg.querySelectorAll(".wire")];
let pinned = null;

function escapeText(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// Keep a selected channel's contract readable without covering the diagram.
const inspection = document.getElementById("inspection");
const choices = document.createElement("select");
choices.setAttribute("aria-label", "Inspect channel");
choices.add(new Option("Choose a channel…", ""));
for (const [key, item] of Object.entries(details)) {
  if (item.guide) choices.add(new Option(item.title, key));
}
if (choices.length > 1) document.getElementById("inspector-intro").after(choices);

function inspect(ref) {
  const item = details[ref];
  choices.value = item?.guide ? ref : "";
  inspection.replaceChildren();
  if (!item) return;
  const h = item.handshake;
  const heading = document.createElement("h2");
  heading.textContent = h
    ? [h.writers.join(" + "), item.title, h.readers.join(" + ")].filter(Boolean).join(" → ")
    : item.title;
  inspection.append(heading);
  if (h) inspection.append(handshake(h));
  const colors = h ? {"FULL when": h.full.color, "EMPTY when": h.empty.color} : {};
  // A role's CTA map already says where it runs.
  const facts = item.guide ||
    {"Docstring": item.doc, "Size": item.size, "Where": item.ctas ? "" : item.where};
  const fields = document.createElement("dl");
  for (const [name, value] of Object.entries(facts)) {
    if (!value) continue;
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = name;
    dd.textContent = value;
    if (colors[name]) dt.classList.add("signal", colors[name]);
    fields.append(dt, dd);
  }
  if (fields.children.length) inspection.append(fields);
  if (item.warps) inspection.append(...roleDrilldown(ref, item));
  const lines = item.lines.filter((line) => line !== item.title);
  if (!lines.length) return;
  const more = document.createElement("details"), summary = document.createElement("summary");
  const raw = document.createElement("pre");
  raw.textContent = lines.join("\n");
  summary.textContent = item.guide ? "Barrier counts, fences & buffers" : "Declaration";
  more.open = !item.guide && !item.warps; // hubs, streams and raw code have nothing else
  more.append(summary, raw);
  inspection.append(more);
}

// A role's drill-down: where its warps sit in each thread block, then every handoff it takes
// part in, as "waits for ... then signals ...".
function roleDrilldown(ref, item) {
  const out = [];
  if (item.ctas) out.push(ctaMap(ref, item));
  if (!item.handoffs.length) return out;
  const side = (s) =>
    `<td><b class="sig ${s.color}">${s.word}</b> ${escapeText(s.by)}<small>${escapeText(s.count)}</small></td>`;
  const rows = item.handoffs.map((row) =>
    `<tr><td><button type="button" class="use ${row.verb} ${row.signals.color}" data-chip="${row.chip}">${row.verb}</button></td>` +
    `<th scope="row"><button type="button" data-chip="${row.chip}">${escapeText(details[row.chip].title)}</button>` +
    (row.slots.length ? `<small>slot ${row.slots.join(", ")}</small>` : "") + "</th>" +
    side(row.waits) + side(row.signals) + "</tr>");
  const table = document.createElement("div");
  table.className = "handoffs";
  table.innerHTML = "<table><thead><tr><td></td><td></td><th>waits for</th><th>then signals</th></tr></thead>" +
    `<tbody>${rows.join("")}</tbody></table><p>Counts are each barrier's declared arrivals, not this role's share.</p>`;
  table.querySelectorAll("[data-chip]").forEach((b) => b.addEventListener("click", () => pin(b.dataset.chip)));
  out.push(table);
  return out;
}

// One strip of warps per thread block of the cluster; the selected role is lit where it runs.
function ctaMap(ref, item) {
  const roles = Object.entries(details).filter(([, r]) => r.warps).sort((a, b) => a[1].warps[0] - b[1].warps[0]);
  const cell = 26, gap = 4, x0 = 96, rowH = 70;
  const xOf = (w) => x0 + w * cell + Math.floor(w / 4) * gap;
  let g = "";
  item.ctas.forEach((cta, rank) => {
    const y = 8 + rank * rowH;
    let crowded = 0; // names wider than their warps drop to lower lines in turn
    g += `<text x="0" y="${y + 15}" class="map-cta">${escapeText(cta.label)}</text>`;
    for (let w = 0; w < item.num_warps; w++) {
      g += `<rect class="map-warp" x="${xOf(w)}" y="${y}" width="${cell - 3}" height="20" rx="2"/>`;
    }
    roles.forEach(([id, r]) => {
      const here = r.ctas[rank].here, first = r.warps[0], last = r.warps[r.warps.length - 1];
      const cls = "map-role" + (id === ref ? " me" : "") + (here ? "" : " absent");
      const right = xOf(last) + cell - 3;
      g += `<rect class="${cls}" x="${xOf(first)}" y="${y}" width="${right - xOf(first)}" height="20" rx="2"/>`;
      const name = here ? r.title : id === ref ? "not here" : "";
      const ly = y + 33 + (name.length * 6.2 > right - xOf(first) ? crowded++ % 3 : 0) * 11;
      g += `<text x="${(xOf(first) + right) / 2}" y="${ly}" text-anchor="middle" class="${cls} map-name">${escapeText(name)}</text>`;
    });
  });
  const map = document.createElement("div"), label = document.createElement("p");
  map.className = "role-map";
  label.className = "figlabel";
  label.textContent = item.where;
  const width = xOf(item.num_warps - 1) + cell + 60, height = item.ctas.length * rowH + 4;
  map.innerHTML = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img">${g}</svg>`;
  map.firstChild.setAttribute("aria-label", `${item.title}: ${item.where}`);
  map.prepend(label);
  return map;
}

// One wire each way: FULL forward in the writer's colour, EMPTY back in the reader's.
function handshake(h) {
  // The drawing sizes itself: long lists of ends push the wires down, long names and labels
  // push the ends apart, and many stages are drawn as one square and a count.
  const wide = (list, px) => Math.max(0, ...list.map((t) => t.length)) * px;
  const most = Math.max(h.writers.length, h.readers.length);
  const mid = Math.max(54, most * 8 + 6), height = 2 * mid + 4;
  const left = wide(h.writers, 8.2) + 4, right = wide(h.readers, 8.2) + 4;
  const labels = [h.full, h.empty].flatMap((s) => [`FULL · ${s.by}`.length * 6.6, s.count.length * 6]);
  const span = Math.max(300, ...labels) + 28, x0 = left + 14, x1 = x0 + span, cx = (x0 + x1) / 2;
  const width = x1 + 14 + right;
  const names = (list, x, anchor) => list.map((n, i) =>
    `<text x="${x}" y="${mid + 5 + (i - (list.length - 1) / 2) * 16}" text-anchor="${anchor}" class="hs-role">${escapeText(n)}</text>`).join("");
  const wire = (y, from, to, side, word, labelY, countY) => {
    const tip = to > from ? 8 : -8;
    return `<g class="hs ${side.color}"><path class="line" d="M${from},${y} H${to - tip}"/><path class="tip" d="M${to},${y} l${-tip},-4.5 v9 z"/>` +
      `<text x="${cx}" y="${labelY}" text-anchor="middle" class="hs-label">${word} · ${escapeText(side.by)}</text></g>` +
      `<text x="${cx}" y="${countY}" text-anchor="middle" class="hs-count">${escapeText(side.count)}</text>`;
  };
  const drawn = h.slots > 12 ? 1 : h.slots, x = cx - (drawn * 20 - 4) / 2;
  let slots = Array.from({length: drawn}, (_, i) =>
    `<rect class="hs-slot ${h.full.color}" x="${x + i * 20}" y="${mid - 8}" width="16" height="16" rx="2"/>`).join("");
  if (drawn < h.slots) slots += `<text x="${x + 22}" y="${mid + 4}" class="hs-count">× ${h.slots}</text>`;
  const el = document.createElement("div");
  el.className = "handshake";
  el.innerHTML = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img">` +
    names(h.writers, left, "end") + names(h.readers, x1 + 14, "start") +
    wire(mid - 16, x0, x1, h.full, "FULL", mid - 25, mid - 40) + slots +
    wire(mid + 16, x1, x0, h.empty, "EMPTY", mid + 33, mid + 48) + "</svg>";
  el.firstChild.setAttribute("aria-label",
    `FULL runs from ${h.writers.join(", ")} to ${h.readers.join(", ")}; EMPTY runs back.`);
  return el;
}
function pin(ref) {
  pinned = ref; inspect(ref); show(ref);
  document.querySelectorAll("[data-pick]").forEach((el) => el.classList.toggle("picked", el.dataset.pick === ref));
}
document.querySelectorAll("[data-pick]").forEach((el) => el.addEventListener("click", () => {
  pin(el.dataset.pick === pinned ? null : el.dataset.pick);
}));
choices.addEventListener("change", () => pin(choices.value || null));

function show(ref, x, y) {
  svg.querySelectorAll(".hl, .sel").forEach((el) => el.classList.remove("hl", "sel"));
  svg.classList.toggle("focus", Boolean(ref));
  if (!ref) {
    tip.hidden = true;
    return;
  }
  const near = new Set([ref]);
  for (const w of wires) {
    const ends = w.dataset.ends.split(" ");
    if (ends.includes(ref)) {
      w.classList.add("hl");
      ends.forEach((e) => near.add(e));
    }
  }
  near.forEach((id) => document.getElementById(id)?.classList.add("hl"));
  document.getElementById(ref)?.classList.add("sel");
  const d = details[ref];
  if (!d || pinned) {
    tip.hidden = true;
    return;
  }
  tip.innerHTML = `<b>${escapeText(d.title)}</b>${escapeText(d.guide?.Channel || d.lines[0] || "")} · click to inspect`;
  tip.hidden = false;
  const left = Math.min(x + 14, window.scrollX + document.documentElement.clientWidth - tip.offsetWidth - 8);
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${y + 16}px`;
}

function target(ev) {
  return ev.target.closest("[data-ref]");
}

svg.addEventListener("mousemove", (ev) => {
  if (pinned || drag) return;
  const el = target(ev);
  show(el ? el.dataset.ref : null, ev.pageX, ev.pageY);
});
svg.addEventListener("mouseleave", () => { if (!pinned) show(null); });
svg.addEventListener("click", (ev) => {
  if (suppressClick) return;
  const el = target(ev);
  pin(el && el.dataset.ref !== pinned ? el.dataset.ref : null);
});
svg.addEventListener("focusin", (ev) => {
  const el = target(ev);
  if (!el || !el.matches(":focus-visible")) return; // clicks pin through their own handler
  pin(el.dataset.ref);
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") pin(null);
});
// Enter or Space selects the focused node; the arrow keys pan a zoomed graph.
svg.addEventListener("keydown", (ev) => {
  const el = target(ev);
  if (el && (ev.key === "Enter" || ev.key === " ")) {
    ev.preventDefault();
    pin(el.dataset.ref); // focus may have selected it already; Escape clears
    return;
  }
  const step = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[ev.key];
  if (!step || zoom === 1) return;
  ev.preventDefault();
  cx += step[0] * home.w / zoom / 8; cy += step[1] * home.h / zoom / 8; viewport();
});

// The graph fills its panel. zoom = 1 shows all of it; the view keeps the panel's aspect so
// zooming and panning never letterbox or clip to the graph's own box.
const home = {w: svg.viewBox.baseVal.width, h: svg.viewBox.baseVal.height}, MAX_FIT = 1.5;
let zoom = 1, cx = home.w / 2, cy = home.h / 2, drag = null, suppressClick = false;
function viewport() {
  let box = svg.getBoundingClientRect();
  if (!box.width) return; // not laid out yet
  // The panel is as tall as the graph needs at this width, within limits. A small graph is
  // enlarged by at most MAX_FIT so its text stays near the page's size.
  const scale = Math.min(MAX_FIT, box.width / home.w);
  const tall = Math.max(180, Math.min(620, innerHeight * 0.6, home.h * scale));
  if (Math.abs(tall - box.height) > 1) {
    svg.style.height = `${tall}px`;
    box = svg.getBoundingClientRect();
  }
  const aspect = box.width / Math.max(1, box.height);
  const fit = home.w / home.h > aspect ? {w: home.w, h: home.w / aspect} : {w: home.h * aspect, h: home.h};
  if (box.width / fit.w > MAX_FIT) { fit.w = box.width / MAX_FIT; fit.h = fit.w / aspect; }
  const w = fit.w / zoom, h = fit.h / zoom;
  // Centre an axis that fits; otherwise keep the view inside the graph.
  const clamp = (c, view, size) => view >= size ? size / 2 : Math.max(view / 2, Math.min(size - view / 2, c));
  cx = clamp(cx, w, home.w); cy = clamp(cy, h, home.h);
  svg.setAttribute("viewBox", [cx - w / 2, cy - h / 2, w, h].join(" "));
  svg.classList.toggle("zoomed", zoom > 1);
  document.getElementById("zoom-out").disabled = zoom === 1;
  document.getElementById("zoom-in").disabled = zoom === 8;
  document.getElementById("zoom-fit").disabled = zoom === 1;
}
function magnify(factor, x = cx, y = cy) {
  const old = zoom; zoom = Math.max(1, Math.min(8, zoom * factor));
  cx = x + (cx - x) * old / zoom; cy = y + (cy - y) * old / zoom; viewport();
}
// Plain scrolling belongs to the page; Ctrl/⌘ + wheel (and trackpad pinch) zooms about the cursor.
svg.addEventListener("wheel", e => {
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault(); tip.hidden = true;
  const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
  magnify(Math.exp(-Math.max(-120, Math.min(120, e.deltaY * (e.deltaMode ? 16 : 1))) * .008), p.x, p.y);
}, {passive:false});
svg.addEventListener("pointerdown", e => {
  suppressClick = false;
  if (zoom > 1 && e.button === 0 && e.isPrimary) drag = {id:e.pointerId, x:e.clientX, y:e.clientY, cx, cy, scale:svg.getScreenCTM().a};
});
svg.addEventListener("pointermove", e => {
  if (!drag || drag.id !== e.pointerId) return;
  const dx = e.clientX-drag.x, dy = e.clientY-drag.y;
  if (!suppressClick && Math.hypot(dx,dy) < 4) return;
  suppressClick = true; tip.hidden = true; svg.setPointerCapture(e.pointerId);
  cx = drag.cx - dx/drag.scale; cy = drag.cy - dy/drag.scale; viewport();
});
for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
  window.addEventListener(event, e => { if (drag?.id === e.pointerId) drag = null; });
}
document.getElementById("zoom-in").onclick = () => magnify(1.5);
document.getElementById("zoom-out").onclick = () => magnify(1/1.5);
document.getElementById("zoom-fit").onclick = () => { zoom = 1; viewport(); };
new ResizeObserver(viewport).observe(svg);
viewport();
