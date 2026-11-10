// Hover shows details and neighbours; click pins, Escape or empty-space click clears.
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
  if (!d) {
    tip.hidden = true;
    return;
  }
  tip.innerHTML = `<b>${escapeText(d.title)}</b>${d.lines.map(escapeText).join("\n")}`;
  tip.hidden = false;
  const left = Math.min(x + 14, window.scrollX + document.documentElement.clientWidth - tip.offsetWidth - 8);
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${y + 16}px`;
}

function target(ev) {
  return ev.target.closest("[data-ref]");
}

svg.addEventListener("mousemove", (ev) => {
  if (pinned) return;
  const el = target(ev);
  show(el ? el.dataset.ref : null, ev.pageX, ev.pageY);
});
svg.addEventListener("mouseleave", () => { if (!pinned) show(null); });
svg.addEventListener("click", (ev) => {
  const el = target(ev);
  pinned = el && el.dataset.ref !== pinned ? el.dataset.ref : null;
  tip.classList.toggle("pinned", Boolean(pinned));
  show(pinned, ev.pageX, ev.pageY);
});
svg.addEventListener("focusin", (ev) => {
  const el = target(ev);
  if (!el || !el.matches(":focus-visible")) return; // clicks pin through their own handler
  const box = el.getBoundingClientRect();
  pinned = el.dataset.ref;
  show(pinned, box.left + window.scrollX, box.bottom + window.scrollY);
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") {
    pinned = null;
    show(null);
  }
});
