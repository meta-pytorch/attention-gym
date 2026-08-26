const PLOTLY_FONT = '"IBM Plex Sans", sans-serif';
const PLOTLY_MONO_FONT = '"IBM Plex Mono", monospace';

const PLOTLY_THEMES = {
  slate: {
    foreground: "#d4d4d4",
    muted: "#999999",
    grid: "rgba(153, 153, 153, 0.18)",
    line: "rgba(153, 153, 153, 0.32)",
    surface: "#161616",
    hoverBackground: "#161616",
    colors: ["#68b5e6", "#e3a15f", "#6f8f7b", "#d36b62", "#d0a85c", "#a58fc9"],
  },
  default: {
    foreground: "#1a2e22",
    muted: "#6a7a6e",
    grid: "rgba(26, 46, 34, 0.14)",
    line: "#d0ccc4",
    surface: "#eae6de",
    hoverBackground: "#eae6de",
    colors: ["#176d9c", "#97551c", "#5f7f67", "#aa413a", "#9a6a22", "#725b91"],
  },
};

function currentPlotlyTheme() {
  const scheme =
    document.body.dataset.mdColorScheme || document.documentElement.dataset.mdColorScheme;
  return PLOTLY_THEMES[scheme] || PLOTLY_THEMES.slate;
}

function installPlotlyFrameStyles(frame, theme) {
  const doc = frame.contentDocument;
  if (!doc) {
    return;
  }
  doc.documentElement.style.background = theme.surface;
  doc.documentElement.style.setProperty("--plot-fg", theme.foreground);
  doc.documentElement.style.setProperty("--plot-muted", theme.muted);
  doc.documentElement.style.setProperty("--plot-line", theme.line);
  doc.documentElement.style.setProperty("--plot-surface", theme.surface);
  theme.colors.forEach((color, index) =>
    doc.documentElement.style.setProperty(`--c${index}`, color),
  );
  doc.body.style.margin = "0";
  doc.body.style.background = theme.surface;
  if (doc.documentElement.dataset.attentionGymStyled === "true") {
    return;
  }
  doc.documentElement.dataset.attentionGymStyled = "true";

  const fontLink = document.querySelector('link[href*="fonts.googleapis.com"]');
  if (fontLink) {
    doc.head.append(fontLink.cloneNode());
  }

  const style = doc.createElement("style");
  style.textContent = `
    .modebar { opacity: 0; transition: opacity 120ms ease; }
    .plot-container:hover .modebar,
    .plot-container:focus-within .modebar { opacity: 1; }
  `;
  doc.head.append(style);

  doc.addEventListener(
    "wheel",
    (event) => {
      if (event.ctrlKey || event.metaKey) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      const unit =
        event.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? 16
          : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
            ? window.innerHeight
            : 1;
      window.scrollBy({
        left: event.deltaX * unit,
        top: event.deltaY * unit,
        behavior: "auto",
      });
    },
    { capture: true, passive: false },
  );
}

function axisTheme(plot, theme) {
  const update = {};
  for (const key of Object.keys(plot.layout)) {
    if (!/^([xyz]axis)\d*$/.test(key)) {
      continue;
    }
    update[`${key}.gridcolor`] = theme.grid;
    update[`${key}.linecolor`] = theme.line;
    update[`${key}.tickcolor`] = theme.line;
    update[`${key}.zerolinecolor`] = theme.line;
    update[`${key}.tickfont`] = { color: theme.muted, family: PLOTLY_FONT, size: 11 };
    update[`${key}.title.font`] = { color: theme.muted, family: PLOTLY_MONO_FONT, size: 11 };
  }
  return update;
}

async function styleNamedTraces(win, plot, theme) {
  const colors = {
    KDA: theme.colors[0],
    "KDA captured": theme.colors[0],
    "KDA exact": theme.colors[0],
    "KDA slowdown": theme.colors[0],
    FA2: theme.colors[1],
    "FA2 captured": theme.colors[1],
    "FA2 exact": theme.colors[1],
    "FA2 slowdown": theme.colors[1],
    FA4: theme.colors[2],
    "FA4 captured": theme.colors[2],
    "FA4 exact": theme.colors[2],
    "FA4 slowdown": theme.colors[2],
    "FA4 + persistent fwd": theme.colors[5],
    "FA4 + persistent fwd captured": theme.colors[5],
    "FA4 + persistent fwd slowdown": theme.colors[5],
    "FA4 exact": theme.colors[2],
    "Ideal exact graph": theme.colors[2],
    "Static worst-case graph": theme.colors[0],
    "Persistent worst-case graph": theme.colors[1],
    "Everything else in KDAAttention": theme.colors[1],
    "Isolated chunk_kda core": theme.colors[2],
    "Full KDAAttention replay": theme.colors[0],
    "lookahead 8": theme.colors[3],
    "lookahead 16": theme.colors[1],
    "lookahead 32": theme.colors[0],
    "lookahead 64": theme.colors[2],
  };
  for (const [index, trace] of plot.data.entries()) {
    const color = colors[trace.name] || theme.colors[index % theme.colors.length];
    const update = { "marker.color": color };
    if (trace.mode?.includes("lines")) {
      update["line.color"] = color;
      update["line.width"] = 2;
      update["marker.size"] = 7;
    }
    if (trace.error_y) {
      update["error_y.color"] = color;
    }
    await win.Plotly.restyle(plot, update, [index]);
  }
}

function chartLayout(plot, theme) {
  const legendFont = { color: theme.muted, family: PLOTLY_FONT, size: 10 };
  const update = {
    ...axisTheme(plot, theme),
    "title.text": "",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: theme.foreground, family: PLOTLY_FONT, size: 12 },
    hoverlabel: {
      bgcolor: theme.hoverBackground,
      bordercolor: theme.line,
      font: { color: theme.foreground, family: PLOTLY_FONT, size: 12 },
    },
  };

  const annotations = (plot.layout.annotations || []).map((annotation) => ({
    ...annotation,
    font: { color: theme.muted, family: PLOTLY_MONO_FONT, size: 12 },
  }));

  return {
    ...update,
    annotations,
    margin: { l: 62, r: 28, t: 54, b: 58, pad: 0 },
    legend: {
      orientation: "h",
      x: 1,
      xanchor: "right",
      y: 1.13,
      yanchor: "bottom",
      groupclick: "togglegroup",
      font: legendFont,
    },
  };
}

async function stylePlotlyFrame(frame) {
  const doc = frame.contentDocument;
  const win = frame.contentWindow;
  const plot = doc?.querySelector(".plotly-graph-div");
  if (!doc || !win?.Plotly || !plot?.layout) {
    window.setTimeout(() => stylePlotlyFrame(frame), 100);
    return;
  }

  const theme = currentPlotlyTheme();
  installPlotlyFrameStyles(frame, theme);
  const layout = chartLayout(plot, theme);
  const annotations = layout.annotations;
  delete layout.annotations;
  await win.Plotly.relayout(plot, layout);
  if (annotations) {
    await win.Plotly.relayout(plot, { annotations });
  }
  await styleNamedTraces(win, plot, theme);
  win.Plotly.Plots.resize(plot);
}

function queuePlotlyFrameStyle(frame) {
  frame.plotlyThemePromise = (frame.plotlyThemePromise || Promise.resolve())
    .then(() => stylePlotlyFrame(frame))
    .catch((error) => console.error("Unable to style Plotly chart", error));
}

function initializePlotlyCharts(root = document) {
  for (const frame of root.querySelectorAll(".plotly-chart__frame")) {
    if (frame.dataset.themeInitialized === "true") {
      continue;
    }
    frame.dataset.themeInitialized = "true";
    frame.addEventListener("load", () => queuePlotlyFrameStyle(frame));
    if (frame.contentDocument?.readyState === "complete") {
      queuePlotlyFrameStyle(frame);
    }
  }
}

const plotlyThemeObserver = new MutationObserver(() => {
  for (const frame of document.querySelectorAll(".plotly-chart__frame")) {
    queuePlotlyFrameStyle(frame);
  }
});
plotlyThemeObserver.observe(document.body, {
  attributes: true,
  attributeFilter: ["data-md-color-scheme"],
});

initializePlotlyCharts();
if (typeof document$ !== "undefined") {
  document$.subscribe(() => initializePlotlyCharts());
}
