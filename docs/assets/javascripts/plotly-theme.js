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

function chartName(frame) {
  return new URL(frame.src, window.location.href).pathname.split("/").at(-1).replace(".html", "");
}

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

async function styleSchedulerTraces(win, plot, theme) {
  const policyColors = {
    static: theme.colors[0],
    early_exit: theme.colors[0],
    persistent: theme.colors[1],
    production_auto: theme.colors[2],
  };
  const distributionStyles = {
    one_long: { dash: "solid", symbol: "circle" },
    uniform: { dash: "dash", symbol: "square" },
    zipf: { dash: "dot", symbol: "diamond" },
  };
  for (const [index, trace] of plot.data.entries()) {
    const [policy, distribution] = trace.name.split(": ");
    const color = policyColors[policy] || theme.colors[index % theme.colors.length];
    const style = distributionStyles[distribution] || distributionStyles.one_long;
    const update = {
      "marker.color": color,
      "marker.symbol": style.symbol,
      "marker.size": 7,
    };
    if (trace.mode?.includes("lines")) {
      update["line.color"] = color;
      update["line.dash"] = style.dash;
      update["line.width"] = 2;
    }
    if (trace.error_y) {
      update["error_y.color"] = color;
    }
    await win.Plotly.restyle(plot, update, [index]);
  }
}

async function styleNamedTraces(win, plot, theme) {
  const colors = {
    "fixed-capacity KDA": theme.colors[0],
    "physical T_max rows": theme.colors[1],
    "exact L rows": theme.colors[2],
    "physical / exact": theme.colors[5],
    "before PR328": theme.colors[3],
    "after PR328": theme.colors[0],
    "after normalized": theme.colors[2],
    "ideal proportional": theme.muted,
    "KDA captured": theme.colors[0],
    "KDA exact": theme.colors[0],
    "KDA slowdown": theme.colors[0],
    "FA2 captured": theme.colors[1],
    "FA2 exact": theme.colors[1],
    "FA2 slowdown": theme.colors[1],
    "FA4 captured": theme.colors[2],
    "FA4 exact": theme.colors[2],
    "FA4 slowdown": theme.colors[2],
    "FA4 + persistent fwd captured": theme.colors[5],
    "FA4 + persistent fwd slowdown": theme.colors[5],
    "FA4 current": theme.colors[1],
    "FA4 persistent forward": theme.colors[0],
    "FA4 exact": theme.colors[2],
    "current / exact": theme.colors[1],
    "persistent / exact": theme.colors[0],
    allocated: theme.colors[0],
    reserved: theme.colors[1],
    "graph-private segments": theme.colors[2],
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

function chartLayout(name, plot, theme) {
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

  if (name === "kda_cuda_graph_scheduler") {
    return {
      ...update,
      annotations,
      hovermode: "closest",
      margin: { l: 62, r: 190, t: 42, b: 58, pad: 0 },
      legend: {
        x: 1.02,
        xanchor: "left",
        y: 1,
        yanchor: "top",
        tracegroupgap: 4,
        groupclick: "togglegroup",
        font: legendFont,
      },
    };
  }

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

  const name = chartName(frame);
  const theme = currentPlotlyTheme();
  installPlotlyFrameStyles(frame, theme);
  const layout = chartLayout(name, plot, theme);
  const annotations = layout.annotations;
  delete layout.annotations;
  await win.Plotly.relayout(plot, layout);
  if (annotations) {
    await win.Plotly.relayout(plot, { annotations });
  }
  if (name === "kda_cuda_graph_scheduler") {
    await styleSchedulerTraces(win, plot, theme);
  } else {
    await styleNamedTraces(win, plot, theme);
  }
  if (name === "kda_cuda_graph_e2e" || name === "kda_cuda_graph_mm_tax") {
    const ratioIndex = plot.data.findIndex((trace) => trace.name === "physical / exact");
    if (ratioIndex >= 0) {
      await win.Plotly.restyle(plot, { showlegend: false, width: 0.12 }, [ratioIndex]);
    }
  }
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
