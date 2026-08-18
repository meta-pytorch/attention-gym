(() => {
  const root = document.getElementById("ci-health-dashboard");
  if (!root) return;

  const dateFormatter = new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
  const healthLabels = {
    success: "Healthy",
    warning: "Degraded",
    failure: "Failing",
    unknown: "No data",
  };

  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const formatPercent = (value) => (value == null ? "—" : `${value.toFixed(1)}%`);

  const formatDuration = (seconds) => {
    if (seconds == null) return "—";
    const totalSeconds = Math.round(seconds);
    if (totalSeconds < 60) return `${totalSeconds}s`;
    const minutes = Math.floor(totalSeconds / 60);
    const remainder = totalSeconds % 60;
    return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
  };

  const formatDate = (value) => (value ? dateFormatter.format(new Date(value)) : "Unknown");

  const healthKind = (summary) => {
    if (!summary || summary.total === 0) return "unknown";
    if (summary.latest?.kind === "failure" || summary.success_rate < 80) return "failure";
    if (summary.success_rate < 95) return "warning";
    return "success";
  };

  const conclusionKind = (conclusion) => {
    if (conclusion === "success") return "success";
    if (["failure", "startup_failure", "stale", "timed_out"].includes(conclusion)) {
      return "failure";
    }
    return "ignored";
  };

  const summaryCards = (data) => {
    const overall = data.overall;
    const status = healthKind(overall);
    return `
      <div class="ci-dashboard-meta">
        <span>${data.runs_per_workflow} recent runs sampled per workflow</span>
        <span>Updated ${formatDate(data.generated_at)}</span>
      </div>
      <section class="ci-summary" aria-label="CI summary">
        <article class="ci-stat ci-stat--${status}">
          <span class="ci-stat__label">Overall status</span>
          <strong>${healthLabels[status]}</strong>
          <span>${overall.latest ? `Latest: ${escapeHtml(overall.latest.conclusion)}` : "No completed runs"}</span>
        </article>
        <article class="ci-stat">
          <span class="ci-stat__label">Pass rate</span>
          <strong>${formatPercent(overall.success_rate)}</strong>
          <span>${overall.successes} passed · ${overall.failures} failed</span>
        </article>
        <article class="ci-stat">
          <span class="ci-stat__label">Run duration</span>
          <strong>${formatDuration(overall.median_duration_seconds)}</strong>
          <span>p95 ${formatDuration(overall.p95_duration_seconds)}</span>
        </article>
      </section>`;
  };

  const trendWorkflows = (workflows) =>
    workflows.filter(
      (workflow) => workflow.timeline.filter((run) => run.kind !== "ignored").length >= 2
    );

  const trendView = (workflows) => {
    const candidates = trendWorkflows(workflows);
    if (!candidates.length) return '<div class="ci-empty">Not enough completed runs to chart.</div>';
    const selected = Math.max(
      candidates.findIndex((workflow) => workflow.failures > 0),
      0
    );
    return `
      <div class="ci-trend-toolbar">
        <label>
          <span>Workflow</span>
          <select id="ci-trend-workflow">
            ${candidates
              .map(
                (workflow, index) =>
                  `<option value="${index}" ${index === selected ? "selected" : ""}>${escapeHtml(workflow.name)}</option>`
              )
              .join("")}
          </select>
        </label>
        <div class="ci-metric-tabs" role="group" aria-label="Chart metric">
          <button type="button" data-metric="pass" class="is-selected">Pass rate</button>
          <button type="button" data-metric="failure">Failure rate</button>
          <button type="button" data-metric="duration">Duration</button>
        </div>
      </div>
      <article class="ci-chart-card ci-chart-card--large">
        <div>
          <div><h2 id="ci-chart-title">Rolling pass rate</h2><span id="ci-chart-note">Latest five counted runs</span></div>
          <span><i class="ci-failure-point"></i> failed commit · click any point to open its run</span>
        </div>
        <div class="ci-chart"><canvas id="ci-trend-chart"></canvas></div>
        <div id="ci-chart-failures" class="ci-chart-failures"></div>
      </article>
      <div id="ci-chart-fallback" class="ci-muted" hidden>Charts could not be loaded.</div>`;
  };

  const jobView = (jobs) => `
    <div class="ci-panel-heading">
      <div><h2>Job timing</h2><span>Median and p95 across the sampled runs</span></div>
      <label class="ci-job-filter">Filter <input id="ci-job-filter" type="search" placeholder="pytest, docs, build…"></label>
    </div>
    <div class="ci-table-wrap">
      <table class="ci-job-table">
        <thead><tr><th>Workflow / job</th><th>Latest</th><th>Success</th><th>Median</th><th>p95</th><th>Samples</th></tr></thead>
        <tbody id="ci-job-rows">
          ${jobs
            .slice()
            .sort((a, b) => (b.p95_duration_seconds || 0) - (a.p95_duration_seconds || 0))
            .map(
              (job) => `
                <tr data-filter="${escapeHtml(`${job.workflow} ${job.name}`.toLowerCase())}">
                  <td><span>${escapeHtml(job.workflow)}</span><strong>${escapeHtml(job.name)}</strong></td>
                  <td><a class="ci-result ci-status--${conclusionKind(job.latest_conclusion)}" href="${escapeHtml(job.latest_url)}">${escapeHtml(job.latest_conclusion || "unknown")}</a></td>
                  <td>${formatPercent(job.success_rate)}</td>
                  <td>${formatDuration(job.median_duration_seconds)}</td>
                  <td>${formatDuration(job.p95_duration_seconds)}</td>
                  <td>${job.total}</td>
                </tr>`
            )
            .join("")}
        </tbody>
      </table>
    </div>`;

  const failureView = (items) => `
    <div class="ci-panel-heading">
      <div><h2>Failing commits</h2><span>Failed jobs, steps, and extracted pytest tests</span></div>
      <span>${items.length ? `${items.length} most recent` : "No recent failures"}</span>
    </div>
    <div class="ci-failures">
      ${
        items.length
          ? items
              .map(
                (failure, index) => `
                  <details class="ci-failure" ${index === 0 ? "open" : ""}>
                    <summary>
                      <span class="ci-failure__icon">×</span>
                      <span><strong>${escapeHtml(failure.workflow)}</strong><small>${escapeHtml(failure.title)} · ${escapeHtml(failure.branch || "unknown branch")} · ${formatDate(failure.created_at)}</small></span>
                      <code>${escapeHtml(failure.sha)}</code>
                    </summary>
                    <div class="ci-failure__body">
                      ${failure.jobs
                        .map(
                          (job) => `
                            <article>
                              <div class="ci-failure__job"><a href="${escapeHtml(job.url)}">${escapeHtml(job.name)} →</a><span>${formatDuration(job.duration_seconds)}</span></div>
                              ${job.failed_steps.length ? `<p><b>Failed steps:</b> ${job.failed_steps.map(escapeHtml).join(", ")}</p>` : ""}
                              ${
                                job.tests.length
                                  ? `<div class="ci-tests">${job.tests.map((test) => `<code>${escapeHtml(test)}</code>`).join("")}</div>`
                                  : '<p class="ci-muted">No pytest node IDs were found in the retained job log.</p>'
                              }
                            </article>`
                        )
                        .join("")}
                      <a class="ci-run-link" href="${escapeHtml(failure.url)}">Open complete workflow run →</a>
                    </div>
                  </details>`
              )
              .join("")
          : '<div class="ci-empty">All sampled runs passed.</div>'
      }
    </div>`;

  const dashboardViews = (data) => `
    <div class="ci-view-tabs" role="tablist" aria-label="CI health views">
      <button type="button" role="tab" aria-selected="true" data-view="trends">Trends</button>
      <button type="button" role="tab" aria-selected="false" data-view="failures">Failures <span>${data.failures.length}</span></button>
      <button type="button" role="tab" aria-selected="false" data-view="jobs">Jobs <span>${data.jobs.length}</span></button>
    </div>
    <section class="ci-view-panel" role="tabpanel" data-panel="trends">${trendView(data.workflows)}</section>
    <section class="ci-view-panel" role="tabpanel" data-panel="failures" hidden>${failureView(data.failures)}</section>
    <section class="ci-view-panel" role="tabpanel" data-panel="jobs" hidden>${jobView(data.jobs)}</section>`;

  const setupViewTabs = () => {
    const buttons = [...document.querySelectorAll("[data-view]")];
    const panels = [...document.querySelectorAll("[data-panel]")];
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        buttons.forEach((item) => item.setAttribute("aria-selected", item === button));
        panels.forEach((panel) => {
          panel.hidden = panel.dataset.panel !== button.dataset.view;
        });
      });
    });
  };

  const setupTrendChart = (workflows) => {
    const candidates = trendWorkflows(workflows);
    if (!candidates.length) return;
    if (!window.Chart) {
      document.getElementById("ci-chart-fallback").hidden = false;
      return;
    }

    const workflowSelect = document.getElementById("ci-trend-workflow");
    const metricButtons = [...document.querySelectorAll("[data-metric]")];
    let selectedWorkflow = Number(workflowSelect.value);
    let metric = "pass";
    let chart;

    const draw = () => {
      chart?.destroy();
      const workflow = candidates[selectedWorkflow];
      const runs = workflow.timeline
        .filter((run) => run.kind !== "ignored")
        .slice()
        .reverse();
      const rolling = (kind) =>
        runs.map((_, index) => {
          const window = runs.slice(Math.max(0, index - 4), index + 1);
          return (window.filter((run) => run.kind === kind).length / window.length) * 100;
        });
      const metrics = {
        pass: {
          title: "Rolling pass rate",
          note: "Latest five counted runs",
          values: rolling("success"),
          percent: true,
          color: "--ci-success",
        },
        failure: {
          title: "Rolling failure rate",
          note: "Latest five counted runs",
          values: rolling("failure"),
          percent: true,
          color: "--ci-failure",
        },
        duration: {
          title: "Run duration",
          note: "Lower is better",
          values: runs.map((run) => run.duration_seconds),
          percent: false,
          color: "--ci-success",
        },
      };
      const selectedMetric = metrics[metric];
      const styles = getComputedStyle(root);
      const colors = {
        border: styles.getPropertyValue("--ci-border").trim(),
        failure: styles.getPropertyValue("--ci-failure").trim(),
        line: styles.getPropertyValue(selectedMetric.color).trim(),
        success: styles.getPropertyValue("--ci-success").trim(),
        text: getComputedStyle(document.documentElement)
          .getPropertyValue("--md-default-fg-color--light")
          .trim(),
      };
      document.getElementById("ci-chart-title").textContent = selectedMetric.title;
      document.getElementById("ci-chart-note").textContent = selectedMetric.note;
      document.getElementById("ci-chart-failures").innerHTML = runs
        .filter((run) => run.kind === "failure")
        .map(
          (run) =>
            `<a href="${escapeHtml(run.url)}"><i></i>${escapeHtml(run.sha)} · ${formatDate(run.created_at)}</a>`
        )
        .join("");

      chart = new window.Chart(document.getElementById("ci-trend-chart"), {
        type: "line",
        data: {
          labels: runs.map((run) => formatDate(run.created_at)),
          datasets: [
            {
              data: selectedMetric.values,
              borderColor: colors.line,
              backgroundColor: `${colors.line}18`,
              borderWidth: 2,
              fill: true,
              pointBackgroundColor: runs.map((run) =>
                run.kind === "failure" ? colors.failure : colors.line
              ),
              pointBorderWidth: 0,
              pointHoverRadius: 8,
              pointRadius: runs.map((run) => (run.kind === "failure" ? 6 : 3)),
              pointStyle: runs.map((run) => (run.kind === "failure" ? "rectRot" : "circle")),
              tension: 0.25,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { intersect: false, mode: "nearest" },
          onClick: (_event, elements) => {
            if (elements.length) window.open(runs[elements[0].index].url, "_blank", "noopener");
          },
          onHover: (event, elements) => {
            event.native.target.style.cursor = elements.length ? "pointer" : "default";
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (context) =>
                  selectedMetric.percent
                    ? `${selectedMetric.title}: ${formatPercent(context.parsed.y)}`
                    : `Duration: ${formatDuration(context.parsed.y)}`,
                afterLabel: (context) => {
                  const run = runs[context.dataIndex];
                  return [`Result: ${run.conclusion}`, `Commit: ${run.sha}`, "Click to open run"];
                },
              },
            },
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: { color: colors.text, maxRotation: 0, maxTicksLimit: 6 },
            },
            y: {
              beginAtZero: true,
              max: selectedMetric.percent ? 100 : undefined,
              grid: { color: colors.border },
              ticks: {
                color: colors.text,
                callback: (value) =>
                  selectedMetric.percent ? `${value}%` : formatDuration(value),
                stepSize: selectedMetric.percent ? 25 : undefined,
              },
            },
          },
        },
      });
    };

    workflowSelect.addEventListener("change", () => {
      selectedWorkflow = Number(workflowSelect.value);
      draw();
    });
    metricButtons.forEach((button) => {
      button.addEventListener("click", () => {
        metric = button.dataset.metric;
        metricButtons.forEach((item) => item.classList.toggle("is-selected", item === button));
        draw();
      });
    });
    new MutationObserver(draw).observe(document.documentElement, {
      attributeFilter: ["data-md-color-scheme"],
    });
    draw();
  };

  const setupJobFilter = () => {
    const input = document.getElementById("ci-job-filter");
    input?.addEventListener("input", () => {
      const query = input.value.trim().toLowerCase();
      document.querySelectorAll("#ci-job-rows tr").forEach((row) => {
        row.hidden = !row.dataset.filter.includes(query);
      });
    });
  };

  const render = (data) => {
    if (!data.overall) throw new Error("CI health data is incomplete");
    root.innerHTML = summaryCards(data) + dashboardViews(data);
    setupViewTabs();
    setupTrendChart(data.workflows);
    setupJobFilter();
  };

  fetch(new URL("../../assets/ci-health.json", window.location.href))
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(render)
    .catch((error) => {
      root.innerHTML = `<div class="ci-health-error"><strong>CI health data is temporarily unavailable.</strong><span>${escapeHtml(error.message)}</span></div>`;
    });
})();
