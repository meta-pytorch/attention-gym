const PERFETTO_ORIGIN = "https://ui.perfetto.dev";
const params = new URLSearchParams(window.location.search);
const iframe = document.querySelector("#perfetto");
const status = document.querySelector("#status");

function showError(error) {
  status.textContent = error instanceof Error ? error.message : String(error);
  status.classList.add("trace-viewer__status--error");
}

function waitForPerfetto() {
  return new Promise((resolve) => {
    const interval = window.setInterval(() => {
      iframe.contentWindow.postMessage("PING", PERFETTO_ORIGIN);
    }, 100);

    function onMessage(event) {
      if (
        event.origin === PERFETTO_ORIGIN &&
        event.source === iframe.contentWindow &&
        event.data === "PONG"
      ) {
        window.clearInterval(interval);
        window.removeEventListener("message", onMessage);
        resolve();
      }
    }

    window.addEventListener("message", onMessage);
  });
}

async function loadTrace() {
  const traceParameter = params.get("trace");
  if (!traceParameter) {
    throw new Error("Missing trace query parameter");
  }

  const traceUrl = new URL(traceParameter, window.location.href);
  if (traceUrl.origin !== window.location.origin) {
    throw new Error("Trace URL must use the documentation origin");
  }

  const title = params.get("title") || "Perfetto trace";
  document.title = title;
  iframe.title = title;
  iframe.src = `${PERFETTO_ORIGIN}/#!/?mode=embedded`;

  const [response] = await Promise.all([fetch(traceUrl), waitForPerfetto()]);
  if (!response.ok) {
    throw new Error(`Unable to load trace: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  iframe.contentWindow.postMessage(
    {
      perfetto: {
        buffer,
        title,
        fileName: traceUrl.pathname.split("/").at(-1),
      },
    },
    PERFETTO_ORIGIN,
    [buffer],
  );
  status.hidden = true;
}

loadTrace().catch(showError);
