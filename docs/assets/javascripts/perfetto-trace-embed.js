function initializeTraceEmbeds(root = document) {
  for (const embed of root.querySelectorAll("[data-trace-embed]")) {
    if (embed.dataset.initialized === "true") {
      continue;
    }
    embed.dataset.initialized = "true";

    const preview = embed.querySelector(".trace-preview");
    const viewer = embed.querySelector(".trace-embed__viewer");
    const closeButton = viewer.querySelector(".trace-embed__close");
    const iframe = viewer.querySelector("iframe");
    const controls = document.createElement("div");
    const fullscreenButton = document.createElement("button");

    controls.className = "trace-embed__controls";
    fullscreenButton.className = "trace-embed__fullscreen";
    fullscreenButton.type = "button";
    fullscreenButton.textContent = "fullscreen";
    fullscreenButton.setAttribute("aria-pressed", "false");
    closeButton.before(controls);
    controls.append(closeButton, fullscreenButton);

    const isFullscreen = () =>
      document.fullscreenElement === viewer ||
      viewer.classList.contains("trace-embed__viewer--fullscreen");

    const updateFullscreenButton = () => {
      const active = isFullscreen();
      fullscreenButton.textContent = active ? "exit fullscreen" : "fullscreen";
      fullscreenButton.setAttribute("aria-pressed", String(active));
    };

    const enterFallbackFullscreen = () => {
      viewer.classList.add("trace-embed__viewer--fullscreen");
      document.body.classList.add("trace-embed-fullscreen-open");
      updateFullscreenButton();
    };

    const exitFallbackFullscreen = () => {
      viewer.classList.remove("trace-embed__viewer--fullscreen");
      document.body.classList.remove("trace-embed-fullscreen-open");
      updateFullscreenButton();
    };

    const leaveFullscreen = async () => {
      if (document.fullscreenElement === viewer) {
        await document.exitFullscreen();
      } else {
        exitFallbackFullscreen();
      }
    };

    preview.addEventListener("click", () => {
      preview.hidden = true;
      preview.setAttribute("aria-expanded", "true");
      viewer.hidden = false;

      if (!iframe.hasAttribute("src")) {
        iframe.src = iframe.dataset.src;
      }
      closeButton.focus();
    });

    closeButton.addEventListener("click", () => {
      void leaveFullscreen().finally(() => {
        viewer.hidden = true;
        preview.hidden = false;
        preview.setAttribute("aria-expanded", "false");
        preview.focus();
      });
    });

    fullscreenButton.addEventListener("click", () => {
      if (isFullscreen()) {
        void leaveFullscreen();
        return;
      }

      if (!viewer.requestFullscreen) {
        enterFallbackFullscreen();
        return;
      }

      void viewer.requestFullscreen().catch(enterFallbackFullscreen);
    });

    document.addEventListener("fullscreenchange", updateFullscreenButton);
    document.addEventListener("keydown", (event) => {
      if (
        event.key === "Escape" &&
        viewer.classList.contains("trace-embed__viewer--fullscreen")
      ) {
        exitFallbackFullscreen();
        fullscreenButton.focus();
      }
    });
  }
}

initializeTraceEmbeds();
