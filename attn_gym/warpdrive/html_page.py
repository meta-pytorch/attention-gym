"""Shared page shell for warpdrive HTML views.

Styles, data and scripts are inline, so pages open from disk without network access.
Fonts prefer IBM Plex and fall back to system faces.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ASSETS = Path(__file__).parent / "html"

_THEME = """
const root = document.documentElement;
root.dataset.theme = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
document.getElementById("theme").addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  document.dispatchEvent(new Event("themechange"));
});
"""


def asset(name: str) -> str:
    return (ASSETS / name).read_text()


def page(title: str, subtitle: str, body: str, script: str, data=None, css: str = "") -> str:
    """Build a complete HTML document with ``body`` below the header.

    ``data`` must be JSON-serializable. Scripts read it with
    ``JSON.parse(document.getElementById("data").textContent)``.
    """
    # Escape '<' so data cannot close or reopen the script element.
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{asset("page.css")}{css}</style>
</head>
<body>
<header>
<h1>{html.escape(title)}</h1>
<span class="sub">{html.escape(subtitle)}</span>
<span class="spacer"></span>
<button id="theme" type="button" title="Toggle light/dark">light / dark</button>
</header>
{body}
<script type="application/json" id="data">{payload}</script>
<script>
{_THEME}{script}</script>
</body>
</html>
"""
