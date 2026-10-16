import re
from html import escape
from itertools import count
from pathlib import PurePosixPath
from urllib.parse import quote

from markupsafe import Markup


def define_env(env) -> None:
    sidenote_counter = count(1)
    capacity_symbols = {
        "T": ("t", "captured token capacity"),
        "T_max": ("t", "maximum captured token capacity"),
        "L": ("l", "active token count"),
        "N": ("n", "captured sequence capacity"),
        "M": ("m", "active sequence count"),
    }

    @env.macro
    def capacity_symbol(symbol: str) -> Markup:
        """Render a consistently colored CUDA Graph capacity symbol."""
        try:
            variant, title = capacity_symbols[symbol]
        except KeyError as error:
            supported = ", ".join(capacity_symbols)
            raise ValueError(
                f"unknown capacity symbol {symbol!r}; expected one of {supported}"
            ) from error
        return Markup(
            f'<code class="capacity-symbol capacity-symbol--{variant}" '
            f'title="{escape(title, quote=True)}">{escape(symbol)}</code>'
        )

    @env.macro
    def sidenote(
        phrase: str,
        note: str,
        caller=None,
        *,
        note_id: str | None = None,
        classes: str = "",
    ) -> Markup:
        """Wrap caller-provided Markdown with a phrase-linked gutter note."""
        if caller is None:
            raise ValueError("sidenote must be used with a Jinja call block")
        body = str(caller()).strip()
        if phrase not in body:
            raise ValueError(f"sidenote phrase not found in caller body: {phrase!r}")

        slug = re.sub(r"[^a-z0-9]+", "-", phrase.lower()).strip("-") or "note"
        resolved_id = note_id or f"{slug}-note-{next(sidenote_counter)}"
        ref = (
            f'<span class="sidenote-ref" tabindex="0" '
            f'aria-describedby="{escape(resolved_id, quote=True)}">{phrase}</span>'
        )
        body = body.replace(phrase, ref, 1)
        sidenote_classes = "sidenote" + (f" {classes}" if classes else "")
        return Markup(
            f'<div class="with-sidenote" markdown="1">\n'
            f'<div markdown="1">\n\n{body}\n\n</div>\n\n'
            f'<aside id="{escape(resolved_id, quote=True)}" '
            f'class="{escape(sidenote_classes, quote=True)}" markdown="1">\n\n'
            f"{note}\n\n</aside>\n</div>"
        )

    @env.macro
    def perfetto_trace(
        trace: str,
        *,
        title: str,
        alt: str,
        element_id: str | None = None,
    ) -> Markup:
        """Render a snapshot-backed, lazily loaded interactive Perfetto trace."""
        basename = trace.removesuffix(".pftrace")
        resolved_id = element_id or f"{basename.replace('_', '-')}-perfetto"
        source_path = PurePosixPath(env.page.file.src_path)
        page_depth = len(source_path.parent.parts) + (source_path.name != "index.md")
        assets = f"{'../' * page_depth}assets"
        encoded_title = quote(title)
        return Markup(
            f'<div class="trace-embed" data-trace-embed>\n'
            f'  <button class="trace-preview" type="button" aria-expanded="false" '
            f'aria-controls="{escape(resolved_id, quote=True)}">\n'
            f'    <img src="{assets}/traces/{escape(basename, quote=True)}.png" '
            f'alt="{escape(alt, quote=True)}" loading="lazy">\n'
            f'    <span class="trace-preview__label">click to interact</span>\n'
            f"  </button>\n"
            f'  <div id="{escape(resolved_id, quote=True)}" class="trace-embed__viewer" hidden>\n'
            f'    <button class="trace-embed__close" type="button">show snapshot</button>\n'
            f'    <iframe class="tutorial-widget" title="{escape(title, quote=True)}" '
            f'loading="lazy" data-src="{assets}/widgets/perfetto-trace/index.html?'
            f'trace=../../traces/{escape(basename, quote=True)}.pftrace&amp;title={encoded_title}">'
            f"</iframe>\n"
            f"  </div>\n"
            f"</div>"
        )

    @env.macro
    def plotly_chart(
        chart: str,
        *,
        title: str,
        height: int = 560,
        element_id: str | None = None,
    ) -> Markup:
        """Embed a generated standalone Plotly chart from the documentation assets."""
        if height < 1:
            raise ValueError(f"plotly chart height must be positive, got {height}")
        frame_height = height + 2  # Include the iframe border without shrinking its viewport.

        basename = chart.removesuffix(".html")
        resolved_id = element_id or f"{basename.replace('_', '-')}-plot"
        source_path = PurePosixPath(env.page.file.src_path)
        page_depth = len(source_path.parent.parts) + (source_path.name != "index.md")
        assets = f"{'../' * page_depth}assets"
        escaped_title = escape(title, quote=True)
        return Markup(
            f'<figure id="{escape(resolved_id, quote=True)}" class="plotly-chart">\n'
            f'  <iframe class="tutorial-widget plotly-chart__frame" '
            f'src="{assets}/plots/{escape(basename, quote=True)}.html" '
            f'title="{escaped_title}" loading="lazy" '
            f'style="height: {frame_height}px"></iframe>\n'
            f"  <figcaption>{escape(title)}</figcaption>\n"
            f"</figure>"
        )

    @env.macro
    def html_widget(
        widget: str,
        *,
        title: str,
        classes: str = "",
    ) -> Markup:
        """Embed a standalone HTML widget with a page-relative asset path."""
        widget_path = PurePosixPath(widget)
        if widget_path.is_absolute() or ".." in widget_path.parts:
            raise ValueError(f"widget path must stay inside docs/assets/widgets: {widget!r}")
        source_path = PurePosixPath(env.page.file.src_path)
        page_depth = len(source_path.parent.parts) + (source_path.name != "index.md")
        assets = f"{'../' * page_depth}assets"
        iframe_classes = "tutorial-widget" + (f" {classes}" if classes else "")
        return Markup(
            f'<iframe class="{escape(iframe_classes, quote=True)}" '
            f'src="{assets}/widgets/{escape(str(widget_path), quote=True)}" '
            f'title="{escape(title, quote=True)}" loading="lazy"></iframe>'
        )
