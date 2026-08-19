# Documentation Authoring

These instructions apply to `docs/` and to runnable examples used as documentation sources.

## Preserve the Author's Voice

- Preserve existing prose exactly unless the author explicitly requests wording changes.
- Do not silently correct spelling, grammar, capitalization, or tone while changing code or layout.
- Treat prose, code, and presentation as separate concerns.
- Read the latest file contents immediately before editing because the author may be editing concurrently.

## Executable Code Snippets

- Keep canonical tutorial code in a runnable file under `examples/`; do not maintain a second copy in Markdown.
- Mark focused regions with PyMdown snippet markers:

  ```python
  # --8<-- [start:example-name]
  example_code()
  # --8<-- [end:example-name]
  ```

- Include a named region from Markdown inside a code fence:

  ````markdown
  ```python
  --8<-- "examples/example.py:example-name"
  ```
  ````

- Split long examples into small named regions instead of showing an entire program at once.
- Keep the complete source directly runnable and give it inexpensive internal correctness checks.
- Run paired GPU examples through `gpu-run auto -- <command>`.

## CUDA Graph Examples

- Warm up the complete captured workload on a side stream before capture. Synchronize dependencies
  between the current stream and that side stream explicitly.
- Capture on a non-default stream. `torch.cuda.graph` selects one automatically; pass an explicit
  stream when the example needs to show that warmup and capture share it.
- Keep static input and output tensors alive for the graph's lifetime. Update values with in-place
  copies before replay instead of replacing tensor objects.
- Keep captured shapes, layouts, control flow, and kernel arguments static. Do not use CPU-GPU
  synchronization such as `.item()` inside capture.

## Code Annotations

- Put annotation markers in the executable source as valid comments:

  ```python
  with profiler(trace_path):  # (1)!
      run_workload()
  ```

- Put the matching explanation immediately after the Markdown code block:

  ```markdown
  1. Explanation of the annotated operation.
  ```

- Prefer annotations for concepts tied to one operation. Use highlighted line ranges only for short,
  stable snippets because line-number-based highlighting drifts as source changes.
- Annotation styling belongs in `docs/stylesheets/extra.css`; preserve visible hover and keyboard-focus
  states so markers read as interactive.
- Keep `.md-annotation` and its code-line ancestors statically positioned. Material computes tooltip
  coordinates from their in-flow offsets, so `position`, `transform`, or right-edge relocation can
  make clicks appear broken by rendering the tooltip offscreen. Highlight the row with Material's
  `.hll`-style negative margin and matching padding instead of moving the annotation control.

## Reusable Documentation Macros

- Shared parameterized documentation components live in repo-root `main.py` and are registered with
  `@env.macro` from `mkdocs-macros-plugin`.
- Use a Jinja call block when the component wraps ordinary Markdown, as `sidenote` does. Use an
  expression macro when the component emits a complete standalone component, as `perfetto_trace`
  does.
- Keep component HTML, accessibility attributes, unique-ID generation, and page-relative asset-path
  calculation inside the macro. Pages should provide only content and semantic parameters.
- Prefer a macro when the same HTML shape appears twice or is likely to be reused across tutorials.
  Use `html_widget` for standalone files under `docs/assets/widgets/` so nested pages get the correct
  relative path. Keep genuinely one-off diagrams and application mount points as explicit HTML.
- When adding a macro, document its copyable invocation in `docs/developers/documentation.md`, add a
  strict-build example, and verify the rendered interaction at desktop and narrow widths.
- Use `capacity_symbol` for explanatory references to `T`, `T_max`, `L`, `N`, and `M` so CUDA Graph
  tutorials preserve the shared capacity-versus-active-work color anchors. Keep equations and code
  literal.

## Sidenotes

- Use sidenotes only for optional context, not required instructions.
- Prefer the `sidenote` macro in `main.py` instead of hand-writing `.with-sidenote`,
  `.sidenote-ref`, IDs, accessibility attributes, and `<aside>` markup. Use a Jinja call block:
  `{% call sidenote("exact phrase", "optional note") %}...{% endcall %}`.
- Keep raw sidenote HTML only for exceptional structures that the macro cannot express cleanly,
  such as a complex hover-media note.
- Keep the dotted phrase marker and gutter note visually linked through synchronized accent ink:
  interacting with either side adds a soft wash behind the phrase and a matching inline-start stripe
  beside the note. Avoid halos or visible gutter boxes.
  Match the wide/narrow switch to Material's `60em` secondary-sidebar breakpoint so the TOC and
  gutter note transition together. On narrow screens, keep the note out of normal flow and reveal it
  only when the exact `.sidenote-ref` phrase is hovered or focused. Avoid nearby sidenotes whose
  gutter content would overlap.
- Use `.sidenote--hover-media` for an image or animation that should appear only while its exact
  phrase is hovered or focused. Store an optimized local animated WebP under `docs/assets/`, link to
  the original source, provide useful alt text, and test both pointer and keyboard interaction.
- Keep nested raw HTML flush-left. Four leading spaces can make MkDocs parse part of the component as
  a code block and prematurely close the surrounding page layout.

## Generated Traces and Other Artifacts

- Store stable documentation artifacts under `docs/assets/`; Perfetto traces belong in
  `docs/assets/traces/`, and standalone Plotly dashboards belong in `docs/assets/plots/`.
- Embed Plotly dashboards with the `plotly_chart` macro. Generate them with
  `include_plotlyjs="cdn"` so committed HTML contains the figure data without bundling several
  megabytes of Plotly runtime. Omit a fixed Plotly layout `width`; use the macro's `height=` only
  when the generated figure is taller than its default 560 px chart height. The macro owns the extra
  iframe-border pixels. Keep site-level palette, typography,
  surfaces, light/dark synchronization, and wheel-to-page scrolling in
  `docs/assets/javascripts/plotly-theme.js`; do not re-enable Plotly scroll-wheel zoom. The script is
  loaded globally by `mkdocs.yml`, so pages only invoke `plotly_chart` and never import it directly.
  Keep generated charts same-origin, responsive, and based on a standard `.plotly-graph-div`.
- Put self-contained interactive artifacts under `docs/assets/widgets/<name>/` and embed them with
  `html_widget`. For the allocator comparison, use
  `html_widget("cuda-graph-memory/max-vs-four-shared-buckets.html", ..., classes="memory-viz-frame")`;
  do not inline the generated HTML in Markdown.
- The CUDA Graph tutorial's reproducibility scripts live under
  `agent_space/cuda_graph_overcapture/`; read its `README.md` and run the same script from the
  intended source checkout via that checkout's isolated `.venv` and `PYTHONPATH=$PWD`.
- Resolve artifact paths from the example file rather than the caller's current working directory.
- Pass an extensionless path to `transformer_nuggets.utils.benchmark.profiler`; it creates parent
  directories and appends `.pftrace` for the default native Perfetto format. When traces will be
  combined with `merge-traces`, export Chrome JSON/JSON.GZ sources instead
  (`trace_format="chrome_json", gzip_trace=True`), then let `merge-traces` write the final native
  `.pftrace`; it does not consume native `.pftrace` inputs.
- Generate CUDA traces outside the documentation build. CI builds documentation on CPU only.
- Regenerate committed traces intentionally and validate the workload before treating the trace as
  representative.
- Pair an interactive trace with a same-basename PNG when the prose needs a stable visual argument.
  The PNG may be annotated externally and replaced without changing the tutorial markup.
- Before capturing a Perfetto snapshot, set the final viewport first because resizing resets track
  expansion. Use Perfetto's global `unfold_more` control, verify every group shows `expand_less`, and
  then capture at high resolution; `2560x700` with device scale `2` produces a useful `5120x1400`
  wide trace image without a large blank lower canvas.
- Use the `perfetto_trace` macro in `main.py` instead of hand-writing `.trace-preview`, iframe, and
  restore-control HTML. Pass the shared PNG/`.pftrace` basename plus a clear title and alt text.
- Set the iframe URL from `data-src` only after expansion so large traces are not fetched on initial
  page load.

## Preview and Validation

- `mkdocs.yml` watches `examples/`, so edits to included source should trigger live rebuilds after the
  preview server has been restarted with the current configuration.
- Preview with `mkdocs serve`.
- Before finalizing, run `mkdocs build --strict`.
- Run Ruff and the executable example separately; a successful documentation build does not validate
  Python behavior.

See `docs/developers/documentation.md` for the supported MkDocs components and syntax examples.
