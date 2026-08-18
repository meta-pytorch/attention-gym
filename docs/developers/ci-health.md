# CI Health

<div class="ci-health-intro">
  <p>
    A live view of recent GitHub Actions runs, job reliability, execution time, and
    test failures. Cancelled, skipped, neutral, and approval-waiting runs are excluded
    from success-rate calculations.
  </p>
</div>

<div id="ci-health-dashboard" class="ci-health" aria-live="polite">
  <div class="ci-health-loading">Loading recent CI activity…</div>
</div>

<script
  src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js"
  integrity="sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ"
  crossorigin="anonymous"
></script>
<script src="../../assets/javascripts/ci-health.js"></script>
