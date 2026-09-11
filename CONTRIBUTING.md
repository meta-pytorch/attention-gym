# Contributing to attention-gym
We want to make contributing to this project as easy and transparent as
possible.

## GPU test environment

Use an isolated `.venv` in each checkout, with the same nightly and dependency override as CI:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132
uv pip install --python .venv/bin/python --prerelease allow -e '.[tests,linear,dev]' -r requirements-test.txt
uv run --no-sync pytest -n 6 test
```

`requirements-test.txt` temporarily pins FlashAttention source for sparse MLA attention sinks;
the published beta30 wheel does not support them. The override is applied on Linux x86_64,
where the test extra installs FlashAttention. Modal runs a small B200 sink forward/backward
preflight before the full suite, stops after a small failure budget, and does not launch the
cuDNN suite if the ordinary suite fails. Its apps run attached so cancelling a superseded
GitHub Actions run also stops the remote work.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to `attention-gym`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
