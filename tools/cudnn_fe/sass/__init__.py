"""SASS equivalence gate for the vendored cudnn-frontend GDN and KDA kernels.

``snapshot`` drives the Attention Gym drivers (``cudnn_fe.gdn``, ``cudnn_fe.kda``,
``cudnn_fe.summary``) on small shapes that select every plan, captures each compiled cubin, and
writes per-label ``.sass``/``.ops``/``.res``/``.fns`` files plus ``ARTIFACTS.txt``, ``CASES.txt``
and ``META.txt``. ``diff`` classifies every kernel of two snapshots as identical, noise or a real
change and exits nonzero on real changes. The GPU-free parsing and classification live in
``sass.py`` and ``diff.py``; ``capture.py`` and ``cases.py`` need CuTeDSL and a CUDA device.

    python -m tools.cudnn_fe.sass snapshot <out> [--tree PATH] [--cases SUBSTR ...]
    python -m tools.cudnn_fe.sass diff <a> <b> [--strict] [--quiet]
"""
