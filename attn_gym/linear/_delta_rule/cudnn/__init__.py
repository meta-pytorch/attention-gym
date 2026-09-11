"""Optional cuDNN implementation shared by delta-rule attention variants.

Importing this package does not load CuTeDSL. Variant adapters lazily load the shared
launchers and vendored Frost kernels only after explicit cuDNN dispatch.
"""
