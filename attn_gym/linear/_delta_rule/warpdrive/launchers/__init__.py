"""Validated launchers of the warpdrive delta-rule backend, shared by the GDN and KDA adapters.

Importing this package does not load CuTeDSL. Variant adapters lazily load these launchers,
and through them the kernels, only after explicit ``warpdrive`` dispatch.
"""
