"""Warpdrive rewrites of NVIDIA cudnn-frontend's Frost GDN/KDA linear-attention kernels.

The ``cudnn`` backend dispatches here: ``drivers`` plan each call and ``bundles`` compile its
kernels into single launches. ``LICENSING.md`` gives each file's licence.
"""
