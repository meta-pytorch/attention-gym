"""Warpdrive rewrites of NVIDIA cudnn-frontend's GDN/KDA linear-attention kernels.

The ``warpdrive`` backend dispatches here: ``launchers`` validate the public layout, ``drivers``
plan each call and ``bundles`` compile its kernels into single launches. ``LICENSING.md``
gives each file's licence.
"""
