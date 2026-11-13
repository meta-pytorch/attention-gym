"""tcgen05 thread-sync fences (PTX ISA 9.7.18.6.3, "Specialized Inter-thread Synchronization").

Besides the standard memory-consistency mechanisms, an asynchronous tcgen05 operation (``mma``,
``cp``, ``shift``, ``ld``, ``st``) is ordered against another thread's tcgen05 operation through
a thread synchronization (mbarrier, named barrier) when the signalling thread runs
``tcgen05.fence::before_thread_sync`` between its operation and the signal, and the waiting
thread runs ``tcgen05.fence::after_thread_sync`` between the wait and its operation; warpdrive
uses this route. A ``tcgen05.commit`` performs the first fence for the MMAs it tracks.

The layer emits both around every managed wait and signal of a role that issues tcgen05
operations: one that takes ``tmem: Tmem`` or signals a channel side as ``Umma``
(``RoleTrace.tcgen05``). Whether one synchronization carries tcgen05 ordering is not local
(it composes through other roles' synchronizations), so every one of such a role's is fenced.
Raw code calls ``tcgen05_before_thread_sync``/``tcgen05_after_thread_sync`` itself.
"""

from __future__ import annotations

from cutlass._mlir.dialects import nvvm


def tcgen05_before_thread_sync():
    """``tcgen05.fence::before_thread_sync``: order this thread's earlier asynchronous tcgen05
    operations (``mma``, ``cp``, ``shift``, ``ld``, ``st``) before its next thread
    synchronization (an mbarrier or named-barrier arrive or sync). Emit it after the
    accesses' completion waits (``tcgen05.wait::ld``/``st``) and before the signal. A
    ``tcgen05.commit`` performs it implicitly for the operations it tracks (PTX ISA
    9.7.18.6.3 and 9.7.18.6.4)."""
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)


def tcgen05_after_thread_sync():
    """``tcgen05.fence::after_thread_sync``: order this thread's later asynchronous tcgen05
    operations after its preceding thread synchronization (an mbarrier wait or barrier
    sync), and so after what other threads ordered before their matching signal. Emit it
    after the wait and before the first tcgen05 operation it guards (PTX ISA 9.7.18.6.3)."""
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)


def before_sync(trace):
    """The layer's ``tcgen05_before_thread_sync()`` ahead of a managed signal: emitted only in
    a role that issues tcgen05 operations (``RoleTrace.tcgen05``)."""
    if trace is not None and trace.tcgen05:
        tcgen05_before_thread_sync()


def after_sync(trace):
    """The layer's ``tcgen05_after_thread_sync()`` after a managed wait, in a role that issues
    tcgen05 operations."""
    if trace is not None and trace.tcgen05:
        tcgen05_after_thread_sync()
