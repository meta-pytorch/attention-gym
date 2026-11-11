"""Host-only kernel factory for declaration tests: nothing built here is compiled.

Traced code shared between test modules would escape the ``jit_cache`` source key (see "Adding
CuTeDSL Kernels" in AGENTS.md), so keep compiled toys in their own test module.
"""

from attn_gym.warpdrive import WarpSpecialized


def probe_kernel(channels, roles, cluster=None) -> WarpSpecialized:
    """A kernel with the given channel factories (attribute name -> factory) and roles."""
    cls = type("Probe", (WarpSpecialized,), dict(roles))

    def init(self):
        for name, make in channels.items():
            setattr(self, name, make())
        if cluster is not None:
            self.cluster = cluster

    cls.__init__ = init
    return cls()
