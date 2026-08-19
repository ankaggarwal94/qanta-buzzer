"""QA-011: env-triggered no-network guard for subprocess CLI runs (R-028).

Injected into child interpreters via PYTHONPATH by tests/_colm_aims_helpers
``run_cli``; active only when COLM_AIMS_TEST_NO_NET=1, so no other test's
subprocesses are affected. AF_UNIX (local IPC) stays allowed, mirroring the
parent-process ``colm_no_network`` fixture.
"""
import os
import socket

if os.environ.get("COLM_AIMS_TEST_NO_NET") == "1":
    _real_connect = socket.socket.connect

    def _guarded_connect(self, address):
        if self.family == socket.AF_UNIX:
            return _real_connect(self, address)
        raise RuntimeError(
            "network disabled in colm_aims tests (R-028 no-network guard)"
        )

    def _guarded_create_connection(*args, **kwargs):
        raise RuntimeError(
            "network disabled in colm_aims tests (R-028 no-network guard)"
        )

    socket.socket.connect = _guarded_connect
    socket.create_connection = _guarded_create_connection
