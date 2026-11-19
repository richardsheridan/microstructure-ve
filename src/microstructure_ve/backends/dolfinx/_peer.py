"""A persistent peer process that solves whole sims with the other PETSc scalar build.

One process holds one PETSc scalar build, so ``run`` hands a sim this build should not solve to a
peer running the interpreter its variable names (``_arithmetic.peer_variable``). The client side
keeps one peer per interpreter path, started on first use and reused across calls; each serves one
request at a time. The server side is this module run as
``<interpreter> -m microstructure_ve.backends.dolfinx._peer <request fd> <reply fd>``.

The channel is a pair of pipes carrying pickles, so the peer's stdout and stderr stay the
caller's. The peer's environment is the caller's minus both interpreter variables, so it solves
with its own build only. On start the peer reports its scalar kind and its ``microstructure_ve``
package directory, and the client refuses a peer of its own build or of another source tree. A
peer exits when its request pipe closes; the client stops its peers at interpreter exit.

A peer leads its own process group, so the caller's terminal interrupt reaches the caller only,
and stopping a peer ends every process it started. Its pool workers die by ``SIGTERM``;
multiprocessing's resource tracker ignores that, and exits, unlinking the pool's semaphores, once
the processes using it are gone.

Only tuples, strings, sims, arrays and pickled exceptions cross the pipe: the server runs as
``__main__``, so a class defined here would not unpickle on the other side.
"""
from __future__ import annotations

import atexit
import contextlib
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing.connection import Connection

import microstructure_ve

from . import _arithmetic as arithmetic
from ._solver import COMPLEX_BUILD, Cancelled, ScalarBuildError

_HANDSHAKE_S = 600.0
"""Longest wait for a starting peer to report, covering a cold import of the FE stack."""

_POLL_S = 0.1
"""Interval at which a waiting caller polls ``cancel`` and the peer's liveness."""

_STOP_S = 10.0
"""Longest wait for a stopped peer's process group to exit before it is killed outright."""


def _package_dir():
    """The resolved directory of the ``microstructure_ve`` package this process imports."""
    return os.path.realpath(os.path.dirname(microstructure_ve.__file__))


class _Peer:
    """A started peer process and the caller's ends of its two pipes."""

    def __init__(self, proc, tx, rx):
        self.proc, self.tx, self.rx = proc, tx, rx
        self.lock = threading.Lock()
        self.clear_pending = False
        self._stop_lock = threading.Lock()
        self._stopped = False

    def _signal_group(self, sig):
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            return False
        return True

    def stop(self):
        """End the peer and every process in its group, wait for them, and close the pipes.

        Only the first call does this; a concurrent one waits for it, and a later one returns.
        """
        with self._stop_lock:
            if self._stopped:
                return
            self._signal_group(signal.SIGTERM)
            deadline = time.monotonic() + _STOP_S
            try:
                self.proc.wait(_STOP_S)
            except subprocess.TimeoutExpired:
                pass
            while (alive := self._signal_group(0)) and time.monotonic() < deadline:
                time.sleep(0.05)
            if alive:
                self._signal_group(signal.SIGKILL)
            self.proc.wait()
            self.tx.close()
            self.rx.close()
            self._stopped = True


_PEERS = {}
_PEERS_LOCK = threading.Lock()


def _refuse(var, path, reason):
    return ScalarBuildError(f"{var}={path} cannot solve this sim: {reason}.")


def _start(var, path):
    """Start and handshake a peer running ``path``; raise ``ScalarBuildError`` if it is unusable."""
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        raise _refuse(var, path, "no executable file at that path")
    request_r, request_w = os.pipe()
    reply_r, reply_w = os.pipe()
    env = {k: v for k, v in os.environ.items()
           if k not in (arithmetic.REAL_PYTHON_VAR, arithmetic.COMPLEX_PYTHON_VAR)}
    try:
        proc = subprocess.Popen(
            [path, "-m", __spec__.name, str(request_r), str(reply_w)],
            pass_fds=(request_r, reply_w), stdin=subprocess.DEVNULL, env=env, process_group=0)
    except OSError as e:
        os.close(request_w)
        os.close(reply_r)
        raise _refuse(var, path, f"it could not be started ({e})") from e
    finally:
        os.close(request_r)
        os.close(reply_w)
    p = _Peer(proc, Connection(request_w, readable=False), Connection(reply_r, writable=False))
    try:
        if not p.rx.poll(_HANDSHAKE_S):
            raise _refuse(var, path, f"it did not report within {_HANDSHAKE_S:.0f} s")
        try:
            complex_peer, package = p.rx.recv()
        except EOFError:
            raise _refuse(var, path, f"it exited with status {proc.wait()} before reporting "
                                     "(its stderr is above)") from None
        if complex_peer == COMPLEX_BUILD:
            kind = "complex" if COMPLEX_BUILD else "real"
            raise _refuse(var, path, f"its PETSc is the {kind}-scalar build, as this one's is")
        if package != _package_dir():
            raise _refuse(var, path, f"it imports microstructure_ve from {package}, not from "
                                     f"{_package_dir()}")
    except BaseException:
        p.stop()
        raise
    return p


def _peer_for(var, path):
    """The live peer for ``path``, started if there is none or the last one died."""
    with _PEERS_LOCK:
        p = _PEERS.get(path)
    if p is not None and p.proc.poll() is not None:
        _drop(path, p)
    with _PEERS_LOCK:
        p = _PEERS.get(path)
        if p is None:
            p = _PEERS[path] = _start(var, path)
        return p


def _drop(path, p):
    """Stop ``p`` and forget it, unless another peer already replaced it."""
    with _PEERS_LOCK:
        if _PEERS.get(path) is p:
            del _PEERS[path]
    p.stop()


@contextlib.contextmanager
def _stopped_on_failure(path, p):
    """Stop ``p`` if the block raises; a broken pipe raises ``RuntimeError`` naming the peer."""
    try:
        yield
    except (EOFError, OSError):
        _drop(path, p)
        raise RuntimeError(f"the peer process {path} exited with status {p.proc.returncode} "
                           "during a request") from None
    except BaseException:
        _drop(path, p)
        raise


def _receive(path, p, cancel=None):
    """Wait for ``p``'s reply to the request in flight and return it, polling ``cancel``."""
    with _stopped_on_failure(path, p):
        while not p.rx.poll(_POLL_S):
            if p.proc.poll() is not None:
                break
            if cancel is not None and cancel():
                raise Cancelled("cancelled by callback")
        return p.rx.recv()


def _request(path, p, message, cancel=None):
    """Send ``message`` to ``p`` and return its reply, polling ``cancel`` while waiting."""
    with _stopped_on_failure(path, p):
        p.tx.send(message)
    return _receive(path, p, cancel)


def _locked_peer(var, path):
    """The live peer for ``path``, its lock held by the caller on return."""
    while True:
        p = _peer_for(var, path)
        p.lock.acquire()
        with _PEERS_LOCK:
            current = _PEERS.get(path) is p
        if current:
            return p
        p.lock.release()  # another call stopped it while this one waited


@contextlib.contextmanager
def sending(var, path, sim, cancel, kwargs):
    """Send ``sim`` to the peer running ``path`` (named by ``var``) and yield a callable that
    waits for the reply, polling ``cancel``, and returns ``(rows, energies)`` or raises the
    peer's exception with its traceback as a note.

    The peer serves only this request until the block exits; a block that exits before the
    reply has arrived stops the peer."""
    p = _locked_peer(var, path)
    try:
        if cancel is not None and cancel():
            raise Cancelled("cancelled by callback")
        _clear_if_pending(path, p)
        with _stopped_on_failure(path, p):
            p.tx.send(("run", sim, kwargs))
        replies = []

        def collect():
            replies.append(_receive(path, p, cancel))
            return _unpack(path, *replies[0])

        try:
            yield collect
        finally:
            if not replies:
                _drop(path, p)
    finally:
        p.lock.release()


def solve(var, path, sim, cancel, kwargs):
    """Solve ``sim`` in the peer running ``path`` (named by ``var``); return
    ``(rows, energies)``, or raise the peer's exception with its traceback as a note."""
    with sending(var, path, sim, cancel, kwargs) as collect:
        return collect()


def _unpack(path, status, payload, text):
    """The payload of an ``"ok"`` reply; otherwise raise the exception the reply carries."""
    if status == "ok":
        return payload
    try:
        exc = pickle.loads(payload) if payload is not None else None
    except Exception:
        exc = None
    if not isinstance(exc, BaseException):
        raise RuntimeError(f"the peer process {path} raised:\n{text}")
    exc.add_note(f"Raised in the peer process {path}:\n{text}")
    raise exc


def _clear_if_pending(path, p):
    """Clear ``p``'s cache if a clear arrived while it served a request; hold ``p.lock``."""
    if p.clear_pending and p.proc.poll() is None:
        p.clear_pending = False
        _request(path, p, ("clear",))


def clear_peers():
    """Clear the FE cache of every live peer, or of a busy one before its next request; start
    none and wait for no request."""
    with _PEERS_LOCK:
        peers = list(_PEERS.items())
    for path, p in peers:
        p.clear_pending = True
        if p.lock.acquire(blocking=False):
            try:
                _clear_if_pending(path, p)
            except RuntimeError:
                pass  # it died; the next routed call replaces it
            finally:
                p.lock.release()


@atexit.register
def _stop_all():
    with _PEERS_LOCK:
        peers = list(_PEERS.values())
        _PEERS.clear()
    for p in peers:
        p.stop()


def _serve(request_fd, reply_fd):
    """Report this build and package directory, then answer requests until the pipe closes."""
    from petsc4py import PETSc

    from . import _run

    rx = Connection(request_fd, writable=False)
    tx = Connection(reply_fd, readable=False)
    tx.send((arithmetic.is_complex_scalar(PETSc.ScalarType), _package_dir()))
    while True:
        try:
            message = rx.recv()
        except EOFError:
            return
        if message[0] == "clear":
            _run.clear_cache()
            tx.send(("ok", None, None))
            continue
        _, sim, kwargs = message
        try:
            out = _run._solve(sim, cancel=None, **kwargs)
        except Exception as e:
            text = traceback.format_exc()
            try:
                payload = pickle.dumps(e)
            except Exception:
                payload = None
            tx.send(("error", payload, text))
        else:
            tx.send(("ok", out, None))


if __name__ == "__main__":
    _serve(int(sys.argv[1]), int(sys.argv[2]))
