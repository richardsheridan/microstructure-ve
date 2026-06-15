"""Regenerate the committed ABAQUS reaction-force oracles under tests/data/.

By default this runs one ABAQUS job per feature-matrix cell (``tests._matrix.matrix_cells``
x {elastic, viscoelastic}) and copies each resulting reaction-force tsv to
``tests/data/oracle_<dim>d_<mode>_<traction>_<bc>_<test_type>.tsv``. These are the numeric
truth the FE backend is validated against as matrix cells go from red to green.

Run from the repo root with the msve interpreter::

    /path/to/msve/python tools/make_oracles.py            # the 64 matrix cells
    /path/to/msve/python tools/make_oracles.py --legacy    # the 3 legacy x-uniaxial oracles

Host realities this runner is built around (do NOT "simplify" them away):

* ABAQUS ``interactive`` segfaults in ``fname_from_piped_fd``/``bcu_cleanup`` when stdin is
  not a live tty. We therefore launch **non-interactively** with stdin redirected from
  /dev/null -- no piped-fd inquiry, no overwrite prompt.
* abq2019 on this RHEL8 host aborts with a harmless **signal-6 during wrap-up even on a
  successful solve**, so process exit codes are meaningless. We never look at them: a job
  *succeeded* iff the ODB reader produces a non-empty tsv. Completion is detected from the
  filesystem (.lck gone), not from the launcher exiting.
* The box is under license contention. A job that cannot check out a license is detected
  from its logs and **retried with 60s x exponential backoff** rather than failing.
* A wedged job is stopped with ``abaqus terminate job=<name>`` (a clean shutdown through
  ABAQUS), never by killing the launcher pid -- that would corrupt ABAQUS's job queue.

Genuinely failed solves (e.g. an under-constrained exotic cell ABAQUS finds singular) are
recorded with an error excerpt and skipped; the batch keeps going and reports them at the
end. Oracles are tiny, so they commit naturally.
"""
import argparse
import pathlib
import random
import re
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests._helpers import oracle_simulation_2d, oracle_simulation_3d  # noqa: E402
from tests._matrix import (  # noqa: E402
    cell_expectations,
    matrix_cells,
    matrix_simulation,
)
from microstructure_ve.backends.abaqus import write_inp, write_odb_reader  # noqa: E402

ABAQUS = "/var/DassaultSystemes/SIMULIA/Commands/abaqus"
WORK_ROOT = REPO / "abaqus-work"
DATA = REPO / "tests" / "data"

REAL_TEST_TYPES = ("elastic", "viscoelastic")

# Markers (case-insensitive) that mean "could not get a license" -> retry, not a real fail.
_LICENSE_RE = re.compile(
    r"license|flexlm|lmgrd|failed to check out|could not .{0,40}license|"
    r"unable to .{0,40}license|queued|licenses? .{0,20}(unavailable|exceeded)",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(r"\*\*\*ERROR|has not been completed|exited with an error", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"COMPLETED SUCCESSFULLY", re.IGNORECASE)

POLL_INTERVAL = 5.0      # s between filesystem sweeps
GRACE = 5.0              # s after launch before a "no lck" is trusted (detach race)
JOB_TIMEOUT = 900.0      # s wall-clock per attempt before terminating a wedged solve
READER_TIMEOUT = 300.0   # s for the odb reader
MAX_ATTEMPTS = 8         # license retries before giving up
BACKOFF_BASE = 60.0      # s; delay = min(BACKOFF_BASE * 2**(attempt-1), BACKOFF_CAP) + jitter
BACKOFF_CAP = 600.0


class Job:
    def __init__(self, name, build, nset, dest):
        self.name = name
        self.build = build            # () -> Simulation
        self.nset = nset              # drive nodeset to extract RF/U from
        self.dest = dest              # tests/data/<name>.tsv on success
        self.attempt = 0
        self.proc = None
        self.logf = None
        self.launched_at = 0.0
        self.not_before = 0.0         # backoff gate

    @property
    def wd(self):
        return WORK_ROOT / self.name


def matrix_jobs():
    jobs = []
    for cell in matrix_cells():
        for tt in REAL_TEST_TYPES:
            name = f"oracle_{cell['dim']}d_{cell['mode']}_{cell['traction']}_{cell['bc']}_{tt}"
            nset = cell_expectations(test_type=tt, **cell)["drive_name"]
            jobs.append(Job(
                name,
                (lambda c=cell, t=tt: matrix_simulation(test_type=t, **c)),
                nset,
                DATA / (name + ".tsv"),
            ))
    return jobs


def legacy_jobs():
    return [
        Job("oracle_2d_confined", lambda: oracle_simulation_2d("confined"), "X1Y0",
            DATA / "oracle_2d_confined.tsv"),
        Job("oracle_2d_free", lambda: oracle_simulation_2d("free"), "X1Y0",
            DATA / "oracle_2d_free.tsv"),
        Job("oracle_3d_elastic", oracle_simulation_3d, "X1Y0Z0",
            DATA / "oracle_3d_elastic.tsv"),
    ]


def _read(path):
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def _job_logs(job):
    """Concatenated launcher/solver logs used for license + error detection."""
    name = job.name
    return "\n".join(
        _read(job.wd / (name + ext))
        for ext in (".launch.log", ".log", ".dat", ".msg", ".sta")
    )


def _error_excerpt(job):
    lines = []
    for ext in (".dat", ".msg", ".log"):
        for ln in _read(job.wd / (job.name + ext)).splitlines():
            if _ERROR_RE.search(ln):
                lines.append(ln.strip())
    return "; ".join(lines[:4]) or "no ***ERROR found; check logs in " + str(job.wd)


def prepare(job):
    """Fresh workdir (no stale .lck/.odb/.sta) + emitted .inp and reader."""
    wd = job.wd
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    write_inp(job.build(), wd / (job.name + ".inp"))
    write_odb_reader(wd / "read_abaqus_odb.py")
    return wd


def launch(job):
    prepare(job)
    job.attempt += 1
    job.logf = open(job.wd / (job.name + ".launch.log"), "wb")
    # non-interactive + /dev/null stdin: the segfault workaround. No exit-code reliance.
    job.proc = subprocess.Popen(
        [ABAQUS, f"job={job.name}", "cpus=1", "ask_delete=OFF"],
        cwd=job.wd, stdin=subprocess.DEVNULL, stdout=job.logf, stderr=subprocess.STDOUT,
    )
    job.launched_at = time.monotonic()


def terminate(job):
    """Stop a wedged job cleanly through ABAQUS (never kill the pid -- that corrupts the
    job queue). ``abaqus terminate`` signals the running solver to shut down."""
    try:
        subprocess.run(
            [ABAQUS, "terminate", f"job={job.name}"], cwd=job.wd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


def timed_out(job):
    return (time.monotonic() - job.launched_at) > JOB_TIMEOUT


def is_settled(job):
    """True once the solve has stopped (launcher exited and the .lck is gone), or timed out."""
    if timed_out(job):
        return True
    if (job.wd / (job.name + ".lck")).exists():
        return False
    if job.proc.poll() is None:
        return False
    # launcher may detach immediately; don't trust a missing .lck in the first few seconds
    return (time.monotonic() - job.launched_at) >= GRACE


def run_reader(job):
    """Run the ODB reader; return the tsv Path iff it produced >= 1 data row."""
    tsv = job.wd / (job.name + "-reaction-force.tsv")
    if tsv.exists():
        tsv.unlink()
    try:
        with open(job.wd / (job.name + ".reader.log"), "wb") as rl:
            subprocess.run(
                [ABAQUS, "python", "read_abaqus_odb.py", job.name, job.nset],
                cwd=job.wd, stdin=subprocess.DEVNULL, stdout=rl, stderr=subprocess.STDOUT,
                timeout=READER_TIMEOUT,
            )
    except subprocess.TimeoutExpired:
        return None
    if tsv.exists() and len(tsv.read_text(errors="ignore").splitlines()) >= 2:
        return tsv
    return None


def classify(job):
    """Outcome of a settled job: ('done', tsv) | ('license', None) | ('failed', msg)."""
    odb = job.wd / (job.name + ".odb")
    if odb.exists():
        # the signal-6 wrap-up abort doesn't stop the .odb being written; try to extract
        tsv = run_reader(job)
        if tsv is not None:
            return "done", tsv
        if _LICENSE_RE.search(_read(job.wd / (job.name + ".reader.log"))):
            return "license", None  # the reader (abaqus python) was itself license-starved
        return "failed", "odb present but reader produced no tsv: " + _error_excerpt(job)
    # no odb: license starvation vs a real pre-/in-solve error (or a termination)
    if _LICENSE_RE.search(_job_logs(job)):
        return "license", None
    if timed_out(job):
        return "failed", "timed out and was terminated before producing an .odb"
    return "failed", _error_excerpt(job)


def _backoff(attempt):
    return min(BACKOFF_BASE * 2 ** (attempt - 1), BACKOFF_CAP) + random.uniform(0, 10)


def run_all(jobs, max_inflight):
    pending, running = list(jobs), []
    done, failed = [], []
    while pending or running:
        now = time.monotonic()
        pending.sort(key=lambda j: j.not_before)
        for job in list(pending):
            if max_inflight and len(running) >= max_inflight:
                break
            if now >= job.not_before:
                launch(job)
                running.append(job)
                pending.remove(job)
                print(f"[launch {job.attempt}] {job.name}", flush=True)

        for job in list(running):
            if not is_settled(job):
                continue
            if timed_out(job):
                terminate(job)  # clean ABAQUS shutdown, not a pid kill
            running.remove(job)
            if job.logf:
                job.logf.close()
                job.logf = None

            outcome, payload = classify(job)
            if outcome == "done":
                shutil.copy(payload, job.dest)
                done.append(job)
                print(f"[done] {job.name} -> tests/data/{job.dest.name}", flush=True)
            elif outcome == "license" and job.attempt < MAX_ATTEMPTS:
                delay = _backoff(job.attempt)
                job.not_before = time.monotonic() + delay
                pending.append(job)
                print(f"[retry] {job.name}: license-starved; backoff {delay:.0f}s "
                      f"(attempt {job.attempt}/{MAX_ATTEMPTS})", flush=True)
            elif outcome == "license":
                failed.append((job, f"license starvation (gave up after {job.attempt} attempts)"))
                print(f"[give-up] {job.name}: license starvation", flush=True)
            else:
                failed.append((job, payload))
                print(f"[FAIL] {job.name}: {payload[:200]}", flush=True)

        if running or pending:
            time.sleep(POLL_INTERVAL)
    return done, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", action="store_true",
                    help="regenerate the 3 legacy x-uniaxial oracles instead of the matrix")
    ap.add_argument("--max-inflight", type=int, default=0,
                    help="cap concurrent ABAQUS jobs (0 = launch all at once, the default)")
    ap.add_argument("--filter", default="",
                    help="only run jobs whose name contains this substring (e.g. viscoelastic)")
    args = ap.parse_args(argv)

    jobs = legacy_jobs() if args.legacy else matrix_jobs()
    if args.filter:
        jobs = [j for j in jobs if args.filter in j.name]
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    if not pathlib.Path(ABAQUS).exists():
        # No ABAQUS here: still emit every .inp + reader so they can be run elsewhere.
        for job in jobs:
            prepare(job)
            print(f"wrote {job.wd / (job.name + '.inp')} (ABAQUS not found; "
                  f"run: abaqus job={job.name} cpus=1 && "
                  f"abaqus python read_abaqus_odb.py {job.name} {job.nset})")
        return

    print(f"running {len(jobs)} ABAQUS jobs "
          f"(max_inflight={args.max_inflight or 'all'})", flush=True)
    done, failed = run_all(jobs, args.max_inflight)

    print("\n==== summary ====")
    print(f"  succeeded: {len(done)}/{len(jobs)}")
    if failed:
        print(f"  FAILED:    {len(failed)}")
        for job, msg in failed:
            print(f"    - {job.name}: {msg[:160]}")
    else:
        print("  all jobs produced an oracle tsv")


if __name__ == "__main__":
    main()
