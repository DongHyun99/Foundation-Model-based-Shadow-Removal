import json
import multiprocessing as mp
import os
import time

from fmsr import tasks as T


def _task(i=0):
    return T.Task("V", "p1", "default", "AISTD", f"img{i}.jpg")


def _finish(out, task, payload=b"x"):
    for p in (T.raw_path(out, task), T.composite_path(out, task)):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / f".{p.name}.{os.getpid()}.tmp"
        tmp.write_bytes(payload)
        os.replace(tmp, p)


def test_paths_and_claims_root(tmp_path):
    t = _task()
    assert T.raw_path(tmp_path, t) == tmp_path / "V/p1/default/raw/AISTD/img0.png"
    assert T.lock_path(tmp_path, t) == tmp_path / ".claims/V/p1/default/AISTD/img0.png.lock"
    assert T.claims_root(tmp_path / "_smoke") == tmp_path / "_smoke/.claims"


def test_claim_is_exclusive_and_released(tmp_path):
    t = _task()
    assert T.try_claim(tmp_path, t, gpu=0)
    assert not T.try_claim(tmp_path, t, gpu=1)  # held by a live owner (this pid)
    T.release(tmp_path, t)
    assert T.try_claim(tmp_path, t, gpu=1)


def test_claim_skips_done_and_failed(tmp_path):
    done, bad = _task(1), _task(2)
    _finish(tmp_path, done)
    assert not T.try_claim(tmp_path, done, gpu=0)
    assert not T.lock_path(tmp_path, done).exists()
    T.mark_failed(tmp_path, bad, "boom", gpu=0)
    assert not T.try_claim(tmp_path, bad, gpu=0)
    assert T.pending(tmp_path, [done, bad, _task(3)]) == [_task(3)]


def _write_lock(path, pid, host=T.HOST):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": pid, "host": host, "gpu": 0, "started": time.time()}))


def _dead_pid():
    p = mp.Process(target=lambda: None)
    p.start()
    p.join()
    return p.pid


def test_dead_owner_lock_is_reclaimed(tmp_path):
    t = _task()
    _write_lock(T.lock_path(tmp_path, t), _dead_pid())
    assert T.try_claim(tmp_path, t, gpu=0)
    assert json.loads(T.lock_path(tmp_path, t).read_text())["pid"] == os.getpid()


def test_live_foreign_host_lock_respected_until_stale_age(tmp_path):
    t = _task()
    lock = T.lock_path(tmp_path, t)
    _write_lock(lock, 1, host="other-host")
    assert not T.try_claim(tmp_path, t, gpu=0)
    old = time.time() - 4000
    os.utime(lock, (old, old))
    assert T.try_claim(tmp_path, t, gpu=0)


def test_empty_lock_stale_after_grace(tmp_path):
    t = _task()
    lock = T.lock_path(tmp_path, t)
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    assert not T.try_claim(tmp_path, t, gpu=0)
    old = time.time() - T.EMPTY_LOCK_GRACE_S - 5
    os.utime(lock, (old, old))
    assert T.try_claim(tmp_path, t, gpu=0)


def _reclaim_worker(out, barrier):
    t = _task()
    barrier.wait()
    if T.try_claim(out, t, gpu=0):
        _finish(out, t, payload=b"deterministic")
        T.release(out, t)


def test_concurrent_reclaim_invariants(tmp_path):
    """Racing reclaimers may duplicate one generation (tolerated) but never corrupt outputs."""
    t = _task()
    _write_lock(T.lock_path(tmp_path, t), _dead_pid())
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(6)
    procs = [ctx.Process(target=_reclaim_worker, args=(tmp_path, barrier)) for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert T.is_done(tmp_path, t)
    for p in (T.raw_path(tmp_path, t), T.composite_path(tmp_path, t)):
        assert p.read_bytes() == b"deterministic"
        assert [x.name for x in p.parent.iterdir()] == [p.name]  # no tmp leftovers


def test_pick_next_rules():
    weights = {"A": 100, "B": 28, "C": 50}
    mins = {"A": 10, "B": 60, "C": 60}
    # A would score 900 but is below its min_remaining; C (5000) beats B (59*28 < 60 -> ineligible).
    assert T.pick_next({"A": 9, "B": 59, "C": 100}, weights, mins) == "C"
    assert T.pick_next({"A": 50, "B": 500, "C": 100}, weights, mins) == "B"  # 14000 > 5000 = 5000
    assert T.pick_next({"A": 50, "B": 500, "C": 100}, weights, mins, exclude=("B",)) == "A"  # tie -> name
    assert T.pick_next({"A": 0, "B": 10, "C": 0}, weights, mins) is None
