"""Task enumeration, existence-based resume, file claims (protocol v2) and work-steal picking.

Layout (OUT_ROOT defaults to outputs/):
  OUT_ROOT/<V>/<P>/<S>/{raw,composite}/<Test>/<stem>.png            results
  OUT_ROOT/.claims/<V>/<P>/<S>/<Test>/<stem>.png.lock|.failed        claims and poison markers

Known, tolerated race: stale reclaim renames by path, so two concurrent reclaimers can both win
after a crash (ABA). The worst case is one duplicate deterministic generation; outputs stay
identical because writes are atomic and inputs/seed are fixed.
"""

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from fmsr.data import list_images, output_name

HOST = socket.gethostname()
EMPTY_LOCK_GRACE_S = 120
DEFAULT_STALE_AFTER_S = 1800


@dataclass(frozen=True)
class Task:
    variant: str
    prompt_id: str
    sampling_id: str
    test: str
    image: str  # source filename in test_A

    @property
    def out_name(self):
        return output_name(self.image)

    @property
    def run_rel(self):
        return Path(self.variant) / self.prompt_id / self.sampling_id


def run_dir(out_root, variant, prompt_id, sampling_id):
    return Path(out_root) / variant / prompt_id / sampling_id


def raw_path(out_root, task):
    return Path(out_root) / task.run_rel / "raw" / task.test / task.out_name


def composite_path(out_root, task):
    return Path(out_root) / task.run_rel / "composite" / task.test / task.out_name


def claims_root(out_root):
    return Path(out_root) / ".claims"


def lock_path(out_root, task):
    return claims_root(out_root) / task.run_rel / task.test / f"{task.out_name}.lock"


def failed_path(out_root, task):
    return claims_root(out_root) / task.run_rel / task.test / f"{task.out_name}.failed"


def is_done(out_root, task):
    return raw_path(out_root, task).exists() and composite_path(out_root, task).exists()


def is_failed(out_root, task):
    return failed_path(out_root, task).exists()


def enumerate_tasks(cfg, variant, prompt_ids=None, sampling_ids=None, limit_per_dataset=None):
    """Prompt-major task order: prompt -> sampling -> test -> sorted images."""
    images = {}
    for test in cfg.datasets.tests:
        names = list_images(cfg.datasets.dir(test, "shadow"))
        images[test] = names[:limit_per_dataset] if limit_per_dataset else names
    tasks = []
    for run in cfg.runs(variant, prompt_ids, sampling_ids):
        for test, names in images.items():
            tasks.extend(Task(variant, run.prompt_id, run.sampling.id, test, n) for n in names)
    return tasks


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_is_stale(lock, stale_after_s=DEFAULT_STALE_AFTER_S, now=None):
    now = time.time() if now is None else now
    try:
        st = lock.stat()
    except FileNotFoundError:
        return False
    age = now - st.st_mtime
    try:
        info = json.loads(lock.read_text())
    except (ValueError, OSError):
        return age > EMPTY_LOCK_GRACE_S
    if info.get("host") == HOST and not _pid_alive(int(info.get("pid", -1))):
        return True
    return age > stale_after_s


def _create_lock(lock, gpu):
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "host": HOST, "gpu": gpu, "started": time.time()}).encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def try_claim(out_root, task, gpu, stale_after_s=DEFAULT_STALE_AFTER_S):
    """Claim a task. Returns True when this process owns it and it still needs work."""
    if is_failed(out_root, task):
        return False
    lock = lock_path(out_root, task)
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            _create_lock(lock, gpu)
        except FileExistsError:
            if not lock_is_stale(lock, stale_after_s):
                return False
            stale = lock.with_name(f"{lock.name}.stale.{os.getpid()}.{time.time_ns()}")
            try:
                os.rename(lock, stale)
            except FileNotFoundError:
                return False  # another reclaimer won
            stale.unlink(missing_ok=True)
            continue
        if is_done(out_root, task):
            release(out_root, task)
            return False
        return True
    return False


def release(out_root, task):
    lock_path(out_root, task).unlink(missing_ok=True)


def mark_failed(out_root, task, error, gpu):
    path = failed_path(out_root, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "host": HOST, "gpu": gpu, "time": time.time(), "error": error}))
    release(out_root, task)


def is_actively_claimed(out_root, task, stale_after_s=DEFAULT_STALE_AFTER_S):
    lock = lock_path(out_root, task)
    return lock.exists() and not lock_is_stale(lock, stale_after_s)


def pending(out_root, tasks):
    """Tasks that are neither done nor poisoned."""
    return [t for t in tasks if not is_done(out_root, t) and not is_failed(out_root, t)]


def remaining_unclaimed(out_root, tasks, stale_after_s=DEFAULT_STALE_AFTER_S):
    return sum(1 for t in pending(out_root, tasks) if not is_actively_claimed(out_root, t, stale_after_s))


def pick_next(remaining, weights, min_remaining, exclude=()):
    """Steal target: argmax remaining*weight over variants with remaining >= min_remaining; ties by name."""
    best = None
    for name in sorted(remaining):
        if name in exclude or remaining[name] <= 0 or remaining[name] < min_remaining[name]:
            continue
        score = remaining[name] * weights[name]
        if best is None or score > best[0]:
            best = (score, name)
    return None if best is None else best[1]
