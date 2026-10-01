"""python test_issue.py -> exits 1 if a key loads another's data."""
import multiprocessing as mp
import shutil
import sys
import tempfile
from pathlib import Path

from numba.core.caching import IndexDataCacheFile

N_PROCS = 24
N_ROUNDS = 5
_STAMP = ("proof", 1)
_BASE = "collision_probe"
_barrier = None


def _init(barrier):
    global _barrier
    _barrier = barrier


def _saver(args):
    idx, cache_dir = args
    cache = IndexDataCacheFile(
        cache_path=cache_dir, filename_base=_BASE, source_stamp=_STAMP
    )
    _barrier.wait()                       # make the saves overlap
    cache.save(f"k{idx}", f"payload-{idx}")


def main():
    ctx = mp.get_context("forkserver")
    total_wrong = 0
    for rnd in range(1, N_ROUNDS + 1):
        cache_dir = tempfile.mkdtemp(prefix="numba-cache-proof-")
        barrier = ctx.Barrier(N_PROCS)
        with ctx.Pool(N_PROCS, initializer=_init, initargs=(barrier,)) as pool:
            pool.map(_saver, [(i, cache_dir) for i in range(N_PROCS)])

        # A FRESH reader, as a later process would be.
        verifier = IndexDataCacheFile(
            cache_path=cache_dir, filename_base=_BASE, source_stamp=_STAMP
        )
        misses = wrong = 0
        for i in range(N_PROCS):
            got = verifier.load(f"k{i}")
            if got is None:
                misses += 1               # benign: recompiles
            elif got != f"payload-{i}":
                wrong += 1               # THE BUG: another key's data
                print(f"    WRONG: key 'k{i}' -> {got!r}")
        nbc = sorted(Path(cache_dir).glob(f"{_BASE}*.nbc"))
        print(
            f"round {rnd}: {len(nbc)} data files for {N_PROCS} distinct keys; "
            f"{misses} misses, {wrong} wrong-payload loads"
        )
        total_wrong += wrong
        shutil.rmtree(cache_dir, ignore_errors=True)
    sys.exit(1 if total_wrong else 0)


if __name__ == "__main__":
    main()
