
import cProfile
from bench_basic import sum2d
from numba.core.rvsdg_frontend import bcinterp

with cProfile.Profile() as pr:
    bcinterp.run_frontend(sum2d)


pr.print_stats(sort='cumtime')

pr.dump_stats("rvsdg_direct_bench.prof")
#### make flamegraph with
# python -m flameprof rvsdg_direct_bench.prof