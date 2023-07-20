from pprint import pprint

from numba import njit
from numba.tests.support import override_config


def sum1d(n):
    c = 0
    for i in range(n):
        c += i
    return c

def test_sum1d_compile(benchmark):
    @benchmark
    def compile():
        njit(sum1d).compile("(intp,)")

def test_sum1d_run(benchmark):
    disp = njit(sum1d)
    disp.compile("(intp,)")

    @benchmark
    def run():
        disp(1024)


def test_sum1d_llvm_pass_info():
    with override_config("LLVM_PASS_TIMINGS", True):
        disp = njit(sum1d)
        disp.compile("(intp,)")

    md = disp.get_metadata(disp.signatures[0])
    timings = md['llvm_pass_timings']
    print(timings)


# -----------------------------------------------------------------------------

def sum2d(n):
    c = 0
    for i in range(n):
        for j in range(i):
            c += i + j
    for i in range(n):
        for j in range(i):
            c += i + j

    return c


def test_sum2d_compile(benchmark):
    @benchmark
    def compile():
        njit(sum2d).compile("(intp,)")

def test_sum2d_run(benchmark):
    disp = njit(sum2d)
    disp.compile("(intp,)")

    @benchmark
    def run():
        disp(1024)


def test_sum2d_llvm_pass_info():
    with override_config("LLVM_PASS_TIMINGS", True):
        disp = njit(sum2d)
        disp.compile("(intp,)")

    md = disp.get_metadata(disp.signatures[0])
    pprint(md)
    timings = md['llvm_pass_timings']
    print(timings)


# def test_sum2d_event_timings():
#     disp = njit(sum2d)
#     disp.compile("(intp,)")
#     md = disp.get_metadata(disp.signatures[0])
#     pprint(md)
