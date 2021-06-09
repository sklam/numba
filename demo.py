import sys

from contextlib import contextmanager

from numba import njit
from numba.core.dispatcher import TargetConfig
from numba._dispatcher import set_use_tls_target_stack
from numba.core.target_extension import (
    dispatcher_registry, CPUDispatcher, CPU, target_registry,
)


# ------------ A "CustomCPU" target ------------


class CustomCPU(CPU):
    pass


class CustomCPUDispatcher(CPUDispatcher):
    pass


target_registry["CustomCPU"] = CustomCPU
dispatcher_registry[target_registry["CustomCPU"]] = CustomCPUDispatcher


# ------------ For switching target ------------


@contextmanager
def switch_target(retarget):
    # __enter__
    tc = TargetConfig()
    tc.push(retarget)
    set_use_tls_target_stack(True)
    yield
    # __exit__
    tc.pop()
    set_use_tls_target_stack(False)


def retarget(cpu_disp):
    kernel = njit(_target="CustomCPU")(cpu_disp.py_func)
    return kernel


# ------------ Functions being tested ------------


@njit(_target="cpu")
def fixed_target(x):
    """
    This has a fixed target to "cpu".
    Cannot be used in "CustomCPU" target.
    """
    print("called fixed")
    return x * 2


@njit
def flex_call_fixed(x):
    """
    This has a flexible target, but uses a fixed target function.
    Cannot be used in "CustomCPU" target.
    """
    print("called flex_call_fixed")
    return fixed_target(x)


@njit
def flex_target(x):
    """
    This has a flexible target.
    Can be used in "CustomCPU" target.
    """
    print("called flex")
    return x + 1


# ------------ Testcases ------------


def case0():
    """
    Running in default (CPU) target.

    This should run fine.
    """

    @njit
    def foo(x):
        x = fixed_target(x)
        x = flex_target(x)
        return x

    r = foo(123)
    print(r)


def case1():
    """
    Running in CustomCPU target.
    This should run fine.
    """

    @njit
    def foo(x):
        x = flex_target(x)
        return x

    with switch_target(retarget):
        r = foo(123)
    print(r)


def case2():
    """
    Running in CustomCPU target.
    The non-nested call into fixed_target should raise error.
    """

    @njit
    def foo(x):
        x = fixed_target(x)
        x = flex_target(x)
        return x

    with switch_target(retarget):
        r = foo(123)
    print(r)


def case3():
    """
    Running in CustomCPU target.
    The nested call into fixed_target should raise error
    """

    @njit
    def foo(x):
        x = flex_call_fixed(x)  # calls fixed_target indirectly
        x = flex_target(x)
        return x

    with switch_target(retarget):
        r = foo(123)
    print(r)


def case4():
    """
    Same as case2 but flex_call_fixed() is invoked outside of CustomCPU target
    before the switch_target.


    This should raise error but it DOES NOT currently.
    A known problem that allows this to work because of cached implementation.

    TODO: put Flags into signature used for in-memory caching.
    """

    flex_call_fixed(123)

    print("after the cpu call".center(80, "="))

    @njit
    def foo(x):
        x = flex_call_fixed(x)  # calls fixed_target indirectly
        x = flex_target(x)
        return x

    with switch_target(retarget):
        r = foo(123)
    print(r)


def main():
    if len(sys.argv) != 2:
        print("python demo.py <case_number>")
        print("   case_number: 0, 1, 2, 3, 4")
    else:
        case_num = sys.argv[1]
        fn = globals()[f"case{case_num}"]
        fn()


if __name__ == "__main__":
    main()
