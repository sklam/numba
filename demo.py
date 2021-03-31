from numba import njit
from numba.core.extending import overload
from numba.core.utils import ConfigStack


def fastmath_status():
    pass


@overload(fastmath_status)
def ov_fastmath_status():
    flags = ConfigStack().top()
    print("HERE")
    val = "Has fastmath" if flags.fastmath else "No fastmath"

    def codegen():
        print(val)

    return codegen


@njit(fastmath=True)
def set_fastmath():
    fastmath_status()


@njit()
def foo():
    fastmath_status()
    set_fastmath()


foo()
