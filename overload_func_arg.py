import dis
import numpy as np
from types import FunctionType

from numba.extending import overload
from numba import njit
from numba import types
import numba


def apply(array, func):
    raise NotImplementedError


@overload(apply)
def ov_apply(array, func):
    print("At ov_apply array :: {}, func :: {}".format(array, func))
    if isinstance(func, types.Dispatcher):
        # This will happen for `add1`
        return lambda array, func: func(array)
    elif isinstance(func, types.MakeFunctionLiteral):
        # This will happen for `mul10`
        # This continues with the code similar to
        # https://github.com/IntelPython/hpat/blob/cf3d90740a4f8384ef5cd8c5a607d0d2365f5423/hpat/hiframes/pd_series_ext.py#L554-L563
        # and the lowering code in
        # https://github.com/IntelPython/hpat/blob/0265f07ff025eb213cad31db6d57b2d538d2c918/hpat/hiframes/dataframe_pass.py#L913
        # But, I don't know if it can be done with `@overload`.
        make_function_node = func.literal_value
        code = make_function_node.code
        fnty = FunctionType(
            code=code,
            globals={},
            name="some_name",
            argdefs=make_function_node.defaults,
            closure=make_function_node.closure,
        )
        jit_it = njit(fnty)
        return lambda array, func: jit_it(array)


@njit
def add1(arr):
    arr = arr.copy()
    for i in range(arr.size):
        arr[i] += 1
    return arr


def main():
    # Example 1
    # The easiest is to refer to a `@njit` decorated function.
    print("Example 1".center(80, "-"))

    @njit
    def test(array):
        array = apply(array, add1)
        return array

    array = np.arange(10)
    r = test(array)
    print(r)

    # Example 2
    # This version uses a inner function defined inside the jitted function.
    # It seems to rely on HPAT specific transformation passes
    print("Example 2".center(80, "-"))

    @njit
    def test2(array):
        mul10 = lambda x: x * 10
        array = apply(array, mul10)
        return array

    array = np.arange(10)
    r = test2(array)
    print(r)


main()
