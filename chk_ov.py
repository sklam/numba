from numba.core.extending import overload
from numba.core import types
from numba.core.datamodel import models
from numba.core.registry import cpu_target
from numba import njit
import numpy as np

def foo():
    pass



class Tensor(types.Array):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"Tensor({self.name})"


models.register_default(Tensor)(models.ArrayModel)

@overload(foo, use_impl_for=True)
def ov_foo_base(x, y):
    if isinstance(x, types.Array) and isinstance(y, types.Array):
        def impl(x, y):
            return "base"
        impl.impl_for = types.Array
        return impl


class OVer:
    def __init__(self, sig, impl):
        self.expected_sig = sig  # this attribute is used as magic
        self.impl = impl


@overload(foo, use_impl_for=True)
def ov_foo_tensor(x, y):
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        def impl(x, y):
            return "tensor"
        impl.impl_for = Tensor
        return impl


tyctx = cpu_target.typing_context
tyctx.refresh()

array_t = types.Array(types.intp, 2, "A")
tensor_t = Tensor(types.intp, 2, "A")
fnty = tyctx.resolve_value_type(foo)
print(fnty)

# Base
out = tyctx.resolve_function_type(fnty, (array_t, array_t), {})
print(out)

# Tensor
out = tyctx.resolve_function_type(fnty, (tensor_t, tensor_t), {})
print(out)

