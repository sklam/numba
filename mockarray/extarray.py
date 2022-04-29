from typing import Tuple
import ctypes

from numba.core import types


import operator
from numba import njit, typeof, types
from numba.extending import (
    register_model,
    models,
    typeof_impl,
    overload,
    overload_method,
    overload_classmethod,
    intrinsic,
    overload_attribute,
)
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core import cgutils
import numpy as np
from numba.core.typing.npydecl import parse_dtype, parse_shape
from numba.np.arrayobj import _parse_empty_args, _empty_nd_impl
from numba.core.imputils import impl_ret_new_ref

from numba.core.unsafe import refcount

import extarray_capi

# ----------------------------------------------------------------------------
# Part 1: Setup basic `ExtArray` class


class ExtArray:
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        handle: extarray_capi.ExtArrayHandlePtr,
    ):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.handle = handle
        self.size = np.prod(self.shape)
        self.nbytes = extarray_capi.getnbytes(handle)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def layout(self):
        # Always C layout
        return "C"

    @property
    def handle_addr(self) -> int:
        return ctypes.cast(self.handle, ctypes.c_void_p).value

    @property
    def data_addr(self) -> int:
        return extarray_capi.getpointer(self.handle)

    def as_numpy(self):
        buf = (ctypes.c_byte * self.nbytes).from_address(
            extarray_capi.getpointer(self.handle)
        )
        return np.ndarray(shape=self.shape, dtype=self.dtype, buffer=buf)

    def __eq__(self, other):
        return all(
            [
                self.shape == other.shape,
                self.dtype == other.dtype,
                self.handle_addr == other.handle_addr,
            ]
        )

    def __repr__(self):
        return f"ExtArray({self.shape}, 0x{self.handle_addr:x})"


def test_extarray_basic():
    nelem = 10
    handle = extarray_capi.alloc(nelem * np.dtype(np.float64).itemsize)
    ea = ExtArray(shape=(nelem,), dtype=np.float64, handle=handle)
    assert ea.handle == handle

    ptr = extarray_capi.getpointer(ea.handle)
    assert ea.handle != ptr
    print("pointer", hex(ptr))

    cbuf = ea.as_numpy()
    print(cbuf)  # uninitialized garbage values

    extarray_capi.free(handle)


# ----------------------------------------------------------------------------
# Part 2: Setup Numba type for ExtArray and numba.typeof


class ExtArrayType(types.Array):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"ExtArrayType({self.name})"

    def as_base_array_type(self):
        return types.Array(
            dtype=self.dtype, ndim=self.ndim, layout=self.layout
        )


@typeof_impl.register(ExtArray)
def typeof_index(val, c):
    return ExtArrayType(typeof(val.dtype), val.ndim, val.layout)


def test_extarraytype_basic():
    nelem = 10
    handle = extarray_capi.alloc(nelem * np.dtype(np.float64).itemsize)
    ea = ExtArray(shape=(nelem,), dtype=np.float64, handle=handle)

    ea_typ = typeof(ea)
    print(ea_typ)

    assert ea_typ.layout == "C"
    assert ea_typ.ndim == 1
    assert ea_typ.dtype == typeof(np.dtype(np.float64))


# ----------------------------------------------------------------------------
# Part 3: Datamodel, box and unbox


@register_model(ExtArrayType)
class _ExtArrayTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            # copied from base array type
            ("meminfo", types.MemInfoPointer(fe_type.dtype)),
            ("parent", types.pyobject),
            ("nitems", types.intp),
            ("itemsize", types.intp),
            ("data", types.CPointer(fe_type.dtype)),
            ("shape", types.UniTuple(types.intp, ndim)),
            ("strides", types.UniTuple(types.intp, ndim)),
            # extra fields
            ("handle", types.voidptr),
        ]
        super().__init__(dmm, fe_type, members)


@unbox(ExtArrayType)
def unbox_extarray(typ, obj, c):
    as_numpy = c.pyapi.object_getattr_string(obj, "as_numpy")
    nparr = c.pyapi.call_function_objargs(as_numpy, ())
    data_array_type = typ.as_base_array_type()
    ary = c.unbox(data_array_type, nparr)
    c.pyapi.decref(nparr)

    data_ary = c.context.make_array(data_array_type)(
        c.context, c.builder, value=ary.value
    )

    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder)
    nativeary.meminfo = data_ary.meminfo
    nativeary.nitems = data_ary.nitems
    nativeary.itemsize = data_ary.itemsize
    nativeary.data = data_ary.data
    nativeary.shape = data_ary.shape
    nativeary.strides = data_ary.strides

    handle_obj = c.pyapi.object_getattr_string(obj, "handle_addr")
    handle = c.unbox(types.uintp, handle_obj).value
    c.pyapi.decref(handle_obj)

    nativeary.handle = c.context.cast(
        c.builder, handle, types.uintp, types.voidptr
    )

    aryptr = nativeary._getpointer()
    return NativeValue(c.builder.load(aryptr), is_error=c.pyapi.c_api_error())


def _box_extarray(array, handle):
    hldr = ctypes.cast(
        ctypes.c_void_p(handle), extarray_capi.ExtArrayHandlePtr
    )
    return ExtArray(shape=array.shape, dtype=array.dtype, handle=hldr)


@box(ExtArrayType)
def box_extarray(typ, val, c):
    base_ary = c.context.make_array(typ.as_base_array_type())(
        c.context, c.builder
    )
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder, value=val)
    cgutils.copy_struct(base_ary, nativeary)

    handle = nativeary.handle
    data_ary = c.box(typ.as_base_array_type(), base_ary._getvalue())

    boxer = c.pyapi.unserialize(c.pyapi.serialize_object(_box_extarray))
    lluintp = c.context.get_value_type(types.uintp)
    handle_obj = c.pyapi.long_from_longlong(
        c.builder.ptrtoint(handle, lluintp)
    )
    retval = c.pyapi.call_function_objargs(boxer, [data_ary, handle_obj])
    c.pyapi.decref(handle_obj)

    return retval


def test_unbox():
    @njit
    def foo(ea):
        print("ea refcount", refcount.get_refcount(ea))

    nelem = 10
    handle = extarray_capi.alloc(nelem * np.dtype(np.float64).itemsize)
    ea = ExtArray(shape=(nelem,), dtype=np.float64, handle=handle)

    foo(ea)


def test_unbox_box():
    @njit
    def foo(ea):
        return ea

    nelem = 10
    handle = extarray_capi.alloc(nelem * np.dtype(np.float64).itemsize)
    ea = ExtArray(shape=(nelem,), dtype=np.float64, handle=handle)

    ret = foo(ea)
    assert ea == ret


# ----------------------------------------------------------------------------
# Part 4: Allocator


def extarray_empty(shape, dtype):
    nelem = np.prod(shape)
    dtype = np.dtype(dtype)
    handle = extarray_capi.alloc(nelem * dtype.itemsize)
    return ExtArray(shape=shape, dtype=dtype, handle=handle)


@overload(other_numpy.empty)
def ol_empty_impl(shape, dtype=None):
    def impl(shape, dtype=None):
        return oat_empty_intrin(shape, dtype)

    return impl


def test_allocator():
    # @njit
    def foo(shape):
        return extarray_empty(shape, dtype=np.float64)

    foo()