from typing import Tuple
import ctypes

from numba.core import types


import operator
from llvmlite import ir as llvmir
from llvmlite import binding as llvm
from numba import njit, typeof
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
from numba.core import typing
import numpy as np
from numba.core.typing.npydecl import parse_dtype, parse_shape
from numba.np.arrayobj import _parse_empty_args, _empty_nd_impl
from numba.core.imputils import impl_ret_new_ref
from numba.core.unsafe import refcount
from numba.np import numpy_support

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
    """This is needed for overloading getitem/setitem"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"ExtArrayType({self.name})"

    def as_base_array_type(self):
        return types.Array(
            dtype=self.dtype, ndim=self.ndim, layout=self.layout
        )

    # Needed for overloading ufunc
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NotImplemented  # disable default ufunc
        if method == "__call__":
            for inp in inputs:
                if not isinstance(inp, (types.Array, types.Number)):
                    return NotImplemented
            # # Ban if all arguments are OtherArrayType
            # if all(isinstance(inp, OtherArrayType) for inp in inputs):
            #     return NotImplemented
            return NotImplemented
            return ExtArrayType
        else:
            return NotImplemented

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if readonly is None:
            readonly = not self.mutable
        return type(self)(dtype=dtype, ndim=ndim, layout=layout, readonly=readonly,
                          aligned=self.aligned)



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


# Call these once to bind the library function to LLVM
llvm.add_symbol(
    "extarray_alloc", ctypes.cast(extarray_capi.alloc, ctypes.c_void_p).value
)

llvm.add_symbol(
    "extarray_getpointer",
    ctypes.cast(extarray_capi.getpointer, ctypes.c_void_p).value,
)

llvm.add_symbol(
    "extarray_make_meminfo",
    ctypes.cast(extarray_capi.make_meminfo, ctypes.c_void_p).value,
)
####

def cast_integer(context, builder, val, fromty, toty):
    # XXX Shouldn't require this.
    if toty.bitwidth == fromty.bitwidth:
        # Just a change of signedness
        return val
    elif toty.bitwidth < fromty.bitwidth:
        # Downcast
        return builder.trunc(val, context.get_value_type(toty))
    elif fromty.signed:
        # Signed upcast
        return builder.sext(val, context.get_value_type(toty))
    else:
        # Unsigned upcast
        return builder.zext(val, context.get_value_type(toty))


@intrinsic
def intrin_alloc(typingctx, allocsize, align):
    """Intrinsic to call into the allocator for Array
    """

    def codegen(context, builder, signature, args):
        [allocsize, align] = args

        # XXX: error are being eaten.
        #      example: replace the next line with `align_u32 = align`
        # align_u32 = cast_integer(
        #     context, builder, align, signature.args[1], types.uint32
        # )
        meminfo = extarray_new_meminfo(builder, allocsize)
        return meminfo

    from numba.core.typing import signature

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = signature(mip, allocsize, align)
    return sig, codegen


@overload_classmethod(ExtArrayType, "_allocate")
def oat_allocate(cls, allocsize, alignment):
    def impl(cls, allocsize, alignment):
        return intrin_alloc(allocsize, alignment)

    return impl

####

def extarray_new_meminfo(builder, nbytes):
    # Call extarray_alloc to allocate a ExtArray handle
    alloc_fnty = llvmir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t])
    alloc_fn = cgutils.get_or_insert_function(
        builder.module, alloc_fnty, "extarray_alloc",
    )
    handle = builder.call(alloc_fn, [nbytes])

    # Make Meminfo
    meminfo_fnty = llvmir.FunctionType(
        cgutils.voidptr_t, [cgutils.voidptr_t]
    )
    meminfo_fn = cgutils.get_or_insert_function(
        builder.module, meminfo_fnty, name="extarray_make_meminfo"
    )
    meminfo = builder.call(meminfo_fn, [handle])
    return meminfo


@intrinsic(prefer_literal=True)
def ext_array_alloc(typingctx, nbytes, nitems, ndim, shape, dtype, itemsize):
    if not isinstance(ndim, types.IntegerLiteral):
        # reject if ndim is not a literal
        return
    # note: skipping error checking for other arguments

    def codegen(context, builder, signature, args):
        [nbytes, nitems, ndim, shape, dtype, itemsize] = args

        # Call extarray_alloc to allocate a ExtArray handle
        alloc_fnty = llvmir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t])
        alloc_fn = cgutils.get_or_insert_function(
            builder.module, alloc_fnty, "extarray_alloc",
        )
        handle = builder.call(alloc_fn, [nbytes])

        # Call extarray_getpointer
        getptr_fnty = llvmir.FunctionType(
            cgutils.voidptr_t, [cgutils.voidptr_t]
        )
        getptr_fn = cgutils.get_or_insert_function(
            builder.module, getptr_fnty, "extarray_getpointer",
        )
        dataptr = builder.call(getptr_fn, [handle])

        nativearycls = context.make_array(signature.return_type)
        nativeary = nativearycls(context, builder)

        # Make Meminfo
        meminfo_fnty = llvmir.FunctionType(
            cgutils.voidptr_t, [cgutils.voidptr_t]
        )
        meminfo_fn = cgutils.get_or_insert_function(
            builder.module, meminfo_fnty, name="extarray_make_meminfo"
        )
        meminfo = builder.call(meminfo_fn, [handle])

        # compute strides
        strides = []
        cur_stride = itemsize
        for s in reversed(cgutils.unpack_tuple(builder, shape)):
            strides.append(cur_stride)
            cur_stride = builder.mul(cur_stride, s)
        strides.reverse()

        # populate array struct (same as populate_array)

        nativeary.meminfo = meminfo
        nativeary.nitems = nitems
        nativeary.itemsize = itemsize
        nativeary.data = builder.bitcast(dataptr, nativeary.data.type)
        nativeary.shape = shape
        nativeary.strides = cgutils.pack_array(builder, strides)
        nativeary.handle = handle

        return nativeary._getvalue()

    arraytype = ExtArrayType(
        ndim=ndim.literal_value, dtype=dtype.dtype, layout="C"
    )
    sig = typing.signature(
        arraytype, nbytes, nitems, ndim, shape, dtype, itemsize
    )
    return sig, codegen


@overload(extarray_empty)
def ol_empty_impl(shape, dtype):
    dtype = numpy_support.as_dtype(dtype.dtype)
    itemsize = dtype.itemsize

    def impl(shape, dtype):
        nelem = np.prod(np.asarray(shape))
        return ext_array_alloc(
            nelem * itemsize, nelem, len(shape), shape, dtype, itemsize
        )

    return impl


def test_allocator():
    @njit
    def foo(shape):
        return extarray_empty(shape, dtype=np.float64)

    shape = (2, 3)
    r = foo(shape)
    arr = r.as_numpy()
    assert arr.shape == shape
    assert arr.dtype == np.dtype(np.float64)
    assert arr.size == np.prod(shape)


# ----------------------------------------------------------------------------
# Part 4: Getitem Setitem


@overload_method(ExtArrayType, "as_numpy")
def extarray_as_numpy(arr):
    def impl(arr):
        return intrin_otherarray_as_numpy(arr)

    return impl


@intrinsic
def intrin_otherarray_as_numpy(typingctx, arr):

    base_arry_t = arr.as_base_array_type()

    def codegen(context, builder, signature, args):
        [arr] = args
        arry_t = signature.args[0]
        nativearycls = context.make_array(arry_t)
        nativeary = nativearycls(context, builder, value=arr)

        base_ary = context.make_array(base_arry_t)(context, builder)
        cgutils.copy_struct(base_ary, nativeary)
        out = base_ary._getvalue()
        context.nrt.incref(builder, base_arry_t, out)
        return out

    from numba.core.typing import signature

    sig = signature(base_arry_t, arr)
    return sig, codegen


# @overload(operator.getitem)
# def ol_getitem_impl(arr, idx):
#     if isinstance(arr, ExtArrayType):

#         def impl(arr, idx):
#             return arr.as_numpy()[idx]

#         return impl


# @overload(operator.setitem)
# def ol_setitem_impl(arr, idx, val):
#     if isinstance(arr, ExtArrayType):

#         def impl(arr, idx, val):
#             nparr = arr.as_numpy()
#             nparr[idx] = val

#         return impl


def test_setitem():
    @njit
    def foo(size):
        arr = extarray_empty((size,), dtype=np.float64)
        for i in range(arr.size):
            arr[i] = i
        return arr

    r = foo(10)
    arr = r.as_numpy()
    np.testing.assert_equal(arr, np.arange(10))


def test_getitem():
    @njit
    def foo(size):
        arr = extarray_empty((size,), dtype=np.float64)
        for i in range(arr.size):
            arr[i] = i
        c = 0
        for i in range(arr.size):
            c += i
        return c

    res = foo(10)
    assert res == np.arange(10).sum()

def test_getitem_slice():
    @njit
    def foo(size):
        arr = extarray_empty((2, size), dtype=np.float64)
        return arr[0]

    res = foo(10)
    print(type(res))
    assert False

# ----------------------------------------------------------------------------
# Part 5: Access to handle address in ExtArray


@intrinsic
def intrin_extarray_handle_addr(typingctx, arr):
    def codegen(context, builder, signature, args):
        [arr] = args
        nativearycls = context.make_array(signature.args[0])
        nativeary = nativearycls(context, builder, value=arr)
        return nativeary.handle

    sig = typing.signature(types.voidptr, arr)
    return sig, codegen


@overload_attribute(ExtArrayType, "handle_addr")
def extarray_handle_addr(arr):
    def get(arr):
        return intrin_extarray_handle_addr(arr)

    return get


def test_handle_addr():
    @njit
    def foo(arr):
        return arr.handle_addr

    arr = extarray_empty((10,), dtype=np.float64)
    handle_addr = foo(arr)
    assert arr.handle_addr == handle_addr


# ----------------------------------------------------------------------------
# Part 6: overload add


@overload(operator.add)
def ol_add_impl(lhs, rhs):
    if isinstance(lhs, ExtArrayType):
        if lhs.dtype != rhs.dtype:
            raise TypeError(
                f"LHS dtype ({lhs.dtype}) != RHS.dtype ({rhs.dtype})"
            )

        def impl(lhs, rhs):
            if lhs.shape != rhs.shape:
                raise ValueError("shape incompatible")
            out = extarray_empty(lhs.shape, lhs.dtype)
            for i in np.ndindex(lhs.shape):
                out[i] = lhs[i] + rhs[i]
            return out

        return impl


@njit
def extarray_arange(size, dtype):
    out = extarray_empty((size,), dtype=np.float64)
    for i in range(size):
        out[i] = i
    return out


def test_add():
    @njit
    def foo(n):
        a = extarray_arange(n, dtype=np.float64)
        b = extarray_arange(n, dtype=np.float64)
        res = a + b
        return res

    n = 12
    res = foo(12)
    assert isinstance(res, ExtArray)
    np.testing.assert_equal(res.as_numpy(), np.arange(n) + np.arange(n))


"""
Investigate if having a ExtArrayType._allocate will fix the segfault
from reusing the getitem/setitem
"""