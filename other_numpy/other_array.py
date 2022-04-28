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
import other_numpy
import numpy as np
from numba.core.typing.npydecl import parse_dtype, parse_shape
from numba.np.arrayobj import _parse_empty_args, _empty_nd_impl
from numba.core.imputils import impl_ret_new_ref


class OtherArray:
    def __init__(self, ndim, layout, data, extra_parameter):
        self._ndim = ndim
        self._layout = layout
        self._data = data
        self._extra_parameter = extra_parameter

    @property
    def ndim(self):
        return self._ndim

    @property
    def layout(self):
        return self._layout

    @property
    def data(self):
        return self._data

    @property
    def extra_parameter(self):
        return self._extra_parameter

    def __repr__(self):
        return (
            f"OtherArray({self.data}, extra_parameter={self.extra_parameter})"
        )


class OtherArrayType(types.Array):
    __use_overload_indexing__ = True

    def __init__(self, *args, **kwargs):
        super(OtherArrayType, self).__init__(*args, **kwargs)
        self.name = f"OtherArrayType{self.name}"

    def get_array_type(self):
        return types.Array(
            dtype=self.dtype, ndim=self.ndim, layout=self.layout
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # return NotImplemented   # disable default ufunc
        if method == "__call__":
            for inp in inputs:
                if not isinstance(inp, (types.Array, types.Number)):
                    return NotImplemented
            # # Ban if all arguments are OtherArrayType
            # if all(isinstance(inp, OtherArrayType) for inp in inputs):
            #     return NotImplemented
            return NotImplemented
            return OtherArrayType
        else:
            return NotImplemented


@register_model(OtherArrayType)
class OtherArrayTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            ("meminfo", types.MemInfoPointer(fe_type.dtype)),
            ("parent", types.pyobject),
            ("nitems", types.intp),
            ("itemsize", types.intp),
            ("data", types.CPointer(fe_type.dtype)),
            ("shape", types.UniTuple(types.intp, ndim)),
            ("strides", types.UniTuple(types.intp, ndim)),
            ("extra_parameter", types.intp),
        ]
        super(OtherArrayTypeModel, self).__init__(dmm, fe_type, members)


@typeof_impl.register(OtherArray)
def typeof_index(val, c):
    return OtherArrayType(typeof(val.data.dtype).dtype, val.ndim, val.layout)


@unbox(OtherArrayType)
def unbox_oat(typ, obj, c):

    buf = c.pyapi.object_getattr_string(obj, "data")
    data_array_type = typ.get_array_type()
    ary = c.unbox(data_array_type, buf)
    c.pyapi.decref(buf)

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

    extra_param_obj = c.pyapi.object_getattr_string(obj, "extra_parameter")
    extra_param = c.unbox(types.intp, extra_param_obj).value
    nativeary.extra_parameter = extra_param

    aryptr = nativeary._getpointer()
    return NativeValue(c.builder.load(aryptr),)


@intrinsic
def intrin_otherarray_attr(typingctx, arr):
    def codegen(context, builder, signature, args):
        [arr] = args
        nativearycls = context.make_array(signature.args[0])
        nativeary = nativearycls(context, builder, value=arr)
        return nativeary.extra_parameter

    from numba.core.typing import signature

    sig = signature(types.intp, arr)
    return sig, codegen


@overload_attribute(OtherArrayType, "extra_parameters")
def array_extra_parameters(arr):
    def get(arr):
        return intrin_otherarray_attr(arr)

    return get


@intrinsic
def intrin_otherarray_data(typingctx, arr):

    base_arry_t = arr.get_array_type()

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


@overload_attribute(OtherArrayType, "data")
def array_data(arr):
    def get(arr):
        return intrin_otherarray_data(arr)

    return get


def _box_oat(array, extra_parameter):
    array_t = typeof(array)
    return OtherArray(array_t.dtype, array_t.layout, array, extra_parameter)


@box(OtherArrayType)
def box_oat(typ, val, c):
    base_ary = c.context.make_array(typ.get_array_type())(c.context, c.builder)
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder, value=val)
    cgutils.copy_struct(base_ary, nativeary)

    extra_param = nativeary.extra_parameter

    data_ary = c.box(typ.get_array_type(), base_ary._getvalue())

    boxer = c.pyapi.unserialize(c.pyapi.serialize_object(_box_oat))
    extra_param_obj = c.pyapi.long_from_longlong(extra_param)
    retval = c.pyapi.call_function_objargs(boxer, [data_ary, extra_param_obj])
    c.pyapi.decref(extra_param_obj)

    return retval


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
        align_u32 = cast_integer(
            context, builder, align, signature.args[1], types.uint32
        )
        meminfo = context.nrt.meminfo_alloc_aligned(
            builder, allocsize, align_u32
        )
        return meminfo

    from numba.core.typing import signature

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = signature(mip, allocsize, align)
    return sig, codegen


@overload_classmethod(OtherArrayType, "_allocate")
def oat_allocate(cls, allocsize, alignment):
    def impl(cls, allocsize, alignment):
        return intrin_alloc(allocsize, alignment)

    return impl


@intrinsic
def oat_empty_intrin(tyctx, shape, dtype):
    if dtype is None or isinstance(dtype, types.NoneType):
        nb_dtype = types.double
    else:
        nb_dtype = parse_dtype(dtype)

    ndim = parse_shape(shape)
    if nb_dtype is not None and ndim is not None:
        sig = OtherArrayType(dtype=nb_dtype, ndim=ndim, layout="C")(
            shape, dtype
        )
    else:
        return None, None

    def codegen(cgctx, builder, sig, llargs):
        arrtype, shapes = _parse_empty_args(cgctx, builder, sig, llargs)
        ary = _empty_nd_impl(cgctx, builder, arrtype, shapes)
        return impl_ret_new_ref(
            cgctx, builder, sig.return_type, ary._getvalue()
        )

    return sig, codegen


@overload(other_numpy.empty)
def ol_empty_impl(shape, dtype=None):
    def impl(shape, dtype=None):
        return oat_empty_intrin(shape, dtype)

    return impl


@overload(operator.add)
def ol_add_impl(lhs, rhs):
    if isinstance(lhs, OtherArrayType):

        def impl(lhs, rhs):
            out = other_numpy.empty(lhs.size)
            out.data[:] = lhs.data + rhs.data
            return lhs

        return impl


@overload(operator.getitem)
def ol_getitem_impl(arr, idx):
    if isinstance(arr, OtherArrayType):

        def impl(arr, idx):
            print("OAT getitem")
            return arr.data[idx]

        return impl


@overload(operator.setitem)
def ol_getitem_impl(arr, idx, val):
    if isinstance(arr, OtherArrayType):

        def impl(arr, idx, val):
            print("OAT setitem")
            arr.data[idx] = val

        return impl


################################################################################


@njit
def foo(arr):
    r = other_numpy.empty(arr.size)

    for i in range(r.size):
        v = arr[i]  # getitem via overload
        r[i] = arr.extra_parameters + v

    arr[:] = r  # setitem via overload
    return arr + r  # add via overload


oa = OtherArray(1, "C", np.ones(5), 1234)

r = foo(oa)
# print(type(r))
print(oa)
print(r)
