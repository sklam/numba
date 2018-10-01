"""
Implement a Ref[T] object to store heap-allocated object of T.
"""
from collections import namedtuple

import llvmlite.ir as llvmir

from numba import types
from numba import typeof
from numba import cgutils
from numba.extending import overload, intrinsic
from numba.datamodel import default_manager, models


class RefType(types.Type):
    def __init__(self, element):
        if not isinstance(element, types.Type):
            raise ValueError("expecting `element` to be a type")
        self.element = element
        name = 'ref!{}'.format(element.name)
        super(RefType, self).__init__(name)


class RefModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        dtype = fe_typ.element
        members = [
            ('meminfo', types.MemInfoPointer(dtype)),
        ]
        super(RefModel, self).__init__(dmm, fe_typ, members)


default_manager.register(RefType, RefModel)


def make(obj):
    pass


def get(ref):
    pass


def put(ref, obj):
    pass


def _imp_dtor(context, module, reftype):
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = llvmir.FunctionType(llvmir.VoidType(),
                                     [llvoidptr, llsize, llvoidptr])

    fname = "_RefDtor.{0}".format(reftype.name)
    dtor_fn = module.get_or_insert_function(dtor_ftype,
                                            name=fname)
    if dtor_fn.is_declaration:
        # define dtor
        builder = llvmir.IRBuilder(dtor_fn.append_basic_block())

        alloc_fe_type = reftype.element
        alloc_type = context.get_value_type(alloc_fe_type)
        # load
        ptr = builder.bitcast(dtor_fn.args[0], alloc_type.as_pointer())
        data = builder.load(ptr)
        # decref
        context.nrt.decref(builder, alloc_fe_type, data)

        builder.ret_void()

    return dtor_fn


@overload(make)
def _overload_make(elem):
    reftype = RefType(elem)

    @intrinsic
    def make_ref(typingctx, elem):
        def codegen(context, builder, signature, args):
            [elemval] = args
            refobj = RefHelper(context, builder, reftype)
            refobj.initialize()
            refobj.set(elemval)
            what = refobj.get_llvm_value()
            return what

        sig = reftype(elem)
        return sig, codegen

    def impl(elem):
        return make_ref(elem)

    return impl


@overload(get)
def _overload_get(reftype):
    elem = reftype.element

    @intrinsic
    def ref_get(typingctx, reftype):
        def codegen(context, builder, signature, args):
            [refvalue] = args
            refobj = RefHelper(context, builder, reftype, value=refvalue)
            return refobj.get()

        sig = elem(reftype)
        return sig, codegen

    def impl(ref):
        return ref_get(ref)

    return impl


@overload(put)
def _overload_put(reftype, elem):
    @intrinsic
    def ref_put(typingctx, reftype, elem):
        def codegen(context, builder, signature, args):

            [refvalue, elemval] = args
            refobj = RefHelper(context, builder, reftype, value=refvalue)
            refobj.set(elemval)
            return refobj.get_llvm_value()

        sig = reftype(reftype, elem)
        return sig, codegen

    def impl(ref, val):
        return ref_put(ref, val)

    return impl


class RefHelper(object):
    def __init__(self, context, builder, reftype, value=None):
        self.context = context
        self.builder = builder
        self.reftype = reftype
        self.refstructcls = cgutils.create_struct_proxy(reftype)
        self.refstruct = self.refstructcls(
            self.context,
            self.builder,
            value=value,
            )

    def get_element_type(self):
        return self.context.get_value_type(self.reftype.element)

    def initialize(self):
        alloc_size = self.context.get_abi_sizeof(self.get_element_type())

        meminfo = self.context.nrt.meminfo_alloc_dtor(
            self.builder,
            self.context.get_constant(types.uintp, alloc_size),
            _imp_dtor(self.context, self.builder.module, self.reftype),
        )
        self.refstruct.meminfo = meminfo

    def get_data_pointer(self):
        meminfo = self.refstruct.meminfo
        data_pointer = self.context.nrt.meminfo_data(self.builder, meminfo)
        data_pointer = self.builder.bitcast(
            data_pointer,
            self.get_element_type().as_pointer(),
            )
        return data_pointer

    def set(self, val):
        self.builder.store(val, self.get_data_pointer())

    def get(self):
        return self.builder.load(self.get_data_pointer())

    def get_llvm_value(self):
        return self.refstruct._getvalue()



