"""
The client.

Requires exe_server.py to be running
"""
import contextlib
import ctypes
import operator
import pickle

import zmq
import numpy as np

from numba import types
from numba.extending import overload, intrinsic
from numba.core.extending_hardware import (
    JitDecorator,
    hardware_registry,
    GPU,
)
from numba.core import registry, utils
from numba.core.dispatcher import Dispatcher
from numba.core.descriptors import TargetDescriptor
from numba.core import cpu, typing, cgutils
from numba.core.base import BaseContext
from numba.core.compiler_lock import global_compiler_lock
from numba.core.utils import cached_property
from numba.core import callconv, decorators
from numba.core.codegen import BaseCPUCodegen, CodeLibrary
from numba.core.imputils import RegistryLoader, Registry
from numba.np import arrayobj
from numba import _dynfunc
import llvmlite.binding as ll
from llvmlite import ir
from llvmlite.llvmpy import core as lc
from numba.core import compiler


# Define a new target, this hardware extends GPU, this places the RPU in the
# hardware hierarchy.
class RPU(GPU):
    ...


# register the rpu hardware hierarchy token in the hardware registry
hardware_registry["rpu"] = RPU


# Define a bare codelibrary that disable all optimizations.
class RemoteCodeLibrary(CodeLibrary):
    def _optimize_functions(self, ll_module):
        pass

    def _optimize_final_module(self):
        pass

    def _finalize_specific(self):
        pass

    def get_asm_str(self):
        return None


class JITRPUCodegen(BaseCPUCodegen):
    # This largely rips off the CPU for ease

    _library_class = RemoteCodeLibrary

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"
        target = ll.Target.from_triple(ll.get_process_triple())
        tm = target.create_target_machine()
        self._tm = tm
        self._target_data = tm.target_data
        self._data_layout = str(tm.target_data)
        self._linking_modules = []

    def _create_empty_module(self, name):
        ir_module = lc.Module(name=name)
        ir_module.triple = ll.get_process_triple()
        if self._data_layout:
            ir_module.data_layout = self._data_layout
        return ir_module

    def _module_pass_manager(self):
        pass

    def _function_pass_manager(self, llvm_module):
        pass

    def _add_module(self, module):
        # Added modules are needed to be linked to the final modules
        print(f"==== {self}._add_module", module.name)
        self._linking_modules.append(module)


# This is the function registry for the rpu, it just has one, this one!
rpu_function_registry = Registry()


# Implement a new context for the RPU target
class RPUContext(BaseContext):
    allow_dynamic_globals = True

    # Overrides
    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    @global_compiler_lock
    def init(self):
        self._internal_codegen = JITRPUCodegen("numba.exec")
        self._remote_exe = RemoteExecutor()
        self.refresh()
        # Add a dummy NRT module to disable the incref/decref
        self.dummy_nrt = ll.parse_assembly(
            """
define void @NRT_decref(i8* noalias nocapture %0){
    ret void
}

define void @NRT_incref(i8* noalias nocapture){
    ret void
}
"""
        )

    def refresh(self):
        registry = rpu_function_registry
        try:
            loader = self._registries[registry]
        except KeyError:
            loader = RegistryLoader(registry)
            self._registries[registry] = loader
        self.install_registry(registry)
        # Also refresh typing context, since @overload declarations can
        # affect it.
        self.typing_context.refresh()

    @property
    def target_data(self):
        return self._internal_codegen.target_data

    def codegen(self):
        return self._internal_codegen

    # Borrow the CPU call conv
    @cached_property
    def call_conv(self):
        return callconv.CPUCallConv(self)

    def create_cpython_wrapper(
        self, library, fndesc, env, call_helper, release_gil=False
    ):
        # Disable cpython wrapper
        pass

    def create_cfunc_wrapper(self, library, fndesc, env, call_helper):
        # The following is a direct copy of the CPU create_cfunc_wrapper
        wrapper_module = self.create_module("cfunc_wrapper")
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        wrapper_callee = wrapper_module.add_function(
            fnty, fndesc.llvm_func_name
        )

        ll_argtypes = [self.get_value_type(ty) for ty in fndesc.argtypes]
        ll_return_type = self.get_value_type(fndesc.restype)
        wrapty = ir.FunctionType(ll_return_type, ll_argtypes)
        wrapfn = wrapper_module.add_function(
            wrapty, fndesc.llvm_cfunc_wrapper_name
        )
        builder = ir.IRBuilder(wrapfn.append_basic_block("entry"))

        status, out = self.call_conv.call_function(
            builder,
            wrapper_callee,
            fndesc.restype,
            fndesc.argtypes,
            wrapfn.args,
            attrs=("noinline",),
        )

        with builder.if_then(status.is_error, likely=False):
            # If (and only if) an error occurred, acquire the GIL
            # and use the interpreter to write out the exception.
            pyapi = self.get_python_api(builder)
            gil_state = pyapi.gil_ensure()
            self.call_conv.raise_error(builder, pyapi, status)
            cstr = self.insert_const_string(builder.module, repr(self))
            strobj = pyapi.string_from_string(cstr)
            pyapi.err_write_unraisable(strobj)
            pyapi.decref(strobj)
            pyapi.gil_release(gil_state)

        builder.ret(out)
        library.add_ir_module(wrapper_module)

    def get_executable(self, library, fndesc, env):
        # This is overridden to create a callable that will offload the actual
        # execution to the execution server.
        @ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.py_object, ctypes.py_object)
        def inner(closure, args):
            print("---Running:", fndesc.qualname, args)
            res = self._offload_to_remote(
                library, fndesc.llvm_cfunc_wrapper_name, fndesc.argtypes, args
            )
            return res

        # Prepare the function for injection into the dispatcher
        addr = ctypes.cast(inner, ctypes.c_void_p).value
        doc = "compiled wrapper for %r" % (fndesc.qualname,)
        cfunc = _dynfunc.make_function(
            fndesc.lookup_module(),
            fndesc.qualname.split(".")[-1],
            doc,
            addr,
            env,
            # objects to keepalive with the function
            (library, inner),
        )
        return cfunc

    def _offload_to_remote(self, library, fname, argtypes, args):
        # This implements the offloading to the remote server.

        # First, complete the linking so that we have a self-contained module.
        mod = library._final_module
        for lm in library.codegen._linking_modules:
            mod.link_in(lm)

        # Then, linkin the dummy NRT library to mask off the incref/decref.
        mod.link_in(self.dummy_nrt)

        # The following is checking for undefined symbols in NRT.
        undefined = 0
        for fn in mod.functions:
            if fn.name.startswith("NRT_") and fn.is_declaration:
                print(f"undefined symbol {fn}")
                undefined += 1
        if undefined:
            raise RuntimeError("undefined NRT symbols")

        # Send to remote server to JIT and execute.
        return self._remote_exe.jit(mod, fname, argtypes, args)


# Defines a class to communicate to the remote server.
class RemoteExecutor:

    port = 5555

    def __init__(self):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{self.port}")

    def jit(self, mod, fname, argtypes, args):
        dat = dict(
            msg="jit", mod=str(mod), fname=fname, argtypes=argtypes, args=args
        )
        packed = pickle.dumps(dat)
        self.socket.send(packed)
        result = pickle.loads(self.socket.recv())
        print(result)
        return result["return"]

    def allocate_array(self, ary):
        dat = dict(msg="allocate", ary=ary)
        packed = pickle.dumps(dat)
        self.socket.send(packed)
        result = pickle.loads(self.socket.recv())
        return result["return"]


# Nested contexts to help with isolatings bits of compilations
class _NestedContext(object):
    _typing_context = None
    _target_context = None

    @contextlib.contextmanager
    def nested(self, typing_context, target_context):
        old_nested = self._typing_context, self._target_context
        try:
            self._typing_context = typing_context
            self._target_context = target_context
            yield
        finally:
            self._typing_context, self._target_context = old_nested


# Implement a RPU TargetDescriptor, this one borrows bits from the CPU
class RPUTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions
    _nested = _NestedContext()

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return RPUContext(self.typing_context, self._target_name)

    @utils.cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return typing.Context()

    @property
    def target_context(self):
        """
        The target context for RPU targets.
        """
        nested = self._nested._target_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for CPU targets.
        """
        nested = self._nested._typing_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_typing_context

    def nested_context(self, typing_context, target_context):
        """
        A context manager temporarily replacing the contexts with the
        given ones, for the current thread of execution.
        """
        return self._nested.nested(typing_context, target_context)


# Create a RPU target instance
rpu_target = RPUTarget("rpu")


# Declare a dispatcher for the RPU target
class RPUDispatcher(Dispatcher):
    targetdescr = rpu_target


# Register a dispatcher for the RPU target, a lot of the code uses this
# internally to work out what to do RE compilation
registry.dispatcher_registry["rpu"] = RPUDispatcher


# Implement a dispatcher for the RPU target
class rjit(JitDecorator):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args):
        assert len(args) < 2
        if args:
            func = args[0]
        else:
            func = self._args[0]
        self.py_func = func
        # wrap in dispatcher
        return self.dispatcher_wrapper()

    def get_dispatcher(self):
        """
        Returns the dispatcher
        """
        return registry.dispatcher_registry["rpu"]

    def dispatcher_wrapper(self):
        disp = self.get_dispatcher()
        # Parse self._kwargs here
        topt = {}
        if "nopython" in self._kwargs:
            topt["nopython"] = True

        # It would be easy to specialise the default compilation pipeline for
        # this target here.
        pipeline_class = compiler.Compiler
        if "pipeline_class" in self._kwargs:
            pipeline_class = self._kwargs["pipeline_class"]
        return disp(
            py_func=self.py_func,
            targetoptions=topt,
            pipeline_class=pipeline_class,
        )


# add it to the decorator registry, this is so e.g. @overload can look up a
# JIT function to do the compilation work.
decorators.jit_registry["rpu"] = rjit

# -------------- Case 1, want to compile for a new target, the RPU ---------
print(" Case 1 - Use RPU target ".center(80, "="))

# In this section you can try commenting one or more of the overloads for
# 'my_func' to explore the effect of having a hardware hierarchy. The hierarchy
# for the RPU target is: RPU -> GPU -> Generic; where -> is 'extends from'.
# As a result, a RPU compiled function will try and use a RPU overload if
# available, if it's not available but there's a GPU version, it will use that
# and finally, if there's no GPU version it will use a generic one if it is
# available. In this case, if the CPU compiled version is used, it will use the
# generic version as it's the only version available for the CPU hardware.


def my_func(x):
    pass


# The RPU target "knows" nothing, add in some primitives for basic things...

# need to register how to lower dummy for @intrinsic
@rpu_function_registry.lower_constant(types.Dummy)
def constant_dummy(context, builder, ty, pyval):
    return context.get_dummy_value()


# and how to deal with IntegerLiteral to Integer casts
@rpu_function_registry.lower_cast(types.IntegerLiteral, types.Integer)
def literal_int_to_number(context, builder, fromty, toty, val):
    lit = context.get_constant_generic(
        builder, fromty.literal_type, fromty.literal_value,
    )
    return context.cast(builder, lit, fromty.literal_type, toty)


# and how to lower an Int constant
@rpu_function_registry.lower_constant(types.Integer)
def const_float(context, builder, ty, pyval):
    lty = context.get_value_type(ty)
    return lty(pyval)


@intrinsic(hardware="rpu")
def intrin_add(tyctx, x, y):
    sig = x(x, y)

    def codegen(cgctx, builder, tyargs, llargs):
        cgutils.printf(builder, "intrin_add() invoked\n")
        return builder.add(*llargs)

    return sig, codegen


# Spell out how to overload 'add', call the rpu specific intrinsic
@overload(operator.add, hardware="gpu")
def ol_add(x, y):
    if isinstance(x, types.Integer) and isinstance(y, types.Integer):

        def impl(x, y):
            return intrin_add(x, y)

        return impl


@intrinsic(hardware="rpu")
def ref_array(tyctx, addr, shape, dtype):
    # This acts like `carray()` by taking the "device address" of the array
    # along with the shape and dtype to create a reference to the array.
    print("ref_array", addr, shape, dtype)
    shape = types.unliteral(shape)
    aryty = types.Array(dtype=dtype.dtype, ndim=len(shape), layout="C")
    sig = aryty(addr, shape, dtype)

    def codegen(cgctx, builder, tyargs, llargs):
        cgutils.printf(builder, "ref_array() invoked\n")
        aryproxy = arrayobj.make_array(aryty)(cgctx, builder)

        [addr, shape, _] = llargs
        llty = cgctx.get_data_type(dtype.dtype)
        ptr = builder.inttoptr(addr, llty.as_pointer())

        itemsize = cgctx.get_constant(types.intp, cgctx.get_abi_sizeof(llty))
        strides = [itemsize]
        out = arrayobj.populate_array(
            aryproxy,
            data=ptr,
            shape=shape,
            strides=strides,
            itemsize=itemsize,
            meminfo=None,
        )
        return out._getvalue()

    return sig, codegen


@intrinsic(hardware="rpu")
def intrin_getitem(tyctx, ary, idx):
    print("intrin_getitem", ary, idx)
    sig = ary.dtype(ary, idx)

    def codegen(cgctx, builder, sig, llargs):
        cgutils.printf(builder, "intrin_getitem() invoked\n")
        [aryty, idxty] = sig.args
        [ary, idx] = llargs
        aryproxy = arrayobj.make_array(aryty)(cgctx, builder, value=ary)
        ptr = builder.gep(aryproxy.data, [idx])
        return builder.load(ptr)

    return sig, codegen


@overload(operator.getitem, hardware="rpu")
def ol_ary_getitem(ary, idx):
    print("getitem", ary, idx)

    def codegen(ary, idx):
        return intrin_getitem(ary, idx)

    return codegen


@overload(my_func, hardware="rpu")
def ol_my_func3(x):
    def impl(x):
        # Exercise arrays
        y = 9
        ary = ref_array(x, (y + 1,), np.intp)
        return ary[3] + ary[8] + ary[2]

    return impl


# -----------------------------------------------------------------------------

remote = RemoteExecutor()

# This is the demonstration function, it calls a version of the overloaded
# 'my_func' function.
@rjit(nopython=True)
def foo(x):
    return my_func(x)


arr = np.arange(10, dtype=np.intp)
# Allocate a "device" array
da = remote.allocate_array(arr)
print("allocate_array", da)
# Invoke foo()
print(f"foo({da})", foo(da))
