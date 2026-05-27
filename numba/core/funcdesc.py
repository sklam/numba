"""
Function descriptors.
"""

from collections import defaultdict
import hashlib
import importlib
import struct
import uuid
import warnings

from numba.core import types, itanium_mangler
from numba.core.errors import NumbaWarning
from numba.core.utils import _dynamic_modname, _dynamic_module

import numpy as np

def default_mangler(name, argtypes, *, abi_tags=(), uid=None):
    return itanium_mangler.mangle(name, argtypes, abi_tags=abi_tags, uid=uid)


def qualifying_prefix(modname, qualname):
    """
    Returns a new string that is used for the first half of the mangled name.
    """
    # XXX choose a different convention for object mode
    return '{}.{}'.format(modname, qualname) if modname else qualname


class FunctionDescriptor(object):
    """
    Base class for function descriptors: an object used to carry
    useful metadata about a natively callable function.

    Note that while `FunctionIdentity` denotes a Python function
    which is being concretely compiled by Numba, `FunctionDescriptor`
    may be more "abstract".
    """
    __slots__ = ('native', 'modname', 'qualname', 'doc', 'typemap',
                 'calltypes', 'args', 'kws', 'restype', 'argtypes',
                 'mangled_name', 'canonical_mangled_name',
                 'unique_name', 'env_name', 'global_dict',
                 'inline', 'noalias', 'abi_tags', 'uid')

    def __init__(self, native, modname, qualname, unique_name, doc,
                 typemap, restype, calltypes, args, kws, mangler=None,
                 argtypes=None, inline=False, noalias=False, env_name=None,
                 global_dict=None, abi_tags=(), uid=None, code=None,
                 closure=None):
        self.native = native
        self.modname = modname
        self.global_dict = global_dict
        self.qualname = qualname
        self.unique_name = unique_name
        self.doc = doc
        # XXX typemap and calltypes should be on the compile result,
        # not the FunctionDescriptor
        self.typemap = typemap
        self.calltypes = calltypes
        self.args = args
        self.kws = kws
        self.restype = restype
        # Argument types
        if argtypes is not None:
            assert isinstance(argtypes, tuple), argtypes
            self.argtypes = argtypes
        else:
            # Get argument types from the type inference result
            # (note the "arg.FOO" convention as used in typeinfer
            self.argtypes = tuple(self.typemap["arg." + a] for a in args)
        mangler = default_mangler if mangler is None else mangler
        # The mangled name *must* be unique, else the wrong function can
        # be chosen at link time.
        qualprefix = qualifying_prefix(self.modname, self.qualname)
        if uid is not None and typemap is not None:
            # canonical name uses the raw counter uid so that recursive callers
            # (which stored that uid during type inference) can still be resolved.
            self.canonical_mangled_name = mangler(
                qualprefix,
                self.argtypes,
                abi_tags=abi_tags,
                uid=uid,
            )
            uid = _FunctionHasher().compute_uid(
                uid,
                typemap,
                code,
                closure,
                qualname,
                global_dict,
            )
        else:
            self.canonical_mangled_name = None
        self.uid = uid
        self.mangled_name = mangler(
            qualprefix,
            self.argtypes,
            abi_tags=abi_tags,
            uid=self.uid,
        )
        if env_name is None:
            env_name = mangler(
                ".NumbaEnv.{}".format(qualprefix),
                self.argtypes,
                abi_tags=abi_tags,
                uid=self.uid,
            )
        self.env_name = env_name
        self.inline = inline
        self.noalias = noalias
        self.abi_tags = abi_tags

    def lookup_globals(self):
        """
        Return the global dictionary of the function.
        It may not match the Module's globals if the function is created
        dynamically (i.e. exec)
        """
        return self.global_dict or self.lookup_module().__dict__

    def lookup_module(self):
        """
        Return the module in which this function is supposed to exist.
        This may be a dummy module if the function was dynamically
        generated or the module can't be found.
        """
        if self.modname == _dynamic_modname:
            return _dynamic_module
        else:
            try:
                # ensure module exist
                return importlib.import_module(self.modname)
            except ImportError:
                return _dynamic_module

    def lookup_function(self):
        """
        Return the original function object described by this object.
        """
        return getattr(self.lookup_module(), self.qualname)

    @property
    def llvm_func_name(self):
        """
        The LLVM-registered name for the raw function.
        """
        return self.mangled_name

    # XXX refactor this

    @property
    def llvm_cpython_wrapper_name(self):
        """
        The LLVM-registered name for a CPython-compatible wrapper of the
        raw function (i.e. a PyCFunctionWithKeywords).
        """
        return itanium_mangler.prepend_namespace(self.mangled_name,
                                                 ns='cpython')

    @property
    def llvm_cfunc_wrapper_name(self):
        """
        The LLVM-registered name for a C-compatible wrapper of the
        raw function.
        """
        return 'cfunc.' + self.mangled_name

    def __repr__(self):
        return "<function descriptor %r>" % (self.unique_name)

    @classmethod
    def _get_function_info(cls, func_ir):
        """
        Returns
        -------
        qualname, unique_name, modname, doc, args, kws, globals

        ``unique_name`` must be a unique name.
        """
        func = func_ir.func_id.func
        qualname = func_ir.func_id.func_qualname
        # XXX to func_id
        modname = func.__module__
        doc = func.__doc__ or ''
        args = tuple(func_ir.arg_names)
        kws = ()        # TODO
        global_dict = None

        if modname is None:
            # Dynamically generated function.
            modname = _dynamic_modname
            # Retain a reference to the dictionary of the function.
            # This disables caching, serialization and pickling.
            global_dict = func_ir.func_id.func.__globals__

        unique_name = func_ir.func_id.unique_name

        return qualname, unique_name, modname, doc, args, kws, global_dict

    @classmethod
    def _from_python_function(cls, func_ir, typemap, restype,
                              calltypes, native, mangler=None,
                              inline=False, noalias=False, abi_tags=()):
        (qualname, unique_name, modname, doc, args, kws, global_dict,
         ) = cls._get_function_info(func_ir)

        self = cls(native, modname, qualname, unique_name, doc,
                   typemap, restype, calltypes,
                   args, kws, mangler=mangler, inline=inline, noalias=noalias,
                   global_dict=global_dict, abi_tags=abi_tags,
                   uid=func_ir.func_id.unique_id,
                   code=func_ir.func_id.code,
                   closure=getattr(func_ir.func_id.func, '__closure__', None))
        return self


class PythonFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for Numba-compiled functions.
    """
    __slots__ = ()

    @classmethod
    def from_specialized_function(cls, func_ir, typemap, restype, calltypes,
                                  mangler, inline, noalias, abi_tags):
        """
        Build a FunctionDescriptor for a given specialization of a Python
        function (in nopython mode).
        """
        return cls._from_python_function(func_ir, typemap, restype, calltypes,
                                         native=True, mangler=mangler,
                                         inline=inline, noalias=noalias,
                                         abi_tags=abi_tags)

    @classmethod
    def from_object_mode_function(cls, func_ir):
        """
        Build a FunctionDescriptor for an object mode variant of a Python
        function.
        """
        typemap = defaultdict(lambda: types.pyobject)
        calltypes = typemap.copy()
        restype = types.pyobject
        return cls._from_python_function(func_ir, typemap, restype, calltypes,
                                         native=False)


class ExternalFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for opaque external functions
    (e.g. raw C functions).
    """
    __slots__ = ()

    def __init__(self, name, restype, argtypes):
        args = ["arg%d" % i for i in range(len(argtypes))]

        def mangler(a, x, abi_tags, uid=None):
            return a
        super(ExternalFunctionDescriptor, self
              ).__init__(native=True, modname=None, qualname=name,
                         unique_name=name, doc='', typemap=None,
                         restype=restype, calltypes=None, args=args,
                         kws=None,
                         mangler=mangler,
                         argtypes=argtypes)


class _FunctionHasher:
    """Stable content hash logic used by FunctionDescriptor."""

    def __init__(self):
        self._h = hashlib.sha256()

    def update(self, data):
        self._h.update(data)

    def hash_closure_cells(self, closure, code, qualname):
        """Feed each closure cell's value into the hasher.

        Returns True if any cell could not be stably hashed (unhashable,
        non-ndarray value), False otherwise.
        """

        has_unstable_cell = False
        freevars = code.co_freevars if code is not None else ()
        for i, cell in enumerate(closure):
            if i < len(freevars):
                self.update(freevars[i].encode())
            try:
                val = cell.cell_contents
            except ValueError:
                # Empty cell (unbound free variable) — stable sentinel.
                self.update("empty_cell".encode())
                continue
            try:
                cell_hash = hash(val)
            except TypeError:
                if isinstance(val, np.ndarray):
                    # numpy arrays are not hashable but are treated as frozen
                    # constants in JIT closures.
                    self.update("ndarray".encode())
                    self.update(val.dtype.str.encode())
                    self.update(str(val.shape).encode())
                    self.update(val.tobytes())
                else:
                    freevar_name = freevars[i] if i < len(freevars) else str(i)
                    warnings.warn(
                        NumbaWarning(
                            f"Cannot obtain a stable hash for closure "
                            f"variable {freevar_name!r} of {qualname!r} "
                            f"(type: {type(val).__name__!r}). The LLVM "
                            f"symbol name will not be stable across Python "
                            f"runs; on-disk caching is disabled for this "
                            f"function."
                        )
                    )
                    self.update(np.uint64(id(val)) & (2**64 - 1))
                    has_unstable_cell = True
            else:
                self.update(np.uint64(cell_hash & (2**64 - 1)))
        return has_unstable_cell

    def compute_uid(
        self, uid, typemap, code, closure, qualname, global_dict
    ) -> str:
        """Compute and return a stable content-hash uid."""
        NHEXCHAR = 16  # 64-bit is plenty
        if code is not None:
            self.update(code.co_code)  # raw bytecode bytes — stable
        # Feed each (varname, type_str) pair in sorted order for determinism.
        for k in sorted(typemap):
            self.update(k.encode())
            self.update(str(typemap[k]).encode())
        has_unstable_cell = (
            self.hash_closure_cells(closure, code, qualname)
            if closure is not None
            else False
        )
        if global_dict is not None or has_unstable_cell:
            # Use a UUID4 to guarantee per-compilation uniqueness for
            # dynamic function s(no global_dict) or functions with unstable cell
            return uuid.uuid4().hex[:NHEXCHAR]
        return self._h.hexdigest()[:NHEXCHAR]
