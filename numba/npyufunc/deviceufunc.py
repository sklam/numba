"""
Implements custom ufunc dispatch mechanism for non-CPU devices.
"""
from __future__ import print_function, absolute_import
import operator
import warnings
from functools import reduce
import numpy as np
from numba.utils import longint, OrderedDict
from numba.utils import IS_PY3
from numba.npyufunc.ufuncbuilder import _BaseUFuncBuilder
from numba import sigutils, types
from numba.typing import signature
from numba.npyufunc.sigparse import parse_signature

if IS_PY3:
    def _exec(codestr, glbls):
        exec(codestr, glbls)
else:
    eval(compile("""
def _exec(codestr, glbls):
    exec codestr in glbls
""",
                 "<_exec>", "exec"))


def _broadcast_axis(a, b):
    """
    Raises
    ------
    ValueError if broadcast fails
    """
    if a == b:
        return a
    elif a == 1:
        return b
    elif b == 1:
        return a
    else:
        raise ValueError("failed to broadcast {0} and {1}".format(a, b))


def _pairwise_broadcast(shape1, shape2):
    """
    Raises
    ------
    ValueError if broadcast fails
    """
    shape1, shape2 = map(tuple, [shape1, shape2])

    while len(shape1) < len(shape2):
        shape1 = (1,) + shape1

    while len(shape1) > len(shape2):
        shape2 = (1,) + shape2

    return tuple(_broadcast_axis(a, b) for a, b in zip(shape1, shape2))


def _multi_broadcast(*shapelist):
    """
    Raises
    ------
    ValueError if broadcast fails
    """
    assert shapelist

    result = shapelist[0]
    others = shapelist[1:]
    try:
        for i, each in enumerate(others, start=1):
            result = _pairwise_broadcast(result, each)
    except ValueError:
        raise ValueError("failed to broadcast argument #{0}".format(i))
    else:
        return result


class UFuncMechanism(object):
    """
    Prepare ufunc arguments for vectorize.
    """
    DEFAULT_STREAM = None
    SUPPORT_DEVICE_SLICING = False

    def __init__(self, typemap, args):
        """Never used directly by user. Invoke by UFuncMechanism.call().
        """
        self.typemap = typemap
        self.args = args
        nargs = len(self.args)
        self.argtypes = [None] * nargs
        self.scalarpos = []
        self.signature = None
        self.arrays = [None] * nargs

    def _fill_arrays(self):
        """
        Get all arguments in array form
        """
        for i, arg in enumerate(self.args):
            if isinstance(arg, np.ndarray):
                self.arrays[i] = arg
            elif self.is_device_array(arg):
                self.arrays[i] = arg
            elif isinstance(arg, (int, longint, float, complex, np.number)):
                # Is scalar
                self.scalarpos.append(i)
            else:
                raise TypeError("argument #%d has invalid type" % (i + 1,))

    def _fill_argtypes(self):
        """
        Get dtypes
        """
        for i, ary in enumerate(self.arrays):
            if ary is not None:
                self.argtypes[i] = ary.dtype

    def _resolve_signature(self):
        """Resolve signature.
        May have ambiguous case.
        """
        matches = []
        # Resolve scalar args exact match first
        if self.scalarpos:
            # Try resolve scalar arguments
            for formaltys in self.typemap:
                match_map = []
                for i, (formal, actual) in enumerate(zip(formaltys,
                                                         self.argtypes)):
                    if actual is None:
                        actual = np.asarray(self.args[i]).dtype

                    match_map.append(actual == formal)

                if all(match_map):
                    matches.append(formaltys)

        # No matching with exact match; try coercing the scalar arguments
        if not matches:
            matches = []
            for formaltys in self.typemap:
                all_matches = all(actual is None or formal == actual
                                  for formal, actual in
                                  zip(formaltys, self.argtypes))
                if all_matches:
                    matches.append(formaltys)

        if not matches:
            raise TypeError("No matching version.  GPU ufunc requires array "
                            "arguments to have the exact types.  This behaves "
                            "like regular ufunc with casting='no'.")

        if len(matches) > 1:
            raise TypeError("Failed to resolve ufunc due to ambiguous "
                            "signature. Too many untyped scalars. "
                            "Use numpy dtype object to type tag.")

        # Try scalar arguments
        self.argtypes = matches[0]

    def _get_actual_args(self):
        """Return the actual arguments
        Casts scalar arguments to numpy.array.
        """
        for i in self.scalarpos:
            self.arrays[i] = np.array([self.args[i]], dtype=self.argtypes[i])

        return self.arrays

    def _broadcast(self, arys):
        """Perform numpy ufunc broadcasting
        """
        shapelist = [a.shape for a in arys]
        shape = _multi_broadcast(*shapelist)

        for i, ary in enumerate(arys):
            if ary.shape == shape:
                pass

            else:
                if self.is_device_array(ary):
                    arys[i] = self.broadcast_device(ary, shape)

                else:
                    ax_differs = [ax for ax in range(len(shape))
                                  if ax >= ary.ndim
                                  or ary.shape[ax] != shape[ax]]

                    missingdim = len(shape) - len(ary.shape)
                    strides = [0] * missingdim + list(ary.strides)

                    for ax in ax_differs:
                        strides[ax] = 0

                    strided = np.lib.stride_tricks.as_strided(ary,
                                                              shape=shape,
                                                              strides=strides)

                    arys[i] = self.force_array_layout(strided)

        return arys

    def get_arguments(self):
        """Prepare and return the arguments for the ufunc.
        Does not call to_device().
        """
        self._fill_arrays()
        self._fill_argtypes()
        self._resolve_signature()
        arys = self._get_actual_args()
        return self._broadcast(arys)

    def get_function(self):
        """Returns (result_dtype, function)
        """
        return self.typemap[self.argtypes]

    def is_device_array(self, obj):
        """Is the `obj` a device array?
        Override in subclass
        """
        return False

    def broadcast_device(self, ary, shape):
        """Handles ondevice broadcasting

        Override in subclass to add support.
        """
        raise NotImplementedError("broadcasting on device is not supported")

    def force_array_layout(self, ary):
        """Ensures array layout met device requirement.

        Override in sublcass
        """
        return ary

    @classmethod
    def call(cls, typemap, args, kws):
        """Perform the entire ufunc call mechanism.
        """
        # Handle keywords
        stream = kws.pop('stream', cls.DEFAULT_STREAM)
        out = kws.pop('out', None)

        if kws:
            warnings.warn("unrecognized keywords: %s" % ', '.join(kws))

        # Begin call resolution
        cr = cls(typemap, args)
        args = cr.get_arguments()
        resty, func = cr.get_function()

        outshape = args[0].shape

        def attempt_ravel(a):
            if cr.SUPPORT_DEVICE_SLICING:
                raise NotImplementedError

            try:
                # Call the `.ravel()` method
                return a.ravel()
            except NotImplementedError:
                # If it is not a device array
                if not cr.is_device_array(a):
                    raise
                # For device array, retry ravel on the host by first
                # copying it back.
                else:
                    hostary = cr.to_host(a, stream).ravel()
                    return cr.to_device(hostary, stream)

        if args[0].ndim > 1:
            args = [attempt_ravel(a) for a in args]

        # Prepare argument on the device
        devarys = []
        any_device = False
        for a in args:
            if cr.is_device_array(a):
                devarys.append(a)
                any_device = True
            else:
                dev_a = cr.to_device(a, stream=stream)
                devarys.append(dev_a)

        # Launch
        shape = args[0].shape
        if out is None:
            # No output is provided
            devout = cr.device_array(shape, resty, stream=stream)

            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)

            if any_device:
                # If any of the arguments are on device,
                # Keep output on the device
                return devout.reshape(outshape)
            else:
                # Otherwise, transfer output back to host
                return devout.copy_to_host().reshape(outshape)

        elif cr.is_device_array(out):
            # If output is provided and it is a device array,
            # Return device array
            if out.ndim > 1:
                out = attempt_ravel(out)
            devout = out
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            return devout.reshape(outshape)

        else:
            # If output is provided and it is a host array,
            # Return host array
            assert out.shape == shape
            assert out.dtype == resty
            devout = cr.device_array(shape, resty, stream=stream)
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            return devout.copy_to_host(out, stream=stream).reshape(outshape)

    def to_device(self, hostary, stream):
        """Implement to device transfer
        Override in subclass
        """
        raise NotImplementedError

    def to_host(self, devary, stream):
        """Implement to host transfer
        Override in subclass
        """
        raise NotImplementedError

    def device_array(self, shape, dtype, stream):
        """Implements device allocation
        Override in subclass
        """
        raise NotImplementedError

    def launch(self, func, count, stream, args):
        """Implements device function invocation
        Override in subclass
        """
        raise NotImplementedError


def to_dtype(ty):
    return np.dtype(str(ty))


class DeviceVectorize(_BaseUFuncBuilder):
    def __init__(self, func, identity=None, targetoptions={}):
        assert not targetoptions
        self.py_func = func
        self.identity = self.parse_identity(identity)
        # { arg_dtype: (return_dtype), cudakernel }
        self.kernelmap = OrderedDict()

    @property
    def pyfunc(self):
        return self.py_func

    def add(self, sig=None, argtypes=None, restype=None):
        # Handle argtypes
        if argtypes is not None:
            warnings.warn("Keyword argument argtypes is deprecated",
                          DeprecationWarning)
            assert sig is None
            if restype is None:
                sig = tuple(argtypes)
            else:
                sig = restype(*argtypes)
        del argtypes
        del restype

        # compile core as device function
        args, return_type = sigutils.normalize_signature(sig)
        devfnsig = signature(return_type, *args)

        funcname = self.pyfunc.__name__
        kernelsource = self._get_kernel_source(self._kernel_template,
                                               devfnsig, funcname)
        corefn, return_type = self._compile_core(devfnsig)
        glbl = self._get_globals(corefn)
        sig = signature(types.void, *([a[:] for a in args] + [return_type[:]]))
        _exec(kernelsource, glbl)

        stager = glbl['__vectorized_%s' % funcname]
        kernel = self._compile_kernel(stager, sig)

        argdtypes = tuple(to_dtype(t) for t in devfnsig.args)
        resdtype = to_dtype(return_type)
        self.kernelmap[tuple(argdtypes)] = resdtype, kernel

    def build_ufunc(self):
        raise NotImplementedError

    def _get_kernel_source(self, template, sig, funcname):
        args = ['a%d' % i for i in range(len(sig.args))]
        fmts = dict(name=funcname,
                    args=', '.join(args),
                    argitems=', '.join('%s[__tid__]' % i for i in args))
        return template.format(**fmts)

    def _compile_core(self, sig):
        raise NotImplementedError

    def _get_globals(self, corefn):
        raise NotImplementedError

    def _compile_kernel(self, fnobj, sig):
        raise NotImplementedError


class DeviceGUFuncVectorize(_BaseUFuncBuilder):
    def __init__(self, func, sig, identity=None, targetoptions={}):
        # Allow nopython flag to be set.
        if not targetoptions.pop('nopython', True):
            raise TypeError("nopython flag must be True")
        # Are there any more target options?
        if targetoptions:
            opts = ', '.join([repr(k) for k in targetoptions.keys()])
            fmt = "The following target options are not supported: {0}"
            raise TypeError(fmt.format(opts))

        self.py_func = func
        self.identity = self.parse_identity(identity)
        self.signature = sig
        self.inputsig, self.outputsig = parse_signature(self.signature)
        assert len(self.outputsig) == 1, "only support 1 output"
        # { arg_dtype: (return_dtype), cudakernel }
        self.kernelmap = OrderedDict()

    @property
    def pyfunc(self):
        return self.py_func

    def add(self, sig=None, argtypes=None, restype=None):
        # Handle argtypes
        if argtypes is not None:
            warnings.warn("Keyword argument argtypes is deprecated",
                          DeprecationWarning)
            assert sig is None
            if restype is None:
                sig = tuple(argtypes)
            else:
                sig = restype(*argtypes)
        del argtypes
        del restype

        indims = [len(x) for x in self.inputsig]
        outdims = [len(x) for x in self.outputsig]

        funcname = self.py_func.__name__
        src = expand_gufunc_template(self._kernel_template, indims,
                                     outdims, funcname)

        glbls = self._get_globals(sig)

        _exec(src, glbls)
        fnobj = glbls['__gufunc_{name}'.format(name=funcname)]

        args, return_type = sigutils.normalize_signature(sig)
        outertys = list(_determine_gufunc_outer_types(args, indims + outdims))
        kernel = self._compile_kernel(fnobj, sig=tuple(outertys))

        dtypes = tuple(np.dtype(str(t.dtype)) for t in outertys)
        self.kernelmap[tuple(dtypes[:-1])] = dtypes[-1], kernel

    def _compile_kernel(self, fnobj, sig):
        raise NotImplementedError

    def _get_globals(self, sig):
        raise NotImplementedError


def _determine_gufunc_outer_types(argtys, dims):
    for at, nd in zip(argtys, dims):
        if isinstance(at, types.Array):
            yield at.copy(ndim=nd + 1)
        else:
            if nd > 0:
                raise ValueError("gufunc signature mismatch: ndim>0 for scalar")
            yield types.Array(dtype=at, ndim=1, layout='A')


def expand_gufunc_template(template, indims, outdims, funcname):
    """Expand gufunc source template
    """
    argdims = indims + outdims
    argnames = ["arg{0}".format(i) for i in range(len(argdims))]
    checkedarg = "min({0})".format(', '.join(["{0}.shape[0]".format(a)
                                              for a in argnames]))
    inputs = [_gen_src_for_indexing(aref, adims, _gen_src_for_input_indexing)
              for aref, adims in zip(argnames, indims)]
    outputs = [_gen_src_for_indexing(aref, adims, _gen_src_for_output_indexing)
               for aref, adims in zip(argnames[len(indims):], outdims)]
    argitems = inputs + outputs
    src = template.format(name=funcname, args=', '.join(argnames),
                          checkedarg=checkedarg,
                          argitems=', '.join(argitems))
    return src


def _gen_src_for_indexing(aref, adims, gen_sliced):
    return "{aref}[{sliced}]".format(aref=aref, sliced=gen_sliced(adims))


def _gen_src_for_input_indexing(adims):
    if adims > 0:
        return _gen_src_for_array_indexing(adims)
    else:
        return '__tid__'


def _gen_src_for_output_indexing(adims):
    if adims > 0:
        return _gen_src_for_array_indexing(adims)
    else:
        return '__tid__:(__tid__ + 1)'


def _gen_src_for_array_indexing(adims):
    return ','.join(['__tid__'] + [':'] * adims)


class GUFuncEngine(object):
    '''Determine how to broadcast and execute a gufunc
    base on input shape and signature
    '''

    @classmethod
    def from_signature(cls, signature):
        return cls(*parse_signature(signature))

    def __init__(self, inputsig, outputsig):
        # signatures
        self.sin = inputsig
        self.sout = outputsig
        # argument count
        self.nin = len(self.sin)
        self.nout = len(self.sout)

    def schedule(self, ishapes):
        if len(ishapes) != self.nin:
            raise TypeError('invalid number of input argument')

        # associate symbol values for input signature
        symbolmap = {}
        outer_shapes = []
        inner_shapes = []

        for argn, (shape, symbols) in enumerate(zip(ishapes, self.sin)):
            argn += 1  # start from 1 for human
            inner_ndim = len(symbols)
            if len(shape) < inner_ndim:
                fmt = "arg #%d: insufficient inner dimension"
                raise ValueError(fmt % (argn,))
            if inner_ndim:
                inner_shape = shape[-inner_ndim:]
                outer_shape = shape[:-inner_ndim]
            else:
                inner_shape = ()
                outer_shape = shape

            for axis, (dim, sym) in enumerate(zip(inner_shape, symbols)):
                axis += len(outer_shape)
                if sym in symbolmap:
                    if symbolmap[sym] != dim:
                        fmt = "arg #%d: shape[%d] mismatch argument"
                        raise ValueError(fmt % (argn, axis))
                symbolmap[sym] = dim

            outer_shapes.append(outer_shape)
            inner_shapes.append(inner_shape)

        # solve output shape
        oshapes = []
        for outsig in self.sout:
            oshape = []
            for sym in outsig:
                oshape.append(symbolmap[sym])
            oshapes.append(tuple(oshape))

        # find the biggest outershape as looping dimension
        sizes = [reduce(operator.mul, s, 1) for s in outer_shapes]
        largest_i = np.argmax(sizes)
        loopdims = outer_shapes[largest_i]

        pinned = [False] * self.nin  # same argument for each iteration
        for i, d in enumerate(outer_shapes):
            if d != loopdims:
                if d == (1,) or d == ():
                    pinned[i] = True
                else:
                    fmt = "arg #%d: outer dimension mismatch"
                    raise ValueError(fmt % (i + 1,))

        return GUFuncSchedule(self, inner_shapes, oshapes, loopdims, pinned)


class GUFuncSchedule(object):
    def __init__(self, parent, ishapes, oshapes, loopdims, pinned):
        self.parent = parent
        # core shapes
        self.ishapes = ishapes
        self.oshapes = oshapes
        # looping dimension
        self.loopdims = loopdims
        self.loopn = reduce(operator.mul, loopdims, 1)
        # flags
        self.pinned = pinned

        self.output_shapes = [loopdims + s for s in oshapes]

    def __str__(self):
        import pprint

        attrs = 'ishapes', 'oshapes', 'loopdims', 'loopn', 'pinned'
        values = [(k, getattr(self, k)) for k in attrs]
        return pprint.pformat(dict(values))


class GenerializedUFunc(object):
    def __init__(self, kernelmap, engine):
        self.kernelmap = kernelmap
        self.engine = engine
        self.max_blocksize = 2 ** 30
        assert self.engine.nout == 1, "only support single output"

    def __call__(self, *args, **kws):
        callsteps = self._call_steps(args, kws)
        callsteps.prepare_inputs()
        indtypes, schedule, outdtype, kernel = self._schedule(
            callsteps.norm_inputs, callsteps.output)
        callsteps.adjust_input_types(indtypes)
        callsteps.allocate_outputs(schedule, outdtype)
        callsteps.prepare_kernel_parameters()
        newparams, newretval = self._broadcast(schedule,
                                               callsteps.kernel_parameters,
                                               callsteps.kernel_returnvalue)
        callsteps.launch_kernel(kernel, schedule.loopn, newparams + [newretval])
        return callsteps.post_process_result()

    def _schedule(self, inputs, out):
        input_shapes = [a.shape for a in inputs]
        schedule = self.engine.schedule(input_shapes)

        # find kernel
        idtypes = tuple(i.dtype for i in inputs)
        try:
            outdtype, kernel = self.kernelmap[idtypes]
        except KeyError:
            # No exact match, then use the first compatbile.
            # This does not match the numpy dispatching exactly.
            # Later, we may just jit a new version for the missing signature.
            idtypes = self._search_matching_signature(idtypes)
            # Select kernel
            outdtype, kernel = self.kernelmap[idtypes]

        # check output
        if out is not None and schedule.output_shapes[0] != out.shape:
            raise ValueError('output shape mismatch')

        return idtypes, schedule, outdtype, kernel

    def _search_matching_signature(self, idtypes):
        """
        Given the input types in `idtypes`, return a compatible sequence of
        types that is defined in `kernelmap`.

        Note: Ordering is guaranteed by `kernelmap` being a OrderedDict
        """
        for sig in self.kernelmap.keys():
            if all(np.can_cast(actual, desired)
                   for actual, desired in zip(sig, idtypes)):
                return sig
        else:
            raise TypeError("no matching signature")

    def _broadcast(self, schedule, params, retval):
        assert schedule.loopn > 0, "zero looping dimension"

        odim = 1 if not schedule.loopdims else schedule.loopn
        newparams = []
        for p, cs in zip(params, schedule.ishapes):
            if not cs and p.size == 1:
                # Broadcast scalar input
                devary = self._broadcast_scalar_input(p, odim)
                newparams.append(devary)
            else:
                # Broadcast vector input
                newparams.append(p.reshape(odim, *cs))
        newretval = retval.reshape(odim, *schedule.oshapes[0])
        return newparams, newretval

    def _broadcast_scalar_input(self, ary, shape):
        raise NotImplementedError


class GUFuncCallSteps(object):
    __slots__ = [
        'args',
        'kwargs',
        'output',
        'norm_inputs',
        'kernel_returnvalue',
        'kernel_parameters',
        '_is_device_array',
        '_need_device_conversion',
    ]

    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

        self.output = self.kwargs.get('out')
        self._is_device_array = [self.is_device_array(a) for a in self.args]
        self._need_device_conversion = not any(self._is_device_array)

        # Normalize inputs
        inputs = []
        for a, isdev in zip(self.args, self._is_device_array):
            if isdev:
                inputs.append(a)
            else:
                inputs.append(np.array(a))
        self.norm_inputs = inputs

    def adjust_input_types(self, indtypes):
        """
        Attempt to cast the inputs to the required types if necessary
        and if they are not device array.

        Side effect: Only affects the element of `norm_inputs` that requires
        a type cast.
        """
        for i, (ity, val) in enumerate(zip(indtypes, self.norm_inputs)):
            if ity != val.dtype:
                if not hasattr(val, 'astype'):
                    msg = ("compatible signature is possible by casting but "
                           "{0} does not support .astype()").format(type(val))
                    raise TypeError(msg)
                # Cast types
                self.norm_inputs[i] = val.astype(ity)

    def allocate_outputs(self, schedule, outdtype):
        # allocate output
        if self._need_device_conversion or self.output is None:
            retval = self.device_array(shape=schedule.output_shapes[0],
                                       dtype=outdtype)
        else:
            retval = self.output
        self.kernel_returnvalue = retval

    def prepare_kernel_parameters(self):
        params = []
        for inp, isdev in zip(self.norm_inputs, self._is_device_array):
            if isdev:
                params.append(inp)
            else:
                params.append(self.to_device(inp))
        assert all(self.is_device_array(a) for a in params)
        self.kernel_parameters = params

    def post_process_result(self):
        if self._need_device_conversion:
            out = self.to_host(self.kernel_returnvalue, self.output)
        elif self.output is None:
            out = self.kernel_returnvalue
        else:
            out = self.output
        return out

    def prepare_inputs(self):
        pass

    def launch_kernel(self, kernel, nelem, args):
        raise NotImplementedError

    def is_device_array(self, obj):
        raise NotImplementedError

    def to_device(self, hostary):
        raise NotImplementedError

    def device_array(self, shape, dtype):
        raise NotImplementedError


class UFuncEngine(object):
    def __init__(self, types, signature):
        self.types = tuple(types)
        self.signature = signature
        inputs, outputs = parse_signature(signature)
        self.sin = tuple(inputs)
        self.sout = tuple(outputs)
        self.nin = len(self.sin)
        self.nout = len(self.sout)
        if len(self.types) != self.nin + self.nout:
            raise TypeError("invalid number of types")

    def schedule(self, args, kwargs):
        # Process arguments
        self._prepare_arguments(args, kwargs)
        self._match_signature()
        self._compute_output_shape()
        # Determine broadcasting
        outer_shapes = self.outer_input_shapes + self.outer_output_shapes
        inner_shapes = self.inner_input_shapes + self.inner_output_shapes
        self.loop_shape = _multi_broadcast((1,), *outer_shapes)
        self.broadcasted_shapes = tuple([self.loop_shape + s
                                         for s in inner_shapes])
        # Cast to arrays of the right type
        values = [self._asarray(v, dtype=ty)
                  for ty, v in zip(self.types, self.inputs + self.outputs)]

        # Broadcast
        self.kernel_args = self._broadcast(values, self.broadcasted_shapes)

    def _prepare_arguments(self, args, kwargs):
        inputs = args[:self.nin]
        outputs = args[self.nin:]
        if len(inputs) != self.nin:
            raise TypeError("invalid number of inputs")

        if 'out' in kwargs:
            if not outputs:
                outputs = [kwargs.pop('out')]
            else:
                raise TypeError("output already specified but `out` is defined")

        if outputs and len(outputs) != self.nout:
            raise TypeError("Invalid number of outputs")

        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.input_shapes = [self._get_shape(i) for i in self.inputs]
        self.output_shapes = [self._get_shape(i) for i in self.outputs]

    def _match_signature(self):
        symbols = {}
        inner, outer = self._associate_symbol_values(symbols, self.sin,
                                                     self.input_shapes)
        self.symbols = symbols
        self.inner_input_shapes = tuple(inner)
        self.outer_input_shapes = tuple(outer)

    def _compute_output_shape(self):
        if self.outputs:
            self._adjust_existing_output()
        else:
            self._allocate_output()

    def _adjust_existing_output(self):
        assert self.outputs, "expect explicit output arguments"
        assert len(self.output_shapes) == len(self.outputs)
        inner, outer = self._associate_symbol_values(self.symbols, self.sout,
                                                     self.output_shapes)
        self.inner_output_shapes = tuple(inner)
        self.outer_output_shapes = tuple(outer)

    def _allocate_output(self):
        raise NotImplementedError

    def _associate_symbol_values(self, symvalues, symbols, shapes):
        outerlist = []
        innerlist = []

        for num, (syms, shape) in enumerate(zip(symbols, shapes), start=1):
            sym_stack = list(reversed(syms))
            shape_stack = list(reversed(shape))

            while sym_stack and shape_stack:
                sym = sym_stack.pop()
                val = shape_stack.pop()
                got = symvalues.get(sym, val)
                if got != val:
                    raise TypeError("symbol {0} mismatch in arg #{1}".format(
                        sym, num))
                symvalues[sym] = got

            if sym_stack:
                raise TypeError(("arg #{0} has invalid shape for signature "
                                 "{1}").format(num, syms))

            outer = tuple([x for x in reversed(shape_stack)])
            outerlist.append(outer)
            innerlist.append(tuple(shape[len(outer):]))

        return innerlist, outerlist

    # ------- can override ---------

    def _get_shape(self, value):
        return getattr(value, 'shape', ())

    def _broadcast(self, arrays, shapes):
        out = []
        for arr, shape in zip(arrays, shapes):
            if arr.shape != shape:
                arr = self._broadcast_reshape(arr, shape)
            out.append(arr)

        return out

    def _broadcast_reshape(self, arr, shape):
        if hasattr(arr, 'reshape'):
            return arr.reshape(shape)
        else:
            msg = "type {0} does not have .reshape()".format(type(arr))
            raise TypeError(msg)

    def _asarray(self, value, dtype):
        return np.asarray(value, dtype=dtype)
