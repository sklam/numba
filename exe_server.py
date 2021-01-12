"""
The execution server.
"""

import zmq

from llvmlite import binding as ll
from numba import _dynfunc
from numba.core.base import _load_global_helpers
from numba.core import types

import ctypes
import pickle
import types as py_types

port = 5555


libpy = ctypes.CDLL(None)

class LLVMExe:
    def __init__(self):
        ll.initialize()
        ll.initialize_all_targets()
        ll.initialize_all_asmprinters()

        _load_global_helpers()

        self._dummy_module = py_types.ModuleType("LLVMExe")

        self._allocated = {}

    def jit(self, mod, fname, argtypes, args):
        target = ll.Target.from_triple(ll.get_process_triple())
        tm = target.create_target_machine(reloc="static", jit=True)

        mod = ll.parse_assembly(mod)
        engine = ll.create_mcjit_compiler(mod, tm)

        print(mod.get_function(fname).module)
        print(f"fname={fname} argtypes={argtypes}")

        cargtys = []
        cargs = []
        for at, av in zip(argtypes, args):
            if isinstance(at, types.Integer):
                cargtys.append(ctypes.c_int64)
                cargs.append(av)
            else:
                raise TypeError(at)

        print('>>>>', cargtys, cargs)

        engine.finalize_object()
        addr = engine.get_function_address(fname)
        print(f"JIT function address: {addr:x}")

        prototype = ctypes.CFUNCTYPE(ctypes.c_int64, *cargtys)
        cfunc = prototype(addr)
        print(f"Execute JIT function with args: {args}")
        return cfunc(*cargs)

    def allocate(self, ary):
        print(f"Allocate array: {ary}")
        addr = ary.ctypes.data
        self._allocated[addr] = ary
        return addr


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")

    le = LLVMExe()
    while True:
        #  Wait for next request from client
        message = socket.recv()
        print(f"Received {len(message)} Bytes")
        dct = pickle.loads(message)
        msg = dct.pop('msg')
        if msg == 'jit':
            result = le.jit(**dct)
            socket.send(pickle.dumps({"return": result}))
        elif msg == 'allocate':
            result = le.allocate(**dct)
            socket.send(pickle.dumps({"return": result}))


if __name__ == '__main__':
    main()
