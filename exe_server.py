"""
The execution server.
"""

import zmq

from llvmlite import binding as ll
from numba import _dynfunc
from numba.core.base import _load_global_helpers


import ctypes
import pickle
import types

port = 5555


libpy = ctypes.CDLL(None)


class LLVMExe:
    def __init__(self):
        ll.initialize()
        ll.initialize_all_targets()
        ll.initialize_all_asmprinters()

        _load_global_helpers()

        self._dummy_module = types.ModuleType("LLVMExe")

    def jit(self, mod, fname, args):
        target = ll.Target.from_triple(ll.get_process_triple())
        tm = target.create_target_machine(reloc="static", jit=True)

        mod = ll.parse_assembly(mod)
        engine = ll.create_mcjit_compiler(mod, tm)

        print(mod.get_function(fname))
        print(f"fname={fname}")

        engine.finalize_object()
        addr = engine.get_function_address(fname)
        print(f"JIT function address: {addr:x}")

        prototype = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)
        cfunc = prototype(addr)
        print(f"Execute JIT function with args: {args}")
        return cfunc(*args)


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
        result = le.jit(**dct)

        socket.send(pickle.dumps({"return": result}))


main()
