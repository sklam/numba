import sys
import dis
import os
import os.path

import logging
from contextlib import contextmanager

import numba
from numba.core import errors

NUMBA_ROOT = os.path.dirname(numba.__file__)

IGNORE_LIST = {
    "core/analysis.py",
    "core/base.py",
    "core/bytecode.py",
    "core/byteflow.py",
    "core/compiler_machinery.py",
    "core/compiler.py",
    "core/consts.py",
    "core/controlflow.py",
    "core/cpu_options.py",
    "core/decorators.py",
    "core/dispatcher.py",
    "core/event.py",
    "core/inline_closurecall.py",
    "core/interpreter.py",
    "core/ir.py",
    "core/options.py",
    "core/rewrites/static_raise.py",
    "core/sigutils.py",
    "core/target_extension.py",
    "core/targetconfig.py",
    "core/typed_passes.py",
    "core/typeinfer.py",
    "core/types/functions.py",
    "core/types/misc.py",
    "core/typing/context.py",
    "core/typing/templates.py",
    "core/typing/typeof.py",
    "core/untyped_passes.py",
    "core/utils.py",
    "core/callconv.py",
    "cloudpickle/cloudpickle.py",
    "core/types/abstract.py",
    "core/pythonapi.py",
    "core/codegen.py",
    "core/datamodel/old_models.py",
    "core/itanium_mangler.py",
    "core/typeconv/typeconv.py",
    "core/ir_utils.py",
}


_processed_functions = set()


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.FileHandler(f"numba_errors_{os.getpid()}.log"))


@contextmanager
def pause_tracing():
    try:
        fn = sys.gettrace()
        sys.settrace(None)
        yield
    finally:
        sys.settrace(fn)


@contextmanager
def start_tracing():
    assert sys.version_info[:2] == (3, 13), "only works on 3.13"

    fn = sys.gettrace()
    if fn is None:
        sys.settrace(_trace_func)
        try:
            yield
        finally:
            sys.settrace(None)
    else:
        yield


def _trace_func(frame, event, arg):
    try:
        if event == "call":
            co = frame.f_code
            filename = co.co_filename
            lineno = frame.f_lineno
            key = filename, lineno
            if filename.startswith(NUMBA_ROOT):
                if key not in _processed_functions:
                    relfile = filename[len(NUMBA_ROOT) + 1 :]
                    # only trace into Numba source code
                    _run_analysis(frame, co, relfile)
                    _processed_functions.add(key)

    except Exception:
        _logger.exception("error raised in _trace_func ")


def _run_analysis(frame, co, relpath):
    inst: dis.Instruction
    bc = dis.Bytecode(co)

    instlist = list(bc)
    for i, inst in enumerate(instlist):
        if inst.opname == "RAISE_VARARGS":
            # The current logic search for pattern `raise Exc()` such that
            # the raise statement and loading of the exception type is on the
            # same line.
            insts = _find_inst_for_line(instlist, i)
            # The first LOAD_GLOBAL must be the load for the callee due to the
            # stack ordering expected by CALL
            if "LOAD_GLOBAL" != insts[0].opname:
                return
            if not any(inst.opname == "CALL" for inst in insts[1:]):
                return
            # Ensure not an `assert` statement
            if any(inst.opname == "LOAD_ASSERTION_ERROR" for inst in insts):
                return

            _process_raise(insts, relpath, inst.line_number)


def _find_inst_for_line(instlist: list[dis.Instruction], offset: int):
    # search backward to find relevant inst on the line
    buf = [instlist[offset]]
    line = buf[-1].line_number
    while offset > 0:
        offset -= 1
        cur = instlist[offset]
        if cur.line_number == line:
            buf.append(cur)
    buf.reverse()
    return buf


def _process_raise(instlist: list[dis.Instruction], filename: str, line: int):
    global_name = instlist[0].argval
    if global_name == "errors" and instlist[1].opname == "LOAD_ATTR":
        # Handle `errors.XYZ``
        global_name = instlist[1].argval

    # Ignore if name matches those in numba.core.errors
    if not hasattr(errors, global_name):
        _logger.info(f"{global_name:30} | {filename}:{line}")


def filter_unique_files(files):
    def strip(line):
        return line.strip()

    combined = set()
    for fp in files:
        with open(fp, "r") as fin:
            combined.update(filter(bool, map(strip, fin)))

    for ln in sorted(combined):
        _, filepath_and_line = ln.split("|")
        filepath, line = filepath_and_line.split(":")
        if filepath.strip() in IGNORE_LIST:
            continue
        print(ln)


main = filter_unique_files

if __name__ == "__main__":
    main(sys.argv[1:])
