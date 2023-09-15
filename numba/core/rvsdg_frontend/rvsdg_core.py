import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator, Sequence, no_type_check
from functools import reduce
from dataclasses import dataclass
import inspect

from numba.core import (
    ir,
    bytecode,
    ir_utils,
    utils,
)
from numba.core.utils import (
    BINOPS_TO_OPERATORS,
    INPLACE_BINOPS_TO_OPERATORS,
)
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow, FlowInfo
from numba_rvsdg.core.datastructures.scfg import SCFG, NameGenerator
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    PythonBytecodeBlock,
    RegionBlock,
    SyntheticBranch,
    SyntheticAssignment,
)
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from .rvsdg.rvsdgutils import RegionVisitor
from .rvsdgir import rvsdgir
rvsdgPort = rvsdgir.rvsdgir.Port


def run_frontend(func):
    sig = utils.pySignature.from_callable(func)
    argnames = tuple(sig.parameters)
    rvsdg = build_rvsdg(func.__code__, argnames)


def _debug_scfg(name, byteflow):
    from .rvsdg.regionrenderer import graph_debugger

    g = ByteFlowRenderer().render_byteflow(byteflow)

    with graph_debugger() as dbg:
        dbg.add_graphviz(name, g)


def build_rvsdg(code, argnames: tuple[str, ...]) -> SCFG:
    byteflow = ByteFlow.from_bytecode(code)
    bcmap = byteflow.scfg.bcmap_from_bytecode(byteflow.bc)
    byteflow = byteflow.restructure()
    _debug_scfg("Raw SCFG", byteflow)

    transformer = ToRvsdgIR(bcmap, argnames)
    transformer.visit_graph(byteflow.scfg, data=None)

    raise AssertionError


@dataclass(frozen=True)
class PyAttrs:
    bcinst: dis.Instruction

    def __str__(self):
        return f"py({self.bcinst.opname})"


@dataclass(frozen=True)
class PyStoreAttrs:
    bcinst: dis.Instruction
    varname: str

    def __str__(self):
        return f"py.store({self.bcinst.opname} {self.varname!r})"




class ToRvsdgIR(RegionVisitor[None]):
    bcmap: dict[int, dis.Instruction]
    def __init__(self, bcmap, argnames):
        super().__init__()
        self.bcmap = bcmap
        self.ir = rvsdgir.Region.make(
            opname="function",
            ins=("env", *[f"arg_[{k}]" for k in argnames]),
            outs=("env", "return_value"),
        )

    def visit_block(self, block: BasicBlock, data: None) -> None:
        if isinstance(block, PythonBytecodeBlock):
            instlist = block.get_instructions(self.bcmap)

            bctorvsdg = BcToRvsdgIR.run(tuple(self.ir.args) , instlist)

            print(bctorvsdg.region.prettyformat())
        else:
            raise NotImplementedError(type(block))

    def visit_loop(self, region: RegionBlock, data: None) -> None:
        raise NotImplementedError(type(block))

    def visit_switch(self, region: RegionBlock, data: None) -> None:
        raise NotImplementedError(type(block))



class BcToRvsdgIR:
    stack: list[rvsdgPort]
    effect: rvsdgPort
    in_effect: rvsdgPort
    varmap: dict[str, rvsdgPort]
    incoming_vars: dict[str, rvsdgPort]
    incoming_stackvars: list[rvsdgPort]
    _kw_names: rvsdgPort | None
    region: rvsdgir.Region

    @classmethod
    def run(
        cls,
        argnames: tuple[str, ...],
        instlist: Sequence[dis.Instruction],
    ):
        cvtr = cls(argnames)
        for inst in instlist:
            cvtr.convert(inst)
        return cvtr

    def __init__(self, argnames: tuple[str, ...]):
        self.stack = []
        self.varmap = {}
        self.incoming_vars = {}
        # self.incoming_stackvars = []
        self._kw_names = None
        self.region = rvsdgir.Region.make(
            opname="block",
            ins=argnames,
            outs=["env", "return_value"],
        )
        self.effect = self.region.args["env"]
        self.in_effect = self.effect

    def push(self, val: rvsdgPort):
        self.stack.append(val)

    def pop(self) -> rvsdgPort:
        return self.stack.pop()

    def peek(self, which: int) -> rvsdgPort:
        popped = [self.pop() for i in range(which)]
        # push everything back in
        for elem in reversed(popped):
            self.push(elem)
        # return the last element
        return popped[-1]

    def top(self) -> rvsdgPort:
        tos = self.pop()
        self.push(tos)
        return tos

    def _decorate_varname(self, varname: str) -> str:
        return f"var.{varname}"

    def store(self, varname: str, value: rvsdgPort):
        self.varmap[varname] = value

    def load(self, varname: str) -> rvsdgPort:
        if varname not in self.varmap:
            vs = self.region.args.add_port(f"var_{varname}")
            self.incoming_vars[varname] = vs
            self.varmap[varname] = vs

        return self.varmap[varname]

    def replace_effect(self, env: rvsdgPort):
        self.effect = env

    def convert(self, inst: dis.Instruction):
        fn = getattr(self, f"op_{inst.opname}")
        fn(inst)

    def set_kw_names(self, kw_vs: rvsdgPort):
        assert self._kw_names is None
        self._kw_names = kw_vs

    def pop_kw_names(self):
        res = self._kw_names
        self._kw_names = None
        return res

    def op_NOP(self, inst: dis.Instruction):
        pass  # no-op

    def op_POP_TOP(self, inst: dis.Instruction):
        self.pop()

    def op_SWAP(self, inst: dis.Instruction):
        s = self.stack
        idx = inst.argval
        s[-1], s[-idx] = s[-idx], s[-1]

    def op_COPY(self, inst: dis.Instruction):
        self.push(self.peek(inst.argval))

    def op_RESUME(self, inst: dis.Instruction):
        pass  # no-op

    def op_COPY_FREE_VARS(self, inst: dis.Instruction):
        pass  # no-op

    def op_PUSH_NULL(self, inst: dis.Instruction):
        op = Op(opname="push_null", bc_inst=inst)
        null = op.add_output("null")
        self.push(null)

    def op_LOAD_GLOBAL(self, inst: dis.Instruction):
        assert isinstance(inst.arg, int)  # for typing
        load_null = inst.arg & 1
        op = self.region.add_simple_op(
            opname="py.global.load",
            ins=["env"],
            outs=["out"],
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        op.ins(env=self.effect)
        if load_null:
            nullop = self.region.add_simple_op(
                opname="py.null",
                ins=(),
                outs=["out"],
                attrs=dict(py=PyAttrs(bcinst=inst)),
            )
            self.push(nullop.outs.out)
        self.push(op.outs.out)

    def op_LOAD_CONST(self, inst: dis.Instruction):
        op = self.region.add_simple_op(
            f"py.const.load",
            ins=(),
            outs=["out"],
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        self.push(op.outs.out)

    def op_STORE_FAST(self, inst: dis.Instruction):
        tos = self.pop()
        varname = self._decorate_varname(inst.argval)
        op = self.region.add_simple_op(
            opname=f"py.store",
            ins=["env", "val"],
            outs=["env", "out"],
            attrs=dict(py=PyStoreAttrs(
                varname=inst.argval,
                bcinst=inst,
            ))
        )
        op.ins(env=self.effect, val=tos)
        self.replace_effect(op.outs.env)
        self.store(varname, op.outs.out)

    def op_LOAD_FAST(self, inst: dis.Instruction):
        varname = self._decorate_varname(inst.argval)
        self.push(self.load(varname))

    def op_LOAD_ATTR(self, inst: dis.Instruction):
        obj = self.pop()
        attr = inst.argval
        op = Op(opname=f"load_attr.{attr}", bc_inst=inst)
        op.add_input("obj", obj)
        self.push(op.add_output("out"))

    def op_LOAD_METHOD(self, inst: dis.Instruction):
        obj = self.pop()
        attr = inst.argval
        op = Op(opname=f"load_method.{attr}", bc_inst=inst)
        op.add_input("obj", obj)
        self.push(op.add_output("null"))
        self.push(op.add_output("out"))

    def op_LOAD_DEREF(self, inst: dis.Instruction):
        op = Op(opname="load_deref", bc_inst=inst)
        self.push(op.add_output("out"))

    def op_PRECALL(self, inst: dis.Instruction):
        pass  # no-op

    def op_KW_NAMES(self, inst: dis.Instruction):
        op = Op(opname="kw_names", bc_inst=inst)
        self.set_kw_names(op.add_output("out"))

    def op_CALL(self, inst: dis.Instruction):
        argc: int = inst.argval
        arg1plus = reversed([self.pop() for _ in range(argc)])
        arg0 = self.pop()  # TODO
        kw_names = self.pop_kw_names()

        args: list[rvsdgPort] = [arg0, *arg1plus]
        callable = self.pop()  # TODO
        opname = "py.call"
        ins = ["env", "callee"] + [f"arg{i}" for i in range(len(args))]
        if kw_names is not None:
            ins += ("kw_names",)
            opname.append("kw")
        op = self.region.add_simple_op(
            opname=opname,
            ins=ins,
            outs=("env", "out"),
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        argsmap = {f"arg{i}": v for i, v in enumerate(args)}
        op.ins(env=self.effect, callee=callable, **argsmap)
        self.replace_effect(op.outs.env)
        self.push(op.outs.out)

    def op_GET_ITER(self, inst: dis.Instruction):
        tos = self.pop()
        op = self.region.add_simple_op(
            opname="py.getiter",
            ins=["env", "val"],
            outs=["env", "out"],
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        op.ins(env=self.effect, val=tos)
        self.replace_effect(op.outs.env)
        self.push(op.outs.out)

    def op_FOR_ITER(self, inst: dis.Instruction):
        tos = self.top()
        op = Op(opname="foriter", bc_inst=inst)
        op.add_input("iter", tos)
        # Store the indvar into an internal variable
        self.store("indvar", op.add_output("indvar"))

    def _binaryop(self, opname: str, inst: dis.Instruction):
        rhs = self.pop()
        lhs = self.pop()
        op = Op(opname=opname, bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_input("lhs", lhs)
        op.add_input("rhs", rhs)
        self.replace_effect(op.add_output("env", is_effect=True))
        self.push(op.add_output("out"))

    def op_BINARY_OP(self, inst: dis.Instruction):
        self._binaryop("binaryop", inst)

    def op_COMPARE_OP(self, inst: dis.Instruction):
        self._binaryop("compareop", inst)

    def op_IS_OP(self, inst: dis.Instruction):
        self._binaryop("is_op", inst)

    def _unaryop(self, opname: str, inst: dis.Instruction):
        op = Op(opname=opname, bc_inst=inst)
        op.add_input("val", self.pop())
        self.push(op.add_output("out"))

    def op_UNARY_NOT(self, inst: dis.Instruction):
        self._unaryop("not", inst)

    def op_BINARY_SUBSCR(self, inst: dis.Instruction):
        index = self.pop()
        target = self.pop()
        op = Op(opname="binary_subscr", bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_input("index", index)
        op.add_input("target", target)
        self.replace_effect(op.add_output("env", is_effect=True))
        self.push(op.add_output("out"))

    def op_STORE_SUBSCR(self, inst: dis.Instruction):
        index = self.pop()
        target = self.pop()
        value = self.pop()
        op = Op(opname="store_subscr", bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_input("index", index)
        op.add_input("target", target)
        op.add_input("value", value)
        self.replace_effect(op.add_output("env", is_effect=True))

    def op_BUILD_TUPLE(self, inst: dis.Instruction):
        count = inst.arg
        assert isinstance(count, int)
        items = list(reversed([self.pop() for _ in range(count)]))
        op = Op(opname="build_tuple", bc_inst=inst)
        for i, it in enumerate(items):
            op.add_input(str(i), it)
        self.push(op.add_output("out"))

    def op_BUILD_SLICE(self, inst: dis.Instruction):
        argc = inst.arg
        if argc == 2:
            tos = self.pop()
            tos1 = self.pop()
            start = tos1
            stop = tos
            step = None
        elif argc == 3:
            tos = self.pop()
            tos1 = self.pop()
            tos2 = self.pop()
            start = tos2
            stop = tos1
            step = tos
        else:
            raise Exception("unreachable")

        op = Op(opname="build_slice", bc_inst=inst)
        op.add_input("start", start)
        op.add_input("stop", stop)
        if step is not None:
            op.add_input("step", step)
        self.push(op.add_output("out"))

    def op_RETURN_VALUE(self, inst: dis.Instruction):
        tos = self.pop()
        op = self.region.add_simple_op(
            opname=f"py.return",
            ins=["env", "val"],
            outs=["env", "res"],
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        op.ins(env=self.effect, val=tos)
        self.replace_effect(op.outs["env"])
        self.region.results(env=op.outs["env"],
                            return_value=op.outs.res)

    def op_RAISE_VARARGS(self, inst: dis.Instruction):
        if inst.arg == 0:
            exc = None
            # # No re-raising within a try-except block.
            # # But we allow bare reraise.
            # if state.has_active_try():
            #     raise UnsupportedError(
            #         "The re-raising of an exception is not yet supported.",
            #         loc=self.get_debug_loc(inst.lineno),
            #     )
            raise NotImplementedError
        elif inst.arg == 1:
            exc = self.pop()
        else:
            raise ValueError("Multiple argument raise is not supported.")
        op = Op(opname="raise_varargs", bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_input("exc", exc)
        self.replace_effect(op.add_output("env", is_effect=True))

    def op_JUMP_FORWARD(self, inst: dis.Instruction):
        pass  # no-op

    def op_JUMP_BACKWARD(self, inst: dis.Instruction):
        pass  # no-op

    def _POP_JUMP_X_IF_Y(self, inst: dis.Instruction, *, opname: str):
        tos = self.pop()
        op = Op(opname, bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_input("pred", tos)
        self.replace_effect(op.add_output("env", is_effect=True))

    def op_POP_JUMP_FORWARD_IF_TRUE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname="jump.if_true")

    def op_POP_JUMP_FORWARD_IF_FALSE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname="jump.if_false")

    def op_POP_JUMP_BACKWARD_IF_TRUE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname="jump.if_true")

    def op_POP_JUMP_BACKWARD_IF_FALSE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname="jump.if_false")

    def op_POP_JUMP_FORWARD_IF_NONE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname="jump.if_none")

    def op_POP_JUMP_FORWARD_IF_NOT_NONE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname="jump.if_not_none")

    def _JUMP_IF_X_OR_POP(self, inst: dis.Instruction, *, opname):
        tos = self.top()
        op = Op(opname, bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_input("pred", tos)
        self.replace_effect(op.add_output("env", is_effect=True))

    def op_JUMP_IF_TRUE_OR_POP(self, inst: dis.Instruction):
        self._JUMP_IF_X_OR_POP(inst, opname="jump.if_true")

    def op_JUMP_IF_FALSE_OR_POP(self, inst: dis.Instruction):
        self._JUMP_IF_X_OR_POP(inst, opname="jump.if_false")

    def op_BEFORE_WITH(self, inst: dis.Instruction):
        ctx_mngr = self.pop()

        op = Op(inst.opname, bc_inst=inst)
        op.add_input("env", self.effect)
        op.add_input("ctxmngr", ctx_mngr)
        self.replace_effect(op.add_output("env", is_effect=True))
        yielded = op.add_output("yielded")
        exitfn = op.add_output("exitfn")

        self.push(exitfn)
        self.push(yielded)
