import dis
import operator
from contextlib import contextmanager
import builtins
from typing import (
    Iterator,
    Sequence,
    Mapping,
    Callable,
    no_type_check,
    Type,
    TypeVar,
    Any,
)
from functools import reduce, cache
from dataclasses import dataclass, replace as _dataclass_replace
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
    SyntheticFill,
    SyntheticTail,
)
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from .rvsdg.rvsdgutils import RegionVisitor
from . import rvsdgir

rvsdgPort = rvsdgir.Port


def run_frontend(func):
    sig = utils.pySignature.from_callable(func)
    argnames = tuple(sig.parameters)
    rvsdg = build_rvsdg(func.__code__, argnames)

    func_id = bytecode.FunctionIdentity.from_function(func)
    func_ir = rvsdg_to_ir(func_id, rvsdg)
    return func_ir


def _debug_scfg(name, byteflow):
    from .rvsdg.regionrenderer import graph_debugger

    g = ByteFlowRenderer().render_byteflow(byteflow)

    with graph_debugger() as dbg:
        dbg.add_graphviz(name, g)


def _debug_dot(name, gvdot):
    from .rvsdg.regionrenderer import graph_debugger

    with graph_debugger() as dbg:
        dbg.add_graphviz(name, gvdot)


def build_rvsdg(code, argnames: tuple[str, ...]) -> rvsdgir.Region:
    from .rvsdg.bc2rvsdg import (
        canonicalize_scfg,
        _scfg_add_conditional_pop_stack,
    )

    byteflow = ByteFlow.from_bytecode(code)
    bcmap = byteflow.scfg.bcmap_from_bytecode(byteflow.bc)
    _scfg_add_conditional_pop_stack(bcmap, byteflow.scfg)
    byteflow = byteflow.restructure()
    canonicalize_scfg(byteflow.scfg)
    _debug_scfg("canonicalized SCFG", byteflow)

    transformer = ToRvsdgIR.run(byteflow.scfg, bcmap, argnames)

    # transformer.ir.prettyprint()
    render_rvsdgir(transformer.ir, "rvsdgir")

    return transformer.ir


def rvsdg_to_ir(
    func_id: bytecode.FunctionIdentity,
    rvsdg: rvsdgir.Region,
) -> ir.FunctionIR:
    rvsdg2ir = RvsdgIRInterp(func_id)
    rvsdg2ir.run(rvsdg)

    for label, blk in rvsdg2ir.blocks.items():
        # print("label", label)
        # blk.dump()
        blk.verify()

    cfg = ir_utils.compute_cfg_from_blocks(rvsdg2ir.blocks)
    if True:
        defs = ir_utils.build_definitions(rvsdg2ir.blocks)
        fir = ir.FunctionIR(
            blocks=rvsdg2ir.blocks,
            is_generator=False,
            func_id=func_id,
            loc=rvsdg2ir.first_loc,
            definitions=defs,
            arg_count=len(func_id.arg_names),  # type: ignore
            arg_names=func_id.arg_names,  # type: ignore
        )
        _debug_dot("initial function IR", fir.render_dot())

    if len(cfg.dead_nodes()) > 0:
        raise Exception("has dead blocks")

    ir_utils.merge_adjacent_blocks(rvsdg2ir.blocks)
    rvsdg2ir.blocks = ir_utils.rename_labels(rvsdg2ir.blocks)
    if True:
        defs = ir_utils.build_definitions(rvsdg2ir.blocks)
        fir = ir.FunctionIR(
            blocks=rvsdg2ir.blocks,
            is_generator=False,
            func_id=func_id,
            loc=rvsdg2ir.first_loc,
            definitions=defs,
            arg_count=len(func_id.arg_names),  # type: ignore
            arg_names=func_id.arg_names,  # type: ignore
        )
        _debug_dot("simplified CFG function IR", fir.render_dot())

    from .bcinterp import _simplify_assignments
    _simplify_assignments(rvsdg2ir.blocks)
    if True:
        defs = ir_utils.build_definitions(rvsdg2ir.blocks)
        fir = ir.FunctionIR(
            blocks=rvsdg2ir.blocks,
            is_generator=False,
            func_id=func_id,
            loc=rvsdg2ir.first_loc,
            definitions=defs,
            arg_count=len(func_id.arg_names),  # type: ignore
            arg_names=func_id.arg_names,  # type: ignore
        )
        _debug_dot("simplified assignments function IR", fir.render_dot())

    # fir.dump()
    return fir


def render_rvsdgir(ir: rvsdgir.Region, name: str):
    from .rvsdg.regionrenderer import (
        GraphBacking,
        GraphNodeMaker,
        GraphvizRendererBackend,
        graph_debugger,
    )

    g = GraphBacking()
    maker = GraphNodeMaker(parent_path=()).subgroup("regionouter")
    render_rvsdgir_region(g, maker, ir)
    g.verify()
    rgr = GraphvizRendererBackend()
    g.render(rgr)
    with graph_debugger() as dbg:
        dbg.add_graphviz(name, rgr.digraph)


def render_rvsdgir_region(g, maker, ir: rvsdgir.Region):
    from .rvsdg.regionrenderer import GraphBacking, GraphNodeMaker, GraphEdge

    assert isinstance(g, GraphBacking)
    assert isinstance(maker, GraphNodeMaker)

    def ident(ref) -> str:
        return str(id(ref))

    args_name = "outputs" + ident(ir._ref)
    results_name = "inputs" + ident(ir._ref)

    g.add_node(
        ident(ir._ref),
        maker.make_node(
            kind="op", data=dict(body=f"{ir.attrs.prettyformat()}")
        ),
    )

    prefix = {"rvsdg.loop": "loop_", "rvsdg.switch": "switch_"}.get(
        ir.opname, "region"
    )
    maker = maker.subgroup(prefix + ident(ir._ref))
    g.add_node(
        args_name,
        maker.make_node(
            kind="ports", ports=tuple(ir.args), data=dict(body=f"args")
        ),
    )
    g.add_node(
        results_name,
        maker.make_node(
            kind="ports", ports=tuple(ir.results), data=dict(body=f"results")
        ),
    )
    is_node_ports = set()

    for op in ir.body.toposorted_ops():
        if isinstance(op, rvsdgir.RegionOp):
            submaker = maker.subgroup("regionouter" + ident(op._ref))
            g.add_node(
                "inputs" + ident(op._ref),
                submaker.make_node(
                    kind="ports", ports=tuple(op.ins), data=dict(body=f"inputs")
                ),
            )
            render_rvsdgir_region(g, submaker, op.subregion)
            g.add_node(
                "outputs" + ident(op._ref),
                submaker.make_node(
                    kind="ports",
                    ports=tuple(op.outs),
                    data=dict(body=f"outputs"),
                ),
            )
            # connect args
            for k in op.ins:
                g.add_edge(
                    src="inputs" + ident(op._ref),
                    dst="outputs" + ident(op.subregion._ref),
                    src_port=k,
                    dst_port=k,
                )
            # connect results
            for k in op.outs:
                g.add_edge(
                    src="inputs" + ident(op.subregion._ref),
                    dst="outputs" + ident(op._ref),
                    src_port=k,
                    dst_port=k,
                )

        else:
            g.add_node(
                ident(op._ref),
                maker.make_node(
                    kind="ported_op",
                    ports=[f"in_{p}" for p in op.ins]
                    + [f"out_{p}" for p in op.outs],
                    data=dict(body=f"{op.attrs.prettyformat()}"),
                ),
            )
            is_node_ports.update(op.ins.list_ports())
            is_node_ports.update(op.outs.list_ports())

    for edge in ir._storage.iter_edges():
        if edge.source in is_node_ports:
            src = ident(edge.source.ref)
            src_port = f"out_{edge.source.portname}"
        else:
            src = "outputs" + ident(edge.source.ref)
            src_port = edge.source.portname
        if edge.target in is_node_ports:
            dst = ident(edge.target.ref)
            dst_port = f"in_{edge.target.portname}"
        else:
            dst = "inputs" + ident(edge.target.ref)
            dst_port = edge.target.portname
        g.add_edge(
            src=src,
            dst=dst,
            src_port=src_port,
            dst_port=dst_port,
        )

    return g


T = TypeVar("T")


def _expect_type(cls: Type[T]):
    def wrap(obj: Any) -> T:
        assert isinstance(obj, cls)
        return obj

    return wrap


def _pretty_bytecode(inst: dis.Instruction) -> str:
    return f"{inst.offset}:{inst.opname}({inst.argval})"


@dataclass(frozen=True)
class PyAttrs:
    bcinst: dis.Instruction

    def __str__(self):
        return f"[{_pretty_bytecode(self.bcinst)}]"



@dataclass(frozen=True)
class PyBinOpAttrs(PyAttrs):
    binop: str

    def __str__(self):
        return f"[{_pretty_bytecode(self.bcinst)}, {self.binop!r}]"



@dataclass(frozen=True)
class PyVarAttrs(PyAttrs):
    varname: str

    def __str__(self):
        return f"[{_pretty_bytecode(self.bcinst)} {self.varname!r}]"



@dataclass(frozen=True)
class PyStoreAttrs(PyVarAttrs):
    pass


@dataclass(frozen=True)
class PyLoadAttrs(PyVarAttrs):
    pass


@dataclass(frozen=True)
class _ToRvsdgIR_Data:
    stack: tuple[str, ...]
    varmap: dict[str, rvsdgPort]
    region: rvsdgir.Region

    def __post_init__(self):
        assert isinstance(self.stack, tuple), "stack must be a tuple"
        # Verify that the stack and the varmap is consistent
        for k in self.stack:
            assert k in self.varmap

    def replace(self, **kwargs) -> "_ToRvsdgIR_Data":
        return _dataclass_replace(self, **kwargs)

    def imported(self) -> "_ToRvsdgIR_Data":
        def repl(k: str) -> str:
            prefix = "export_"
            if k.startswith(prefix):
                return f"import_{k[len(prefix):]}"
            return k

        varmap = {repl(name): port for name, port in self.varmap.items()}
        stack = tuple(repl(name) for name in self.stack)
        return self.replace(varmap=varmap, stack=stack)

    def exported(self) -> "_ToRvsdgIR_Data":
        def repl(k: str) -> str:
            prefix = "import_"
            if k.startswith(prefix):
                return f"export_{k[len(prefix):]}"
            return k

        varmap = {repl(name): port for name, port in self.varmap.items()}
        stack = tuple(repl(name) for name in self.stack)
        return self.replace(varmap=varmap, stack=stack)

    def nest(self, region_opname, fn, **kwargs) -> "_ToRvsdg_Data":
        imported = self.imported()
        region_op = self.region.add_subregion(
            opname=region_opname, ins=imported.varmap.keys(), outs=(), **kwargs
        )
        region_op.ins(**imported.varmap)

        subregion = region_op.subregion
        inner_data = imported.replace(
            region=subregion, varmap=dict(**subregion.args)
        )

        inner_data = fn(inner_data)

        for k, v in inner_data.varmap.items():
            subregion.results.add_port(k)
            subregion.results.connect(k, v)

        out_varmap = {k: region_op.outs[k] for k in inner_data.varmap}
        return self.replace(stack=inner_data.stack, varmap=dict(**out_varmap))


class ToRvsdgIR(RegionVisitor[_ToRvsdgIR_Data]):
    bcmap: dict[int, dis.Instruction]

    @classmethod
    def run(cls, scfg, bcmap, argnames):
        inst = cls(bcmap, argnames)
        varmap = {"env": inst.ir.args["env"]}
        for k in argnames:
            varmap[_decorate_varname(k)] = inst.ir.args[f"arg_{k}"]
        data = _ToRvsdgIR_Data(stack=(), varmap=varmap, region=inst.ir)
        data = inst.visit_graph(scfg, data=data)

        # Connect output
        inst.ir.results(**data.varmap)
        return inst

    def __init__(self, bcmap, argnames):
        super().__init__()
        self.bcmap = bcmap
        self.ir = rvsdgir.Region.make(
            opname="function",
            ins=("env", *[f"arg_{k}" for k in argnames]),
            outs=("env", "return_value"),
        )
        self._backedge_label = 0
        self._switch_label = 0

        # XXX: THIS IS NEEDED BECAUSE numba-rvsdg is not providing a cpvar name
        self._switch_cp_stack = []

    def visit_block(
        self, block: BasicBlock, data: _ToRvsdgIR_Data
    ) -> _ToRvsdgIR_Data:
        from .rvsdg.bc2rvsdg import ExtraBasicBlock

        if isinstance(block, PythonBytecodeBlock):
            instlist = block.get_instructions(self.bcmap)

            imported = data.imported()
            bctorvsdg, stack, varmap = BcToRvsdgIR.run(
                block.name,
                data.region,
                imported.stack,
                imported.varmap,
                instlist,
                switch_cp_stack=self._switch_cp_stack,
            )
            return _ToRvsdgIR_Data(
                stack=tuple(stack), varmap=varmap, region=data.region
            )

        elif isinstance(block, (SyntheticFill, SyntheticTail)):
            # no-op
            return data.exported()

        elif isinstance(block, SyntheticAssignment):
            # Add and export control variables
            cur_varmap = data.varmap.copy()
            for k, v in block.variable_assignment.items():
                region = data.region
                op = region.add_simple_op(
                    "rvsdg.cpvar", ins=(), outs=["cp"], attrs={"cpval": int(v)}
                )
                cur_varmap[k] = op.outs.cp
            return data.replace(varmap=cur_varmap).exported()
        elif isinstance(block, SyntheticBranch):
            cur_varmap = data.varmap.copy()
            region = data.region
            # Remove lifetime of the CP variable
            cpvar = cur_varmap.pop(block.variable)
            if block.variable.startswith("backedge"):
                def _handle_backedge(block, region, cpvar):
                    # Assume CP in {0, 1}
                    bvt = set(block.branch_value_table)
                    assert bvt == {0, 1}, bvt
                    [backedge] = block.backedges
                    if block.branch_value_table[0] == backedge:
                        # active lo
                        negate = False
                    else:
                        # active hi
                        negate = True

                    # Search backward for the parent loop
                    parent = region
                    while parent.opname != "rvsdg.loop":
                        parent = parent.get_parent()
                    cp = parent.attrs.extras["cp"]
                    parent.attrs.extras["scfg_bvt"] = block.branch_value_table

                    if negate:
                        negop = region.add_simple_op(
                                    "rvsdg.negate",
                                    ins=["val"],
                                    outs=["out"],
                                )
                        negop.ins(val=cpvar)
                        cpvar = negop.outs.out
                    return cpvar, cp

                cpvar, cp = _handle_backedge(block, region, cpvar)
            else:
                # Handle normal switch
                def _handle_switch(block, region, cpvar):
                    cp = block.variable
                    self._switch_cp_stack[-1] = block.variable
                    return cpvar, cp
                cpvar, cp = _handle_switch(block, region, cpvar)

            op = region.add_simple_op(
                "rvsdg.setcpvar",
                ins=["env", "cp"],
                outs=["env"],
                attrs={"cp": cp,
                       "scfg_bvt": block.branch_value_table},
            )
            op.ins(env=cur_varmap["env"], cp=cpvar)
            cur_varmap["env"] = op.outs.env

            return data.replace(varmap=cur_varmap)

        elif isinstance(block, ExtraBasicBlock):
            stack = list(data.stack)
            varmap = data.varmap.copy()
            for opname in block.inst_list:
                if opname == "POP":
                    del varmap[stack.pop()]
                elif opname == "FOR_ITER_STORE_INDVAR":
                    # push "indvar" variable into top of stack
                    stkname = f"import_{len(stack)}"
                    stack.append(stkname)
                    varmap[stkname] = varmap.pop("indvar")
                else:
                    raise NotImplementedError(opname)
            return data.replace(stack=tuple(stack), varmap=varmap).exported()

        # otherwise
        raise NotImplementedError(type(block))

    def visit_loop(
        self, region: RegionBlock, data: _ToRvsdgIR_Data
    ) -> _ToRvsdgIR_Data:
        assert isinstance(region, RegionBlock) and region.kind == "loop"

        def _emit_loop(data):
            return self.visit_linear(region, data)

        return data.nest(
            "rvsdg.loop", _emit_loop, attrs={"cp": self.get_backedge_label()}
        )

    def visit_switch(
        self, region: RegionBlock, data: _ToRvsdgIR_Data
    ) -> _ToRvsdgIR_Data:
        assert isinstance(region, RegionBlock) and region.kind == "switch"

        def _emit_switch_body(inner_data: _ToRvsdgIR_Data) -> _ToRvsdgIR_Data:
            # Emit header
            header = region.header
            header_block = region.subregion[header]
            inner_data = self.visit_linear(header_block, inner_data)
            header_targets = header_block._jump_targets
            # Get BVT
            bvt = None
            for blk in header_block.subregion.graph.values():
                if isinstance(blk, SyntheticBranch):
                    bvt = blk.branch_value_table
            if bvt is None:
                # XXX because numba-rvsdg is not setting it in the FOR_ITER case
                bvt = {i: target for i, target in enumerate(header_targets)}

            # Emit branches
            def _emit_branches(inner_data: _ToRvsdgIR_Data) -> _ToRvsdgIR_Data:
                imported_inner_data = inner_data.imported()
                swt_region_op = imported_inner_data.region.add_subregion(
                    opname="rvsdg.switch",
                    ins=list(imported_inner_data.varmap.keys()),
                    outs=(),
                    attrs={"cp": self._switch_cp_stack[-1],
                           "scfg_name": region.name,
                           "scfg_bvt": bvt},
                )
                swt_region_op.ins(**imported_inner_data.varmap)
                case_data = imported_inner_data.replace(
                    region=swt_region_op.subregion,
                    varmap=dict(**swt_region_op.subregion.args),
                )

                data_foreach_case = []

                branches = [region.subregion.graph[k] for k in header_targets]

                for i, blk in enumerate(branches):
                    assert blk.kind == "branch"

                    def _add_case_block(data):
                        return self.visit_linear(blk, data)

                    data_foreach_case.append(
                        case_data.nest(
                            "rvsdg.case", _add_case_block, attrs={"case": i, "scfg_name": blk.name}
                        )
                    )

                # Merge stack
                merged_nstack = max(
                    len(each.stack) for each in data_foreach_case
                )
                merged_stack = [f"export_{i}" for i in range(merged_nstack)]
                # Merge varmaps
                merging_varmaps = []
                merging_non_stack_vars: set[str] = set()
                for each in data_foreach_case:
                    cur_non_stack = {
                        k: v
                        for k, v in each.varmap.items()
                        if k not in each.stack
                    }
                    merging_varmaps.append(cur_non_stack)
                    merging_non_stack_vars.update(cur_non_stack)
                for k in merging_non_stack_vars:
                    swt_region_op.subregion.results.add_port(k)
                for each_varmap in merging_varmaps:
                    swt_region_op.subregion.results(**each_varmap)
                for k in merged_stack:
                    swt_region_op.subregion.results.add_port(k)
                for each in data_foreach_case:
                    # The reversed() is to align the stack to the right.
                    #      A: export_0, export_1
                    #      B:   null,   export_0
                    # Merged:    0,          1
                    # print("-----")
                    # print('merged', merged_stack)
                    # print('      ', each.stack)
                    # if len(each.stack) != len(merged_stack):
                    #     breakpoint()
                    for k, stk in zip(reversed(merged_stack), reversed(each.stack)):
                        port = each.varmap[stk]
                        swt_region_op.subregion.results.connect(k, port)

                out_varmap = {
                    k: swt_region_op.outs[k] for k in swt_region_op.outs
                }
                return inner_data.replace(
                    stack=tuple(merged_stack), varmap=dict(**out_varmap)
                )

            inner_data = _emit_branches(inner_data)

            exiting = region.exiting
            exiting_block = region.subregion[exiting]
            inner_data = self.visit_linear(exiting_block, inner_data)
            return inner_data

        # setup
        self._switch_cp_stack.append(self.get_switch_label())
        # emit
        out = _emit_switch_body(data)
        # teardown
        self._switch_cp_stack.pop()
        return out

    def get_backedge_label(self) -> str:
        self._backedge_label += 1
        return f"backedge_label_{self._backedge_label}"

    def get_switch_label(self) -> str:
        self._switch_label += 1
        return f"switch_label_{self._switch_label}"


def _decorate_varname(varname: str) -> str:
    return f"var_{varname}"


class BcToRvsdgIR:
    stack: list[rvsdgPort]
    varmap: dict[str, rvsdgPort]
    _kw_names: rvsdgPort | None
    region: rvsdgir.Region
    _switch_cp_stack: Sequence[str]

    @classmethod
    def run(
        cls,
        scfg_name: str,
        parent: rvsdgir.Region,
        stack: Sequence[str],
        varmap: dict[str, rvsdgPort],
        instlist: Sequence[dis.Instruction],
        switch_cp_stack: Sequence[str],
    ):
        inst = cls(scfg_name, parent, stack, varmap, switch_cp_stack)
        for bc in instlist:
            inst.convert(bc)
        # terminate live vars there are imported stack
        dead_inc_stack = []
        for k in stack:
            if k in inst.region.args:
                if inst.region.args[k] == inst.varmap.get(k):
                    dead_inc_stack.append(k)

        # add output ports from live vars
        exported_varmap = {}
        for k, v in inst.varmap.items():
            if k not in dead_inc_stack:
                if k not in inst.region.results:
                    inst.region.results.add_port(k)
                    inst.region.results.connect(k, v)
                exported_varmap[k] = inst.region_op.outs[k]
        # add outputs ports from stack vars
        exported_stack = []
        for i, v in enumerate(inst.stack):
            k = f"export_{i}"
            inst.region.results.add_port(k)
            inst.region.results.connect(k, v)
            exported_stack.append(k)

        return inst, exported_stack, dict(**inst.region_op.outs)

    def __init__(
        self,
        scfg_name: str,
        parent: rvsdgir.Region,
        stack: Sequence[str],
        varmap: Mapping[str, rvsdgPort],
        switch_cp_stack: Sequence[str],
    ):
        self.varmap = {}
        self._kw_names = None
        self._switch_cp_stack = switch_cp_stack
        self.region_op = parent.add_subregion(
            opname="block",
            ins=varmap.keys(),
            outs=(),
            attrs={"scfg_name": scfg_name}
        )
        self.region = self.region_op.subregion
        self.stack = [self.region.args[k] for k in stack]

        # Map live-vars to inputs
        for k in varmap:
            self.varmap[k] = self.region.args[k]
        self.region_op.ins(**varmap)

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

    def store(self, varname: str, value: rvsdgPort):
        self.varmap[varname] = value

    def load(self, varname: str) -> rvsdgPort:
        if varname not in self.varmap:
            vs = self.region.args.add_port(varname)
            self.incoming_vars[varname] = vs
            self.varmap[varname] = vs

        return self.varmap[varname]

    @property
    def effect(self) -> rvsdgPort:
        return self.varmap["env"]

    def replace_effect(self, env: rvsdgPort):
        self.varmap["env"] = env

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

    def _top_switch_cp(self) -> str:
        return self._switch_cp_stack[-1]

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
            f"py.const",
            ins=(),
            outs=["out"],
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        self.push(op.outs.out)

    def op_STORE_FAST(self, inst: dis.Instruction):
        tos = self.pop()
        varname = _decorate_varname(inst.argval)
        op = self.region.add_simple_op(
            opname=f"py.store",
            ins=["env", "val"],
            outs=["env", "out"],
            attrs=dict(
                py=PyStoreAttrs(
                    varname=inst.argval,
                    bcinst=inst,
                )
            ),
        )
        op.ins(env=self.effect, val=tos)
        self.replace_effect(op.outs.env)
        self.store(varname, op.outs.out)

    def op_LOAD_FAST(self, inst: dis.Instruction):
        varname = _decorate_varname(inst.argval)
        op = self.region.add_simple_op(
            "py.load",
            ins=("env", "val"),
            outs=("env", "out"),
            attrs=dict(
                py=PyLoadAttrs(
                    varname=inst.argval,
                    bcinst=inst,
                )
            )
        )
        op.ins(val=self.load(varname), env=self.effect)
        self.replace_effect(op.outs.env)
        self.push(op.outs.out)

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
        op = self.region.add_simple_op(
            opname="py.foriter",
            ins=["env", "iter"],
            outs=["env", "indvar", "itervalid"],
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        op.ins(env=self.effect, iter=tos)

        self.store("indvar", op.outs.indvar)

        negop = self.region.add_simple_op(
            "rvsdg.negate",
            ins=["val"],
            outs=["out"],
        )
        negop.ins(val=op.outs.itervalid)
        cpvar = negop.outs.out

        setcpop = self.region.add_simple_op(
            "rvsdg.setcpvar",
            ins=("env", "cp"),
            outs=["env"],
            attrs={"cp": self._top_switch_cp()},
        )
        setcpop.ins(env=op.outs.env, cp=cpvar)
        self.replace_effect(setcpop.outs.env)

    def _binaryop(self, opname: str, inst: dis.Instruction):
        rhs = self.pop()
        lhs = self.pop()
        binop = dis._nb_ops[inst.argval][1]
        op = self.region.add_simple_op(
            opname="py.binop",
            ins=["env", "lhs", "rhs"],
            outs=["env", "out"],
            attrs=dict(py=PyBinOpAttrs(bcinst=inst, binop=binop)),
        )
        op.ins(env=self.effect, lhs=lhs, rhs=rhs)
        self.replace_effect(op.outs.env)
        self.push(op.outs.out)

    def op_BINARY_OP(self, inst: dis.Instruction):
        self._binaryop("binaryop", inst)

    def op_COMPARE_OP(self, inst: dis.Instruction):
        rhs = self.pop()
        lhs = self.pop()
        binop = inst.argval
        op = self.region.add_simple_op(
            opname="py.cmpop",
            ins=["env", "lhs", "rhs"],
            outs=["env", "out"],
            attrs=dict(py=PyBinOpAttrs(bcinst=inst, binop=binop)),
        )
        op.ins(env=self.effect, lhs=lhs, rhs=rhs)
        self.replace_effect(op.outs.env)
        self.push(op.outs.out)

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
        self.replace_effect(op.outs.env)
        env = self.varmap["env"]
        self.varmap.clear()
        self.stack.clear()
        self.varmap["env"] = env
        self.varmap["return_value"] = op.outs.res

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

    def _jump_if(self, pred: rvsdgPort):
        cp = self._switch_cp_stack[-1]
        cpop = self.region.add_simple_op(
            "rvsdg.setcpvar",
            ins=["env", "cp"],
            outs=["env"],
            attrs=dict(cp=cp),
        )
        cpop.ins(env=self.effect, cp=pred)
        self.replace_effect(cpop.outs.env)

    def op_POP_JUMP_FORWARD_IF_TRUE(self, inst: dis.Instruction):
        predop = self.region.add_simple_op(
            "py.predicate",
            ins=["val"],
            outs=["out"],
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        predop.ins(val=self.pop())
        self._jump_if(predop.outs.out)

    def op_POP_JUMP_FORWARD_IF_FALSE(self, inst: dis.Instruction):
        predop = self.region.add_simple_op(
            "py.not",
            ins=["val"],
            outs=["out"],
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        predop.ins(val=self.pop())
        self._jump_if(predop.outs.out)

    op_POP_JUMP_BACKWARD_IF_TRUE = op_POP_JUMP_FORWARD_IF_TRUE
    op_POP_JUMP_BACKWARD_IF_FALSE = op_POP_JUMP_FORWARD_IF_FALSE

    # def op_POP_JUMP_FORWARD_IF_NONE(self, inst: dis.Instruction):
    #     self._POP_JUMP_X_IF_Y(inst, opname="py.jump.if_none")

    # def op_POP_JUMP_FORWARD_IF_NOT_NONE(self, inst: dis.Instruction):
    #     self._POP_JUMP_X_IF_Y(inst, opname="py.jump.if_not_none")

    def op_JUMP_IF_TRUE_OR_POP(self, inst: dis.Instruction):
        tos = self.top()
        predop = self.region.add_simple_op(
            "py.predicate",
            ins=["val"],
            outs=["out"],
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        predop.ins(val=tos)
        self._jump_if(predop.outs.out)

    def op_JUMP_IF_FALSE_OR_POP(self, inst: dis.Instruction):
        tos = self.top()
        predop = self.region.add_simple_op(
            "py.not",
            ins=["val"],
            outs=["out"],
            attrs=dict(py=PyAttrs(bcinst=inst)),
        )
        predop.ins(val=tos)
        self._jump_if(predop.outs.out)

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


class BaseInterp:
    blocks: dict[int, ir.Block]
    func_id: bytecode.FunctionIdentity
    local_scope: ir.Scope
    global_scope: ir.Scope
    _portdata: dict[rvsdgPort, ir.Var]
    _current_block: ir.Block | None
    branch_predicate: ir.Var | None
    _label_map: dict[str, int]
    _emit_debug_print = False
    _region_stack: list[rvsdgir.Region]
    _cpmap: dict[str, ir.Var]
    _region_blockmap: dict[rvsdgir.Region, int]

    # _ret_name = ".retval"

    def __init__(self, func_id):
        self.func_id = func_id
        self.loc = self.first_loc = ir.Loc.from_function_id(func_id)
        self.global_scope = ir.Scope(parent=None, loc=self.loc)
        self.local_scope = ir.Scope(parent=self.global_scope, loc=self.loc)
        self.blocks = {}
        self._portdata = {}
        self._current_block = None
        self._label_map = {}
        self._region_stack = []
        self._cpmap = {}
        self._region_blockmap = {}

    def get_global_value(self, name):
        """THIS IS COPIED from interpreter.py

        Get a global value from the func_global (first) or
        as a builtins (second).  If both failed, return a ir.UNDEFINED.
        """
        try:
            return self.func_id.func.__globals__[name]
        except KeyError:
            return getattr(builtins, name, ir.UNDEFINED)

    def store_var(self, val, port: rvsdgPort, varname: str) -> ir.Var:
        # Set redefine=False to match current non-SSA behavior
        value = self.store(val, varname, redefine=False)
        self.write_port(self._region, port, value)
        return value

    def store_port(self, val, port: rvsdgPort) -> ir.Var:
        varname = f"${port.portname}"
        value = self.store(val, varname)
        self.write_port(self._region, port, value)
        return value

    def store_phi_port(
        self, block_prefix: str, val: ir.Var, port: rvsdgPort
    ) -> ir.Var:
        value = self.store_phi(block_prefix, val, port)
        self.write_port(self._region, port, value)
        return value

    def store_phi(
        self, block_prefix: str, val: ir.Var, port: rvsdgPort
    ) -> ir.Var:
        return self.store(
            val, f"$phi_{block_prefix}_{port.portname}", redefine=False
        )

    def store(self, value, name, *, redefine=True, block=None) -> ir.Var:
        target: ir.Var
        # The following `if` is to reduce the number of assignments
        if isinstance(value, ir.Var) and redefine and name.startswith("$"):
            return value
        if redefine:
            target = self.local_scope.redefine(name, loc=self.loc)
        else:
            target = self.local_scope.get_or_define(name, loc=self.loc)
        stmt = ir.Assign(value=value, target=target, loc=self.loc)
        self.append(stmt, block=block)
        return target

    def write_port(
        self, region: rvsdgir.Region, port: rvsdgPort, value: ir.Var,
        *, dependents=True,
    ):
        self._portdata[port] = value
        # also store into aliases
        if dependents:
            for dep in region.iter_port_dependents(port):
                self._portdata[dep] = value

    def read_port(self, port: rvsdgPort) -> ir.Var:
        return self._portdata[port]

    def append(self, stmt: ir.Stmt, block=None):
        if block is None:
            block = self.current_block
        if block.is_terminated:
            block.insert_before_terminator(stmt)
        else:
            block.append(stmt)

    def _get_temp_label(self) -> int:
        num = len(self._label_map)
        assert num not in self._label_map
        self._label_map[f"annoy.{num}"] = num
        return num

    @contextmanager
    def set_block(self, label: int) -> Iterator[ir.Block]:
        """A context manager that set the current block for other IR building
        methods.
        """
        block: ir.Block
        block = self.blocks[label]
        assert not block.is_terminated
        old = self._current_block
        self._current_block = block
        try:
            yield block
        finally:
            self._current_block = old

    @property
    def current_block(self) -> ir.Block:
        out = self._current_block
        assert out is not None
        return out

    @property
    def _region(self) -> rvsdgir.Region:
        return self._region_stack[-1]

    @contextmanager
    def _push_region(self, region: rvsdgir.Region):
        self._region_stack.append(region)
        try:
            yield
        finally:
            self._region_stack.pop()


class PyOpHandler(BaseInterp):
    def py_null(self, op: rvsdgir.SimpleOp, attrs: PyAttrs):
        pass

    def py_global_load(self, op: rvsdgir.SimpleOp, attrs: PyAttrs):
        gvname = attrs.bcinst.argval
        value = self.get_global_value(gvname)
        # TODO: handle non scalar
        const = ir.Global(gvname, value, loc=self.loc)
        self.store_port(const, op.outs.out)

    def py_call(self, op: rvsdgir.SimpleOp, attrs: PyAttrs):
        [_env, callee_or_null, arg0_or_callee, *args] = op.ins.values()

        producer = self._region.get_producer(callee_or_null)
        if producer.opname == "py.null":
            callee = arg0_or_callee
        else:
            callee = callee_or_null
            args = (arg0_or_callee, *args)

        calleevars = self.read_port(callee)
        argvars = [self.read_port(p) for p in args]
        kwargs = ()

        expr = ir.Expr.call(calleevars, argvars, kwargs, loc=self.loc)
        self.store_port(expr, op.outs.out)

    def _py_intr_unary(
        self,
        op: rvsdgir.SimpleOp,
        attrs: PyAttrs,
        cb: Callable[[ir.Var], ir.Expr],
    ):
        self.store_port(cb(self.read_port(op.ins.val)), op.outs.out)

    def py_getiter(self, op: rvsdgir.SimpleOp, attrs: PyAttrs):
        self._py_intr_unary(
            op, attrs, lambda v: ir.Expr.getiter(value=v, loc=self.loc)
        )

    def py_foriter(self, op: rvsdgir.SimpleOp, attrs: PyAttrs):
        pairval = ir.Expr.iternext(
            value=self.read_port(op.ins.iter), loc=self.loc
        )
        pair = self.store(pairval, "$foriter")

        iternext = ir.Expr.pair_first(value=pair, loc=self.loc)
        indval = self.store(iternext, "$foriter.indvar")
        self.store_port(indval, op.outs.indvar)

        isvalid = ir.Expr.pair_second(value=pair, loc=self.loc)
        self.store_port(isvalid, op.outs.itervalid)

    def py_load(self, op: rvsdgir.SimpleOp, attrs: PyLoadAttrs):
        # TODO: insert metadata for user debugging
        # Otherwise, this is just a passthrough
        self.store_port(self.read_port(op.ins.val), op.outs.out)

    def py_store(self, op: rvsdgir.SimpleOp, attrs: PyStoreAttrs):
        # TODO: insert metadata for user debugging
        # Otherwise, this is just a passthrough
        self.store_var(self.read_port(op.ins.val), op.outs.out,
                       varname=attrs.varname)

    def py_const(self, op: rvsdgir.SimpleOp, attrs: PyAttrs):
        const = ir.Const(value=attrs.bcinst.argval, loc=self.loc)
        self.store_port(const, op.outs.out)

    def py_return(self, op: rvsdgir.SimpleOp, attrs: PyAttrs):
        return_value = self.read_port(op.ins.val)
        self.store_port(return_value, op.outs.res)

    def py_binop(self, op: rvsdgir.SimpleOp, attrs: PyBinOpAttrs):
        binop = attrs.binop
        if "=" in binop:
            self._inplace_binop(binop[:-1], op)
        else:
            self._binop(binop, op)

    def _inplace_binop(self, binop: str, op: rvsdgir.SimpleOp):
        fn_immuop = BINOPS_TO_OPERATORS[binop]
        fn_op = INPLACE_BINOPS_TO_OPERATORS[binop + '=']
        lhs = self.read_port(op.ins.lhs)
        rhs = self.read_port(op.ins.rhs)
        expr = ir.Expr.inplace_binop(fn_op, fn_immuop, lhs, rhs, loc=self.loc)
        self.store_port(expr, op.outs.out)

    def _binop(self, binop: str, op: rvsdgir.SimpleOp):
        fn_op = BINOPS_TO_OPERATORS[binop]
        lhs = self.read_port(op.ins.lhs)
        rhs = self.read_port(op.ins.rhs)
        expr = ir.Expr.binop(fn_op, lhs, rhs, loc=self.loc)
        self.store_port(expr, op.outs.out)

    def py_cmpop(self, op: rvsdgir.SimpleOp, attrs: PyBinOpAttrs):
        cmpop = attrs.binop
        self._binop(cmpop, op)

    def py_predicate(self, op: rvsdgir.SimpleOp, attrs: PyAttrs):
        val = self.read_port(op.ins.val)
        boolfn = ir.Global("bool", bool, loc=self.loc)
        boolvar = self.store(boolfn, "$bool")
        pred = ir.Expr.call(boolvar, (val,), (), loc=self.loc)
        self.store_port(pred, op.outs.out)

    def py_not(self, op: rvsdgir.SimpleOp, attrs: PyAttrs):
        val = self.read_port(op.ins.val)
        notfn = ir.Const(operator.not_, loc=self.loc)
        notvar = self.store(notfn, "$not")
        pred = ir.Expr.call(notvar, (val,), (), loc=self.loc)
        self.store_port(pred, op.outs.out)


class RvsdgOpHandler(BaseInterp):
    def rvsdg_cpvar(self, op: rvsdgir.SimpleOp):
        cpval = ir.Const(value=op.attrs.extras["cpval"], loc=self.loc)
        self.store_port(cpval, op.outs.cp)

    def rvsdg_setcpvar(self, op: rvsdgir.SimpleOp):
        cpname = op.attrs.extras["cp"]
        cpval = self.read_port(op.ins.cp)
        self._cpmap[cpname] = cpval

    def rvsdg_negate(self, op: rvsdgir.SimpleOp):
        val = self.read_port(op.ins.val)
        not_fn = ir.Const(operator.not_, loc=self.loc)
        res = ir.Expr.call(self.store(not_fn, "$not"), (val,), (), loc=self.loc)
        self.store_port(res, op.outs.out)


class RvsdgIRInterp(PyOpHandler, RvsdgOpHandler):
    def run(self, region: rvsdgir.Region):
        assert region.opname == "function"

        # Prepare ir.Blocks into self.blocks
        def prepare_block(region: rvsdgir.Region):
            """Recursively assign ir.Blocks to each region"""
            label = self._region_blockmap[region] = self._get_temp_label()
            self.blocks[label] = ir.Block(scope=self.local_scope, loc=self.loc)
            for op in region.body.toposorted_ops():
                if isinstance(op, rvsdgir.RegionOp):
                    subregion = op.subregion
                    prepare_block(subregion)
                elif isinstance(op, rvsdgir.SimpleOp) and op.opname.startswith(
                    "py."
                ):
                    self._set_loc_from_py_op(op)

        prepare_block(region)

        # Emit instructions
        with self._push_region(region):
            with self.set_block(self._region_blockmap[region]):
                for i, k in enumerate(self.func_id.arg_names):  # type: ignore
                    val = ir.Arg(index=i, name=k, loc=self.loc)
                    port = region.args[f"arg_{k}"]
                    self.store_port(val, port)

            last_label = self.emit_linear_region(region)

        with self.set_block(last_label):
            retval = self.read_port(region.results.return_value)
            self.append(ir.Return(retval, loc=self.loc))

    def emit_region(self, region: rvsdgir.Region) -> int:
        """
        Behavior:
        - Emit to the starting block for the region.
        """
        with self._push_region(region):
            if region.opname == "rvsdg.case":
                return self.emit_linear_region(region)
            elif region.opname in {"block", "rvsdg.loop"}:
                label = self.emit_linear_region(region)
                assert not self.blocks[label].is_terminated
                if region.opname == "rvsdg.loop":
                    with self.set_block(label):
                        # make phi nodes connection from the end of the loop
                        prefix = f"loop_{self._region_blockmap[region]}"
                        for k, resport in region.results.items():
                            if not _is_env(resport):
                                k = (
                                    "import_" + k.split("_", 1)[1]
                                    if k.startswith("export_")
                                    else k
                                )
                                if k in region.args:  # is looping?
                                    value = self.read_port(resport)
                                    argport = region.args[k]
                                    self.store_phi(prefix, value, argport)
                    # Wire up the tail jump for the loop
                    cpvar = region.attrs.extras["cp"]
                    pred = self._cpmap[cpvar]

                    succ_label = self._inject_internal_block()
                    self.append(
                        ir.Branch(
                            pred,
                            self._region_blockmap[region],
                            succ_label,
                            loc=self.loc,
                        ),
                        self.blocks[label],
                    )

                    return succ_label
                else:
                    return label
            elif region.opname == "rvsdg.switch":
                # The body of the switch must contains "case" regions
                def prep_cases():
                    cases = list(
                        map(
                            _expect_type(rvsdgir.RegionOp),
                            region.body.toposorted_ops(),
                        )
                    )
                    case_map = {case.attrs.extras["scfg_name"]: case
                                for case in cases}
                    bvt = region.attrs.extras["scfg_bvt"]

                    cp_case_map = {k: case_map[v] for k, v in sorted(bvt.items())}
                    assert all(case.opname == "rvsdg.case" for case in cp_case_map.values())
                    return cp_case_map

                cp_case_map = prep_cases()
                if len(cp_case_map) == 2:
                    assert tuple(cp_case_map.keys()) == (0, 1)
                    # Emit branch to the cases
                    cpval = self._cpmap[region.attrs.extras["cp"]]
                    label1 = self._region_blockmap[cp_case_map[1].subregion]
                    label0 = self._region_blockmap[cp_case_map[0].subregion]
                    br = ir.Branch(
                        cpval, truebr=label1, falsebr=label0, loc=self.loc
                    )
                    swt_label = self._region_blockmap[region]
                    self.append(br, self.blocks[swt_label])
                else:
                    # Emit jump tree for more than 2 targets
                    assert len(cp_case_map) > 2
                    labels = [(k, self._region_blockmap[v.subregion]) for k, v in cp_case_map.items()]

                    blocks = []
                    # For all but last labels
                    for _ in range(len(labels) - 1):
                        blocks.append(self._inject_internal_block())

                    # Jump into the first block
                    jmp = ir.Jump(blocks[-1], loc=self.loc)
                    swt_label = self._region_blockmap[region]
                    self.append(jmp, self.blocks[swt_label])

                    cpval = self._cpmap[region.attrs.extras["cp"]]
                    # Handle jump tree
                    while blocks:
                        cp_expect, cp_label = labels.pop()
                        cur_label = blocks.pop()
                        with self.set_block(cur_label):
                            const = self.store(ir.Const(cp_expect, loc=self.loc), "$.const")
                            cmp = ir.Expr.binop(operator.eq, const, cpval, loc=self.loc)
                            pred = self.store(cmp, "$.cmp")

                            if not blocks:
                                _, falsebr = labels.pop()
                            else:
                                falsebr = blocks[-1]
                            br = ir.Branch(
                                cond=pred,
                                truebr=cp_label,
                                falsebr=falsebr,
                                loc=self.loc,
                            )
                            self.append(br)
                # Prepare successor
                succ_label = self._inject_internal_block()
                # Emit each case as linear regions
                done = set()
                for case in cp_case_map.values():
                    if case in done:
                        continue
                    done.add(case)
                    last_label = self._emit_region_call(
                        swt_label, region, case, needs_jump=False
                    )
                    self.append(
                        ir.Jump(succ_label, loc=self.loc),
                        self.blocks[last_label],
                    )
                return succ_label
            else:
                raise NotImplementedError(
                    f"unknown region.opname == {region.opname!r}"
                )

    def emit_linear_region(self, region: rvsdgir.Region) -> int:
        """
        Returns label of the last block, which is not terminated.

        Behavior:
        - Emit to the starting block for the region.
        """
        label = self._region_blockmap[region]
        body = region.body.toposorted_ops()
        # Emit debug print for region.args
        if self._emit_debug_print:
            with self.set_block(label):
                self.debug_print(f"Enter {region.attrs.prettyformat()}")
                for k, v in region.args.items():
                    if k != "env":
                        self.debug_print(f"  {k}:", self.read_port(v))
        # Emit region body
        for op in body:
            if isinstance(op, rvsdgir.RegionOp):
                label = self._emit_region_call(label, region, op)
            elif op.opname.startswith("py."):
                # python operations
                with self.set_block(label):
                    self.emit_py_op(op)
            elif op.opname.startswith("rvsdg."):
                with self.set_block(label):
                    self.emit_rvsdg_op(op)
            else:
                raise NotImplementedError(op)

        # Emit debug print for region.results
        if self._emit_debug_print:
            with self.set_block(label):
                self.debug_print(f"Exit {region.attrs.prettyformat()}")
                for k, v in region.results.items():
                    if k != "env":
                        self.debug_print(f"  {k}:", self.read_port(v))
        return label

    def emit_py_op(self, op):
        fname = op.opname.replace(".", "_")
        fn = getattr(self, fname, None)
        if fn is None:
            raise NotImplementedError(
                f"{fname!r} not defined to emit {op.attrs.prettyformat()}"
            )
        else:
            self._set_loc_from_py_op(op)
            with _exc_note(lambda: f"failed in {fn}; source: {self.loc}"):
                fn(op, op.attrs.extras["py"])

    def emit_rvsdg_op(self, op):
        fname = op.opname.replace(".", "_")
        fn = getattr(self, fname, None)
        if fn is None:
            raise NotImplementedError(
                f"{fname!r} not defined to emit {op.attrs.prettyformat()}"
            )
        else:
            with _exc_note(lambda: f"failed in {fn}; source: {self.loc}"):
                fn(op)

    def _set_loc_from_py_op(self, op: rvsdgir.SimpleOp):
        assert op.opname.startswith("py.")
        pyattrs = op.attrs.extras["py"]
        assert isinstance(pyattrs, PyAttrs)
        pos = pyattrs.bcinst.positions
        if pos is not None:
            self.loc = self.loc.with_lineno(pos.lineno, pos.col_offset)

    def _inject_internal_block(self) -> int:
        label = self._get_temp_label()
        self.blocks[label] = ir.Block(self.local_scope, self.loc)
        return label

    def _emit_region_call(
        self,
        label: int,
        region: rvsdgir.Region,
        op: rvsdgir.RegionOp,
        *,
        needs_jump=True,
    ) -> int:
        if op.subregion.opname != "rvsdg.loop":
            # map ins to args
            for port, argport in zip(
                op.ins.values(), op.subregion.args.values(), strict=True
            ):
                if not _is_env(port):
                    value = self.read_port(port)
                    self.write_port(op.subregion, argport, value)
        else:
            with self.set_block(label):
                # map ins to args
                prefix = f"loop_{self._region_blockmap[op.subregion]}"
                for port, argport in zip(
                    op.ins.values(), op.subregion.args.values(), strict=True
                ):
                    if not _is_env(port):
                        value = self.read_port(port)
                        var = self.store_phi_port(prefix, value, argport)
                        self.write_port(op.subregion, argport, var)

        if needs_jump:
            with self.set_block(label):
                self.append(
                    ir.Jump(self._region_blockmap[op.subregion], loc=self.loc)
                )

        label = self.emit_region(op.subregion)

        # map results to outs
        for resport, port in zip(
            op.subregion.results.values(), op.outs.values(), strict=True
        ):
            if not _is_env(port):
                value = self.read_port(resport)
                self.write_port(region, port, value)

        if op.opname == "rvsdg.switch":
            # Add debug print at end of a switch
            if self._emit_debug_print:
                # Debug print the output of the switch
                with self.set_block(label):
                    self.debug_print(f"Switch output {op.attrs.prettyformat()}")
                    for k, v in op.outs.items():
                        if k != "env":
                            self.debug_print(f"  {k}:", self.read_port(v))

        # handle closing of a case
        if region.opname == "rvsdg.switch":
            # make phi nodes that merge the results from all cases
            assert op.opname == "rvsdg.case", op.opname
            post_case_label = label
            with self.set_block(post_case_label):
                prefix = "switch_" + region.attrs.extras['scfg_name']
                edges = list(region.iter_edges().filter_by_source({*op.outs.values()}))
                for edge in edges:
                    if edge.target.portname != "env":
                        val = self.read_port(edge.source)
                        var = self.store_phi(prefix, val, edge.target)
                        self.write_port(region, edge.target, var)
            label = post_case_label
        return label

    def debug_print(self, *args):
        vars = []
        for arg in args:
            if isinstance(arg, str):
                const = ir.Const(value=arg, loc=self.loc)
                vars.append(self.store(const, "$debugmsg"))
            else:
                vars.append(arg)
        self.append(ir.Print(tuple(vars), vararg=None, loc=self.loc))


@contextmanager
def _exc_note(make_msg: Callable[[], str]) -> Iterator[None]:
    """Context-manager to add a note to the active exception if one occurs."""
    try:
        yield
    except Exception as e:
        e.add_note(make_msg())
        raise


def _is_env(port: rvsdgPort) -> bool:
    """
    Note: we use a naming convention to identify the State port.
    """
    return port.portname == "env"
