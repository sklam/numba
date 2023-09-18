import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator, Sequence, Mapping, no_type_check
from functools import reduce
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


def _debug_scfg(name, byteflow):
    from .rvsdg.regionrenderer import graph_debugger

    g = ByteFlowRenderer().render_byteflow(byteflow)

    with graph_debugger() as dbg:
        dbg.add_graphviz(name, g)


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

    transformer.ir.prettyprint()
    render_rvsdgir(transformer.ir, "rvsdgir")

    return transformer.ir


def rvsdg_to_ir(
    func_id: bytecode.FunctionIdentity, rvsdg: rvsdgir.Region,
) -> ir.FunctionIR:
    rvsdg2ir = RvsdgIRInterp(func_id)
    rvsdg2ir.run(rvsdg)

    raise NotImplementedError
    for blk in rvsdg2ir.blocks.values():
        blk.verify()

    # rvsdg2ir.blocks = ir_utils.simplify_CFG(rvsdg2ir.blocks)
    cfg = ir_utils.compute_cfg_from_blocks(rvsdg2ir.blocks)
    if len(cfg.dead_nodes()) > 0:
        if DEBUG_GRAPH:
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
            fir.render_dot().view()
        raise Exception("has dead blocks")

    # for dead in cfg.dead_nodes():
    #     del rvsdg2ir.blocks[dead]
    ir_utils.merge_adjacent_blocks(rvsdg2ir.blocks)
    rvsdg2ir.blocks = ir_utils.rename_labels(rvsdg2ir.blocks)
    _simplify_assignments(rvsdg2ir.blocks)
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
    # fir.dump()
    if DEBUG_GRAPH:
        fir.render_dot().view()
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
    rgr = GraphvizRendererBackend()
    g.render(rgr)
    with graph_debugger() as dbg:
        dbg.add_graphviz(name, rgr.digraph)


def render_rvsdgir_region(g, maker, ir: rvsdgir.Region):
    from .rvsdg.regionrenderer import GraphBacking, GraphNodeMaker, GraphEdge

    g: GraphBacking

    def ident(ref) -> str:
        return str(ref)

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
    maker: GraphNodeMaker = maker.subgroup(prefix + ident(ir._ref))
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
            inputs_name = "inputs" + ident(op._ref)
            node_name = ident(op._ref)
            outputs_name = "outputs" + ident(op._ref)

            opmaker = maker.subgroup("box_" + ident(op._ref))
            g.add_node(
                inputs_name,
                opmaker.make_node(
                    kind="ports",
                    ports=tuple(op.ins),
                    data=dict(body="ins"),
                ),
            )

            g.add_node(
                node_name,
                opmaker.make_node(
                    kind="op",
                    data=dict(body=f"{op.attrs.prettyformat()}"),
                ),
            )

            g.add_node(
                outputs_name,
                opmaker.make_node(
                    kind="ports",
                    ports=tuple(op.outs),
                    data=dict(body="outs"),
                ),
            )
            g.add_edge(src=inputs_name, dst=node_name, kind="meta")
            g.add_edge(src=node_name, dst=outputs_name, kind="meta")

    for edge in ir._storage.iter_edges():
        src = "outputs" + ident(edge.source.ref)
        dst = "inputs" + ident(edge.target.ref)
        g.add_edge(
            src=src,
            dst=dst,
            src_port=edge.source.portname,
            dst_port=edge.target.portname,
        )

    return g


def _pretty_bytecode(inst: dis.Instruction) -> str:
    return f"{inst.offset}:{inst.opname}({inst.argval})"


@dataclass(frozen=True)
class PyAttrs:
    bcinst: dis.Instruction

    def __str__(self):
        return f"[{_pretty_bytecode(self.bcinst)}]"


@dataclass(frozen=True)
class PyStoreAttrs(PyAttrs):
    varname: str

    def __str__(self):
        return f"[{_pretty_bytecode(self.bcinst)} {self.varname!r}]"


@dataclass(frozen=True)
class _ToRvsdgIR_Data:
    stack: tuple[str, ...]
    varmap: dict[str, rvsdgPort]
    region: rvsdgir.Region

    def __post_init__(self):
        assert isinstance(self.stack, tuple), "stack must be a tuple"

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

    def nest(self, region_opname, fn, **kwargs) -> "_ToRvsdg_Data":
        imported = self.imported()
        region_op = self.region.add_subregion(
            opname=region_opname, ins=imported.varmap.keys(), outs=(), **kwargs
        )
        region_op.ins(**imported.varmap)

        subregion = region_op.subregion
        inner_data = self.replace(
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
                data.region,
                imported.stack,
                imported.varmap,
                instlist,
                switch_cp_stack=self._switch_cp_stack,
            )
            return _ToRvsdgIR_Data(
                stack=tuple(stack), varmap=varmap, region=data.region
            )

        elif isinstance(block, SyntheticFill):
            # no-op
            return data

        elif isinstance(block, SyntheticAssignment):
            # Add and export control variables
            cur_varmap = data.varmap.copy()
            for k, v in block.variable_assignment.items():
                region = data.region
                op = region.add_simple_op(
                    "rvsdg.cpvar", ins=(), outs=["cp"], attrs={"cpval": int(v)}
                )
                cur_varmap[k] = op.outs.cp
            return data.replace(varmap=cur_varmap)
        elif isinstance(block, SyntheticBranch):
            cur_varmap = data.varmap.copy()
            region = data.region
            assert block.variable.startswith("backedge")
            # Search backward for the parent loop
            parent = region
            while parent.opname != "rvsdg.loop":
                parent = parent.get_parent()

            # Remove lifetime of the CP variable
            cpvar = cur_varmap.pop(block.variable)
            op = region.add_simple_op(
                "rvsdg.setcpvar",
                ins=["env", "cp"],
                outs=["env"],
                attrs={"cp": parent.attrs.extras["cp"]},
            )
            op.ins(env=cur_varmap["env"], cp=cpvar)
            cur_varmap["env"] = op.outs.env

            return data.replace(varmap=cur_varmap)

        elif isinstance(block, ExtraBasicBlock):
            # TODO
            return data

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

        def _emit_switch_body(
            inner_data: _ToRvsdgIR_Data, switch_label: str
        ) -> _ToRvsdgIR_Data:
            # Emit header
            header = region.header
            header_block = region.subregion[header]
            inner_data = self.visit_linear(header_block, inner_data)

            # Emit branches
            def _emit_branches(inner_data: _ToRvsdgIR_Data) -> _ToRvsdgIR_Data:
                imported_inner_data = inner_data.imported()
                cases_region_op = imported_inner_data.region.add_subregion(
                    opname="rvsdg.switch",
                    ins=list(imported_inner_data.varmap.keys()),
                    outs=(),
                    attrs={"cp": switch_label},
                )
                cases_region_op.ins(**imported_inner_data.varmap)
                case_data = imported_inner_data.replace(
                    region=cases_region_op.subregion,
                    varmap=dict(**cases_region_op.subregion.args),
                )

                data_foreach_case = []
                branches = filter(
                    lambda blk: blk.kind == "branch",
                    region.subregion.graph.values(),
                )
                for i, blk in enumerate(branches):

                    def _add_case_block(data):
                        return self.visit_linear(blk, data)

                    data_foreach_case.append(
                        case_data.nest(
                            "rvsdg.case", _add_case_block, attrs={"case": i}
                        )
                    )

                # Merge stack
                merged_nstack = min(
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
                    cases_region_op.subregion.results.add_port(k)
                for each_varmap in merging_varmaps:
                    cases_region_op.subregion.results(**each_varmap)
                for k in merged_stack:
                    cases_region_op.subregion.results.add_port(k)
                for each in data_foreach_case:
                    for k, stk in zip(merged_stack, each.stack):
                        port = each.varmap[stk]
                        cases_region_op.subregion.results.connect(k, port)

                out_varmap = {
                    k: cases_region_op.outs[k] for k in cases_region_op.outs
                }
                return inner_data.replace(varmap=dict(**out_varmap))

            inner_data = _emit_branches(inner_data)

            exiting = region.exiting
            exiting_block = region.subregion[exiting]
            inner_data = self.visit_linear(exiting_block, inner_data)
            return inner_data

        # setup
        switch_label = self.get_switch_label()
        self._switch_cp_stack.append(switch_label)
        # emit
        out = _emit_switch_body(data, switch_label)
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
        parent: rvsdgir.Region,
        stack: Sequence[str],
        varmap: dict[str, rvsdgPort],
        instlist: Sequence[dis.Instruction],
        switch_cp_stack: Sequence[str],
    ):
        inst = cls(parent, stack, varmap, switch_cp_stack)
        for bc in instlist:
            inst.convert(bc)
        # add output ports from live vars
        exported_varmap = {}
        for k, v in inst.varmap.items():
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
            f"py.const.load",
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
        op = self.region.add_simple_op(
            opname="py.foriter",
            ins=["env", "iter"],
            outs=["env", "next"],
            attrs=dict(py=PyAttrs(bcinst=inst), cp=self._top_switch_cp()),
        )
        op.ins(env=self.effect, iter=tos)
        self.replace_effect(op.outs.env)
        self.push(op.outs.next)

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


class BaseInterp:
    blocks: dict[int, ir.Block]
    func_id: bytecode.FunctionIdentity
    local_scope: ir.Scope
    global_scope: ir.Scope
    portdata: dict[rvsdgPort, ir.Var]
    _current_block: ir.Block | None
    last_block_label: int | None
    branch_predicate: ir.Var | None
    _label_map: dict[str, int]
    _emit_debug_print = False
    _region_stack: list[rvsdgir.Region]

    # _ret_name = ".retval"

    def __init__(self, func_id):
        self.func_id = func_id
        self.loc = self.first_loc = ir.Loc.from_function_id(func_id)
        self.global_scope = ir.Scope(parent=None, loc=self.loc)
        self.local_scope = ir.Scope(parent=self.global_scope, loc=self.loc)
        self.blocks = {}
        self.portdata = {}
        self._current_block = None
        self.last_block_label = None
        self._label_map = {}
        self._region_stack = []

    def get_global_value(self, name):
        """THIS IS COPIED from interpreter.py

        Get a global value from the func_global (first) or
        as a builtins (second).  If both failed, return a ir.UNDEFINED.
        """
        try:
            return self.func_id.func.__globals__[name]
        except KeyError:
            return getattr(builtins, name, ir.UNDEFINED)

    def store_port(self, val, port: rvsdgPort):
        value = self.store(val, f"${port.portname}")
        self.write_port(self._region, port, value)

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

    def write_port(self, region: rvsdgir.Region, port: rvsdgPort, value: ir.Var):
        self.portdata[port] = value
        # also store into aliases
        aliases = region.get_port_alias(port)
        for alias in aliases:
            self.portdata[alias] = value

    def read_port(self, port: rvsdgPort) -> ir.Var:
        return self.portdata[port]

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
    def set_block(self, label: int, block: ir.Block) -> Iterator[ir.Block]:
        """A context manager that set the current block for other IR building
        methods.

        In addition,

        - It closes any existing block in ``last_block_label`` by jumping to the
          new block.
        - If there is a existing block, it will be restored as the current block
          after the context manager.
        """
        if self.last_block_label is not None:
            last_block = self.blocks[self.last_block_label]
            if not last_block.is_terminated:
                last_block.append(ir.Jump(label, loc=self.loc))

            if self._emit_debug_print:
                print("begin dump last blk".center(80, "-"))
                last_block.dump()
                print("end dump last blk".center(80, "="))

        self.blocks[label] = block
        old = self._current_block
        self._current_block = block
        try:
            yield block
        finally:
            self.last_block_label = label
            self._current_block = old
            # dump
            if self._emit_debug_print:
                print(f"begin dump blk: {label}".center(80, "-"))
                block.dump()
                print("end dump blk".center(80, "="))

    @property
    def current_block(self) -> ir.Block:
        out = self._current_block
        assert out is not None
        return out

    @property
    def _region(self)-> rvsdgir.Region:
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


class RvsdgIRInterp(PyOpHandler):

    def run(self, region: rvsdgir.Region):
        assert region.opname == "function"

        with self._push_region(region):
            label = self._get_temp_label()
            with self.set_block(
                label, ir.Block(scope=self.local_scope, loc=self.loc)
            ):
                for i, k in enumerate(self.func_id.arg_names):  # type: ignore
                    val = ir.Arg(index=i, name=k, loc=self.loc)
                    port = region.args[f"arg_{k}"]
                    self.store_port(val, port)

            self.emit_linear_region(region)

    def emit_region(self, region: rvsdgir.Region):
        with self._push_region(region):
            if region.opname in {"block"}:
                self.emit_linear_region(region)

    def emit_linear_region(self, region: rvsdgir.Region):
        label = self._get_temp_label()
        with self.set_block(
            label, ir.Block(scope=self.local_scope, loc=self.loc)
        ):
            for op in region.body.toposorted_ops():
                if isinstance(op, rvsdgir.RegionOp):
                    # map ins to args
                    for port, argport in zip(op.ins.values(), op.subregion.args.values(), strict=True):
                        if port.portname != "env":
                            value = self.portdata[port]
                            self.write_port(op.subregion, argport, value)
                    self.emit_region(op.subregion)
                elif op.opname.startswith("py."):
                    # python operations
                    self.emit_py_op(op)
                else:
                    raise NotImplementedError(op)

    def emit_py_op(self, op):
        fname = op.opname.replace('.', '_')
        fn = getattr(self, fname, None)
        if fn is None:
            raise NotImplementedError(f"{fname!r} not defined to emit {op.attrs.prettyformat()}")
        else:
            pyattrs: PyAttrs = op.attrs.extras["py"]
            pos = pyattrs.bcinst.positions
            if pos is not None:
                self.loc = self.loc.with_lineno(pos.lineno, pos.col_offset)
            try:
                fn(op, op.attrs.extras["py"])
            except Exception as e:
                e.add_note(f"failed in {fn}; source: {self.loc}")
                raise
