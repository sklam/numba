"""
Utilities for processing RVSDG
"""
import sys
from dataclasses import dataclass, field
from collections import deque
from collections.abc import Mapping
from typing import Any, Iterator, Sequence, Union

from numba_rvsdg.core.datastructures.basic_block import BasicBlock, RegionBlock
from .regionpasses import RegionVisitor
from .bc2rvsdg import DDGBlock, ValueState, Op, DDGProtocol, SCFG


@dataclass(frozen=True)
class Def:
    """A definition of a value/state."""

    parent: DDGProtocol = field(hash=False)


@dataclass(frozen=True)
class VSDef(Def):
    """A definition associated with a ValueState."""

    vs: ValueState

    def __repr__(self) -> str:
        assert isinstance(self.parent, DDGBlock)  # for mypy
        buf = [f"Def({self.vs.name}"]
        if self.vs.parent is not None:
            p = self.vs.parent
            buf.append(f"from {p.short_identity()}")
        buf.append(f"in {self.parent.name})")
        return " ".join(buf)


@dataclass(frozen=True)
class ArgDef(Def):
    """A definition associated with a argument."""

    name: str

    def __repr__(self) -> str:
        return f"Def(arg: {self.name})"


@dataclass(frozen=True)
class PhiDef(Def):
    """A definition associated with a phi node."""

    name: str
    incomings: list[Def] = field(default_factory=list, hash=False)

    def insert_incoming(self, df: Def):
        self.incomings.append(df)

    def __repr__(self) -> str:
        assert isinstance(self.parent, BasicBlock)  # for mypy
        return f"Def(phi, {self.name}, {self.parent.name})"


@dataclass(frozen=True)
class PortDef(Def):
    """A definition associated with a port."""

    name: str

    def __repr__(self) -> str:
        assert isinstance(self.parent, BasicBlock)  # for mypy
        return f"Def(port, {self.name}, {self.parent.name})"


@dataclass(frozen=True)
class Use:
    """A use of a definition."""

    ...


@dataclass(frozen=True)
class OpUse(Use):
    """A use of a definition by an operation."""

    op: Op


@dataclass(frozen=True)
class PortUse(Use):
    """A use of a definition by a port."""

    vsdef: Def


class ForwardedDefs(list[Union[Def, "ForwardedDefs"]]):
    """ForwardedDefs is a list that can contain Def objects or nested ForwardedDefs.
    It is used to represent chains of definitions that are forwarded from block
    to block in dataflow analysis.

    get_edges() traverses the chains and extracts the def-use edges between the
    Def objects. It will recursively process nested ForwardedDefs chains.
    """

    def verify(self):
        """Check that the nesting structure is valid - either all elements are
        ForwardedDefs or only the last element is.

        """
        if self:
            is_nested = [isinstance(x, ForwardedDefs) for x in self]
            if not (all(is_nested) or not any(is_nested[:-1])):
                raise ValueError(
                    "malformed: requires that either all elements "
                    "are or only the last element is ForwardedDefs or "
                )

    def pformat(self, indent=0) -> str:
        """Pretty print the ForwardedDefs chains."""
        buf = []
        if self:
            prefix = " " * indent
            for i, elem in enumerate(self):
                leading = f"{prefix}{i}: "
                if isinstance(elem, ForwardedDefs):
                    buf.append(leading)
                    buf.append(elem.pformat(indent=len(leading)))
                else:
                    buf.append(f"{leading}{elem}")
        return "\n".join(buf)

    def get_edges(self, _prev=None) -> list[tuple[str, str]]:
        """Traverse the chains and extracts the def-use edges between the
        Def objects. It will recursively process nested ForwardedDefs chains.
        """
        buf = []
        prev = _prev
        for elem in self:
            if isinstance(elem, Def):
                if isinstance(elem, VSDef):
                    cur = elem.vs.short_identity()
                    if prev is not None:
                        buf.append((prev, cur))
                    prev = cur
            else:
                assert isinstance(elem, ForwardedDefs)
                edges = elem.get_edges(_prev=prev)
                buf.extend(edges)
        return buf


class UseDefs:
    """Provide analysis to track definitions and uses of values in a data
    dependency graph.
    """

    _defmap: dict[ValueState, VSDef]
    _phis: list[PhiDef]
    _ports: list[PortDef]
    _argmap: dict[str, ArgDef]
    _uses: dict[Def, list[Use]]

    def __init__(self) -> None:
        self._defmap = {}
        self._phis = []
        self._ports = []
        self._argmap = {}
        self._uses = {}

    # Builder API

    def get_or_insert_vs_def(self, vs: ValueState, block: DDGProtocol) -> VSDef:
        return self._defmap.setdefault(vs, VSDef(block, vs))

    def insert_phi_def(self, name: str, block: DDGProtocol) -> PhiDef:
        phidef = PhiDef(block, name)
        self._phis.append(phidef)
        return phidef

    def insert_port_def(self, name: str, block: DDGProtocol) -> PortDef:
        portdef = PortDef(block, name)
        self._ports.append(portdef)
        return portdef

    def get_or_insert_arg_def(self, name: str, parent: DDGProtocol) -> ArgDef:
        return self._argmap.setdefault(name, ArgDef(parent, name))

    def insert_op_use(self, vs: Def, op: Op) -> None:
        self._uses.setdefault(vs, []).append(OpUse(op))

    def insert_port_use(self, src: Def, dst: Def) -> None:
        self._uses.setdefault(src, []).append(PortUse(dst))

    # Analysis API

    def lookup(self, vs_name: str) -> ValueState:
        for vs in self._defmap.keys():
            if vs_name == vs.short_identity():
                return vs
        raise KeyError(vs_name)

    def get_vs_def(self, vs: ValueState) -> VSDef:
        return self._defmap[vs]

    def get_uses(self, df: Def) -> tuple[Use, ...]:
        return tuple(self._uses.get(df, ()))

    def get_forwarded_vs_chain(self, arg: ValueState | Def) -> ForwardedDefs:
        """Trace forwards from a ValueState/Def to build a nested list of Defs

        Parameters
        ----------
        arg : ValueState|Def
            The starting ValueState or Def to trace from.

        Returns
        -------
        list[Def|list]
            A nested list tracing the forwarded Defs. Defs are appended to
            output, lists represent multiple uses.
        """
        df: Def = self.get_vs_def(arg) if isinstance(arg, ValueState) else arg
        results: ForwardedDefs = ForwardedDefs([df])
        todos = deque([(df, results)])

        def handle_use(
            use: Use, output: ForwardedDefs
        ) -> Sequence[tuple[Def, ForwardedDefs]]:
            df: Def
            if isinstance(use, OpUse):
                if use.op.opname in {
                    "stack.export",
                    "stack.incoming",
                    "var.incoming",
                }:
                    [out] = use.op.outputs
                    df = self.get_vs_def(out)
                    output.append(df)
                    return [(df, output)]
            elif isinstance(use, PortUse):
                df = use.vsdef
                output.append(df)
                return [(df, output)]
            return ()

        while todos:
            df, output = todos.popleft()
            uses = self.get_uses(df)
            if len(uses) > 1:
                nest = ForwardedDefs()
                output.append(nest)
                for use in uses:
                    inner = ForwardedDefs()
                    nest.append(inner)
                    todos.extend(handle_use(use, inner))
            elif uses:
                [use] = uses
                todos.extend(handle_use(use, output))

        results.verify()
        return results


class UseDefFrontier(Mapping[str, Def]):
    """UseDefFrontier provides a read-only Mapping interface to a dictionary
    mapping variable names to reaching Definitions. This is used in dataflow
    analysis to represent the frontier of reachable definitions at a program
    point.
    """

    _dct: Mapping[str, Def]

    def __init__(self, dct: Mapping[str, Def]):
        self._dct = dct

    def __getitem__(self, k) -> Def:
        return self._dct[k]

    def __len__(self) -> int:
        return len(self._dct)

    def __iter__(self) -> Iterator[str]:
        return iter(self._dct)


class ComputeUseDefs(RegionVisitor[UseDefFrontier]):
    """
    Find all value-state definitions and uses across all regions
    """

    usedefs: UseDefs

    def __init__(self):
        super().__init__()
        self.usedefs = UseDefs()

    def run(self, graph: SCFG) -> UseDefs:
        # preload arguments
        ud = self.usedefs
        head: DDGProtocol = graph[graph.find_head()]
        args = {
            k: ud.get_or_insert_arg_def(k, head) for k in head.incoming_states
        }
        frontier = UseDefFrontier(args)
        self.visit_graph(graph, data=frontier)
        return self.usedefs

    def visit_block(
        self, block: BasicBlock, data: UseDefFrontier
    ) -> UseDefFrontier:
        usedefs = self.usedefs
        if isinstance(block, DDGBlock):
            ddginfo = block.get_analysis()
            frontier = {data[k]: vs for k, vs in block.in_vars.items()}
            for from_def, to_vs in frontier.items():
                usedefs.insert_port_use(
                    from_def, usedefs.get_or_insert_vs_def(to_vs, block)
                )
            for op in ddginfo.all_ops:
                for vs in op.inputs:
                    usedefs.insert_op_use(
                        usedefs.get_or_insert_vs_def(vs, block), op
                    )
            return UseDefFrontier(
                {
                    k: usedefs.get_or_insert_vs_def(vs, block)
                    for k, vs in block.out_vars.items()
                }
            )
        elif isinstance(block, DDGProtocol):
            out = {}
            for k in block.incoming_states:
                from_def = data[k]
                to_def = usedefs.insert_port_def(k, block)
                usedefs.insert_port_use(from_def, to_def)
                out[k] = to_def
            return UseDefFrontier(out)
        else:
            return data

    def visit_loop(
        self, region: RegionBlock, data: UseDefFrontier
    ) -> UseDefFrontier:
        return super().visit_linear(region, data)

    def visit_switch(
        self, region: RegionBlock, data: UseDefFrontier
    ) -> UseDefFrontier:
        data = self.visit_linear(region.subregion[region.header], data)

        to_merge = []
        for blk in region.subregion.graph.values():
            if blk.kind == "branch":
                to_merge.append(
                    self.visit_linear(region.subregion[blk.name], data)
                )

        tail_blk: DDGProtocol = region.subregion[region.exiting]
        phis = {
            k: self.usedefs.insert_phi_def(k, tail_blk)
            for k in tail_blk.incoming_states
        }
        for frontier in to_merge:
            for k, phi in phis.items():
                inc_def = frontier[k]
                phi.insert_incoming(inc_def)
                self.usedefs.insert_port_use(inc_def, phi)

        return self.visit_linear(
            region.subregion[region.exiting], UseDefFrontier(phis)
        )
