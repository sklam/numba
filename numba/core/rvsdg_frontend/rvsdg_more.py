from numba.core import (
    utils,
)

from .import rvsdg_core
from . import rvsdgir

def run_frontend(func):
    sig = utils.pySignature.from_callable(func)
    argnames = tuple(sig.parameters)
    rvsdg = rvsdg_core.build_rvsdg(func.__code__, argnames)
    simplify_rvsdg(rvsdg)

    rvsdg_core.render_rvsdgir(rvsdg, "rvsdgir-simplified")

def simplify_rvsdg(region: rvsdgir.Region):
    simplify_port_use(region)

def simplify_port_use(region: rvsdgir.Region,
                      dead_outs: set[str]=frozenset()):
    toposorted = list(region.body.toposorted_ops())



    # handle dead outs
    print(region.attrs.prettyformat(), dead_outs)

    for k in dead_outs:
        print('   remove out', k)
        region.results.remove_port(k)

    region.remove_dangling_edges()
    # handle body
    while toposorted:
        node = toposorted.pop()


        if isinstance(node, rvsdgir.RegionOp):
            # find dead output ports
            out_ports = set(node.outs.list_ports())
            used_ports = set(region.iter_edges().iter_ports())
            unused_outs = out_ports - used_ports
            if node.opname == "rvsdg.loop":
                dead_outs={p.portname for p in unused_outs
                           if p.portname not in node.ins}
            else:
                dead_outs={p.portname for p in unused_outs}
            simplify_port_use(node.subregion,
                              dead_outs=dead_outs)

            region.remove_dangling_edges()
        # else:
        #     pass

    # handle unused args
    region.remove_dangling_edges()

    if region.opname not in {"rvsdg.loop", "function"}:
        print("HERE", region.attrs.prettyformat())

        used_ports = set(region.iter_edges().iter_ports())
        arg_ports = set(region.args.list_ports())
        unused_args = arg_ports - used_ports
        if unused_args:
            for p in unused_args:
                print("         remove", p.portname)
                region.args.remove_port(p.portname)
            region.remove_dangling_edges()

    # find passthrus
    if region.has_parent():
        regop = region.get_parent_regionop()
        parent = region.get_parent()

        edge_from_args = set(region.iter_edges().filter_by_source(set(region.args.list_ports())))
        edge_to_results = set(region.iter_edges().filter_by_target(set(region.results.list_ports())))

        passthrus = edge_from_args & edge_to_results
        writtento = edge_to_results - edge_from_args
        readonly = edge_from_args - edge_to_results
        if passthrus:

            readonly_ins = {p.source.portname for p in readonly}
            written_outs = {p.target.portname for p in writtento}
            passthrus_ins: set[str] = set()
            passthrus_outs: set[str] = set()
            for edge in passthrus:
                passthrus_ins.add(edge.source.portname)
                passthrus_outs.add(edge.target.portname)

            regop_shortcuts = dict()
            for k, v in zip(passthrus_ins, passthrus_outs, strict=True):
                regop_shortcuts[regop.ins[k]] = regop.outs[v]

            for k in passthrus_ins - readonly_ins:
                region.args.remove_port(k)
            for k in passthrus_outs - written_outs:
                region.results.remove_port(k)

            region.remove_dangling_edges()
            keep_ports: set[rvsdgir.Port] = set()
            keep_ports.union([regop.ins[k] for k in readonly_ins])
            keep_ports.union([regop.outs[k] for k in written_outs])
            parent.shortcut_edges(regop_shortcuts, keep_ports)