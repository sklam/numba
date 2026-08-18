#!/usr/bin/env python
"""Build Graphviz DOT graphs from imports.json (see extract_imports.py).

Nodes are modules. An edge A -> B means "module A imports module B":
the arrow source (tail) is the importing module and the arrow head
points at the module being imported, i.e. arrows point from a module
toward its dependencies (read A -> B as "A depends on B").

Numba modules sit in nested clusters following the
subpackage hierarchy. Colors:
  light blue  = numba module          (edge: solid steel blue)
  pale blue   = numba module outside --filter ("other numba" cluster)
  dashed blue = numba native/unresolved extension module (shown as PKG.*)
  orange      = third-party package   (edge: dashed orange)
  gray        = stdlib module         (edge: dashed gray)

Usage:
    python build_dot.py imports.json out.dot --filter PREFIX
                                             [--imports NAME]
                                             [--exclude PREFIX]...

--filter numba.core keeps only edges sourced under numba.core, showing
those modules (clustered by subpackage) plus everything they import
(render with fdp).
--imports numpy additionally restricts the graph to modules that
directly import numpy and the intra-numba edges between them (the
numpy node itself is omitted, since every shown module imports it by
construction).
--exclude numba.cuda drops modules under the prefix from the graph
entirely (repeatable). numba.tests is always excluded.
"""

import argparse
import json
import sys

STDLIB = sys.stdlib_module_names  # Python >= 3.10

C_NUMBA = ('#dce9f7', '#336699')   # fill, border
C_OTHER = ('#f4f8fc', '#7a9cc6')   # numba module outside --filter
C_NATIVE = ('#eef3f8', '#336699')
C_THIRD = ('#ffe0b3', '#cc7a00')
C_STDLIB = ('#eeeeee', '#888888')
E_INTRA = '#336699'
E_THIRD = '#cc7a00'
E_STDLIB = '#999999'


def longest_known(name, known):
    parts = name.split('.')
    while parts:
        cand = '.'.join(parts)
        if cand in known:
            return cand, len(parts) == len(name.split('.'))
        parts.pop()
    return None, False


def classify_edges(modules, known, root_pkg):
    intra = set()       # (src, dst) both numba modules
    native = set()      # dst is unresolved numba submodule (C ext, re-export)
    ext = set()         # (src, top_level) outside numba
    for mod, info in modules.items():
        for imp in info['imports']:
            top = imp.split('.')[0]
            if top == root_pkg:
                tgt, exact = longest_known(imp, known)
                if tgt is None:
                    continue
                if not exact and len(imp.split('.')) > 1:
                    # unresolved numba submodule: C extension or re-export.
                    # The '.*' suffix keeps the synthetic aggregate node
                    # distinct from the real package module of the same
                    # prefix (otherwise graphviz errors when that package
                    # is itself in the focus clusters).
                    tgt = '.'.join(imp.split('.')[:2]) + '.*'
                    if tgt != mod:
                        native.add((mod, tgt))
                elif tgt != mod:
                    intra.add((mod, tgt))
            else:
                ext.add((mod, top))
    return intra, native, ext


def under(mod, prefix):
    return mod == prefix or mod.startswith(prefix + '.')


def apply_filter(focus, intra, native, ext):
    """Keep only edges whose source is in the focus set."""
    keep = lambda E: {(s, d) for s, d in E if s in focus}
    return keep(intra), keep(native), keep(ext)


def build_tree(nodes):
    """Nested dict: path tuple -> {'nodes': [...], 'children': {}}."""
    tree = {'nodes': [], 'children': {}}

    def ensure(parts):
        t = tree
        for p in parts:
            t = t['children'].setdefault(p, {'nodes': [], 'children': {}})
        return t

    for mod in nodes:
        parts = mod.split('.')
        parent = parts[:-1] if len(parts) > 1 else []
        ensure(parent)['nodes'].append(mod)
    return tree


def emit_node(lines, indent, mod, label, colors, style='filled'):
    fill, border = colors
    pre = '  ' * indent
    lines.append(f'{pre}"{mod}" [label="{label}", style="{style}", '
                 f'fillcolor="{fill}", color="{border}"];')


def emit_flat_cluster(lines, indent, cid, label, nodes, colors,
                      style='filled', label_of=None):
    if not nodes:
        return
    pre = '  ' * indent
    _, border = colors
    lines.append(f'{pre}subgraph "{cid}" {{')
    lines.append(f'{pre}  label="{label}"; color="{border}"; style=rounded;')
    for n in nodes:
        emit_node(lines, indent + 1, n, label_of(n) if label_of else n,
                  colors, style)
    lines.append(f'{pre}}}')


def emit_tree(tree, path, lines, indent, label_of):
    pre = '  ' * indent
    for mod in sorted(tree['nodes']):
        emit_node(lines, indent, mod, label_of(mod), C_NUMBA)
    for name, sub in sorted(tree['children'].items()):
        cpath = path + (name,)
        cid = 'cluster_' + '_'.join(cpath)
        clabel = '.'.join(cpath)
        lines.append(f'{pre}subgraph "{cid}" {{')
        lines.append(f'{pre}  label="{clabel}"; color="#aaaaaa"; '
                     f'fontcolor="#444444"; style=rounded;')
        emit_tree(sub, cpath, lines, indent + 1, label_of)
        lines.append(f'{pre}}}')


def emit_focus_tree(tree, prefix, lines, label_of):
    """Emit one top-level cluster for PREFIX wrapping its subpackage tree."""
    parts = prefix.split('.')
    node = tree
    for p in parts:
        node = node['children'].get(p, {'nodes': [], 'children': {}})
    parent = tree
    for p in parts[:-1]:
        parent = parent['children'].get(p, {'nodes': [], 'children': {}})
    cid = 'cluster_' + '_'.join(parts)
    lines.append(f'  subgraph "{cid}" {{')
    lines.append(f'    label="{prefix}"; color="#aaaaaa"; '
                 f'fontcolor="#444444"; style=rounded;')
    # the prefix module itself lives one tree level up
    if prefix in parent['nodes']:
        emit_node(lines, 2, prefix, label_of(prefix), C_NUMBA)
    emit_tree(node, tuple(parts), lines, 2, label_of)
    lines.append('  }')


def short_label(mod):
    parts = mod.split('.')
    return parts[-1] + '/__init__' if is_pkg_node(mod) else parts[-1]


_PKG_NODES = set()


def is_pkg_node(mod):
    return mod in _PKG_NODES


def native_label(mod):
    parts = mod.split('.')
    return parts[-2] + '.*' if parts[-1] == '*' else parts[-1]


def write_full(root_pkg, intra, native, ext, out, prefix, focus):
    numba_nodes = sorted(focus)
    other = sorted(d for d in {d for _, d in intra}
                   if not under(d, prefix))
    tree = build_tree(numba_nodes)
    ext_third = sorted({d for _, d in ext} - STDLIB)
    ext_std = sorted({d for _, d in ext} & STDLIB)
    native_nodes = sorted({d for _, d in native})

    L = ['digraph numba_imports {',
         '  graph [compound=true, overlap=false, splines=true, '
         'bgcolor=white, newrank=true];',
         '  node [shape=box, fontsize=10, fontname="Helvetica"];',
         '  edge [arrowsize=0.6];']
    emit_focus_tree(tree, prefix, L, short_label)
    emit_flat_cluster(L, 1, 'cluster_numba_other',
                      f'{root_pkg} (outside {prefix})', other, C_OTHER)
    emit_flat_cluster(L, 1, 'cluster_numba_native', 'native / unresolved',
                      native_nodes, C_NATIVE, style='filled,dashed',
                      label_of=native_label)
    emit_flat_cluster(L, 1, 'cluster_external_thirdparty', 'third-party',
                      ext_third, C_THIRD)
    emit_flat_cluster(L, 1, 'cluster_external_stdlib', 'stdlib', ext_std,
                      C_STDLIB)

    for s, d in sorted(intra):
        L.append(f'  "{s}" -> "{d}" [color="{E_INTRA}"];')
    for s, d in sorted(native):
        L.append(f'  "{s}" -> "{d}" [color="{E_INTRA}", style=dashed];')
    for s, d in sorted(ext):
        c = E_STDLIB if d in STDLIB else E_THIRD
        L.append(f'  "{s}" -> "{d}" [color="{c}", style=dashed];')
    L.append('}')
    out.write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json_in')
    ap.add_argument('dot_out')
    ap.add_argument('--filter', metavar='PREFIX', required=True,
                    help='show only edges sourced from modules under PREFIX '
                         '(e.g. numba.core), plus everything they import')
    ap.add_argument('--imports', metavar='NAME', default=None,
                    help='show only numba modules that directly import NAME '
                         '(e.g. numpy) and the edges between those modules')
    ap.add_argument('--exclude', metavar='PREFIX', action='append',
                    default=[],
                    help='drop modules under PREFIX from the graph '
                         '(repeatable, e.g. --exclude numba.cuda)')
    args = ap.parse_args()

    with open(args.json_in) as f:
        data = json.load(f)
    root_pkg = data['package']
    modules = {m: i for m, i in data['modules'].items()
               if not m.startswith(root_pkg + '.tests')}
    # drop excluded sources; `known` stays complete so longest_known()
    # still resolves imports of excluded modules correctly
    modules = {m: i for m, i in modules.items()
               if not any(under(m, p) for p in args.exclude)}
    known = set(data['modules'])

    global _PKG_NODES
    _PKG_NODES = {m for m, i in data['modules'].items() if i['is_package']}

    intra, native, ext = classify_edges(modules, known, root_pkg)

    focus = {m for m in modules if under(m, args.filter)}
    if not focus:
        ap.error(f'--filter {args.filter!r} matched no modules')
    intra, native, ext = apply_filter(focus, intra, native, ext)

    if args.imports:
        target = args.imports
        importers = {s for s, d in ext if d == target}
        if not importers:
            ap.error(f'--imports {args.imports!r} matched no modules')
        focus = importers
        # keep only edges between importing modules; the target node/edges
        # are omitted since every shown module imports NAME by construction
        intra = {(s, d) for s, d in intra if s in focus and d in focus}
        native = {(s, d) for s, d in native if s in focus and d in focus}
        ext = set()

    if args.exclude:
        # drop edges whose target is under an excluded prefix
        def drop_excluded(E):
            return {(s, d) for s, d in E
                    if not any(under(d, p) for p in args.exclude)}
        intra, native, ext = (drop_excluded(intra), drop_excluded(native),
                              drop_excluded(ext))

    with open(args.dot_out, 'w') as out:
        write_full(root_pkg, intra, native, ext, out, args.filter, focus)

    from collections import Counter
    top_ext = Counter(d for _, d in ext).most_common(10)
    print(f'modules={len(modules)}'
          + (f' focus={len(focus)}' if focus else '')
          + f' intra_edges={len(intra)} native_edges={len(native)} '
            f'external_edges={len(ext)}')
    print('top external:', ', '.join(f'{n}({c})' for n, c in top_ext))


if __name__ == '__main__':
    main()
