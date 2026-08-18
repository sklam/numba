#!/usr/bin/env python
"""AST-based static import extraction for the numba codebase.

Walks all .py files under a root package, extracts import statements
with the `ast` module (exact syntax, so comments, docstrings and
string literals never produce false matches), resolves relative
imports to absolute dotted names, and writes a per-module import map
to JSON.

Usage:
    python extract_imports.py <root_dir> <package> <output.json>
Example:
    python extract_imports.py numba numba imports.json

Notes:
- `from X import a, b` records both `X` and candidate submodules
  `X.a`, `X.b`. Downstream resolves candidates by longest-prefix match
  against the discovered module set, so attributes fall back to `X`.
- Conditional/lazy imports inside functions and if/try blocks are
  included (static view): the visitor descends into all statements.
- `from __future__ import ...` is skipped (not a real dependency).
"""

import ast
import json
import os
import sys


def resolve(level, rest, mod_parts, is_pkg):
    """Resolve a relative from-import target to an absolute dotted name."""
    if level == 0:
        return rest
    base = list(mod_parts) if is_pkg else list(mod_parts[:-1])
    if level > 1:
        drop = level - 1
        base = base[:-drop] if drop <= len(base) else []
    if rest:
        base.extend(p for p in rest.split('.') if p)
    return '.'.join(base)


class ImportVisitor(ast.NodeVisitor):
    """Collect absolute dotted names imported by one module.

    Descends into every statement, so imports inside functions,
    methods and if/try blocks (lazy/conditional imports) are included.
    """

    def __init__(self, mod_parts, is_pkg):
        self.mod_parts = mod_parts
        self.is_pkg = is_pkg
        self.imports = set()

    def visit_Import(self, node):
        # import a, b.c as d
        for alias in node.names:
            self.imports.add(alias.name)

    def visit_ImportFrom(self, node):
        # from X import a, b  (possibly relative: node.level > 0)
        if node.module == '__future__':
            return
        base = resolve(node.level, node.module or '',
                       self.mod_parts, self.is_pkg)
        if base:
            self.imports.add(base)
        for alias in node.names:
            if alias.name == '*':
                continue
            self.imports.add(base + '.' + alias.name
                             if base else alias.name)


def module_name_for(path, root_dir, package):
    rel = os.path.relpath(path, os.path.dirname(root_dir))
    parts = rel[:-3].split(os.sep)  # strip .py
    is_pkg = parts[-1] == '__init__'
    if is_pkg:
        parts = parts[:-1]
    return '.'.join(parts), is_pkg


def extract_file(path, root_dir, package):
    mod, is_pkg = module_name_for(path, root_dir, package)
    with open(path, encoding='utf-8', errors='replace') as f:
        tree = ast.parse(f.read(), filename=path)
    visitor = ImportVisitor(mod.split('.'), is_pkg)
    visitor.visit(tree)
    imports = visitor.imports
    imports.discard(mod)  # never self-import
    return mod, is_pkg, sorted(imports)


def main():
    root_dir, package, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    modules = {}
    errors = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in sorted(filenames):
            if not fn.endswith('.py'):
                continue
            path = os.path.join(dirpath, fn)
            try:
                mod, is_pkg, imports = extract_file(path, root_dir, package)
            except SyntaxError as e:
                errors.append(f'{path}: {e}')
                continue
            modules[mod] = {
                'file': path,
                'is_package': is_pkg,
                'imports': imports,
            }
    with open(out_json, 'w') as f:
        json.dump({'package': package, 'modules': modules}, f,
                  indent=1, sort_keys=True)
    print(f'{len(modules)} modules -> {out_json}')
    for err in errors:
        print(f'WARNING: skipped unparseable file: {err}', file=sys.stderr)


if __name__ == '__main__':
    main()
