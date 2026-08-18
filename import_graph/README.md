# import_graph

Static import-graph tooling for the numba codebase: extract per-module
imports to JSON, then turn that into a focused Graphviz DOT graph.

- `extract_imports.py` — AST-based import extractor (no regex; comments,
  docstrings and string literals never match). Writes a per-module import
  map to JSON.
- `build_dot.py` — builds a DOT graph from the JSON, focused on a
  subpackage via `--filter`, with optional `--imports` / `--exclude`
  narrowing. Numba modules are drawn in nested clusters following the
  subpackage hierarchy.

Requires Python >= 3.10 and `graphviz` (`fdp` or `dot`) for rendering.

## Graph semantics

Nodes are modules. An edge `A -> B` means "module A imports module B":
the arrow source (tail) is the importer, the head points at the module
being imported — read it as "A depends on B".

Colors:

| Style | Meaning | Edge |
|---|---|---|
| light blue box | numba module under `--filter` | solid steel blue |
| pale blue box | numba module outside `--filter` | solid steel blue |
| dashed blue box | numba native/unresolved extension module | dashed steel blue |
| orange box | third-party package | dashed orange |
| gray box | stdlib module | dashed gray |

## extract_imports.py

```
python import_graph/extract_imports.py <root_dir> <package> <output.json>
```

Example:

```
python import_graph/extract_imports.py numba numba import_graph/imports.json
# -> 745 modules -> import_graph/imports.json
```

Notes:

- `from X import a, b` records both `X` and candidate submodules `X.a`,
  `X.b`; `build_dot.py` resolves candidates by longest-prefix match
  against the discovered module set, so plain attributes fall back to `X`.
- Conditional/lazy imports inside functions and if/try blocks are
  included (static view).
- `from __future__ import ...` is skipped; unparseable files are skipped
  with a warning on stderr.

## build_dot.py

```
python import_graph/build_dot.py imports.json out.dot --filter PREFIX
                                 [--imports NAME] [--exclude PREFIX]...
```

- `--filter PREFIX` (required): keep only edges sourced from modules
  under PREFIX, showing those modules (clustered by subpackage) plus
  everything they import.
- `--imports NAME`: additionally restrict to modules that directly
  import NAME and the intra-numba edges between them. The NAME node
  itself is omitted — every shown module imports it by construction.
- `--exclude PREFIX`: drop modules under PREFIX entirely (repeatable).
- `numba.tests` is always excluded.

Prints stats (focus size, edge counts, top external packages) to stdout.

Render with `fdp` (`dot` is slower but better layout for hierarchical data):

```
fdp -Tsvg out.dot -o out.svg
```

## Example workflow: numpy-focused graph

Which numba modules import numpy directly, and how do those modules
import each other (excluding the CUDA subtree)?

```
# 1. extract (only needs re-running when the source tree changes)
python import_graph/extract_imports.py numba numba import_graph/imports.json

# 2. build the focused DOT
python import_graph/build_dot.py \
    import_graph/imports.json import_graph/graph_numpy.dot \
    --filter numba --imports numpy --exclude numba.cuda
# -> modules=263 focus=49 intra_edges=114 native_edges=0 external_edges=0

# 3. render
fdp -Tsvg import_graph/graph_numpy.dot -o import_graph/graph_numpy.svg
fdp -Tpng import_graph/graph_numpy.dot -o import_graph/graph_numpy.png
```

Variations:

```
# everything numba.core imports (external/native clusters included)
python import_graph/build_dot.py \
    import_graph/imports.json import_graph/graph_core.dot --filter numba.core

# numpy importers within numba.core only (answer: just ir_utils)
python import_graph/build_dot.py \
    import_graph/imports.json graph.dot \
    --filter numba.core --imports numpy --exclude numba.cuda
# -> focus=1 intra_edges=0
```
