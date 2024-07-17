import sys
from collections import defaultdict
from pprint import pprint

from numba import jit
import numpy as np
from numba.misc.dict_watcher import ChangeWatcher, DictWatcherManager

import typing

import chk_module_watcher as chk  # which is this file


CONSTANT_A = 1
CONSTANT_B = True
CONSTANT_C = False


@jit
def bar():
    a = chk.CONSTANT_A
    b = chk.CONSTANT_C
    return a, b, np.arange(3), np.arange(0, 4, dtype=np.float32)


@jit
def foo():
    a = CONSTANT_A
    b = CONSTANT_B
    return a, b, bar()


class Entry(typing.NamedTuple):
    dct: dict
    mod: str
    attr: str


def install_watchers(mod_consts):
    class PrintChange(ChangeWatcher):
        def on_change(self, key, value):
            print("... changing", key, "to", value)

    dct_attrs = defaultdict(list)
    for mod, attr in mod_consts:
        dct = sys.modules[mod].__dict__
        dct_attrs[id(dct)].append(Entry(dct, mod, attr))

    manager = DictWatcherManager()
    for entries in dct_attrs.values():
        dcts, _mods, attrs = zip(*entries)
        attrs = frozenset(attrs)
        print(f"-- {hex(id(dcts[0]))} ----- {attrs}")
        manager.watch(dcts[0], PrintChange(keys=attrs))


def test():
    r = foo()
    mod_consts = foo.get_referenced_globals(foo.signatures[0])
    pprint(mod_consts)

    install_watchers(mod_consts)

    global CONSTANT_A, bar
    CONSTANT_A = 123

    del chk.CONSTANT_C

    @jit
    def bar():
        return 123456


if __name__ == "__main__":
    test()
