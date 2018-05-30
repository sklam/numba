"""
Typing for scipy specific features
"""
from .containers import _HeterogenousTuple


class LowLevelCallable(_HeterogenousTuple):
    """
    Items:
        0: pycapsule
        1: function
        2: user_data
    """
    def __init__(self, llc):
        from numba import typeof, types

        pycap, function, user_data = llc
        sig = llc.signature
        elems = [
            types.pyobject,     # pycapsule
            typeof(function),   # function
            types.none,         # userdata (ignored)
            ]
        name = '{}({})'.format(type(self).__name__, llc)
        super(LowLevelCallable, self).__init__(name)
        self.types = elems
        self._signature = sig
