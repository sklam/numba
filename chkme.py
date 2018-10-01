import numpy as np

from numba.unsafe import ref
from numba import njit
from numba import types
from collections import namedtuple


userpod = namedtuple('userpod', ['value', 'buf'])

@njit
def bar(podref, newval, newbuf):
    ref.put(podref, userpod(value=newval, buf=newbuf))
    return podref


class OpaqueUserPod(ref.OpaqueRef):
    pass


cast_as_opaque, cast_as_userpod = OpaqueUserPod.casters


@njit
def foo(v, b):
    pod = userpod(value=v, buf=b)
    podref = ref.make(pod)
    pod = ref.get(podref)
    podref = bar(podref, 123, pod.buf)

    opaque = cast_as_opaque(podref)
    return pod, ref.get(podref)


r = foo(1, np.asarray([1., 2.]))

print(r)
