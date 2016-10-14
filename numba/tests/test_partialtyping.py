import numpy as np

from numba import njit
from .support import unittest


class TestPartialTyping(unittest.TestCase):

    def test_recursion(self):
        @njit
        def foo(arr):
            if arr.ndim > 1:
                tmp = 0
                for i in range(arr.shape[0]):
                    tmp += foo(arr[i])
                return tmp
            else:
                return arr.sum()

        shape = (1, 2, 3, 4)
        arr = np.arange(np.prod(shape)).reshape(shape)
        self.assertEqual(foo(arr), foo.py_func(arr))

    def test_bad_return(self):
        @njit
        def foo(arr, flag):
            if flag:
                return arr.ndim
            tmp = 2
            return tmp

        self.assertEqual(foo(12, False), 2)
        with self.assertRaises(TypeError) as raises:
            foo(12, True)

        self.assertIn("Unknown attribute 'ndim' of type int64",
                      str(raises.exception))

if __name__ == '__main__':
    unittest.main()
