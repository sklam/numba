import numpy as np

from numba import njit
from numba.errors import TypingError
from .support import unittest


class TestPartialTyping(unittest.TestCase):

    def test_recursion(self):
        @njit(partial_typing=True)
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
        @njit(partial_typing=True)
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

    def test_bad_untyped_single_return(self):
        @njit(partial_typing=True)
        def foo(arr):
            c = 0
            for i in range(10):
                c += i
            c += arr.ndim
            return c

        with self.assertRaises(TypingError) as raises:
            # passing in an integer to trigger untyped error
            foo(10)

        self.assertIn("Unknown attribute 'ndim' of type int64",
                      str(raises.exception))

    def test_bad_untyped_multi_return(self):
        @njit(partial_typing=True)
        def foo(arr):
            c = arr.shape[0]
            for i in range(arr.ndim):
                c += i
            return c

        foo(10)

    def test_untyped_multi_paths(self):
        @njit(partial_typing=True)
        def foo(arr, n):
            c = 0
            for i in range(n):
                c += arr.shape[i]
            return c

        arr = np.ones((1, 2, 3))
        # normal operation
        self.assertEqual(foo(arr, arr.ndim), np.asarray(arr.shape).sum())
        # invalid type but ok
        self.assertEqual(foo(5, 0), 0)
        # invalid type but fail
        with self.assertRaises(TypeError) as raises:
            foo(5, 1)
        self.assertIn("Unknown attribute 'shape' of type int64",
                      str(raises.exception))


if __name__ == '__main__':
    unittest.main()
