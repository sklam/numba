from __future__ import absolute_import, print_function, division

import numpy as np

from numba import unittest_support as unittest
from ..support import TestCase

from numba.npyufunc.deviceufunc import UFuncEngine


class TestUFuncEngine(TestCase):
    def test_scalar_output_explicit_output(self):
        ufe = UFuncEngine(types=(np.float64, np.float32, np.intp),
                          signature="(m, n),(p)->()")

        arr0 = np.arange(10).reshape(2, 5)
        arr1 = np.arange(10)
        out = np.zeros(1)

        args = arr0, arr1, out
        kwargs = {}

        ufe.schedule(args, kwargs)

        self.assertIs(arr0, ufe.inputs[0])
        self.assertIs(arr1, ufe.inputs[1])
        self.assertIs(out, ufe.outputs[0])

        self.assertEqual(len(ufe.inputs), 2)
        self.assertEqual(len(ufe.outputs), 1)

        self.assertEqual(ufe.symbols['m'], arr0.shape[0])
        self.assertEqual(ufe.symbols['n'], arr0.shape[1])
        self.assertEqual(ufe.symbols['p'], arr1.shape[0])

        self.assertEqual(set(ufe.symbols.keys()), set(['m', 'n', 'p']))

        self.assertEqual(len(ufe.inner_input_shapes), 2)
        self.assertEqual(len(ufe.outer_input_shapes), 2)

        self.assertEqual(ufe.inner_input_shapes[0], arr0.shape)
        self.assertEqual(ufe.inner_input_shapes[1], arr1.shape)

        self.assertEqual(ufe.outer_input_shapes[0], ())
        self.assertEqual(ufe.outer_input_shapes[1], ())

        self.assertEqual(len(ufe.inner_output_shapes), 1)
        self.assertEqual(len(ufe.outer_output_shapes), 1)

        self.assertEqual(ufe.inner_output_shapes[0], ())
        self.assertEqual(ufe.outer_output_shapes[0], (1,))

        self.assertEqual(ufe.loop_shape, (1,))

        self.assertEqual(len(ufe.broadcasted_shapes), 3)
        self.assertEqual(ufe.broadcasted_shapes[0], (1,) + arr0.shape)
        self.assertEqual(ufe.broadcasted_shapes[1], (1,) + arr1.shape)

        self.assertEqual(len(ufe.kernel_args), 3)
        self.assertEqual(ufe.kernel_args[0].shape, (1, 2, 5))
        self.assertEqual(ufe.kernel_args[1].shape, (1, 10))
        self.assertEqual(ufe.kernel_args[2].shape, (1,))

        self.assertEqual(ufe.kernel_args[0].dtype, np.float64)
        self.assertEqual(ufe.kernel_args[1].dtype, np.float32)
        self.assertEqual(ufe.kernel_args[2].dtype, np.intp)

        self.assertFalse(ufe.implicit_outputs)

    def test_scalar_output_implicit_output(self):
        ufe = UFuncEngine(types=(np.float64, np.float32, np.intp),
                          signature="(m, n),(p)->()")

        arr0 = np.arange(10).reshape(2, 5)
        arr1 = np.arange(10)

        args = arr0, arr1
        kwargs = {}

        ufe.schedule(args, kwargs)

        self.assertIs(arr0, ufe.inputs[0])
        self.assertIs(arr1, ufe.inputs[1])

        self.assertEqual(len(ufe.inputs), 2)
        self.assertEqual(len(ufe.outputs), 1)

        self.assertEqual(ufe.symbols['m'], arr0.shape[0])
        self.assertEqual(ufe.symbols['n'], arr0.shape[1])
        self.assertEqual(ufe.symbols['p'], arr1.shape[0])

        self.assertEqual(set(ufe.symbols.keys()), set(['m', 'n', 'p']))

        self.assertEqual(len(ufe.inner_input_shapes), 2)
        self.assertEqual(len(ufe.outer_input_shapes), 2)

        self.assertEqual(ufe.inner_input_shapes[0], arr0.shape)
        self.assertEqual(ufe.inner_input_shapes[1], arr1.shape)

        self.assertEqual(ufe.outer_input_shapes[0], ())
        self.assertEqual(ufe.outer_input_shapes[1], ())

        self.assertEqual(len(ufe.inner_output_shapes), 1)
        self.assertEqual(len(ufe.outer_output_shapes), 1)

        self.assertEqual(ufe.inner_output_shapes[0], ())
        self.assertEqual(ufe.outer_output_shapes[0], (1,))

        self.assertEqual(ufe.loop_shape, (1,))

        self.assertEqual(len(ufe.broadcasted_shapes), 3)
        self.assertEqual(ufe.broadcasted_shapes[0], (1,) + arr0.shape)
        self.assertEqual(ufe.broadcasted_shapes[1], (1,) + arr1.shape)

        self.assertEqual(len(ufe.kernel_args), 3)
        self.assertEqual(ufe.kernel_args[0].shape, (1, 2, 5))
        self.assertEqual(ufe.kernel_args[1].shape, (1, 10))
        self.assertEqual(ufe.kernel_args[2].shape, (1,))

        self.assertEqual(ufe.kernel_args[0].dtype, np.float64)
        self.assertEqual(ufe.kernel_args[1].dtype, np.float32)
        self.assertEqual(ufe.kernel_args[2].dtype, np.intp)

        self.assertTrue(ufe.implicit_outputs)

if __name__ == '__main__':
    unittest.main()
