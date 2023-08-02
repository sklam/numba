"""
Tests specific for the RVSDG-frontend
"""
import unittest

from numba import njit
from numba.tests.support import (
    TestCase,
    skip_unless_rvsdg_frontend_enabled,
)


@skip_unless_rvsdg_frontend_enabled
class TestRVSDGFrontend(TestCase):
    """Test RVSDG-frontend specific problems"""

    # Test join-return series exercise problem related to RETURN_VALUE
    # bytecode not unified by numba-rvsdg
    def test_join_returns_1(self):
        @njit
        def udt(x):
            if x:
                return 1
            else:
                return 2

        self.assertEqual(udt(True), udt.py_func(True))
        self.assertEqual(udt(False), udt.py_func(False))

    def test_join_returns_2(self):
        @njit
        def udt(x):
            if x:
                pass
            else:
                return 2

        for x in [True, False]:
            self.assertEqual(udt(x), udt.py_func(x))

    def test_join_returns_3(self):
        @njit
        def udt(x):
            if x > 0:
                if x > 2:
                    return 1
            else:
                return 2
            return 3

        for x in [0, 1, 2]:
            self.assertEqual(udt(x), udt.py_func(x))


if __name__ == "__main__":
    unittest.main()
