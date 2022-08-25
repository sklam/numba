from numba import njit


@njit(myflag="show_me")
def foo(x):
    return x


foo(100)