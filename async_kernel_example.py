from __future__ import print_function

import logging
import argparse

import numpy as np

from numba import hsa


def run(ntimes):

    @hsa.jit("int32[:], int32[:]")
    def add1_kernel(dst, src):
        """
        A simple kernel that set the destination array to dst[i] = (src[i] + 1)
        """
        i = hsa.get_global_id(0)
        if i < dst.size:
            dst[i] = src[i] + 1

    print('context info: %s' % hsa.get_context().agent)

    blksz = 256
    gridsz = 10**5
    nitems = blksz * gridsz

    arr = np.arange(nitems, dtype=np.int32)

    print('Make coarsegrain CPU array for source data')
    # Allocate a coarsegrain array on the CPU.  The resulting object is a
    # subclass of numpy array.
    coarse_arr = hsa.coarsegrain_array(shape=arr.shape, dtype=arr.dtype)
    coarse_arr[:] = arr  # initialize to *arr*

    print('Make coarsegrain CPU array for result data')
    coarse_res_arr = hsa.coarsegrain_array(shape=arr.shape, dtype=arr.dtype)
    coarse_res_arr[:] = 0  # zero-initialized

    print("Make stream to create async command pipeline")
    stream = hsa.stream()

    print('Make GPU result array')
    gpu_res_arr = hsa.device_array_like(coarse_arr)

    print('Make GPU source array and copy source data from CPU asynchronously')
    gpu_arr = hsa.to_device(coarse_arr, stream=stream)

    # Launch kernel *ntimes*
    for i in range(ntimes):
        print('Launch kernel: %d' % i)

        # launch kernel with *stream* asynchronously
        add1_kernel[gridsz, blksz, stream](gpu_res_arr, gpu_arr)

        # copy GPU result array to GPU source array asynchronously
        gpu_arr.copy_to_device(gpu_res_arr, stream=stream)

    print('Copy GPU result array back to the CPU result array asynchronously')
    gpu_res_arr.copy_to_host(coarse_res_arr, stream=stream)

    # The expected result
    expect = coarse_arr + ntimes

    # The following check is likely to fail because the async copy hasn't finish
    print("Is the result correct before sync: %s" % np.all(coarse_res_arr == expect))

    print("Synchronize")
    stream.synchronize()

    # The previous synchronization call has ensured all previous async copy
    # and async kernel launch have completed.
    # The following check MUST pass.
    print("Is the result correct after sync: %s" % np.all(coarse_res_arr == expect))


description = """
An example to demonstrate the HSA async copy and async kernel launches.

This script launches a sequence of async memory transfers and kernel launches.
The result is checked before and after the synchronization point to reveal
the effect of asynchronous calls.
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('ntimes', type=int, nargs='?', default=20,
                        help='number of kernel launches')
    parser.add_argument('--debug', dest='debug', action='store_const',
                        const=logging.DEBUG, default=None,
                        help='trigger debug logging to see driver API calls')

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    run(args.ntimes)
