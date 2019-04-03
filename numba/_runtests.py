from __future__ import print_function

import json
import re
import sys
import trace


def _main(argv, **kwds):
    from numba.testing import run_tests
    # This helper function assumes the first element of argv
    # is the name of the calling program.
    # The 'main' API function is invoked in-process, and thus
    # will synthesize that name.

    if '--failed-first' in argv:
        # Failed first
        argv.remove('--failed-first')
        return _FailedFirstRunner().main(argv, kwds)
    elif '--last-failed' in argv:
        argv.remove('--last-failed')
        return _FailedFirstRunner(last_failed=True).main(argv, kwds)
    else:
        import time
        ignoredirs = [sys.prefix]
        tracer = trace.Trace(count=True, trace=False, ignoredirs=ignoredirs)
        ts = time.time()
        out = tracer.runfunc(run_tests, argv, defaultTest='numba.tests', **kwds)
        te = time.time()
        successful = out.wasSuccessful()
        if successful:
            tests = [a for a in argv[1:] if not a.startswith('-')]
            write_traced_results(tests, tracer.results(), te - ts)
        return successful
        # return run_tests(argv, defaultTest='numba.tests',
        #                  **kwds).wasSuccessful()


def write_traced_results(tests, covresults, exectime):
    import numba
    import os.path
    import time
    import pickle
    moddir = os.path.abspath(os.path.dirname(os.path.dirname(numba.__file__)))

    buf = []
    for (fpath, lineno), counts in covresults.counts.items():
        if fpath.startswith(moddir):
            fpath = os.path.relpath(fpath, start=moddir)
            buf.append((fpath, lineno, counts))

    os.makedirs('./report', exist_ok=True)
    with open('./report/cov{}'.format(time.time()), 'wb') as fout:
        pickle.dump({'tests': tests, 'trace': buf, 'exectime': exectime}, fout)


def main(*argv, **kwds):
    """keyword arguments are accepted for backward compatiblity only.
    See `numba.testing.run_tests()` documentation for details."""
    return _main(['<main>'] + list(argv), **kwds)


class _FailedFirstRunner(object):
    """
    Test Runner to handle the failed-first (--failed-first) option.
    """
    cache_filename = '.runtests_lastfailed'

    def __init__(self, last_failed=False):
        self.last_failed = last_failed

    def main(self, argv, kwds):
        from numba.testing import run_tests
        prog = argv[0]
        argv = argv[1:]
        flags = [a for a in argv if a.startswith('-')]

        all_tests, failed_tests = self.find_last_failed(argv)
        # Prepare tests to run
        if failed_tests:
            ft = "There were {} previously failed tests"
            print(ft.format(len(failed_tests)))
            remaing_tests = [t for t in all_tests
                             if t not in failed_tests]
            if self.last_failed:
                tests = list(failed_tests)
            else:
                tests = failed_tests + remaing_tests
        else:
            if self.last_failed:
                tests = []
            else:
                tests = list(all_tests)

        if not tests:
            print("No tests to run")
            return True
        # Run the testsuite
        print("Running {} tests".format(len(tests)))
        print('Flags', flags)
        result = run_tests([prog] + flags + tests, **kwds)
        if not self.last_failed:
            self.save_failed_tests(result, all_tests)
        return result.wasSuccessful()

    def save_failed_tests(self, result, all_tests):
        print("Saving failed tests to {}".format(self.cache_filename))
        cache = []
        # Find failed tests
        failed = set()
        for case in result.errors + result.failures:
            failed.add(case[0].id())
        # Build cache
        for t in all_tests:
            if t in failed:
                cache.append(t)
        # Write cache
        with open(self.cache_filename, 'w') as fout:
            json.dump(cache, fout)

    def find_last_failed(self, argv):
        from numba.tests.support import captured_output

        # Find all tests
        listargv = ['-l'] + [a for a in argv if not a.startswith('-')]
        with captured_output("stdout") as stream:
            main(*listargv)

            pat = re.compile(r"^(\w+\.)+\w+$")
            lines = stream.getvalue().splitlines()
        all_tests = [x for x in lines if pat.match(x) is not None]

        try:
            fobj = open(self.cache_filename)
        except IOError:
            failed_tests = []
        else:
            with fobj as fin:
                failed_tests = json.load(fin)
        return all_tests, failed_tests
