import subprocess as subp
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

def main():
    Path("testlogs/passed").mkdir(parents=True, exist_ok=True)
    Path("testlogs/failed").mkdir(parents=True, exist_ok=True)

    listing = subp.check_output("python runtests.py --list", shell=True, encoding='utf8')
    tests = []
    for line in listing.splitlines():
        if line.startswith("numba.tests"):
            tests.append(line)
        else:
            print(line)

    groups = defaultdict(list)
    for t in tests:
        comps = t.split('.')
        k = ".".join(comps[:4])
        groups[k].append(t)

    def runner(testgroup):
        cmdargs = " ".join(groups[testgroup])
        cmd = f"python runtests.py -m=1 -vb {cmdargs}"
        print("RUNNING", testgroup)
        try:
            output = subp.check_output(cmd.split(), stderr=subp.STDOUT, stdin=subp.DEVNULL)
        except subp.CalledProcessError as e:
            output = e.output
            status = "failed"
        else:
            status = "passed"

        with open(f"testlogs/{status}/{testgroup}.log", "wb") as fout:
            fout.write(output)

        return testgroup

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as pool:
        futures = [pool.submit(runner, testgroup) for testgroup in groups]
        for future in as_completed(futures):
            print("DONE", future.result())
    print("END")


if __name__ == "__main__":
    main()
