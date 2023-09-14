import re
from pprint import pprint
from pathlib import Path


REGEX_ERROR = re.compile(r"(errors|failures|unexpectedSuccess)=(\d+)")


def main():
    failed = {}
    timedout = {}

    for file in Path("testlogs/failed").glob("*.log"):
        with open(file, "r") as fin:
            body = fin.read().strip()
        print(file)
        lines = body.splitlines()
        cur_timedout = []
        for ln in reversed(lines):
            if ln.startswith("- "):
                cur_timedout.append(ln.strip("- '\""))
            else:
                break
        if cur_timedout:
            timedout[file] = cur_timedout
        else:
            last_line = lines[-1]
            if last_line.startswith("Parallel: "):
                last_line = lines[-2]
            assert last_line.startswith("FAILED"), file
            failed[file] = last_line

    pprint(failed)
    pprint(timedout)

    stats = {"timedout": sum(len(v) for v in timedout.values())}
    for _fp, msg in failed.items():
        gather_failed_stats(msg, stats)

    pprint(stats)

def gather_failed_stats(line, stats):
    for fail_type, count in REGEX_ERROR.findall(line):
        orig = stats.setdefault(fail_type, 0)
        stats[fail_type] = orig + int(count)



if __name__ == "__main__":
    main()
