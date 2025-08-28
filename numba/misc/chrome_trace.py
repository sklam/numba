"""

Command-line Interface
----------------------

This module can also be used as a command-line tool to combine Chrome trace
files:

    python -m numba.misc.chrome_trace combine <base_filename>

This will combine all files matching the pattern ``<base_filename>.*`` into a
single trace file and clean up the individual files.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import string
import time
from dataclasses import dataclass
from typing import NamedTuple, Optional, Generator


class _FormatPart(NamedTuple):
    """Represents a parsed part of a format string from
    string.Formatter.parse()."""

    literal_text: str
    field_name: Optional[str]
    format_spec: Optional[str]
    conversion: Optional[str]


@dataclass(frozen=True)
class ChromeTraceConfig:
    filename_pattern: str
    support_multiprocessing: bool
    parsed_parts: list[_FormatPart]

    @classmethod
    def parse(cls, usertext) -> ChromeTraceConfig:
        # Parse the format string to detect patterns and gather field names
        formatter = string.Formatter()
        parsed_parts = [
            _FormatPart(*part) for part in formatter.parse(usertext)
        ]
        field_names = [
            part.field_name for part in parsed_parts if part.field_name
        ]

        # Check if pattern has format fields
        has_patterns = bool(field_names)

        # For now, return basic config - can be extended based on needs
        return cls(
            filename_pattern=usertext,
            support_multiprocessing=has_patterns,
            parsed_parts=parsed_parts,
        )

    def __bool__(self):
        return bool(self.filename_pattern)

    def __str__(self):
        return self.apply()

    def apply(self) -> str:
        """Returns the filename after replacing the patterns.

        Supported patterns are:

        - "{pid}": for ``os.getpid()``
        - "{ts}": for ``time.time()``
        """
        # Check if the pattern has format fields
        if not any(p.field_name is not None for p in self.parsed_parts):
            return self.filename_pattern

        format_vars = {"pid": os.getpid(), "ts": time.time()}
        try:
            return self.filename_pattern.format(**format_vars)
        except KeyError as e:
            raise ValueError(
                f"Unsupported format field '{e.args[0]}' in pattern "
                f"'{self.filename_pattern}'. "
                f"Supported fields are: {', '.join(format_vars.keys())}"
            )

    def glob(self) -> Generator[pathlib.Path, None, None]:
        format_vars = {
            "pid": "*",
            "ts": "*",
        }
        pat = self.filename_pattern.format(**format_vars)
        return pathlib.Path(".").glob(pat)

    def combined(self) -> str:
        """Returns the filename for the combined output."""
        format_vars = {
            "pid": "pid",
            "ts": "ts",
        }
        return self.filename_pattern.format(**format_vars)


def main():
    """
    Command-line interface for combining Chrome trace files.

    Usage:
        python -m numba.core.event combine <base_filename>
    """
    parser = argparse.ArgumentParser(
        description="Combine chrome trace files", prog="numba.core.event"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    combine_parser = subparsers.add_parser(
        "combine", help="Combine multiple chrome trace files into one"
    )
    combine_parser.add_argument(
        "chrome_trace_file",
        help="Base filename for the chrome trace files to combine",
    )

    args = parser.parse_args()

    if args.command == "combine":
        ctc = ChromeTraceConfig.parse(args.chrome_trace_file)
        print(f"Combining chrome trace files: {ctc}")

        evs = []
        parts = list(ctc.glob())

        if not parts:
            print(f"No files found matching pattern: {ctc}")
            return

        print(f"Found {len(parts)} file(s) to combine:")
        for part in parts:
            print(f"  - {part}")

        print("\nReading and combining events...")
        for filepart in parts:
            print(f"Reading {filepart}...")
            with open(filepart) as fin:
                file_events = json.load(fin)
                evs.extend(file_events)
                print(f"  Added {len(file_events)} events from {filepart}")

        print(f"\nTotal events collected: {len(evs)}")
        output_filename = ctc.combined()
        print(f"Writing combined events to: {output_filename}")
        with open(output_filename, "w") as fout:
            json.dump(evs, fout)
        print(f"Successfully wrote {len(evs)} events to {output_filename}")

        print("\nCleaning up individual files...")
        for path in parts:
            print(f"Removing {path}...")
            path.unlink()

        print("\nCombine operation completed successfully!")
        print(f"Combined {len(parts)} files into {output_filename}")
    else:
        raise RuntimeError("unreachable")


if __name__ == "__main__":
    main()
