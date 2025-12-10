#!/usr/bin/env python3
"""
Flatten multi-line 'operator_revise_array' blocks into a single line.

Usage:
  python3 flatten_operator_array.py <input_file> [output_file]

- If output_file is omitted, the script will write to <input_file>.flat next to the input.
- The script preserves all other lines unchanged.
- A block starts with a line beginning with 'operator_revise_array' and continues with one or more lines
  that contain only numbers separated by spaces (possibly with leading spaces) until a separator line
  such as '------------------' or a new header like 'output_vector_after' or another non-numeric content.
"""

import sys
import os
import re
from typing import List

HEADER_OP = 'operator_revise_array'
SEPARATORS = {
    '------------------',
}
NEXT_HEADERS = {
    'output_vector_after',
    'output_vector_before',
    'operator_revise_array',  # another header begins
}

number_line_re = re.compile(r"^\s*(?:-?\d+)(?:\s+-?\d+)*\s*$")


def is_number_line(s: str) -> bool:
    """Return True if the line consists of integers separated by spaces (with optional leading spaces)."""
    return bool(number_line_re.match(s))


def flatten_file(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].rstrip('\n')

        if line.startswith(HEADER_OP):
            # Emit the header line as-is (normalized single space after header)
            header = HEADER_OP
            out.append(header)

            # Collect subsequent numeric lines
            nums: List[str] = []
            i += 1
            while i < n:
                peek = lines[i].rstrip('\n')
                if peek in SEPARATORS:
                    break
                # Stop if next header or non-numeric content appears
                if any(peek.startswith(h) for h in NEXT_HEADERS) or not is_number_line(peek):
                    break
                # Split and collect numbers
                nums.extend(peek.strip().split())
                i += 1

            # Write flattened single line with numbers
            if nums:
                out.append(f" { ' '.join(nums) }")
            else:
                # No numbers found; keep a blank after header for clarity
                out.append(" ")

            # Do not increment i here if we broke on a separator/next header; the loop will handle it
            continue

        # Not an operator header: keep the line unchanged
        out.append(line)
        i += 1

    # Re-append trailing newline characters uniformly
    return [l + '\n' for l in out]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 flatten_operator_array.py <input_file> [output_file]", file=sys.stderr)
        sys.exit(2)

    inp = sys.argv[1]
    if not os.path.isfile(inp):
        print(f"Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) >= 3:
        outp = sys.argv[2]
    else:
        base, ext = os.path.splitext(inp)
        outp = base + '.flat' + (ext if ext else '')

    with open(inp, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    flattened = flatten_file(lines)

    with open(outp, 'w', encoding='utf-8') as f:
        f.writelines(flattened)

    print(f"Written flattened output to: {outp}")


if __name__ == '__main__':
    main()
