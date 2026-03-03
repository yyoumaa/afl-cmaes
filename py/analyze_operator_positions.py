#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze operator usage per position for each `newcase`.

Input format (from /cma-log/output_for_ana.txt):

newcase
output_vector_after
<score>
operator_revise_array
<op_id> <pos>
<op_id> <pos>
...
------------------
... (more executions) ...

For each `newcase`, this script aggregates, for every position, how many
times each operator is used. The result format is:

newcase1
0:8(1)
1:8(2)
2:7(4),8(5)
...
---------------------------------------------
newcase2
...
"""

import argparse
import collections
from typing import Dict, List, Tuple


def parse_output_file(path: str) -> Dict[int, Dict[int, collections.Counter]]:
    """
    Parse the given log file.

    Returns:
        stats: dict
            {
              newcase_idx: {
                position: Counter({op_id: count, ...}),
                ...
              },
              ...
            }
    """
    stats: Dict[int, Dict[int, collections.Counter]] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    cur_case = 0
    last_was_newcase = False  # 合并连续的 newcase，视为同一个 case

    while i < len(lines):
        line = lines[i].strip()

        # Start of a new case
        if line == "newcase":
            # 连续出现的 newcase 视为同一个 case（和训练脚本保持一致）
            if not last_was_newcase:
                cur_case += 1
                stats.setdefault(cur_case, {})
            last_was_newcase = True
            i += 1
            continue
        else:
            # 一旦遇到非 newcase 行，就结束“连续 newcase”状态
            if line != "":
                last_was_newcase = False

        # We only care about blocks that have operator_revise_array
        if line == "operator_revise_array":
            if cur_case == 0:
                # operator_revise_array before any newcase -> ignore
                i += 1
                continue

            i += 1
            # Read until separator or a new block marker
            while i < len(lines):
                l = lines[i].strip()
                if l in ("------------------", "newcase", "output_vector_after", ""):
                    # end of this execution's operator list
                    break

                parts = l.split()
                if len(parts) >= 2:
                    try:
                        op_id = int(parts[0])
                        pos = int(parts[1])
                    except ValueError:
                        # Malformed line, skip
                        i += 1
                        continue

                    case_stats = stats.setdefault(cur_case, {})
                    pos_counter = case_stats.setdefault(pos, collections.Counter())
                    pos_counter[op_id] += 1

                i += 1
            # do not consume the separator / marker here; outer loop will handle it
            continue

        i += 1

    return stats


def format_stats(
    stats: Dict[int, Dict[int, collections.Counter]],
    max_pos: int = None,
) -> List[str]:
    """
    Format statistics into the desired textual representation.

    Args:
        stats: parsed statistics
        max_pos: if given, only positions <= max_pos are considered;
                 if None, use the maximum position observed for that newcase.

    Returns:
        List of lines (without trailing newlines).
    """
    lines: List[str] = []

    # Process newcases in order
    for case_idx in sorted(stats.keys()):
        case_stats = stats[case_idx]

        lines.append(f"newcase{case_idx}")

        if not case_stats:
            lines.append("---------------------------------------------")
            continue

        # Determine position range
        max_observed_pos = max(case_stats.keys())
        upper = max_observed_pos if max_pos is None else min(max_observed_pos, max_pos)

        # Iterate positions in ascending order, only emit those that have data
        for pos in range(0, upper + 1):
            if pos not in case_stats:
                continue
            counter = case_stats[pos]
            # Format as: pos:op1(count1),op2(count2)
            # operators sorted by id for stability
            parts: List[str] = []
            for op_id in sorted(counter.keys()):
                parts.append(f"{op_id}({counter[op_id]})")
            ops_str = ",".join(parts)
            lines.append(f"{pos}:{ops_str}")

        lines.append("---------------------------------------------")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze operator usage per position for each newcase "
        "from output_for_ana-style logs.",
    )
    parser.add_argument(
        "input_file",
        help="Path to the log file (e.g. /cma-log/output_for_ana.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--max-pos",
        type=int,
        default=None,
        help="Maximum position index to include (default: all observed positions).",
    )

    args = parser.parse_args()

    stats = parse_output_file(args.input_file)
    lines = format_stats(stats, max_pos=args.max_pos)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    else:
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()

