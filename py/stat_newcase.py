#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

def parse_file(filepath):
    # stats[newcase][position][operator] = count
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    current_case = None
    in_operator_block = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("newcase"):
                current_case = line
                in_operator_block = False
                continue

            if line.startswith("------------------"):
                in_operator_block = False
                continue

            if line == "operator_revise_array":
                in_operator_block = True
                continue

            if line == "output_vector_after" or line.isdigit():
                continue

            if in_operator_block and current_case:
                try:
                    operator, position = map(int, line.split())
                    stats[current_case][position][operator] += 1
                except ValueError:
                    pass

    return stats


def write_result(stats, out_path, max_position=2000, top_n=3):
    with open(out_path, "w", encoding="utf-8") as f:
        for case in stats:
            f.write(case + "\n")

            # 原始统计
            for pos in range(0, max_position + 1):
                if pos in stats[case]:
                    ops = stats[case][pos]
                    op_str = ",".join(
                        f"{op}({cnt})"
                        for op, cnt in sorted(ops.items())
                    )
                    f.write(f"{pos}:{op_str}\n")

            # Top-N 展示
            f.write(f"\n[Top {top_n} operators per position]\n")
            for pos in sorted(stats[case].keys()):
                ops = stats[case][pos]

                # 按出现次数降序，次数相同按算子编号升序
                top_ops = sorted(
                    ops.items(),
                    key=lambda x: (-x[1], x[0])
                )[:top_n]

                op_list = ",".join(str(op) for op, _ in top_ops)
                f.write(f"({pos}):({op_list})\n")

            f.write("---------------------------------------------\n")


def main():
    if len(sys.argv) < 2:
        print("用法: python stat_newcase.py input.txt [output.txt] [top_n]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "stat_result.txt"
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    stats = parse_file(input_file)
    write_result(stats, output_file, top_n=top_n)

    print(f"统计完成，结果已写入: {output_file}")


if __name__ == "__main__":
    main()
