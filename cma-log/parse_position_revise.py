#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import csv
from collections import defaultdict

def parse_floats_from_line(line):
    return [float(x) for x in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line)]

def parse_ints_from_line(line):
    return [int(x) for x in re.findall(r'-?\d+', line)]

def is_matrix_row(line):
    # 视为矩阵行的条件：含至少一个数字，并且没有字母或冒号 (粗略判定)
    return bool(re.search(r'-?\d+', line)) and (':' not in line) and (re.search(r'[A-Za-z]', line) is None)

def process_case(case_text, case_idx):
    """
    解析单个 case 文本，返回结构：
    {
      '>': { 'matrices': [ (per_score_list, flat_matrix_list), ... ], 'max_len': int },
      '<': { similar }
    }
    """
    lines = case_text.splitlines()
    i = 0
    n = len(lines)

    results = {'>': [], '<': []}  # each element: (stage_finds_per_score(list18), flat_matrix(list of ints))
    while i < n:
        line = lines[i].strip()
        if line.startswith('prox_score_after-prox_score_before>0'):
            sign = '>'
            i += 1
        elif line.startswith('prox_score_after-prox_score_before<0'):
            sign = '<'
            i += 1
        else:
            i += 1
            continue

        # After we found a sign marker，向下查找 stage_finds_per_score 和 position_revise
        per_score = None
        matrix = None

        # search until we hit either another prox marker or next case end
        j = i
        while j < n:
            l = lines[j].strip()
            # stop if next prox marker (start a new block)
            if l.startswith('prox_score_after-prox_score_before>0') or l.startswith('prox_score_after-prox_score_before<0') or l.startswith('-----------next case----------'):
                break
            if l.startswith('stage_finds_per_score:'):
                # parse floats in the remainder of this line and possibly following lines until next label
                # usually it's on same line
                # attempt to parse numbers on this line
                rest = lines[j][lines[j].find(':')+1:]
                nums = parse_floats_from_line(rest)
                # if not enough, check next lines until we've collected 18 or hit a blank/other label
                k = j + 1
                while len(nums) < 18 and k < n:
                    nxt = lines[k].strip()
                    if nxt == '' or ':' in nxt:
                        break
                    extra = parse_floats_from_line(nxt)
                    if not extra:
                        break
                    nums.extend(extra)
                    k += 1
                per_score = nums[:18] if len(nums) >= 18 else nums
                j = k
                continue

            if l.startswith('position_revise:'):
                # collect following rows that look like matrix rows
                k = j + 1
                rows = []
                while k < n:
                    rowline = lines[k].strip()
                    # stop collection if line empty or looks like a label or next prox marker or next case
                    if rowline == '':
                        # could be an empty line between rows; but many examples have blank line then rows.
                        # if next line contains digits we continue; else break
                        # peek ahead
                        if k+1 < n and is_matrix_row(lines[k+1]):
                            k += 1
                            continue
                        else:
                            break
                    if rowline.startswith('prox_score_after-prox_score_before') or rowline.startswith('stage_') or rowline.startswith('prox_score_before') or rowline.startswith('-----------next case----------'):
                        break
                    if is_matrix_row(rowline):
                        ints = parse_ints_from_line(rowline)
                        if ints:
                            rows.append(ints)
                            k += 1
                            continue
                        else:
                            break
                    else:
                        break
                # flatten rows row-major
                flat = []
                for r in rows:
                    flat.extend(r)
                matrix = flat
                j = k
                continue

            j += 1

        # move i to j (so outer loop continues after processed block)
        i = j

        # Only accept this block if we have both per_score (18) and a matrix (non-empty)
        if per_score is not None and len(per_score) >= 18 and matrix:
            results[sign].append((per_score[:18], matrix))
        else:
            # 不满足要求则忽略（题目要求：只要有这两个标记出现之后的这些数据，没有这两个标记的话出现的类似的数据不算数）
            # 如果缺少某一项就跳过
            pass

    # compute max lengths per sign
    meta = {}
    for s in ['>', '<']:
        maxlen = 0
        for per_score, mat in results[s]:
            if len(mat) > maxlen:
                maxlen = len(mat)
        meta[s] = {'items': results[s], 'max_len': maxlen}
    return meta

def aggregate_case(meta):
    """
    meta: output from process_case
    returns aggregate dictionaries:
      counts[sign][pos_index][value] -> count
      sums[sign][pos_index][value] -> sumscore
      max_len per sign
    """
    counts = {'>': defaultdict(lambda: defaultdict(int)), '<': defaultdict(lambda: defaultdict(int))}
    sums = {'>': defaultdict(lambda: defaultdict(float)), '<': defaultdict(lambda: defaultdict(float))}
    max_len = {'>': meta['>']['max_len'], '<': meta['<']['max_len']}

    for sign in ['>', '<']:
        for per_score, mat in meta[sign]['items']:
            # per_score: list of 18 floats
            # mat: flat list of ints
            for idx, val in enumerate(mat):
                # idx is 0-based position index
                counts[sign][idx][val] += 1
                if val != -1:
                    # val should be in 0..17; check
                    if 0 <= val < len(per_score):
                        sums[sign][idx][val] += per_score[val]
                    else:
                        # 超界则忽略
                        pass
    return counts, sums, max_len

def write_report_and_csv(all_cases_aggregates, out_txt_path):
    """
    all_cases_aggregates: list of tuples (case_idx, counts, sums, max_len)
    写文本报告到 out_txt_path，CSV 输出到同名 .csv
    CSV 列: case, position (1-based), value, count_gt0, sum_gt0, count_lt0, sum_lt0
    """
    txt_lines = []
    csv_rows = []
    for case_idx, counts, sums, max_len in all_cases_aggregates:
        txt_lines.append(f"-----------next case {case_idx}----------")
        for sign_label, sign_name in [('>', 'prox_score_after-prox_score_before>0'), ('<', 'prox_score_after-prox_score_before<0')]:
            txt_lines.append(sign_name)
            txt_lines.append(f"最长数组长度：{max_len[sign_label]}")
            txt_lines.append("每个位置各个数字出现次数")
            # positions 1..max_len[sign_label]
            for pos in range(max_len[sign_label]):
                # build tuples for values -1..17
                tup_strs = []
                for v in range(-1, 18):
                    cnt = counts[sign_label].get(pos, {}).get(v, 0)
                    sum_sign = sums[sign_label].get(pos, {}).get(v, 0.0)
                    # we also want the other sign's sum as z in each tuple per你的模板
                    other_sign = '<' if sign_label == '>' else '>'
                    other_sum = sums[other_sign].get(pos, {}).get(v, 0.0)
                    # tuple format: (value,x,y,z) where x=count, y=sum_for_this_sign, z=sum_for_other_sign
                    tup_strs.append(f"({v},{cnt},{format(sum_sign, '.6f')},{format(other_sum, '.6f')})")
                    # fill csv rows (we'll produce combined csv showing both signs counts/sums)
                txt_lines.append(f"位置 {pos+1}: " + ",".join(tup_strs))
            txt_lines.append("")  # blank line

        # prepare csv rows per position and value; combine counts and sums for both signs
        max_pos = max(max_len['>'], max_len['<'])
        for pos in range(max_pos):
            for v in range(-1, 18):
                cnt_gt = counts['>'].get(pos, {}).get(v, 0)
                sum_gt = sums['>'].get(pos, {}).get(v, 0.0)
                cnt_lt = counts['<'].get(pos, {}).get(v, 0)
                sum_lt = sums['<'].get(pos, {}).get(v, 0.0)
                csv_rows.append({
                    'case': case_idx,
                    'position': pos+1,
                    'value': v,
                    'count_gt0': cnt_gt,
                    'sum_gt0': f"{sum_gt:.6f}",
                    'count_lt0': cnt_lt,
                    'sum_lt0': f"{sum_lt:.6f}"
                })

    # write text file
    with open(out_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(txt_lines))

    # write csv file (same name but .csv)
    csv_path = re.sub(r'\.[^.]+$', '.csv', out_txt_path)
    if csv_path == out_txt_path:
        csv_path = out_txt_path + '.csv'

    fieldnames = ['case', 'position', 'value', 'count_gt0', 'sum_gt0', 'count_lt0', 'sum_lt0']
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    return csv_path

def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_position_revise.py input.txt output.txt")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # split cases by delimiter
    parts = re.split(r'(?m)^-{11,}next case-{10,}$|(?m)^-----------next case----------$', content)
    # 上面正则尽量兼容不同数量的 '-'。如果分割后首项为空或内容非 case，可把非空项作为 case。
    cases = []
    for p in parts:
        if p.strip() != '':
            cases.append(p)

    all_cases_aggregates = []
    for idx, case_text in enumerate(cases, start=1):
        meta = process_case(case_text, idx)
        counts, sums, max_len = aggregate_case(meta)
        all_cases_aggregates.append((idx, counts, sums, max_len))

    csv_path = write_report_and_csv(all_cases_aggregates, out_path)
    print(f"写入报告: {out_path}")
    print(f"同时生成 CSV 文件: {csv_path}")

if __name__ == '__main__':
    main()
