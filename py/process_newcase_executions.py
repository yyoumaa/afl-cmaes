#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理 fuzzing 日志文件，按 newcase 分割，统计每次执行后的位置-算子使用情况。

输入格式：
- newcase 标记新的 case（连续 newcase 合并）
- ------------------ 分隔不同的执行
- 每次执行包含 output_vector_after + reward 和 operator_revise_array + 算子列表

输出：
- 目录下每个文件对应一个 newcase
- 每个文件包含每次执行后的统计信息和选择的算子数组
"""

import argparse
import collections
import os
from typing import Dict, List, Tuple, Optional


def parse_log_file(log_path: str) -> Dict[int, List[List[Tuple[int, int]]]]:
    """
    解析日志文件，按 newcase 和 execution 组织数据。
    
    Returns:
        {
            newcase_idx: [
                [(op_id, pos), ...],  # 第一次执行
                [(op_id, pos), ...],  # 第二次执行
                ...
            ],
            ...
        }
    """
    result: Dict[int, List[List[Tuple[int, int]]]] = {}
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    i = 0
    cur_case = 0
    last_was_newcase = False
    current_execution: List[Tuple[int, int]] = []
    in_operator_array = False
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 处理 newcase
        if line == "newcase":
            # 连续 newcase 合并
            if not last_was_newcase:
                cur_case += 1
                result.setdefault(cur_case, [])
            last_was_newcase = True
            i += 1
            continue
        else:
            if line != "":
                last_was_newcase = False
        
        # 处理执行分隔符
        if line == "------------------":
            # 当前执行结束，保存
            if current_execution:
                result.setdefault(cur_case, []).append(current_execution)
                current_execution = []
            in_operator_array = False
            i += 1
            continue
        
        # 处理 output_vector_after（新执行开始）
        if line == "output_vector_after":
            # 保存上一个执行（如果有）
            if current_execution:
                result.setdefault(cur_case, []).append(current_execution)
            current_execution = []
            in_operator_array = False
            i += 1
            continue
        
        # 处理 operator_revise_array
        if line == "operator_revise_array":
            if cur_case == 0:
                i += 1
                continue
            in_operator_array = True
            i += 1
            continue
        
        # 解析算子行
        if in_operator_array and cur_case > 0:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    op_id = int(parts[0])
                    pos = int(parts[1])
                    # 忽略无效值 4294967295
                    if pos != 4294967295:
                        current_execution.append((op_id, pos))
                except ValueError:
                    pass
        
        i += 1
    
    # 处理文件末尾的执行
    if current_execution:
        result.setdefault(cur_case, []).append(current_execution)
    
    return result


def get_top_n_operators(counter: collections.Counter, n: int) -> List[int]:
    """
    获取频率前 n 的算子（如果频率相同，按算子ID排序）。
    
    Returns:
        算子ID列表，按频率降序，频率相同时按ID升序
    """
    if not counter:
        return []
    
    # 按 (频率, -算子ID) 排序，频率高的在前，频率相同时ID小的在前
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return [op_id for op_id, _ in sorted_items[:n]]


def find_max_connected_region(
    positions: List[int],
    pos_candidates: Dict[int, List[int]],
    n: int
) -> List[Optional[int]]:
    """
    找到能形成最大连通区域的算子选择序列。
    
    算法思路（动态规划）：
    1. 对于每个位置，候选算子是前n个高频算子
    2. dp[i][op] = 选择算子 op 在位置 i 时，从位置 0 到 i 的：
       - 总连通区域长度（所有连续相同算子的块的总长度）
       - 当前连通块的长度（如果和前一个位置相同，则累加；否则重新开始）
    3. 选择总连通区域长度最大的方案
    
    Args:
        positions: 所有有数据的位置列表（已排序）
        pos_candidates: {位置: [候选算子列表]}
        n: 每个位置考虑的前n个高频算子
    
    Returns:
        每个位置选择的算子，如果没有数据则为 None
    """
    if not positions:
        return []
    
    num_positions = len(positions)
    
    # dp[i][op] = (总连通长度, 当前连通块长度, 前一个位置选择的算子)
    # 总连通长度 = 所有连续相同算子的块的总长度
    # 当前连通块长度 = 如果和前一个位置相同，则累加；否则重新开始
    dp: Dict[Tuple[int, int], Tuple[int, int, Optional[int]]] = {}
    
    # 初始化第一个位置
    first_pos = positions[0]
    for op in pos_candidates.get(first_pos, []):
        dp[(0, op)] = (1, 1, None)  # 总长度1，当前块长度1
    
    # 动态规划
    for i in range(1, num_positions):
        pos = positions[i]
        candidates = pos_candidates.get(pos, [])
        
        for op in candidates:
            best_total = 0
            best_current = 1
            best_prev_op = None
            
            # 尝试从前一个位置的所有候选转移
            prev_pos = positions[i - 1]
            prev_candidates = pos_candidates.get(prev_pos, [])
            
            for prev_op in prev_candidates:
                if (i - 1, prev_op) not in dp:
                    continue
                
                prev_total, prev_current, _ = dp[(i - 1, prev_op)]
                
                # 如果前一个位置也选择了 op，当前连通块可以扩展
                if prev_op == op:
                    new_current = prev_current + 1
                    # 总长度增加1（因为当前块扩展了1）
                    new_total = prev_total + 1
                else:
                    # 否则，当前位置开始新的连通区域
                    new_current = 1
                    # 总长度增加1（新增一个长度为1的块）
                    new_total = prev_total + 1
                
                if new_total > best_total:
                    best_total = new_total
                    best_current = new_current
                    best_prev_op = prev_op
            
            dp[(i, op)] = (best_total, best_current, best_prev_op)
    
    # 回溯找到最优路径
    if not dp:
        return [None] * num_positions
    
    # 找到最后一个位置的最佳选择（总连通长度最大）
    last_idx = num_positions - 1
    last_pos = positions[last_idx]
    best_last_op = None
    best_final_total = 0
    
    for op in pos_candidates.get(last_pos, []):
        if (last_idx, op) in dp:
            total, _, _ = dp[(last_idx, op)]
            if total > best_final_total:
                best_final_total = total
                best_last_op = op
    
    if best_last_op is None:
        return [None] * num_positions
    
    # 回溯构建路径
    result = [None] * num_positions
    current_op = best_last_op
    current_idx = last_idx
    
    while current_idx >= 0:
        result[current_idx] = current_op
        if current_idx == 0:
            break
        _, _, prev_op = dp[(current_idx, current_op)]
        if prev_op is None:
            break
        current_op = prev_op
        current_idx -= 1
    
    return result


def format_execution_stats(
    cumulative_stats: Dict[int, collections.Counter],
    max_pos: Optional[int] = None
) -> List[str]:
    """
    格式化执行统计信息。
    
    Returns:
        格式化的行列表，如 ["3:10(2)", "4:5(1),7(1)", ...]
    """
    lines = []
    
    if not cumulative_stats:
        return lines
    
    max_observed_pos = max(cumulative_stats.keys())
    upper = max_observed_pos if max_pos is None else min(max_observed_pos, max_pos)
    
    for pos in range(0, upper + 1):
        if pos not in cumulative_stats:
            continue
        counter = cumulative_stats[pos]
        parts = []
        for op_id in sorted(counter.keys()):
            parts.append(f"{op_id}({counter[op_id]})")
        ops_str = ",".join(parts)
        lines.append(f"{pos}:{ops_str}")
    
    return lines


def generate_operator_array(
    cumulative_stats: Dict[int, collections.Counter],
    n: int,
    max_pos: Optional[int] = None
) -> List[Optional[int]]:
    """
    生成算子选择数组。
    
    Args:
        cumulative_stats: 累积的位置-算子统计
        n: 每个位置考虑的前n个高频算子
        max_pos: 最大位置（如果为None，使用观察到的最大位置）
    
    Returns:
        每个位置选择的算子，没有数据的位置为 None（输出时用 '-' 表示）
    """
    if not cumulative_stats:
        return []
    
    max_observed_pos = max(cumulative_stats.keys())
    upper = max_observed_pos if max_pos is None else min(max_observed_pos, max_pos)
    
    # 收集所有有数据的位置
    positions = [pos for pos in range(0, upper + 1) if pos in cumulative_stats]
    
    if not positions:
        return []
    
    # 为每个位置获取前n个候选算子
    pos_candidates = {}
    for pos in positions:
        counter = cumulative_stats[pos]
        candidates = get_top_n_operators(counter, n)
        if candidates:
            pos_candidates[pos] = candidates
    
    # 使用动态规划找到最大连通区域
    selected_ops = find_max_connected_region(positions, pos_candidates, n)
    
    # 构建完整数组（包括没有数据的位置）
    result = []
    for pos in range(0, upper + 1):
        if pos in positions:
            idx = positions.index(pos)
            result.append(selected_ops[idx])
        else:
            result.append(None)
    
    return result


def format_operator_array(arr: List[Optional[int]]) -> str:
    """
    格式化算子数组为字符串，None 用 '-' 表示。
    """
    return "[" + ",".join(str(x) if x is not None else "-" for x in arr) + "]"


def process_newcase(
    case_idx: int,
    executions: List[List[Tuple[int, int]]],
    output_dir: str,
    n: int,
    max_pos: Optional[int] = None
) -> None:
    """
    处理一个 newcase 的所有执行，生成输出文件。
    """
    output_lines = [f"newcase{case_idx}"]
    
    # 累积统计（跨所有执行）
    cumulative_stats: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    
    # 处理每次执行
    for exec_idx, execution in enumerate(executions):
        # 更新累积统计
        for op_id, pos in execution:
            cumulative_stats[pos][op_id] += 1
        
        # 输出分隔符
        output_lines.append("------------------")
        
        # 输出当前累积统计
        stats_lines = format_execution_stats(cumulative_stats, max_pos)
        output_lines.extend(stats_lines)
        
        # 生成并输出算子数组
        op_array = generate_operator_array(cumulative_stats, n, max_pos)
        output_lines.append(format_operator_array(op_array))
    
    # 写入文件
    output_file = os.path.join(output_dir, f"newcase{case_idx}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="处理 fuzzing 日志，按 newcase 分割并统计每次执行后的位置-算子使用情况"
    )
    parser.add_argument(
        "input_file",
        help="输入日志文件路径（如 /cma-log/output_for_ana.txt）"
    )
    parser.add_argument(
        "output_dir",
        help="输出目录路径"
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=2,
        help="每个位置考虑的前n个高频算子（默认: 2）"
    )
    parser.add_argument(
        "--max-pos",
        type=int,
        default=None,
        help="最大位置索引（默认: 使用观察到的最大位置）"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析日志文件
    print(f"正在解析日志文件: {args.input_file}")
    parsed_data = parse_log_file(args.input_file)
    
    print(f"找到 {len(parsed_data)} 个 newcase")
    
    # 处理每个 newcase
    for case_idx in sorted(parsed_data.keys()):
        executions = parsed_data[case_idx]
        print(f"处理 newcase{case_idx}，包含 {len(executions)} 次执行")
        process_newcase(case_idx, executions, args.output_dir, args.top_n, args.max_pos)
    
    print(f"处理完成！输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
