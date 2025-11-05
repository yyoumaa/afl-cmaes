#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage_score_plot.py

读取日志文件中每组 prox_score_after-prox_score_before>0 后
第一个出现的 stage_finds_score_all: 下一行数据，
提取前 16 个维度（保留两位小数），绘制折线图并保存。
可以通过 -n 参数限制绘制前 N 组数据。
"""

import re
import argparse
import itertools
import matplotlib.pyplot as plt
from itertools import cycle

colors = [f'C{i}' for i in range(10)]
markers = ['o', 's']
# 先展开成 16 对 (color, marker)
combos = list(itertools.product(colors, markers))  # 长度 16

def parse_file(filename):
    """
    扫描文件，提取每次 prox_score 后第一个 stage_finds_score_all: 对应的数值行，
    返回一个 M x 16 的列表（M 组数据，每组 16 维）。
    """
    data,biaojis= [],[]
    group_idx = 0          # 已完成的组计数
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
         # ---------- biaoji 处理 ----------
        if 'biaoji' == line or line == 'biaoji':
            # 把“最近一次已完成的组”记作标记
            if group_idx > 0:
                biaojis.append(group_idx)
            i += 1
            continue

        if 'prox_score_after-prox_score_before>0' in lines[i]:
            # 往后找第一个 stage_finds_score_all:
            j = i + 1
            while j < len(lines):
                if 'stage_finds_score_all:' in lines[j]:
                    # 读取下一行
                    if j + 1 < len(lines):
                        nums = re.findall(r'[-+]?\d*\.\d+|\d+', lines[j+1])
                        # 转 float，保留两位，去掉最后两个全 0 维度
                        vals = [round(float(x), 2) for x in nums[:-2]]
                        if len(vals) >= 16:
                            data.append(vals[:16])
                            group_idx += 1
                    i = j
                    break
                j += 1
        i += 1

    return data,biaojis


def plot_data(data, biaojis,output_path):
    """
    data: list of length M, 每项是一个长度 16 的 list
    """
    if not data:
        print("⚠️ 没有在文件中找到符合条件的数据。")
        return

    num_groups = len(data)
    x = list(range(1, num_groups + 1))

    plt.figure(figsize=(10, 6))
    for dim in range(16):
        y = [group[dim] for group in data]
        color = colors[dim % len(colors)]
        marker = markers[dim // len(colors)]  # 0–9 用 'o'，10–15 用 's'
        plt.plot(
            x, y,
            label=f'Dim {dim+1}',
            color=color,
            marker=marker,
            markersize=3,
            linewidth=1
        )

      # 画标记竖线
    for m in biaojis:
        plt.axvline(x=m, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)


    plt.xlabel('Group Index')
    plt.ylabel('Score')
    plt.title('Stage Finds Score by Group Index')
    plt.legend(ncol=2, fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ 折线图已保存至：{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="绘制 stage_finds_score_all 各维度随组序号的折线图")
    parser.add_argument('input_file',
                        help="输入的文本文件路径")
    parser.add_argument('-o', '--output', default='plot.png',
                        help="输出图片文件名，默认 plot.png")
    parser.add_argument('-n', '--max-groups', type=int, default=None,
                        help="仅绘制前 N 组数据（默认全部）")
    args = parser.parse_args()

    data,biaojis = parse_file(args.input_file)
    # 如果指定了最大组数，则截取前 N 组
    if args.max_groups is not None:
        data = data[:args.max_groups]

    plot_data(data,biaojis, args.output)

if __name__ == '__main__':
    main()
