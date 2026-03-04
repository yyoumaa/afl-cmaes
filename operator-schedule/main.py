"""
主要处理文件 尽量全部手写吧别用ai。。。
1. 读取数据文件
    1.1 直接读取数据文件 ✅
    1.2 改成共享内存的设计来读取数据文件 todo
2. 进行一阶段的处理和分析
    2.1 算子分类 ✅
    2.2 使用bandit来选择算子族 ✅
    2.3 使用afl的在线数据分析
3. 进行二阶段的处理和分析
    3.1 进行SMR区域划分
    3.2 使用bandit来选择算子和区域
"""
import os
import sys
import time
import argparse
import random
import logging
import pickle
import gzip
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import json
import time
from pathlib import Path


@dataclass
class Block:
    """一个区块 = 一个样本"""
    score: float
    pairs: List[Tuple[int, int]]  # [(op, pos), ...]

@dataclass
class Case:
    blocks: List[Block] = field(default_factory=list)

@dataclass
class ParsedData:
    cases: List[Case] = field(default_factory=list)

OPERATOR_FAM_NUM=5 #6个族

# 各算法每个 block 后的选择记录，key=算法名, value=list[int]
# show_xxx_log() 里往这里追加，plot_choices() 统一画图
choice_history: Dict[str, List[int]] = {}

# ---------- LinUCB Context Feature 配置 ----------
TARGET_DISTANCE = 190.0  # 到达目标点的总距离（后续根据实际情况修改）
REWARD_WINDOW_SIZE = 20  # reward momentum 计算的滑动窗口大小
UCB_C = 1.0 #UCB的探索系数
case_number_label=[] #在线模式下数据的case标记

# ---------- 算子分类映射 ----------
class OpFamily(IntEnum):
    """六个算子族，既可用数字（OpFamily.FAMILY_A == 0）也可用名称（OpFamily.FAMILY_A.name == 'FAMILY_A'）"""
    Numeric_Adjustment = 0  
    Noise_Perturbation = 1
    Structural_Modification = 2
    Local_Reuse = 3
    Chunk_Overwrite = 4
    # Structural_Resegmentation = 5

# 18个算子(0~17) -> 所属族
OP_TO_FAMILY: Dict[int, OpFamily] = {
    0:  OpFamily.Noise_Perturbation,
    1:  OpFamily.Numeric_Adjustment,
    2:  OpFamily.Numeric_Adjustment,
    3:  OpFamily.Numeric_Adjustment,
    4:  OpFamily.Numeric_Adjustment,
    5:  OpFamily.Numeric_Adjustment,
    6:  OpFamily.Numeric_Adjustment,
    7:  OpFamily.Numeric_Adjustment,
    8:  OpFamily.Numeric_Adjustment,
    9:  OpFamily.Numeric_Adjustment,
    10: OpFamily.Noise_Perturbation,
    11: OpFamily.Structural_Modification,
    12: OpFamily.Structural_Modification,
    13: OpFamily.Local_Reuse,
    14: OpFamily.Chunk_Overwrite,
    15: OpFamily.Chunk_Overwrite,
    16: OpFamily.Structural_Modification,
    # 17: OpFamily.Structural_Resegmentation
}

# 反向索引：族 -> 包含哪些算子id
FAMILY_TO_OPS: Dict[OpFamily, List[int]] = {}
for _op, _fam in OP_TO_FAMILY.items():
    FAMILY_TO_OPS.setdefault(_fam, []).append(_op)

"""
  # 数字方式                                            
  OpFamily.FAMILY_A == 0        # True                                      
  int(OpFamily.FAMILY_A)        # 0                              
                                                                            
  # 字符串方式
  OpFamily.FAMILY_A.name        # 'FAMILY_A'
  OpFamily['FAMILY_B']          # OpFamily.FAMILY_B

  # 查某个算子属于哪个族
  OP_TO_FAMILY[3]               # OpFamily.FAMILY_B
  OP_TO_FAMILY[3].name          # 'FAMILY_B'
  int(OP_TO_FAMILY[3])          # 1

  # 查某个族包含哪些算子
  FAMILY_TO_OPS[OpFamily.FAMILY_A]  # [0, 1, 2]"
"""


class LinUCBArm:
    """LinUCB算法中每个arm（族）的统计信息"""
    def __init__(self, d: int, alpha: float = 1.0):
        """
        Args:
            d: context特征维度
            alpha: 探索参数，类似UCB的c
        """
        self.d = d
        self.alpha = alpha

        # A = D^T D + I_d (d x d矩阵)
        self.A = np.identity(d)

        # b = D^T c (d维向量)
        self.b = np.zeros(d)

        # A的逆矩阵（缓存，避免重复计算）
        self.A_inv = np.identity(d)

    def get_ucb(self, context: np.ndarray) -> float:
        """
        计算该arm在给定context下的UCB值

        UCB = theta^T * x + alpha * sqrt(x^T * A^-1 * x)
        其中 theta = A^-1 * b
        """
        # theta = A^-1 * b
        theta = self.A_inv.dot(self.b)

        # 期望reward: theta^T * x
        expected_reward = theta.dot(context)

        # 不确定性: alpha * sqrt(x^T * A^-1 * x)
        uncertainty = self.alpha * np.sqrt(context.dot(self.A_inv).dot(context))

        ucb = expected_reward + uncertainty
        return ucb

    def update(self, context: np.ndarray, reward: float):
        """
        更新该arm的统计信息

        Args:
            context: 当前的context向量
            reward: 观察到的reward
        """
        # A = A + x * x^T
        self.A += np.outer(context, context)

        # b = b + r * x
        self.b += reward * context

        # 更新A的逆矩阵
        self.A_inv = np.linalg.inv(self.A)

class OnlineScheduler:
    """统一的在线调度器，支持三种算法模式"""

    def __init__(self, mode='greedy', batch_size=100, alpha=1.0):
        """
        mode: 'greedy', 'ucb', 'linucb'
        batch_size: 每批处理多少个blocks
        alpha: LinUCB的探索参数
        """
        self.mode = mode
        self.batch_size = batch_size
        self.alpha = alpha

        # 累积历史数据（所有算法共享）
        self.all_blocks = []  # 存储所有历史blocks
        self.block_count = 0

        # 算法特定的状态
        if mode == 'greedy':
            # 每个族：[选择次数, 累积reward]
            self.stats = [[0, 0.0] for _ in range(OPERATOR_FAM_NUM)]

        elif mode == 'ucb':
            # UCB需要：每个族的选择次数和平均reward
            self.counts = [0] * OPERATOR_FAM_NUM
            self.values = [0.0] * OPERATOR_FAM_NUM

        elif mode == 'linucb':
            # LinUCB需要：每个族一个arm
            self.arms = {fam: LinUCBArm(d=2, alpha=self.alpha)
                        for fam in range(OPERATOR_FAM_NUM)}
            self.recent_rewards = []  # 用于计算momentum

        else:
            raise ValueError(f"Unknown mode: {mode}")

        logging.info(f"OnlineScheduler initialized: mode={mode}, batch_size={batch_size}")


    def make_decision(self, block):
        if self.mode == 'greedy':
            # 算每个族的平均reward，选最高的（epsilon探索）
            epsilon = 0.01
            if random.random() < epsilon:
                return random.randint(0, OPERATOR_FAM_NUM - 1)
            avg = [s[1]/s[0] if s[0] > 0 else 0.0 for s in self.stats]
            return avg.index(max(avg))

        elif self.mode == 'ucb':
            t = sum(self.counts)
            c = UCB_C
            if t == 0:
                return random.randint(0, OPERATOR_FAM_NUM - 1)
            scores = []
            for i in range(OPERATOR_FAM_NUM):
                if self.counts[i] == 0:
                    scores.append(float('inf'))
                else:
                    avg = self.values[i] / self.counts[i]
                    bonus = math.sqrt(math.log(t) / self.counts[i])
                    scores.append(avg + c * bonus)
            return scores.index(max(scores))

        elif self.mode == 'linucb':
            distance_feat = compute_distance_feature(block.score)
            momentum_feat = compute_reward_momentum(self.recent_rewards, self.block_count)
            context = np.array([distance_feat, momentum_feat])
            ucb_values = {fam: self.arms[fam].get_ucb(context) for fam in range(OPERATOR_FAM_NUM)}
            return max(ucb_values, key=ucb_values.get)

    def update(self, block, chosen):
        # 找出block中实际使用的族（被动学习模式）
        families_in_block = set(fam for fam, pos in block.pairs)

        if self.mode == 'greedy':
            # 更新所有实际使用的族
            for fam in families_in_block:
                self.stats[fam][0] += 1
                self.stats[fam][1] += block.score

        elif self.mode == 'ucb':
            # 更新所有实际使用的族
            for fam in families_in_block:
                self.counts[fam] += 1
                self.values[fam] += block.score

        elif self.mode == 'linucb':
            distance_feat = compute_distance_feature(block.score)
            momentum_feat = compute_reward_momentum(self.recent_rewards, self.block_count)
            context = np.array([distance_feat, momentum_feat])
            # 更新所有实际使用的族
            for fam in families_in_block:
                self.arms[fam].update(context, block.score)
            self.recent_rewards.append(block.score)

        self.block_count += 1

    def process_batch(self, batch_blocks):
        decisions = []
        for block in batch_blocks:
            chosen = self.make_decision(block)
            decisions.append(chosen)
            self.update(block, chosen)
        logging.info(f"[{self.mode}] processed batch of {len(batch_blocks)} blocks, total={self.block_count}")
        return decisions





# ---------- 数据解析----------
def parse_source(source_path: str) -> ParsedData:
    """
    解析数据文件，返回 ParsedData 结构。

    文件格式：
    - 连续出现的 `newcase` 视为同一个 case 的开始（相邻多行合并为一次）。
    - 每个 case 下包含多个区块，形如：
        output_vector_after\n
        <score-number>\n
        operator_revise_array\n
        <op pos> [<op pos> ...]  # 可多行、且一行可出现多对数字\n
        ...\n
        ------------------
      score-number 为该区块的收益分数。
      op 为算子 id（0..16），pos 为位置（>=0），op/pos 为 4294967295 则忽略。

    返回 ParsedData，其中每个 Block 保存 score 和过滤后的有效 (op, pos) 对。
    """

    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"source file not found: {source_path}")

    KEY_NEWCASE = 'newcase'
    KEY_AFTER = 'output_vector_after'
    KEY_OP = 'operator_revise_array'
    SEP = '------------------'

    INVALID = 4294967295
    OP_DIM = 17
    POS_CAP = 500

    def parse_first_number(s: str) -> float:
        for tok in s.strip().split():
            try:
                return float(tok)
            except Exception:
                continue
        return None

    def parse_int_tokens(s: str) -> List[int]:
        vals: List[int] = []
        for tok in s.strip().split():
            try:
                vals.append(int(tok))
            except Exception:
                continue
        return vals

    def filter_pairs(raw_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """过滤无效对：哨兵值、负数、超出算子维度"""
        valid = []
        for op, pos in raw_pairs:
            try:
                if op == INVALID or pos == INVALID:
                    continue
                if op < 0 or pos < 0:
                    continue
                if op >= OP_DIM:
                    continue
                if pos >= POS_CAP:
                    continue
                valid.append((int(op), int(pos)))
            except Exception:
                continue
        return valid

    result = ParsedData()
    cur_case: Case = None
    last_token = None

    with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
        while True:
            line = f.readline()
            if not line:
                break
            s = line.rstrip('\n')
            if not s:
                continue

            # 处理 newcase（连续合并）
            if s.startswith(KEY_NEWCASE):
                if cur_case is None or last_token != KEY_NEWCASE:
                    if cur_case is not None and len(cur_case.blocks) > 0:
                        result.cases.append(cur_case)
                    cur_case = Case()
                last_token = KEY_NEWCASE
                continue

            # 若文件开头没有显式 newcase，也允许解析
            if cur_case is None:
                cur_case = Case()

            # 区块起点：output_vector_after
            if s.startswith(KEY_AFTER):
                tail = s[len(KEY_AFTER):].strip()
                score = parse_first_number(tail)
                if score is None:
                    pos = f.tell()
                    nline = f.readline()
                    if nline:
                        score = parse_first_number(nline.rstrip('\n'))
                    if score is None:
                        if nline:
                            f.seek(pos)
                        last_token = KEY_AFTER
                        continue

                # 向前扫描直到遇到 KEY_OP
                while True:
                    pos = f.tell()
                    nline = f.readline()
                    if not nline:
                        break
                    ns = nline.rstrip('\n')
                    if ns.startswith(KEY_OP):
                        pairs: List[Tuple[int, int]] = []
                        op_tail = ns[len(KEY_OP):]
                        ints = parse_int_tokens(op_tail)
                        for i in range(0, len(ints) - 1, 2):
                            pairs.append((ints[i], ints[i + 1]))
                        # 持续读取后续行
                        while True:
                            pos2 = f.tell()
                            l2 = f.readline()
                            if not l2:
                                break
                            s2 = l2.rstrip('\n')
                            if s2 == SEP or s2.startswith(KEY_AFTER) or s2.startswith(KEY_NEWCASE):
                                f.seek(pos2)
                                break
                            if s2.startswith(KEY_OP):
                                tail2 = s2[len(KEY_OP):]
                                ints = parse_int_tokens(tail2)
                                for i in range(0, len(ints) - 1, 2):
                                    pairs.append((ints[i], ints[i + 1]))
                                continue
                            ints = parse_int_tokens(s2)
                            for i in range(0, len(ints) - 1, 2):
                                pairs.append((ints[i], ints[i + 1]))

                        valid_pairs = filter_pairs(pairs)
                        cur_case.blocks.append(Block(score=float(score), pairs=valid_pairs))
                        break
                    elif ns == SEP or ns.startswith(KEY_AFTER) or ns.startswith(KEY_NEWCASE):
                        f.seek(pos)
                        break
                    else:
                        continue

                last_token = KEY_AFTER
                continue

            last_token = 'other'

    # 文件结束，追加最后一个 case
    if cur_case is not None and len(cur_case.blocks) > 0:
        result.cases.append(cur_case)

    total_blocks = sum(len(c.blocks) for c in result.cases)
    if total_blocks == 0:
        raise ValueError("No samples parsed from source file; please check the input format.")

    logging.info("parse_source: parsed %d cases, %d blocks total", len(result.cases), total_blocks)
    return result

def parse_source_v2(source_path: str) -> ParsedData:
    """
    解析新版数据格式：

    文件格式：
    - 使用 `--newcase--` 分割多个 case（连续多个视为同一个 case 起点）
    - 每行代表一个 block：
        score op1 pos1 op2 pos2 ...

    返回 ParsedData，其中每个 Block 保存：
        - score (float)
        - 过滤后的有效 (op, pos) 对
    """

    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"source file not found: {source_path}")

    KEY_NEWCASE = '--newcase--'
    INVALID = 4294967295
    OP_DIM = 17
    POS_CAP = 500

    def parse_tokens(line: str) -> List[float]:
        vals = []
        for tok in line.strip().split():
            try:
                vals.append(float(tok))
            except Exception:
                continue
        return vals

    def filter_pairs(raw_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """过滤无效对"""
        valid = []
        for op, pos in raw_pairs:
            try:
                if op == INVALID or pos == INVALID:
                    continue
                if op < 0 or pos < 0:
                    continue
                if op >= OP_DIM:
                    continue
                if pos >= POS_CAP:
                    continue
                valid.append((int(op), int(pos)))
            except Exception:
                continue
        return valid

    result = ParsedData()
    cur_case: Case = None
    last_token = None

    with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 处理 newcase（连续合并）
            if line.startswith(KEY_NEWCASE):
                if cur_case is None or last_token != KEY_NEWCASE:
                    if cur_case is not None and len(cur_case.blocks) > 0:
                        result.cases.append(cur_case)
                    cur_case = Case()
                last_token = KEY_NEWCASE
                continue

            # 如果文件开头没有 newcase
            if cur_case is None:
                cur_case = Case()

            tokens = line.split()
            if not tokens:
                continue

            # 第一项必须是 score
            try:
                score = float(tokens[0])
            except Exception:
                continue

            # 解析后续 op pos
            ints: List[int] = []
            for tok in tokens[1:]:
                try:
                    ints.append(int(tok))
                except Exception:
                    continue

            raw_pairs: List[Tuple[int, int]] = []
            for i in range(0, len(ints) - 1, 2):
                raw_pairs.append((ints[i], ints[i + 1]))

            valid_pairs = filter_pairs(raw_pairs)
            
            if len(valid_pairs) > 0:
                avg_score = score / len(valid_pairs)
            else:
                avg_score = 0  
            cur_case.blocks.append(
                Block(score=float(avg_score), pairs=valid_pairs)
            )
            # cur_case.blocks.append(
            #     Block(score=float(score), pairs=valid_pairs)
            # )

            last_token = 'block'

    # 文件结束，追加最后一个 case
    if cur_case is not None and len(cur_case.blocks) > 0:
        result.cases.append(cur_case)

    total_blocks = sum(len(c.blocks) for c in result.cases)
    if total_blocks == 0:
        raise ValueError("No samples parsed from source file; please check the input format.")

    logging.info(
        "parse_source_v2: parsed %d cases, %d blocks total",
        len(result.cases),
        total_blocks
    )

    return result

#命令行得到数据文件
arp = argparse.ArgumentParser()
arp.add_argument("--data_file", type=str, required=True)
arp.add_argument("--output_file", type=str, required=True)
arp.add_argument("--online", type=str, default=None,
                choices=['greedy', 'ucb', 'linucb'],
                help="在线模式，指定算法")
arp.add_argument("--batch_size", type=int, default=100)
args = arp.parse_args()

# 全局变量声明（离线模式下会被赋值）
parsed_data = None
total_blocks = 0
total_cases = 0
SELECT_CASE = 0
SELECT_BLOCK = 0

def print_data():
    logging.info("Parsed source: %d cases, %d blocks", len(parsed_data.cases), total_blocks)


# 后续换成共享内存的话注释掉上面代码-----------------------

triaged_data = ParsedData()
def triage_operators():  # 按照算子族分类
    """对每一个区块，把 pairs 里的 op 替换成所属族编号"""
    for case in parsed_data.cases:
        cur_case = Case()
        for block in case.blocks:
            new_pairs = []
            for i in range(len(block.pairs)):
                op, pos = block.pairs[i]
                fam = int(OP_TO_FAMILY[op])  # op -> 族编号
                new_pairs.append((fam, pos))
            cur_case.blocks.append(Block(score=block.score, pairs=new_pairs))
        triaged_data.cases.append(cur_case)                 


def data_to_use_for_greedy(n_blocks=None,n_cases=None):
    """统计历史数据中前 n_blocks 个 block 的每个族被选次数和累计收益，返回 [[N, Q], ...] 长度为 OPERATOR_FAM_NUM"""
    if n_blocks is None:
        n_blocks = SELECT_BLOCK
    if n_cases is None:
        n_cases = SELECT_CASE
    # stats[i] = [被选次数, 累计收益]
    stats = [[0, 0.0] for _ in range(OPERATOR_FAM_NUM)]

    block_count = 0
    case_count = 0
    for case in triaged_data.cases:
        if case_count >= n_cases:
                return stats
        for block in case.blocks:
            if block_count >= n_blocks:
                return stats
            # 找出这个 block 涉及了哪些族（去重）
            families_in_block = set()
            for op, pos in block.pairs:
                families_in_block.add(op)  # triage后 op 已经是族编号了
            # 每个涉及的族都 +1次，+score
            for fam in families_in_block:
                stats[fam][0] += 1
                stats[fam][1] += block.score
            block_count += 1
        case_count += 1

    return stats
 
#  1. 调用 data_to_use_for_greedy() 拿到每个族的 [N, Q]（被选次数、累计收益）                                                                  
#   2. 算每个族的平均收益 Q/N，没被选过的族给 0.0
#   3. random.random() < epsilon → 随机选一个族（探索）                                                                                         
#   4. 否则 → 选平均收益最高的那个族（利用）
#   5. 返回选中的族编号
def greedy(epsilon=0.3):
    """
    ε-greedy 选择算子族。
    1. 用历史数据初始化每个族的 N（被选次数）和 Q（累计收益）
    2. 以 ε 概率随机探索，以 1-ε 概率选平均收益最高的族
    返回选中的族编号 (int)
    """
    stats = data_to_use_for_greedy()  # stats[i] = [N, Q]

    # 计算每个族的平均收益，N==0 的族给 0（优先被探索）
    avg_reward = []
    for i in range(OPERATOR_FAM_NUM):
        if stats[i][0] > 0:
            avg_reward.append(stats[i][1] / stats[i][0])
        else:
            avg_reward.append(0.0)

    # ε-greedy 选择
    if random.random() < epsilon:
        # 探索：随机选一个族
        chosen = random.randint(0, OPERATOR_FAM_NUM - 1)
    else:
        # 利用：选平均收益最高的族
        best_val = max(avg_reward)
        chosen = avg_reward.index(best_val)

    logging.info("greedy: avg_reward=%s, chosen=%d (%s)", avg_reward, chosen, OpFamily(chosen).name)
    return chosen


def show_linucb_log(alpha=1.0):
    """
    打印LinUCB的详细选择日志，展示context特征和每个族的UCB值
    """
    print("\n=== LinUCB Detailed Log ===")

    # 初始化
    context_dim = 2
    arms = {fam: LinUCBArm(d=context_dim, alpha=alpha) for fam in range(OPERATOR_FAM_NUM)}

    # 收集reward历史
    reward_history = []
    for case in triaged_data.cases:
        for block in case.blocks:
            reward_history.append(block.score)

    block_count = 0
    case_count = 0
    choices = []

    for case_id, case in enumerate(triaged_data.cases):
        if case_count >= SELECT_CASE:
            break
        for blk_id, block in enumerate(case.blocks):
            if block_count >= SELECT_BLOCK:
                choice_history["LinUCB"] = choices
                return

            # 计算context
            distance_feat = compute_distance_feature(block.score)
            momentum_feat = compute_reward_momentum(reward_history, block_count)
            context = np.array([distance_feat, momentum_feat])

            # 计算每个族的UCB
            ucb_values = {}
            for fam in range(OPERATOR_FAM_NUM):
                ucb_values[fam] = arms[fam].get_ucb(context)

            # 选择最大UCB的族
            chosen_family = max(ucb_values, key=ucb_values.get)
            choices.append(chosen_family)

            # 打印日志（打印所有block）
            print(f"\nBlock {block_count} (Case {case_id}, Block {blk_id}):")
            print(f"  Score: {block.score:.2f}")
            print(f"  Context: [distance={distance_feat:.4f}, momentum={momentum_feat:.6f}]")
            print(f"  LinUCB values:")
            for fam in range(OPERATOR_FAM_NUM):
                marker = " <-- CHOSEN" if fam == chosen_family else ""
                print(f"    {OpFamily(fam).name:30s}: {ucb_values[fam]:8.4f}{marker}")

            # 更新选中的族
            arms[chosen_family].update(context, block.score)

            block_count += 1

        case_count += 1

    # 保存到全局choice_history
    choice_history['LinUCB'] = choices

    print(f"\n总共处理了 {len(choices)} 个blocks")
    print(f"各族被选择次数:")
    for fam in range(OPERATOR_FAM_NUM):
        count = choices.count(fam)
        print(f"  {OpFamily(fam).name:30s}: {count:4d} ({count/len(choices)*100:.1f}%)")


def show_greedy_log():
    """逐 block 累加统计，打印每个时刻各族的平均收益和 greedy 会选谁"""
    stats = [[0, 0.0] for _ in range(OPERATOR_FAM_NUM)]
    block_count = 0
    case_count = 0
    choices = []

    print(f"{'block':>6}  ", end="")
    for i in range(OPERATOR_FAM_NUM):
        print(f"{OpFamily(i).name:>25}", end="  ")
    print("  chosen")
    print("-" * (8 + 27 * OPERATOR_FAM_NUM + 10))

    
    for case in triaged_data.cases:
        if case_count >= SELECT_CASE:
            break
        for block in case.blocks:
            if block_count >= SELECT_BLOCK:
                choice_history["greedy"] = choices
                return
            # 累加这个 block 的数据
            families_in_block = set()
            for op, pos in block.pairs:
                families_in_block.add(op)
            for fam in families_in_block:
                stats[fam][0] += 1
                stats[fam][1] += block.score
            block_count += 1

            # 计算当前各族平均收益
            avg_reward = []
            for i in range(OPERATOR_FAM_NUM):
                if stats[i][0] > 0:
                    avg_reward.append(stats[i][1] / stats[i][0])
                else:
                    avg_reward.append(0.0)

            # 当前 greedy 会选谁（纯 exploit，不加随机）
            best_val = max(avg_reward)
            chosen = avg_reward.index(best_val)
            choices.append(chosen)

            # 打印这一行
            print(f"{block_count:>6}  ", end="")
            for val in avg_reward:
                print(f"{val:>25.4f}", end="  ")
            print(f"  {OpFamily(chosen).name}")
        case_count += 1

    choice_history["greedy"] = choices


def UCB(c=1.0):
    """
    UCB (Upper Confidence Bound) 选择算子族。

    核心思想：给每个族打一个综合分 = 平均收益 + 探索奖励
      score[i] = Q[i]/N[i] + c * sqrt( ln(t) / N[i] )

    其中：
      Q[i]/N[i] — 族 i 的历史平均收益（利用项：选过去表现好的）
      c * sqrt(ln(t)/N[i]) — 置信上界（探索项：没怎么试过的族得分更高）
        t  = 所有族被选的总次数
        N[i] = 族 i 被选的次数
        c  = 探索系数，越大越倾向探索，一般取 1.0~2.0

    为什么有效：
      - N[i] 小 → 探索项大 → 自动去试不熟悉的族
      - N[i] 大 → 探索项趋近 0 → 由平均收益主导，收敛到最优
      - 不需要像 ε-greedy 那样靠随机探索，探索是自适应的

    参数:
      c: 探索系数，默认 1.0

    返回: 选中的族编号 (int)
    """
    stats = data_to_use_for_greedy()  # stats[i] = [N, Q]

    # t = 所有族被选的总次数
    t = sum(stats[i][0] for i in range(OPERATOR_FAM_NUM))

    # 如果还没有任何数据，随机选一个
    if t == 0:
        chosen = random.randint(0, OPERATOR_FAM_NUM - 1)
        logging.info("UCB: 无历史数据，随机选择 %d (%s)", chosen, OpFamily(chosen).name)
        return chosen

    ucb_scores = []
    for i in range(OPERATOR_FAM_NUM):
        if stats[i][0] == 0:
            # 从没被选过的族，给无穷大分数 → 优先被选
            ucb_scores.append(float('inf'))
        else:
            avg = stats[i][1] / stats[i][0]           # 平均收益（利用项）
            bonus = c * math.sqrt(math.log(t) / stats[i][0])  # 置信上界（探索项）
            ucb_scores.append(avg + bonus)

    # 选 UCB 分数最高的族
    best_val = max(ucb_scores)
    chosen = ucb_scores.index(best_val)

    logging.info("UCB: scores=%s, chosen=%d (%s)",
                 [f"{s:.4f}" if s != float('inf') else "inf" for s in ucb_scores],
                 chosen, OpFamily(chosen).name)
    return chosen


def show_ucb_log(c=1.0):
    """逐 block 累加统计，打印每个时刻各族的 UCB 分数和会选谁"""
    stats = [[0, 0.0] for _ in range(OPERATOR_FAM_NUM)]
    block_count = 0
    case_count = 0
    choices = []

    print(f"\n{'block':>6}  ", end="")
    for i in range(OPERATOR_FAM_NUM):
        print(f"{OpFamily(i).name:>25}", end="  ")
    print("  chosen")
    print("-" * (8 + 27 * OPERATOR_FAM_NUM + 10))

    for case in triaged_data.cases:
        if case_count >= SELECT_CASE:
            break
        for block in case.blocks:
            if block_count >= SELECT_BLOCK:
                choice_history["UCB"] = choices
                return
            # 累加这个 block 的数据
            families_in_block = set()
            for op, pos in block.pairs:
                families_in_block.add(op)
            for fam in families_in_block:
                stats[fam][0] += 1
                stats[fam][1] += block.score
            block_count += 1

            # 计算当前各族 UCB 分数
            t = sum(stats[i][0] for i in range(OPERATOR_FAM_NUM))
            ucb_scores = []
            for i in range(OPERATOR_FAM_NUM):
                if stats[i][0] == 0:
                    ucb_scores.append(float('inf'))
                else:
                    avg = stats[i][1] / stats[i][0]
                    bonus = c * math.sqrt(math.log(t) / stats[i][0])
                    ucb_scores.append(avg + bonus)

            # 选分数最高的
            best_val = max(ucb_scores)
            chosen = ucb_scores.index(best_val)
            choices.append(chosen)

            # 打印这一行
            print(f"{block_count:>6}  ", end="")
            for val in ucb_scores:
                if val == float('inf'):
                    print(f"{'inf':>25}", end="  ")
                else:
                    print(f"{val:>25.4f}", end="  ")
            print(f"  {OpFamily(chosen).name}")
        case_count += 1

    choice_history["UCB"] = choices


def plot_choices(output_file):
    """
    画出 choice_history 里所有算法的选择轨迹。
    横坐标：block 序号
    纵坐标：选择的族编号
    每个算法一条线，之后加新算法只要往 choice_history 里存数据就自动多一条线。
    """
    if not choice_history:
        print("choice_history 为空，没有数据可画")
        return

    plt.figure(figsize=(14, 5))
    for name, choices in choice_history.items():
        plt.plot(range(1, len(choices) + 1), choices, label=name, alpha=0.7)

    plt.xlabel("block")
    plt.ylabel("chosen family")
    # y 轴刻度用族名
    plt.yticks(range(OPERATOR_FAM_NUM), [OpFamily(i).name for i in range(OPERATOR_FAM_NUM)])
    plt.legend()
    plt.title("Algorithm Choices Over Blocks")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"图表已保存到 {output_file}")
    plt.close()



def compute_distance_feature(current_score: float) -> float:
    """
    计算距离目标点的接近程度特征

    Args:
        current_score: 当前block的score（代表已经走了多少分）

    Returns:
        接近程度特征值，范围 [0, 1]
        - 接近 1.0 = 还在初始阶段，离目标很远
        - 接近 0.0 = 已经很接近目标了
        - 如果超过目标距离，返回 0.0
    """
    if current_score >= TARGET_DISTANCE:
        return 0.0

    # 剩余距离 / 总距离
    remaining_distance = TARGET_DISTANCE - current_score
    distance_feature = remaining_distance / TARGET_DISTANCE

    return distance_feature


def compute_reward_momentum(reward_history: List[float], current_index: int) -> float:
    """
    计算reward的变化趋势（momentum）

    Args:
        reward_history: 所有block的reward历史（就是score列表）
        current_index: 当前block的索引

    Returns:
        reward momentum值
        - > 0: 最近有进展，应该利用当前策略
        - ≈ 0: 停滞，应该探索新策略
        - < 0: 退步
    """
    # 如果数据不够两个窗口，返回0（中性）
    if current_index < REWARD_WINDOW_SIZE:
        return 0.0

    # recent_window: 最近的REWARD_WINDOW_SIZE个reward
    recent_start = max(0, current_index - REWARD_WINDOW_SIZE)
    recent_window = reward_history[recent_start:current_index]

    # baseline_window: 再往前的REWARD_WINDOW_SIZE个reward
    baseline_start = max(0, recent_start - REWARD_WINDOW_SIZE)
    baseline_end = recent_start
    baseline_window = reward_history[baseline_start:baseline_end]

    # 如果baseline窗口不够，返回0
    if len(baseline_window) < REWARD_WINDOW_SIZE // 2:  # 至少要有一半数据
        return 0.0

    recent_avg = sum(recent_window) / len(recent_window) if recent_window else 0.0
    baseline_avg = sum(baseline_window) / len(baseline_window) if baseline_window else 0.0

    # momentum = (最近平均 - 基线平均) / 窗口大小，归一化
    momentum = (recent_avg - baseline_avg) / REWARD_WINDOW_SIZE

    return momentum


def LinUCB():
    """
    LinUCB算法：带上下文的bandit

    对每个block：
    1. 计算context特征 [distance_feature, momentum_feature]
    2. 为每个族计算UCB值
    3. 选择UCB最大的族
    4. 用该族的实际reward更新模型
    """
    print("\n=== LinUCB (Contextual Bandit) ===")

    # 初始化每个族的LinUCB arm
    context_dim = 2  # [distance_feature, momentum_feature]
    alpha = 1.0  # 探索参数，可调
    arms = {fam: LinUCBArm(d=context_dim, alpha=alpha) for fam in range(OPERATOR_FAM_NUM)}

    # 收集所有block的score作为reward_history
    reward_history = []
    for case in triaged_data.cases:
        for block in case.blocks:
            reward_history.append(block.score)

    # 遍历每个block，做选择和更新
    choices = []
    block_idx = 0

    for case in triaged_data.cases:
        for block in case.blocks:
            # 1. 计算context特征
            distance_feat = compute_distance_feature(block.score)
            momentum_feat = compute_reward_momentum(reward_history, block_idx)
            context = np.array([distance_feat, momentum_feat])

            # 2. 为每个族计算UCB
            ucb_values = {}
            for fam in range(OPERATOR_FAM_NUM):
                ucb_values[fam] = arms[fam].get_ucb(context)

            # 3. 选择UCB最大的族
            chosen_family = max(ucb_values, key=ucb_values.get)
            choices.append(chosen_family)

            # 4. 用实际reward更新选中的族
            # 这里的reward就是block.score（在离线数据中我们假设这是观察到的reward）
            arms[chosen_family].update(context, block.score)

            block_idx += 1

    # 保存选择历史用于画图
    choice_history['LinUCB'] = choices

    print(f"总共处理了 {len(choices)} 个blocks")
    print(f"各族被选择次数: {dict(zip(range(OPERATOR_FAM_NUM), [choices.count(i) for i in range(OPERATOR_FAM_NUM)]))}")

    return choices

def policy():
    pass

def parse_batch_lines(lines):
    """
    解析 AFL 写过来的批量数据文件。

    格式（每个block一行）：
        score op1 pos1 op2 pos2 ...

    例如：
        3.5 2 100 5 200
        1.2 10 50
    """
    blocks = []
    case_lines = "--newcase--"
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) < 1:
            continue
        if tokens[0] == case_lines: #先跳过，后续加上case的记录处理
            continue
        score = float(tokens[0])
        pairs = []
        for i in range(1, len(tokens) - 1, 2):
            op = int(tokens[i])
            pos = int(tokens[i + 1])
            fam = int(OP_TO_FAMILY.get(op, -1))
            if fam >= 0:
                pairs.append((fam, pos))
        blocks.append(Block(score=score, pairs=pairs))
    return blocks

def run_online(mode='greedy', batch_size=100,
                input_file='/two-stage/afl_to_python.txt',
                decision_file='/two-stage/python_to_afl.txt'):
    scheduler = OnlineScheduler(mode=mode, batch_size=batch_size)
    lines_read = 0  # 已读行数（偏移量）
    """
    在线模式主循环：
    1. 等待 AFL 写够一个batch
    2. 读取一批 blocks
    3. 用 scheduler 做决策
    4. 把决策写到 output_file
    """
    while True:
        # 读取整个文件，跳过已处理的行
        if not os.path.exists(input_file):
            time.sleep(0.01)
            continue

        with open(input_file, 'r') as f:
            all_lines = f.readlines()

        new_lines = all_lines[lines_read:]

        # 不够一个batch就继续等
        if len(new_lines) < batch_size:
            time.sleep(0.01)
            continue

        # 取一个batch
        batch_lines = new_lines[:batch_size]

        # 防止重复处理：再读一次确认行数没变
        time.sleep(0.05)  # 等50ms确保文件写完整
        with open(input_file, 'r') as f:
            verify_lines = f.readlines()
        if len(verify_lines) < lines_read + batch_size:
            # 文件还在写入中，等下一轮
            continue

        batch_blocks = parse_batch_lines(batch_lines)
        decisions = scheduler.process_batch(batch_blocks)
        lines_read += len(batch_lines)

        # 决策追加写入（也不删除）
        with open(decision_file, 'a') as f:
            for d in decisions:
                f.write(f"{d}\n")



def select_operator_families():
    # 1. 每个族一个 UCB / ε-greedy 统计器
    greedy()
    UCB()

    # 2. 线性 contextual bandit（LinUCB）
    LinUCB()

    # 3. 小 MLP 输出每个族的 logit，再用 policy-gradient style更新
    policy()

    
def main():
    global parsed_data, total_blocks, total_cases, SELECT_CASE, SELECT_BLOCK

    if args.online:
        run_online(mode=args.online, batch_size=args.batch_size)
        return  # 在线模式不走下面的离线逻辑

    # 离线模式：解析数据文件
    data_file = args.data_file
    cache_file = data_file + ".pkl.gz"
    if os.path.exists(cache_file):
        with gzip.open(cache_file, "rb") as f:
            parsed_data = pickle.load(f)
        logging.info("从缓存加载: %s", cache_file)
    else:
        parsed_data = parse_source_v2(data_file)
        with gzip.open(cache_file, "wb") as f:
            pickle.dump(parsed_data, f)
        logging.info("解析完成，缓存已保存: %s", cache_file)

    total_blocks = sum(len(c.blocks) for c in parsed_data.cases)
    total_cases = len(parsed_data.cases)
    SELECT_CASE = total_cases  # 选几个case的数据，默认全选
    SELECT_BLOCK = total_blocks  # 选前n个block的数据，默认全选

    # 读取数据文件，保存到全局关键数据结构
    # process_data()
    print_data()

    #算子分类
    triage_operators()


    #使用bandit来选择算子族
    select_operator_families()

    show_greedy_log()
    show_ucb_log()
    show_linucb_log()

    plot_choices(args.output_file)

    #进行SMR区域划分
    # divide_smr_regions()

    #使用bandit来选择算子和区域
    # select_operators_and_regions()


if __name__ == "__main__":
    main()