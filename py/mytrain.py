# -*- coding: utf-8 -*-
"""
Operator-level policy learning for fuzzing
------------------------------------------
Learn to select mutation operators using a neural network.

Workflow:
1. Parse fuzzing log file:
   (context_vector, operator_id, reward)
2. Train a policy network with operator embeddings
3. Use the network to guide next operator selection
"""

import os
import sys
import time
import random
import argparse
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =====================
# Global configuration
# =====================

RAND_SEED = int(time.time())
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_OPERATORS = 15
CONTEXT_DIM = 10       # 上下文向量暂时用10维度
OP_EMBED_DIM = 10       # operator embedding dim: each operator has 10-dim embedding
HIDDEN_DIM = 128

BATCH_SIZE = 64
EPOCHS =40 
LEARNING_RATE = 1e-3

# =====================
# Global configuration
# =====================

#解析数据文件，之后有了共享内存就不用了，暂时先用来得到输入方便调试
def parse_fuzz_log(log_path):
    """
    Parse fuzzing log file.

    Format:
    - 'newcase' marks the start of a new case (consecutive 'newcase' lines are merged)
    - '------------------' separates different executions within a case
    - Each execution block:
      * 'output_vector_after' followed by reward value
      * 'operator_revise_array' followed by lines: 'operator_id value'
      * Lines with value == 4294967295 are ignored

    Return:
        contexts:  np.ndarray [N, CONTEXT_DIM]  (16x10=160 dim, placeholder)
        ops:       np.ndarray [N]              (operator id: 0~15)
        rewards:   np.ndarray [N]              (float reward)
    """
    INVALID_VALUE = 4294967295  # 0xFFFFFFFF
    
    ops_list = []
    rewards_list = []

    # 临时收集一次 execution 的算子列表
    exec_ops = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    in_newcase = False
    current_reward = None
    in_operator_array = False

    def flush_execution():
        """将当前 execution 的算子按均分奖励写入全局列表。"""
        nonlocal exec_ops, current_reward
        if current_reward is None or len(exec_ops) == 0:
            exec_ops = []
            current_reward = None
            return
        k = len(exec_ops)
        shared_reward = current_reward / k
        for op_id in exec_ops:
            ops_list.append(op_id)
            rewards_list.append(shared_reward)
        exec_ops = []
        current_reward = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Handle consecutive 'newcase' lines (merge them)
        if line == 'newcase':
            if not in_newcase:
                in_newcase = True
            i += 1
            continue
        
        # Reset newcase flag when we see separator or other content
        if line == '------------------':
            # 一个 execution 结束，冲刷水桶
            flush_execution()
            in_newcase = False
            in_operator_array = False
            i += 1
            continue
        
        # Parse output_vector_after and reward
        if line == 'output_vector_after':
            # 进入新的 execution，先冲刷上一段（如果有未冲刷）
            flush_execution()
            in_operator_array = False
            exec_ops = []
            if i + 1 < len(lines):
                try:
                    current_reward = float(lines[i + 1].strip())
                except ValueError:
                    current_reward = None
                i += 2
                continue
        
        # Parse operator_revise_array
        if line == 'operator_revise_array':
            in_operator_array = True
            i += 1
            continue
        
        # Parse operator lines (only when in operator array and reward is set)
        if in_operator_array and current_reward is not None:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    op_id = int(parts[0])
                    value = int(parts[1])
                    
                    # Only record if value is not invalid
                    if value != INVALID_VALUE:
                        exec_ops.append(op_id)
                except ValueError:
                    pass  # Skip invalid lines
        
        i += 1
    
    # 处理文件末尾可能未冲刷的 execution
    flush_execution()

    # Convert to numpy arrays
    N = len(ops_list)
    ops = np.array(ops_list, dtype=np.int64)
    rewards = np.array(rewards_list, dtype=np.float32)
    
    # Generate random contexts: 16x10 = 160 dimensions (placeholder, will be replaced later)
    contexts = np.random.randn(N, CONTEXT_DIM).astype(np.float32)
    
    print(f"Parsed {N} operator-reward pairs from {log_path}")
    
    return contexts, ops, rewards

# =====================
# Dataset
# =====================

class OperatorDataset(Dataset):
    def __init__(self, contexts, ops, rewards):
        self.contexts = torch.from_numpy(contexts)
        self.ops = torch.from_numpy(ops).long()
        self.rewards = torch.from_numpy(rewards)

    def __len__(self):
        return len(self.ops)

    def __getitem__(self, idx):
        return (
            self.contexts[idx],
            self.ops[idx],
            self.rewards[idx],
        )


# =====================
# Model
# =====================

class OperatorPolicyNet(nn.Module):
    """
    Policy network:
        scores(context) = [score(op_0 | context), ..., score(op_15 | context)]
    
    Input: context (16x10=160 dim)
    Output: scores [16] - expected rewards for all 16 operators under current context
    
    Operator embeddings are learned parameters.
    """

    def __init__(self, context_dim, num_ops, op_embed_dim, hidden_dim):
        super().__init__()
        self.num_ops = num_ops

        # Operator embeddings: [num_ops, op_embed_dim]
        self.op_embedding = nn.Embedding(num_ops, op_embed_dim)

        # Network: context + op_embedding -> score
        # Input dim = context_dim + op_embed_dim
        self.net = nn.Sequential(
            nn.Linear(context_dim + op_embed_dim, hidden_dim), #10+10->128
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), #128->128
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # output single score  # 128 -> 1
        )

    def forward(self, context): #scores[b, k] = 在第 b 个 context 下，选择第 k 个算子的预期好坏

        """
        Args:
            context: Tensor [B, context_dim]  (16x10=160 dim)

        Returns:
            scores: Tensor [B, num_ops]  (scores for all operators under each context)
        """
        #
        B = context.size(0)  #B是批次大小（如 64）
        
        # Expand context for all operators: [B, context_dim] -> [B, num_ops, context_dim]
        context_expanded = context.unsqueeze(1).repeat(1, self.num_ops, 1)  # [B, num_ops, context_dim]
        
        # Get all operator embeddings: [num_ops, op_embed_dim]
        all_op_ids = torch.arange(self.num_ops, device=context.device)  # [num_ops]
        all_op_embed = self.op_embedding(all_op_ids)  # [num_ops, op_embed_dim]
        
        # Expand operator embeddings for batch: [num_ops, op_embed_dim] -> [B, num_ops, op_embed_dim]
        all_op_embed = all_op_embed.unsqueeze(0).repeat(B, 1, 1)  # [B, num_ops, op_embed_dim]
        
        # Concatenate context and operator embeddings
        x = torch.cat([context_expanded, all_op_embed], dim=-1)  # [B, num_ops, context_dim + op_embed_dim]
        
        # Reshape for network: [B, num_ops, input_dim] -> [B*num_ops, input_dim]
        x_flat = x.view(B * self.num_ops, -1)  # [B*num_ops, context_dim + op_embed_dim]
        
        # Predict scores for all operators
        scores_flat = self.net(x_flat).squeeze(-1)  # [B*num_ops]
        
        # Reshape back: [B*num_ops] -> [B, num_ops]
        scores = scores_flat.view(B, self.num_ops)  # [B, num_ops]
        
        return scores


# =====================
# Training
# =====================

def train(model, dataloader, eval_contexts=None, eval_ops=None, eval_rewards=None):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    all_predicted_scores = []
    all_rewards = []
    all_chosen_ops = []
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        epoch_predicted_scores = []
        epoch_rewards = []
        epoch_chosen_ops = []

        for context, op, reward in dataloader:
            context = context.to(DEVICE)
            op = op.to(DEVICE)
            reward = reward.to(DEVICE)

            optimizer.zero_grad()

            # Predict scores for all operators under current context
            # all_scores: [B, num_ops], 每行是当前 context 下 16 个算子的 score
            all_scores = model(context)

            # 计算每个算子的 log 概率：log softmax 引入"算子之间的竞争"
            # log_probs: [B, num_ops]
            log_probs = torch.log_softmax(all_scores, dim=1)

            # 只取当前实际选择的算子的 log 概率
            # op: [B] -> [B, 1]，gather 之后 squeeze 回 [B]
            chosen_log_prob = log_probs.gather(1, op.unsqueeze(1)).squeeze(1)  # [B]

            # Policy gradient 风格的损失：
            # 如果 reward 大，希望 log_prob 大（概率更高），所以损失是 -logp * reward
            # 这里 reward 可以是正负都行：正 reward 强化该算子，负 reward 惩罚该算子
            loss = -(chosen_log_prob * reward).mean()
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Collect predictions for analysis (only in last epoch to save memory)
            if epoch == EPOCHS - 1:
                with torch.no_grad():
                    chosen_scores = all_scores.gather(1, op.unsqueeze(1)).squeeze(1)
                    epoch_predicted_scores.extend(chosen_scores.cpu().numpy())
                    epoch_rewards.extend(reward.cpu().numpy())
                    epoch_chosen_ops.extend(op.cpu().numpy())
            else:
                # For intermediate epochs, collect a small sample for statistics
                with torch.no_grad():
                    chosen_scores = all_scores.gather(1, op.unsqueeze(1)).squeeze(1)
                    epoch_predicted_scores.extend(chosen_scores.cpu().numpy()[:10])  # Only first 10
                    epoch_rewards.extend(reward.cpu().numpy()[:10])
                    epoch_chosen_ops.extend(op.cpu().numpy()[:10])

        # Print epoch summary
        print(f"[Epoch {epoch}] loss = {total_loss / len(dataloader):.6f}")
        
        # Show prediction statistics
        if epoch % 2 == 0 or epoch == EPOCHS - 1:  # Every 2 epochs or last epoch
            if len(epoch_predicted_scores) > 0:
                epoch_predicted_scores = np.array(epoch_predicted_scores)
                epoch_rewards = np.array(epoch_rewards)
                epoch_chosen_ops = np.array(epoch_chosen_ops)

                    
                # MSE and correlation (for reference, but not what we optimize)
                correlation = np.corrcoef(epoch_predicted_scores, epoch_rewards)[0, 1]
                mse = np.mean((epoch_predicted_scores - epoch_rewards) ** 2)
                
                # Policy gradient metric: average (log_prob * reward) - this is what we maximize
                # Since we optimize -log_prob * reward, higher values mean better policy
                # We approximate this by: log(softmax(score)) * reward
                # For a single score, log_prob ≈ score - log(sum(exp(scores)))
                # We use a simplified version: score relative to mean
                normalized_scores = epoch_predicted_scores - epoch_predicted_scores.mean()
                policy_gradient_metric = np.mean(normalized_scores * epoch_rewards)
                
                print(f"  Policy metric (logp*reward): {policy_gradient_metric:.4f}, "
                      f"MSE={mse:.4f}, Correlation={correlation:.4f}")
        
        # Store last epoch predictions
        if epoch == EPOCHS - 1:
            all_predicted_scores = epoch_predicted_scores
            all_rewards = epoch_rewards
            all_chosen_ops = epoch_chosen_ops
    
    return all_predicted_scores, all_rewards, all_chosen_ops



# def display_result():#展示训练的各种结果，评估模型好坏

def display_parse_data(contexts, ops, rewards):#展示解析的数据
    # =====================
    # Check parsed data
    # =====================
    print("\n" + "=" * 60)
    print("Checking parsed data...")
    print("=" * 60)
    print(f"Total samples: {len(ops)}")
    print(f"Context shape: {contexts.shape}")
    print(f"Operators shape: {ops.shape}")
    print(f"Rewards shape: {rewards.shape}")
    
    # Print first 10 samples
    print("\nFirst 10 samples (context, operator, reward):")
    print("-" * 80)
    print(f"{'Index':<8} {'Operator':<10} {'Reward':<12} {'Context (first 5 dims)':<30}")
    print("-" * 80)
    for i in range(min(50, len(ops))):
        ctx_preview = ", ".join([f"{val:.3f}" for val in contexts[i][:5]])
        print(f"{i:<8} {ops[i]:<10} {rewards[i]:<12.4f} [{ctx_preview}...]")
    
    # Print last 10 samples
    if len(ops) > 10:
        print("\nLast 10 samples (context, operator, reward):")
        print("-" * 80)
        print(f"{'Index':<8} {'Operator':<10} {'Reward':<12} {'Context (first 5 dims)':<30}")
        print("-" * 80)
        for i in range(max(0, len(ops) - 50), len(ops)):
            ctx_preview = ", ".join([f"{val:.3f}" for val in contexts[i][:5]])
            print(f"{i:<8} {ops[i]:<10} {rewards[i]:<12.4f} [{ctx_preview}...]")
    
    # Print statistics
    print("\nData statistics:")
    print(f"  Operators range: [{ops.min()}, {ops.max()}]")
    print(f"  Rewards range: [{rewards.min():.4f}, {rewards.max():.4f}]")
    print(f"  Rewards mean: {rewards.mean():.4f}")
    print(f"  Rewards std: {rewards.std():.4f}")
    
    # Operator distribution
    unique_ops, counts = np.unique(ops, return_counts=True)
    print(f"\nOperator distribution:")
    print(f"{'Operator':<10} {'Count':<10} {'Percentage':<12}")
    print("-" * 35)
    for op_id, count in zip(unique_ops, counts):
        percentage = count / len(ops) * 100
        print(f"{op_id:<10} {count:<10} {percentage:<12.2f}%")
    
    print("=" * 60)

def display_result(final_embeds, initial_embeds, pred_scores, true_rewards, chosen_ops, contexts, model, out_dir):
    """
    Display training results and analysis.
    
    Args:
        final_embeds: Final operator embeddings after training [num_ops, op_embed_dim]
        initial_embeds: Initial operator embeddings before training [num_ops, op_embed_dim]
        pred_scores: Predicted scores from model [N]
        true_rewards: True rewards from data [N]
        chosen_ops: Chosen operator IDs [N]
        contexts: Context vectors [N, context_dim]
        model: Trained model
        out_dir: Output directory for saving results
    """
    # =====================
    # 1. Operator Embeddings Analysis
    # =====================
    print("\n" + "-" * 60)
    print("Operator Embeddings Analysis")
    print("-" * 60)
    
    print(f"\nEmbedding shape: {final_embeds.shape} (16 operators × {OP_EMBED_DIM} dims)")
    
    # Embedding statistics
    print("\nEmbedding statistics per operator:")
    print(f"{'Op':<4} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Norm':<10}")
    print("-" * 60)
    for op_id in range(NUM_OPERATORS):
        embed = final_embeds[op_id]
        print(f"{op_id:<4} {embed.mean():<10.4f} {embed.std():<10.4f} "
              f"{embed.min():<10.4f} {embed.max():<10.4f} {np.linalg.norm(embed):<10.4f}")
    
    # Print each operator's embedding vector
    print("\n" + "-" * 60)
    print("Operator Embedding Vectors:")
    print("-" * 60)
    for op_id in range(NUM_OPERATORS):
        embed = final_embeds[op_id]
        # Format embedding as a readable vector
        embed_str = " ".join([f"{val:8.4f}" for val in embed])
        print(f"Operator {op_id:2d}: [{embed_str}]")
    
    # Embedding similarity matrix (cosine similarity)
    print("\nOperator Embedding Similarity Matrix (cosine similarity):")
    similarity_matrix = np.zeros((NUM_OPERATORS, NUM_OPERATORS))
    for i in range(NUM_OPERATORS):
        for j in range(NUM_OPERATORS):
            # Cosine similarity
            dot_product = np.dot(final_embeds[i], final_embeds[j])
            norm_i = np.linalg.norm(final_embeds[i])
            norm_j = np.linalg.norm(final_embeds[j])
            similarity_matrix[i, j] = dot_product / (norm_i * norm_j + 1e-8)
    
    print("     ", end="")
    for i in range(NUM_OPERATORS):
        print(f"{i:>6}", end="")
    print()
    for i in range(NUM_OPERATORS):
        print(f"Op {i:2d}:", end="")
        for j in range(NUM_OPERATORS):
            print(f"{similarity_matrix[i, j]:6.3f}", end="")
        print()
    
    # Find most similar operator pairs
    print("\nMost similar operator pairs (top 5):")
    pairs = []
    for i in range(NUM_OPERATORS):
        for j in range(i + 1, NUM_OPERATORS):
            pairs.append((i, j, similarity_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for i, (op1, op2, sim) in enumerate(pairs[:5]):
        print(f"  {i+1}. Op {op1:2d} <-> Op {op2:2d}: {sim:.4f}")
    
    # Embedding change from initial to final
    embed_change = np.linalg.norm(final_embeds - initial_embeds, axis=1)
    print("\nEmbedding change (L2 norm from initial):")
    print(f"{'Op':<4} {'Change':<10}")
    print("-" * 20)
    for op_id in range(NUM_OPERATORS):
        print(f"{op_id:<4} {embed_change[op_id]:<10.4f}")
    
    # =====================
    # 2. Prediction Analysis
    # =====================
    print("\n" + "-" * 60)
    print("Prediction Analysis")
    print("-" * 60)
    
    pred_scores = np.array(pred_scores)
    true_rewards = np.array(true_rewards)
    chosen_ops = np.array(chosen_ops)
    
    # Overall statistics
    # Note: MSE/MAE are for reference only - we optimize policy gradient loss, not MSE
    mse = np.mean((pred_scores - true_rewards) ** 2)
    mae = np.mean(np.abs(pred_scores - true_rewards))
    correlation = np.corrcoef(pred_scores, true_rewards)[0, 1]
    
    print(f"\nOverall prediction metrics:")
    print(f"  Note: We optimize policy gradient loss (-log_prob * reward), not MSE")
    print(f"  MSE (reference):     {mse:.6f}")
    print(f"  MAE (reference):    {mae:.6f}")
    print(f"  Correlation:         {correlation:.4f}")
    
    # Policy gradient metric: what we actually optimize
    # We need to recompute log_probs properly from model
    # For now, compute a proxy: average (normalized_score * reward)
    # This approximates the policy gradient objective
    normalized_scores = pred_scores - pred_scores.mean()
    policy_gradient_metric = np.mean(normalized_scores * true_rewards)
    print(f"  Policy metric (proxy): {policy_gradient_metric:.4f} (higher is better)")
    
    # Per-operator prediction statistics
    print(f"\nPer-operator prediction statistics:")
    print(f"{'Op':<4} {'Count':<8} {'Avg Pred':<12} {'Avg True':<12} {'MSE':<12}")
    print("-" * 60)
    for op_id in range(NUM_OPERATORS):
        mask = chosen_ops == op_id
        if mask.sum() > 0:
            op_preds = pred_scores[mask]
            op_rewards = true_rewards[mask]
            op_mse = np.mean((op_preds - op_rewards) ** 2)
            print(f"{op_id:<4} {mask.sum():<8} {op_preds.mean():<12.4f} "
                  f"{op_rewards.mean():<12.4f} {op_mse:<12.6f}")
    
    # Sample predictions
    print(f"\nSample predictions (first 10):")
    print(f"{'Idx':<6} {'Op':<4} {'Pred Score':<12} {'True Reward':<12} {'Diff':<12}")
    print("-" * 60)
    for i in range(min(10, len(pred_scores))):
        print(f"{i:<6} {chosen_ops[i]:<4} {pred_scores[i]:<12.4f} "
              f"{true_rewards[i]:<12.4f} {pred_scores[i]-true_rewards[i]:<12.4f}")
    
    # =====================
    # 2.5. Operator Priority Analysis
    # =====================
    print("\n" + "=" * 60)
    print("Operator Priority Analysis - Which operators to prioritize?")
    print("=" * 60)
    
    # Collect per-operator statistics
    op_stats = []
    for op_id in range(NUM_OPERATORS):
        mask = chosen_ops == op_id
        if mask.sum() > 0:
            op_preds = pred_scores[mask]
            op_rewards = true_rewards[mask]
            op_stats.append({
                'op_id': op_id,
                'count': mask.sum(),
                'avg_reward': op_rewards.mean(),
                'total_reward': op_rewards.sum(),
                'avg_pred': op_preds.mean(),
                'std_reward': op_rewards.std(),
                'usage_rate': mask.sum() / len(chosen_ops) * 100
            })
        else:
            op_stats.append({
                'op_id': op_id,
                'count': 0,
                'avg_reward': 0.0,
                'total_reward': 0.0,
                'avg_pred': 0.0,
                'std_reward': 0.0,
                'usage_rate': 0.0
            })
    
    # Sort by average reward (descending)
    op_stats_sorted_by_reward = sorted(op_stats, key=lambda x: x['avg_reward'], reverse=True)
    
    print("\n" + "-" * 60)
    print("Operators ranked by AVERAGE REWARD (best performers first):")
    print("-" * 60)
    print(f"{'Rank':<6} {'Op':<4} {'Avg Reward':<12} {'Usage':<10} {'Count':<10} {'Std':<10}")
    print("-" * 60)
    for rank, stat in enumerate(op_stats_sorted_by_reward, 1):
        if stat['count'] > 0:
            print(f"{rank:<6} {stat['op_id']:<4} {stat['avg_reward']:<12.4f} "
                  f"{stat['usage_rate']:<10.2f}% {stat['count']:<10} {stat['std_reward']:<10.4f}")
    
    # Sort by total reward contribution
    op_stats_sorted_by_total = sorted(op_stats, key=lambda x: x['total_reward'], reverse=True)
    
    print("\n" + "-" * 60)
    print("Operators ranked by TOTAL REWARD CONTRIBUTION:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Op':<4} {'Total Reward':<14} {'Avg Reward':<12} {'Usage':<10}")
    print("-" * 60)
    for rank, stat in enumerate(op_stats_sorted_by_total[:10], 1):  # Top 10
        if stat['count'] > 0:
            print(f"{rank:<6} {stat['op_id']:<4} {stat['total_reward']:<14.2f} "
                  f"{stat['avg_reward']:<12.4f} {stat['usage_rate']:<10.2f}%")
    
    # Sort by usage frequency
    op_stats_sorted_by_usage = sorted(op_stats, key=lambda x: x['count'], reverse=True)
    
    print("\n" + "-" * 60)
    print("Operators ranked by USAGE FREQUENCY:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Op':<4} {'Usage':<10} {'Count':<10} {'Avg Reward':<12}")
    print("-" * 60)
    for rank, stat in enumerate(op_stats_sorted_by_usage[:10], 1):  # Top 10
        if stat['count'] > 0:
            print(f"{rank:<6} {stat['op_id']:<4} {stat['usage_rate']:<10.2f}% "
                  f"{stat['count']:<10} {stat['avg_reward']:<12.4f}")
    
    # Recommendation based on multiple factors
    print("\n" + "-" * 60)
    print("RECOMMENDATION: Top operators to prioritize")
    print("-" * 60)
    
    # Calculate a composite score: avg_reward * usage_rate (weighted by performance)
    for stat in op_stats:
        if stat['count'] > 0:
            # Composite score: balance between average reward and reliability (inverse of std)
            reliability = 1.0 / (stat['std_reward'] + 1e-6)  # Lower std = higher reliability
            stat['composite_score'] = stat['avg_reward'] * (1 + reliability * 0.1) * stat['usage_rate']
        else:
            stat['composite_score'] = 0.0
    
    op_stats_sorted_by_composite = sorted(op_stats, key=lambda x: x['composite_score'], reverse=True)
    
    print("\nTop 5 operators to prioritize (based on composite score):")
    print("Composite score = avg_reward × (1 + reliability) × usage_rate")
    print(f"{'Rank':<6} {'Op':<4} {'Composite':<12} {'Avg Reward':<12} {'Usage':<10} {'Count':<10}")
    print("-" * 70)
    for rank, stat in enumerate(op_stats_sorted_by_composite[:5], 1):
        if stat['count'] > 0:
            print(f"{rank:<6} {stat['op_id']:<4} {stat['composite_score']:<12.2f} "
                  f"{stat['avg_reward']:<12.4f} {stat['usage_rate']:<10.2f}% {stat['count']:<10}")
    
    # Also show worst performers
    print("\nBottom 3 operators (consider reducing usage):")
    print(f"{'Rank':<6} {'Op':<4} {'Avg Reward':<12} {'Usage':<10} {'Count':<10}")
    print("-" * 50)
    for rank, stat in enumerate(op_stats_sorted_by_reward[-3:], 1):
        if stat['count'] > 0:
            print(f"{rank:<6} {stat['op_id']:<4} {stat['avg_reward']:<12.4f} "
                  f"{stat['usage_rate']:<10.2f}% {stat['count']:<10}")
    
    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY:")
    print("-" * 60)
    top_3_ops = [stat['op_id'] for stat in op_stats_sorted_by_reward[:3] if stat['count'] > 0]
    print(f"✓ Best average reward: Operators {', '.join(map(str, top_3_ops))}")
    
    top_3_total = [stat['op_id'] for stat in op_stats_sorted_by_total[:3] if stat['count'] > 0]
    print(f"✓ Highest total contribution: Operators {', '.join(map(str, top_3_total))}")
    
    top_3_composite = [stat['op_id'] for stat in op_stats_sorted_by_composite[:3] if stat['count'] > 0]
    print(f"✓ Recommended priority: Operators {', '.join(map(str, top_3_composite))}")
    
    # =====================
    # 3. Save Results
    # =====================
    print("\n" + "-" * 60)
    print("Saving results...")
    print("-" * 60)
    
    # Save embeddings
    embed_path = os.path.join(out_dir, "operator_embeddings.npy")
    np.save(embed_path, final_embeds)
    print(f"✓ Saved operator embeddings to: {embed_path}")
    
    # Save similarity matrix
    sim_path = os.path.join(out_dir, "operator_similarity_matrix.npy")
    np.save(sim_path, similarity_matrix)
    print(f"✓ Saved similarity matrix to: {sim_path}")
    
    # Save prediction results
    results = {
        'predicted_scores': pred_scores,
        'true_rewards': true_rewards,
        'chosen_operators': chosen_ops,
        'mse': mse,
        'mae': mae,
        'correlation': correlation
    }
    results_path = os.path.join(out_dir, "prediction_results.npz")
    np.savez(results_path, **results)
    print(f"✓ Saved prediction results to: {results_path}")
    
    # =====================
    # 4. Model Recommendation Demo
    # =====================
    print("\n" + "=" * 60)
    print("Model Recommendation Demo")
    print("=" * 60)
    
    # Use model to predict on some sample contexts
    model.eval()
    with torch.no_grad():
        # Get a few sample contexts from the dataset
        sample_indices = np.random.choice(len(contexts), min(5, len(contexts)), replace=False)
        sample_contexts = contexts[sample_indices]
        sample_contexts_tensor = torch.from_numpy(sample_contexts).float().to(DEVICE)
        
        # Get model predictions
        all_scores = model(sample_contexts_tensor)  # [B, num_ops]
        probs = torch.softmax(all_scores, dim=1)  # [B, num_ops] - probability distribution
        
        print("\nSample recommendations (using trained model):")
        print("-" * 80)
        for i, idx in enumerate(sample_indices):
            scores = all_scores[i].cpu().numpy()  # [num_ops]
            prob_dist = probs[i].cpu().numpy()  # [num_ops]
            
            # Get recommended operator (highest probability)
            recommended_op = np.argmax(prob_dist)
            recommended_prob = prob_dist[recommended_op]
            recommended_score = scores[recommended_op]
            
            print(f"\nSample {i+1} (context index {idx}):")
            print(f"  Recommended operator: {recommended_op} (probability: {recommended_prob:.4f}, score: {recommended_score:.4f})")
            
            # Show top 3 operators
            top3_indices = np.argsort(prob_dist)[-3:][::-1]
            print(f"  Top 3 operators:")
            for rank, op_id in enumerate(top3_indices, 1):
                print(f"    {rank}. Operator {op_id:2d}: prob={prob_dist[op_id]:.4f}, score={scores[op_id]:.4f}")
    
    # Overall recommendation statistics
    print("\n" + "-" * 60)
    print("Overall Model Recommendation Statistics:")
    print("-" * 60)
    
    # Predict on all contexts (or a large sample)
    model.eval()
    with torch.no_grad():
        # Use a subset for efficiency
        eval_size = min(1000, len(contexts))
        eval_indices = np.random.choice(len(contexts), eval_size, replace=False)
        eval_contexts = contexts[eval_indices]
        eval_contexts_tensor = torch.from_numpy(eval_contexts).float().to(DEVICE)
        
        all_scores = model(eval_contexts_tensor)  # [eval_size, num_ops]
        probs = torch.softmax(all_scores, dim=1)  # [eval_size, num_ops]
        
        # Count how many times each operator is recommended
        recommended_ops = torch.argmax(probs, dim=1).cpu().numpy()  # [eval_size]
        recommendation_counts = np.bincount(recommended_ops, minlength=NUM_OPERATORS)
        recommendation_rates = recommendation_counts / eval_size * 100
        
        print(f"\nModel recommendation frequency (based on {eval_size} random contexts):")
        print(f"{'Op':<4} {'Recommendation Rate':<20} {'Count':<10}")
        print("-" * 40)
        for op_id in range(NUM_OPERATORS):
            print(f"{op_id:<4} {recommendation_rates[op_id]:<20.2f}% {recommendation_counts[op_id]:<10}")
        
        # Compare with actual best operators
        print(f"\nComparison with actual best operators (by avg reward):")
        print(f"{'Op':<4} {'Model Rec Rate':<18} {'Actual Avg Reward':<18} {'Match?':<10}")
        print("-" * 55)
        for op_id in range(NUM_OPERATORS):
            mask = chosen_ops == op_id
            actual_avg_reward = true_rewards[mask].mean() if mask.sum() > 0 else 0.0
            match = "✓" if recommendation_rates[op_id] > 5.0 and actual_avg_reward > 2.0 else "✗"
            print(f"{op_id:<4} {recommendation_rates[op_id]:<18.2f}% {actual_avg_reward:<18.4f} {match:<10}")
    
    print("\n" + "=" * 60)
    print("All results saved successfully!")
    print("=" * 60)

def run(args):
    contexts, ops, rewards = parse_fuzz_log(args.log)
    # display_parse_data(contexts, ops, rewards)
    dataset = OperatorDataset(contexts, ops, rewards)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = OperatorPolicyNet(
        context_dim=CONTEXT_DIM,       # 10
        num_ops=NUM_OPERATORS,         # 16
        op_embed_dim=OP_EMBED_DIM,     # 10 (each operator has 10-dim embedding)
        hidden_dim=HIDDEN_DIM,         # 128
    ).to(DEVICE)

    print(model)
    print("\n" + "=" * 60)
    print("Training started...")
    print("=" * 60)
    
    # Store initial embeddings for comparison (make a deep copy)
    initial_embeds = model.op_embedding.weight.detach().cpu().numpy().copy()
    
    # Train model
    pred_scores, true_rewards, chosen_ops = train(model, loader)
    
    print("\n" + "=" * 60)
    print("Training completed. Generating results...")
    print("=" * 60) 
    
    # Get final embeddings
    model.eval()
    final_embeds = model.op_embedding.weight.detach().cpu().numpy()

    # Call display_result with all required parameters
    display_result(
        final_embeds=final_embeds,
        initial_embeds=initial_embeds,
        pred_scores=pred_scores,
        true_rewards=true_rewards,
        chosen_ops=chosen_ops,
        contexts=contexts,
        model=model,
        out_dir=args.out_dir
    )



# =====================
# Main
# =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="fuzzing log file")
    parser.add_argument("--out-dir", default="./out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()