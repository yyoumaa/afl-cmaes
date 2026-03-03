# Fuzzing Context 信息建议

## 1. 当前测试用例信息（queue_cur）

### 基础特征
- **len** (u32): 输入文件长度
  - 意义：不同长度的输入可能需要不同的变异策略
- **prox_score** (u64): 接近度分数
  - 意义：反映测试用例的"价值"，高分用例可能需要更精细的变异
- **depth** (u64): 路径深度
  - 意义：深度越深，可能需要更激进的变异策略
- **exec_us** (u64): 执行时间（微秒）
  - 意义：执行时间长的用例可能需要不同的策略
- **bitmap_size** (u32): bitmap 中设置的位数
  - 意义：反映覆盖了多少代码路径
- **exec_cksum** (u32): 执行轨迹的校验和
  - 意义：可以区分不同的执行路径

### 状态标志
- **was_fuzzed** (u8): 是否已经被 fuzz 过
- **passed_det** (u8): 是否通过了确定性阶段
- **has_new_cov** (u8): 是否触发了新覆盖
- **var_behavior** (u8): 是否有可变行为
- **favored** (u8): 是否被标记为 favored

## 2. 全局 Fuzzing 状态

### 进度信息
- **total_execs** (u64): 总执行次数
  - 意义：fuzzing 的进度，早期和后期可能需要不同策略
- **queue_cycle** (u64): 队列轮次
  - 意义：当前是第几轮 fuzzing
- **cycles_wo_finds** (u64): 连续无新发现的轮次
  - 意义：如果长时间无发现，可能需要改变策略
- **queued_paths** (u32): 队列中的路径数
  - 意义：队列大小反映 fuzzing 的成熟度

### 发现统计
- **unique_crashes** (u64): 唯一崩溃数
- **unique_hangs** (u64): 唯一挂起数
- **unique_tmouts** (u64): 唯一超时数

## 3. 算子历史表现（stage_finds_*）

### 每个算子的统计
- **stage_finds_score_all[operator_num]**: 每个算子累计增益
  - 意义：哪些算子历史表现好
- **stage_finds_per_score[operator_num]**: 每个算子平均增加分数
  - 意义：算子的平均效果
- **stage_finds_times[operator_num]**: 每个算子变异前执行次数
  - 意义：算子的使用频率

## 4. 输出向量（output_vector）

### 当前状态（被注释，但可以启用）
- **output_vector_before[OUTPUT_DIM_SIZE]**: 操作前的输出向量
  - 意义：反映当前 fuzzing 状态（11维）
- **output_vector_after[OUTPUT_DIM_SIZE]**: 操作后的输出向量
  - 意义：反映操作后的状态

## 5. 推荐 Context 向量设计

### 方案 A：精简版（10维）
```python
context = [
    queue_cur->len / max_len,              # 归一化输入长度 [0, 1]
    queue_cur->prox_score / max_prox,     # 归一化接近度分数 [0, 1]
    queue_cur->depth / max_depth,         # 归一化深度 [0, 1]
    log(queue_cur->exec_us + 1) / 10,    # 归一化执行时间 [0, ~1]
    queue_cur->bitmap_size / MAP_SIZE,    # 归一化 bitmap 大小 [0, 1]
    total_execs / 1e6,                    # 归一化总执行次数 [0, ~1]
    queue_cycle / 1000,                   # 归一化队列轮次 [0, ~1]
    cycles_wo_finds / 100,                # 归一化无发现轮次 [0, ~1]
    queued_paths / 10000,                 # 归一化队列大小 [0, ~1]
    unique_crashes / 1000,                # 归一化崩溃数 [0, ~1]
]
```

### 方案 B：完整版（20+维）
```python
context = [
    # 测试用例特征 (6维)
    queue_cur->len / max_len,
    queue_cur->prox_score / max_prox,
    queue_cur->depth / max_depth,
    log(queue_cur->exec_us + 1) / 10,
    queue_cur->bitmap_size / MAP_SIZE,
    queue_cur->exec_cksum / UINT32_MAX,
    
    # 状态标志 (5维)
    queue_cur->was_fuzzed,
    queue_cur->passed_det,
    queue_cur->has_new_cov,
    queue_cur->var_behavior,
    queue_cur->favored,
    
    # 全局状态 (5维)
    total_execs / 1e6,
    queue_cycle / 1000,
    cycles_wo_finds / 100,
    queued_paths / 10000,
    unique_crashes / 1000,
    
    # 算子历史表现（可选，需要额外处理）
    # stage_finds_score_all[0] / max_score,
    # stage_finds_score_all[1] / max_score,
    # ...
]
```

### 方案 C：使用输出向量（11维）
```python
# 如果启用 output_vector_before
context = output_vector_before  # 直接使用 11 维输出向量
```

## 6. 修改建议

### 在 afl-fuzz.c 中记录 context

在 `fprintf(fp_output,"output_vector_after\n");` 之前添加：

```c
// 记录 context 信息
fprintf(fp_output,"context_vector\n");
fprintf(fp_output,"%u ", queue_cur->len);                    // 0: 输入长度
fprintf(fp_output,"%llu ", queue_cur->prox_score);           // 1: 接近度分数
fprintf(fp_output,"%llu ", queue_cur->depth);                // 2: 深度
fprintf(fp_output,"%llu ", queue_cur->exec_us);              // 3: 执行时间
fprintf(fp_output,"%u ", queue_cur->bitmap_size);            // 4: bitmap 大小
fprintf(fp_output,"%llu ", total_execs);                     // 5: 总执行次数
fprintf(fp_output,"%llu ", queue_cycle);                     // 6: 队列轮次
fprintf(fp_output,"%llu ", cycles_wo_finds);                  // 7: 无发现轮次
fprintf(fp_output,"%u ", queued_paths);                      // 8: 队列大小
fprintf(fp_output,"%llu ", unique_crashes);                   // 9: 崩溃数
fprintf(fp_output,"\n");
```

### 在 Python 中解析 context

修改 `parse_fuzz_log` 函数，解析 `context_vector` 行。
