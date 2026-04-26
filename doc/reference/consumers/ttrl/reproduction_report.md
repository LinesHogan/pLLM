# TTRL 复现评估报告

## 结论

当前 tLLM TTRL consumer 是一个无 verl 依赖的 TTRL-inspired 最小原语实现，不是端到端 TTRL 复现。它保留了“同一 prompt 多样本多数投票生成伪奖励”和“用伪奖励做一次轻量 test-time 更新”的方向，并补齐了数学 toycase 需要的 boxed answer 抽取/normalize、多数标签/比例结构化结果、vote/sample 分离索引、以及 toy `GenerationTraceBatch -> RewardOutcomeBatch` 输出；但没有复现上游的完整 test-time RL 训练闭环：伪 ground truth 写回 verl batch、PPO/GRPO actor 更新和指标回填。

因此，评估结论是：核心思想部分对齐，工程实现刻意脱离 verl；若目标是可运行的 TTRL 算法复现，当前实现仍有关键差距。

## 上游机制摘要

上游 TTRL 的核心目标是在没有真实标签的测试数据上训练：先对每个 prompt 生成多个 rollouts，用多数投票得到伪 ground truth，再把该伪标签交给 reward function 给训练样本打 reward，最后走 verl 的 PPO/GRPO actor 更新。

关键实现位于 `<upstream>/verl/verl/trainer/ppo/ttrl_utils.py`：

- `select_top_k_per_prompt` 从每个 prompt 的 `n_votes_per_prompt` 个投票样本中选前 `n_samples_per_prompt` 个进入训练。
- `apply_ttrl_gt` 解码每个 rollout，按 prompt 分组，调用 `_batch_majority_vote` 得到每个 prompt 的 majority label，并把 batch 中的 `reward_model.ground_truth` 替换成 majority label，同时保留原始 ground truth 只用于指标。
- `_majority_vote` 不是对完整 completion 文本投票，而是先 `extract_answer`，过滤无法抽取答案的样本，再 `simplify_expression_string`，最后 `Counter(...).most_common(1)`。
- reward 侧位于 `<upstream>/verl/verl/utils/reward_score/ttrl_math/__init__.py`，`compute_score` 会从模型响应中抽取 boxed answer，并用 mathd/sympy/latex 等规则和伪 ground truth 比较，返回 `score`、`format_score`、`acc`、`pred` 等信息。
- 训练入口在 `<upstream>/verl/verl/trainer/ppo/ray_trainer.py` 的 TTRL 分支：先按 `n_votes_per_prompt` 生成，再应用 majority gt，随后下采样训练 rollouts，并调用正常 actor update。训练后再恢复 original gt 计算 TTRL 诊断指标。

上游公开脚本默认依赖 8 卡、大 batch、长 response、verl/Ray/FSDP/vLLM 训练栈。TTRL 的算法目的不是引入 verl 本身，而是保留“无标签测试样本 -> self-generated majority reward -> online RL update”的闭环。

## tLLM 映射

本仓库实现位于 `tllm/consumers/ttrl/consumer.py`：

- `TTRLSample` 保存 `prompt_idx`、`sample_idx`、`completion`。
- `majority_vote_rewards` 按 `prompt_idx` 分组，只对最后一个 `\boxed{...}` answer 抽取/normalize 后计数，并给 majority label 匹配样本 reward=1，其余 boxed 样本 reward=0。
- unboxed completion 不作为可判定答案参与投票；结构化结果和 reward outcome 会标记 `missing-boxed-answer`。
- `build_ttrl_reward_result` 返回 majority label、majority ratio、每个样本的 normalized boxed label、reward、skip reason、以及被选中的训练索引。
- `build_ttrl_reward_outcome_batch` 将完成的 `GenerationTraceBatch` 转换为 `RewardOutcomeBatch`，保留 `request_id`、`prompt_idx`、`sample_idx`、`output_index`、`group_id` 等 identity 字段。
- `build_ttrl_closed_loop_intent` 将同一 trace 转换为 reward、retry control、update barrier 三类 runtime-native intent batch；多数派样本被 accept，非多数派或缺失答案样本走 retry/skip 路径。
- `select_ttrl_training_indices` 对齐上游 `select_top_k_per_prompt`，支持从 `n_votes_per_prompt` 个投票样本中选择前 `n_samples_per_prompt` 个训练样本。
- `TTRLClosedLoopIntent.training_rewards()` 只暴露被上游式 selection 选中的训练 reward，因此 toy adapter 更新计数不会把纯投票样本误当成训练样本。
- `TTRLEngine` 用一个 `LowRankAdapter(in_dim=hidden_size, out_dim=1)` 预测 scalar reward，并以 MSE 训练。
- `TTRLConsumer.flows()` 声明 completed generation trace 到 reward outcome 的 out-of-band toy flow；`rewards_for_samples`、`reward_result_for_samples`、`reward_outcome_batch_for_trace`、`write_reward_outcomes_for_trace` 和 `update_from_rewards` 暴露了 reward 构造、结构化结果、outcome 输出和 adapter 更新 API。
- 单元测试覆盖了 package export、多数奖励、tie-breaking、无 verl import、adapter 更新计数和 disabled no-op。

这满足了“不要引入 verl”的约束；测试也明确保证导入 tLLM TTRL 模块不会加载 `verl`。

## 对齐情况

已对齐的部分：

- 对同一 prompt 的多 completion 做多数投票。
- 多数投票基于 boxed answer 抽取/normalize 后的答案；不同推理文本但同 boxed answer 会被合并。
- 裸文本答案不会被当成可判定答案；unboxed completion 会被 skip。
- 给多数派样本分配正 reward，少数派样本分配零 reward。
- 将 completed generation trace 转成 reward outcome batch，并保留 identity 字段。
- 暴露 majority label、majority ratio、sample label、reward、selected indices。
- 支持 `n_votes_per_prompt` / `n_samples_per_prompt` 的上游式下采样索引。
- 不依赖 verl。
- 支持轻量本地更新，并记录 `update_count`、`loss_count`、`reward_count`。

未对齐或仅弱对齐的部分：

- 上游 reward 是“模型响应 vs majority pseudo ground truth”的 task reward；tLLM 现在只对 boxed/math toycase 做 toy 对齐，仍缺少完整 latex/sympy/mathd grader/self-check。
- 上游更新 actor policy；tLLM 只训练一个本地低秩 scalar score adapter。当前 trace-to-outcome flow 只产出 reward outcome，adapter 更新本身不会自动改变后续 generation。
- 上游保留 original gt 做诊断指标；tLLM 只有 majority ratio，没有 label accuracy、reward accuracy、pass@n 等完整指标。

## 原仓库轻量复现可行性

我尝试在现有 repo virtualenv 中导入和运行上游核心逻辑。完整上游入口不适合轻量运行，原因是：

- verl PPO 入口依赖 Ray/Hydra/OmegaConf/FSDP/vLLM 等训练栈。
- 当前环境缺少 `omegaconf`，上游包初始化还缺 `pkg_resources`。
- 上游 `ttrl_math` reward 依赖 `latex2sympy2_extended`，当前环境未安装。
- 官方脚本面向多卡训练，不适合本评估的轻量 smoke。

因此没有运行完整上游 PPO case。为避免引入依赖或下载，我用内存 stub 绕开上游可选 math 依赖，直接加载上游 `ttrl_utils.py`，只验证其多数投票和样本选择语义。

## 执行过的检查

使用 repo virtualenv 运行：

- `python -m pytest -q test/consumers/ttrl/test_ttrl_consumer_unit.py`
  - 早期更新结果：`12 passed`
  - 当前 closed-loop adoption 结果：`17 passed`
- `python -m pytest -q test/consumers/ttrl test/test_runtime_generation_trace_unit.py test/test_runtime_reward_outcome_unit.py`
  - 更新后结果：`21 passed`
- `python -m pytest -q test/consumers/ttrl test/consumers/fttt`
  - 当前结果：`34 passed`
- 依赖探测：
  - `torch`、`transformers`、`vllm`、`sympy`、`ray` 可用。
  - `latex2sympy2_extended`、`omegaconf` 不可用。
- 本地缓存模型探测：
  - 存在 `Qwen/Qwen2.5-0.5B-Instruct` 等缓存模型。
- 上游 toycase：
  - 直接加载上游 `ttrl_utils.py`，用内存 stub 提供 `extract_answer`、`simplify_expression_string`、`grade`。
- 真实模型 smoke：
  - 使用本地缓存的 `Qwen/Qwen2.5-0.5B-Instruct`，离线加载，GPU 上生成 4 个短 completion，再调用 tLLM reward 和 adapter update。

## Toycase 结果

构造三个 completion：

- `First derivation. Therefore \boxed{4}`
- `Second derivation. Final: \boxed{4}`
- `Wrong path. Therefore \boxed{5}`

上游多数投票语义：

- majority label: `4`
- majority ratio: `0.6666666666666666`
- reward: `[1.0, 1.0, 0.0]`

tLLM 更新后按 boxed answer 抽取/normalize 后答案投票：

- reward: `[1.0, 1.0, 0.0]`
- majority label: `4`
- majority ratio: `0.6666666666666666`

unboxed skip toycase：

- `Answer: 4`
- `Therefore \boxed{5}`
- `Answer: 4`

tLLM boxed-only 结果：

- reward list: `[0.0, 1.0, 0.0]`
- majority label: `5`
- unboxed samples: `skip_reason=missing-boxed-answer`

trace-to-outcome toycase：

- 输入为 `GenerationTraceBatch` 的 completed entries。
- 输出为 `RewardOutcomeBatch`。
- `request_id`、`prompt_idx`、`sample_idx`、`output_index`、`group_id` 保持 trace identity。
- boxed majority 样本写入 `judge=ttrl_boxed_majority`、`reward=1.0`、`is_correct=True`。
- boxed minority 样本写入 `reward=0.0`、`is_correct=False`。
- unboxed 样本写入 `reward=None`、`is_correct=None`、`skip_reason=missing-boxed-answer`。

closed-loop intent toycase：

- 输入为包含同一 `group_id` 的 completed generation trace。
- 输出同时包含 `RewardOutcomeBatch`、`RetryControlBatch` 和 `UpdateBarrierBatch`。
- majority completion 的 retry action 为 `accept`。
- non-majority completion 的 retry action 为 `retry`，并带 `retry_after_update=True`。
- update barrier 的 `ready_count` 只统计上游式 selected training rows。
- `consume_bundle()` 在可选 sinks 存在时会写入 reward、retry 和 update barrier 三类 batch。

样本选择对照：

- 上游 `select_top_k_per_prompt` 对 2 个 prompt、每个 4 个投票样本、每个保留 2 个训练样本，返回索引 `[0, 1, 4, 5]`。
- tLLM `select_ttrl_training_indices` 返回同样索引 `[0, 1, 4, 5]`。

adapter update：

- 用 `[1.0, 1.0, 0.0]` 更新 tLLM adapter 后，参数发生变化。
- stats 为 `update_count=1`、`loss_count=1`、`reward_count=3`。

## 真实模型 smoke 结果

使用本地缓存 `Qwen/Qwen2.5-0.5B-Instruct` 运行两个极小离线生成：

1. 普通 boxed prompt，采样 4 条、最多 32 token：
   - 只有 1 条输出完整 `\boxed{4}`，其余被截断或未 boxed。
   - tLLM full-text reward: `[1.0, 0.0, 0.0, 0.0]`
   - answer-preprocessed reward: `[1.0, 0.0, 0.0, 0.0]`
   - adapter stats: `update_count=1`、`loss_count=1`、`reward_count=4`

2. 更受控的短格式 prompt，采样 4 条、最多 16 token：
   - responses: 全部为 `\boxed{4}`
   - extracted answers: `['4', '4', '4', '4']`
   - tLLM reward: `[1.0, 1.0, 1.0, 1.0]`

该 smoke 说明本地模型、reward 构造和 adapter 更新链路可以跑通；但它没有证明端到端 TTRL 复现，因为没有执行上游 PPO/GRPO actor update，也没有把 adapter 反馈到 generation。

## 风险和建议

主要风险：

- 把当前 consumer 称为“TTRL 复现”会过度声明。更准确的说法是“TTRL-inspired majority reward utility plus bounded score-adapter update”。
- 当前 boxed answer normalizer 覆盖 toycase，但没有上游完整数学等价判断；复杂 latex/sympy 等价表达式仍可能被拆成不同 label。
- adapter 更新没有接入 tLLM producer-consumer flows，对后续 token 分布没有直接影响。

建议若要提升为更接近 TTRL 的复现：

- 增加可插拔 grader/self-check，覆盖上游 mathd/sympy/latex 等数学等价。
- 明确 adapter 如何影响后续 generation；否则把它定位为诊断/score head，而不是 test-time RL policy update。
- 增加端到端小模型 smoke，将结构化 reward 结果传入轻量 adapter update 并验证统计计数。
