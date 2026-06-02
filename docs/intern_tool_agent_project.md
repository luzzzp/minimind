# MiniMind-ToolRL: 工具调用智能体偏好对齐与强化学习训练框架

## 项目定位

面向数学计算、单位换算、天气查询、时间查询、汇率换算、文本翻译等可验证工具调用场景，基于 MiniMind 小语言模型构建 SFT -> DPO -> GRPO 的训练 pipeline。项目重点解决小模型在工具调用任务中的错工具、错参数、格式不稳定、最终答案不可验证等问题。

## 数据构建

运行：

```bash
cd scripts
python build_intern_tool_data.py --seed 42 --sft_size 5000 --dpo_size 2000 --rl_size 1000 --eval_size 200 --repair_ratio 0.15
```

产物：

- `dataset/intern_tool_sft.jsonl`: 标准工具调用轨迹，用于 SFT。
- `dataset/intern_tool_dpo.jsonl`: chosen/rejected 偏好对，用于 DPO。
- `dataset/intern_tool_agent.jsonl`: 带 tools 和 gt 的 rollout prompts，用于 GRPO。
- `dataset/intern_tool_eval.jsonl`: held-out 自动评测集。

DPO 负样本覆盖错误工具、缺失参数、近似错误参数、无工具硬答、格式异常和最终答案错误。SFT 数据使用结构化 `tool_calls` 字段，并混入约 15% 格式修复样本。GRPO 数据保留 `expected_tools`、`expected_arguments`、`expected_answer_values`，用于规则 Reward 和端到端评测。

## Reward 设计

`trainer/train_agent.py` 新增：

```bash
--reward_mode rule|rm|hybrid
--reward_model_path none
```

默认 `rule`，不依赖外部 Reward Model。规则 Reward 由以下部分组成：

- JSON / `<tool_call>` 标签合法性。
- 工具名称是否与 `expected_tools` 完全一致。
- 参数是否与 `expected_arguments` 完全一致。
- 最终答案是否命中 `expected_answer_values`。
- thinking 闭合、输出长度、重复 n-gram、额外工具调用、格式错误、未完成多轮调用惩罚。

如果后续接入外部 RM，可使用 `--reward_mode hybrid --reward_model_path /path/to/rm`。

## 4090 训练命令

环境：

```bash
conda create -n minimind python=3.10 -y
conda activate minimind
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

训练：

```bash
cd trainer
python train_full_sft.py --from_weight pretrain --save_weight intern_sft_v2 --data_path ../dataset/intern_tool_sft.jsonl --epochs 2 --batch_size 24 --max_seq_len 1024 --save_interval 200

python train_dpo.py --from_weight intern_sft_v2 --save_weight intern_dpo_v2 --data_path ../dataset/intern_tool_dpo.jsonl --epochs 2 --batch_size 8 --max_seq_len 1024 --learning_rate 1e-7 --beta 0.15 --save_interval 100

python train_agent.py --from_weight intern_dpo_v2 --save_weight intern_grpo_v2 --data_path ../dataset/intern_tool_agent.jsonl --epochs 1 --batch_size 2 --num_generations 4 --max_seq_len 1024 --max_gen_len 384 --loss_type grpo --reward_mode rule --reward_model_path none --save_interval 20
```

OOM 时优先调整：

- `--max_gen_len 384 -> 256`
- `--num_generations 4 -> 2`
- `--batch_size 2 -> 1`

## 自动化评测

```bash
cd scripts
python eval_intern_tool_agent.py --weight full_sft --eval_path ../dataset/intern_tool_eval.jsonl --save_badcases ../badcases_full_sft.jsonl
python eval_intern_tool_agent.py --weight intern_sft_v2 --eval_path ../dataset/intern_tool_eval.jsonl --save_badcases ../badcases_sft_v2.jsonl
python eval_intern_tool_agent.py --weight intern_dpo_v2 --eval_path ../dataset/intern_tool_eval.jsonl --save_badcases ../badcases_dpo_v2.jsonl
python eval_intern_tool_agent.py --weight intern_grpo_v2 --eval_path ../dataset/intern_tool_eval.jsonl --save_badcases ../badcases_grpo_v2.jsonl
```

输出指标：

- `json_valid`: 工具调用 JSON 合法率。
- `tool_name_ok`: 工具选择准确率。
- `arg_ok`: 参数准确率。
- `answer_ok`: 最终答案命中率。
- `success`: 端到端成功率。
- `format_bad`: 格式违规率。
- `avg_response_chars`: 平均回复长度。
- `tokens_per_second`: 生成速度。
- `failure_reasons`: 失败原因分布，用于 badcase 分析。

v2 目标：`success >= 60%`，`format_bad <= 3%`，`arg_ok >= 90%`，`answer_ok >= 60%`。

## 消融实验表

| 实验 | json_valid | tool_name_ok | arg_ok | answer_ok | success | format_bad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_sft baseline | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| intern_sft_v2 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| intern_dpo_v2 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| intern_grpo_v2 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| intern_grpo, no DPO | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| intern_grpo, CISPO loss | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |

## 简历描述

**MiniMind-ToolRL：基于 MiniMind 的工具调用智能体偏好对齐与强化学习训练框架**  
自主研发 / 个人项目　　　　　　　　　　　　　　　　　　　　　　　　　　2026.05 -- 至今

**项目描述：** 面向数学计算、单位换算、天气查询、时间查询、汇率换算、文本翻译等可验证工具调用场景，基于 MiniMind 小语言模型构建从监督微调到偏好对齐再到强化学习优化的完整训练 pipeline。项目围绕工具选择错误、参数生成不稳定、格式不规范、最终答案不可验证等问题，设计 **SFT -> DPO -> GRPO** 三阶段训练流程，并通过规则化 Reward 与自动化评测闭环提升模型在工具调用任务中的端到端成功率与输出规范性。

**承担工作：**

- **训练 Pipeline 构建：** 基于 MiniMind 原生训练框架，设计并实现工具智能体对齐流程，依次完成 SFT 指令微调、DPO 偏好对齐与 GRPO 强化学习优化；将 Pretrain 作为基座初始化阶段，重点强化模型在工具调用、参数生成和结果整合中的稳定性。
- **数据工程与偏好数据构建：** 针对多工具调用场景构造高质量合成数据，覆盖单工具、多工具、错误工具、错误参数、无工具硬答、格式异常等样本类型；自动生成 SFT 标准轨迹、DPO chosen/rejected 偏好对与 GRPO rollout prompts，形成可复现的数据构建链路。
- **Reward 函数设计：** 设计面向工具调用的规则化 Reward，将 JSON 格式合法性、工具选择准确性、参数校验、工具执行结果命中、最终答案匹配、重复惩罚、长度约束和未完成惩罚统一为优化信号，降低对外部 Reward Model 的依赖，并缓解 Reward Hacking 与格式投机问题。
- **消融实验与训练调优：** 对比 SFT-only、SFT+DPO、SFT+DPO+GRPO 等不同训练阶段效果；进一步针对 Reward 组成、DPO beta、GRPO/CISPO loss、num_generations、max_gen_len 等关键参数开展消融实验，分析不同策略对工具调用准确率和输出稳定性的影响。
- **自动化评测与效果验证：** 搭建非交互式评测脚本，在 held-out 工具任务集上统计工具调用合法率、工具选择准确率、参数准确率、最终答案准确率、端到端成功率、格式违规率与平均生成长度；最终在单张 4090 上完成训练与评测闭环，并将实测指标沉淀为简历量化结果。

量化指标在训练完成后替换为真实结果，不提前编造提升百分比。
