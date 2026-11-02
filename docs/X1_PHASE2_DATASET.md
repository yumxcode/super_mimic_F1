# X1 Phase-2 数据规模与训练预算决策

日期：2026-11-02。本文回答两个问题：第二阶段用多大规模的动作库、训多少迭代、花多少钱。
依据：SuperMimic 论文（arXiv 2610.28480）+ allmimic 仓库（Shimingyi/allmimic，本项目上游）实测数字
（提取报告：`.meta-agent/research/subtask-c7cedd6d/report.md`），以及本项目 4090D 单卡实测
（TASK_20261102_068：0.55 s/iter = 6.55k iter/h，¥5.76/h，24 GB 显存）。

## 1. 论文与上游仓库的数字

| 来源 | 数据 | 训练 | 结果 |
|---|---|---|---|
| SuperMimic 论文（G1） | BONES-SEED 264.56 h / 129,514 clips（过滤自 142,220） | 1×H100 3 h | 超过 SONIC 的 return 0.90，129,514 段中 25 段失败 |
| SuperMimic 论文（G1） | 同上 | 2×H100 9.8 h | 0/129,514 失败 |
| SuperMimic 规模消融 | 1 h → 264 h 库（264×） | 0.54 → 1.89 H100-h | 到目标 return 的算力增长 3.5× 而已：**库规模买的是 clip 覆盖率，不是 return** |
| allmimic README（1×4090） | LAFAN1 40 clips（约 10 min） | 0.42 h → 0.85；**0.83 h → 0/40 失败**（return 0.91） | 单卡、几十段数据即可全库零失败 |
| allmimic README（2×4090） | BONES-SEED 全量 176k iter / 6.6 h | 26.7k iter/h（双卡） | 306 段硬动作 0 失败；OOD hold-out 100%/0.884 |
| 本项目一阶段（4090D） | 41 clips / 2.06 min | 3000 iter / 0.63 h | in-sample EVAL **0.9205，0/41 失败** |

结论要点：
- 论文 264 h/129k 段是"覆盖率"口径（目标是库内每段都被跟踪）；**return 水平由几百段即可达到**
  （消融端点 1 h 库已接近饱和）。
- 一阶段 41 段库、3000 iter 已经 0/41 失败、EVAL 0.9205——说明该管线在"走"这一活动上数据量不是瓶颈。
- 二阶段引入走/慢跑/跑/转身/舞蹈等多活动，段数需求由**活动多样性与 held-out 泛化评测**决定，
  而不是 return。

## 2. 决策

### 数据规模：目标 **330 段左右净入库（过验收门后），约 33-40 分钟**，held-out 12%

| 活动类别 | 目标净段数 | 源（CMU 为主） | 说明 |
|---|---|---|---|
| 行走（含变速/持物） | ~150 | 07/08/16/35/38/49/82/91（已有 41 段种子）+ 补充同族 | 主体分布 |
| 慢跑/跑步 | ~90 | 09（run ×12）、35 族快走、HDM05 jog/run | 大摆臂、屈肘、腾空相 |
| 转身/侧移/变向 | ~40 | 16 族（stepping/turns）、CMU 49 | 偏航跟踪 |
| 舞蹈/全身大幅 | ~50 | 137（dance）、143 等 | 手臂过头（可达边界压力测试）、节奏变化 |

- 每段沿用一阶段窗口协议：6-10 s 稳态窗口，50 Hz 重采样。
- held-out：**不同 subject/活动的 40 段（≈12%）**单独入库（如 91 族行走若不入训则整族 held-out、
  部分 09 run、部分 137 dance），供 sim2sim 泛化评估；held-out 段同样过验收门。
- 存储红线（契约）：SMPL-H 原始 npz（~0.5 MB/段，峰值 ~200 MB）→ 转 stageii → 批量重定向 →
  **原始与中间文件用完即删**；最终只保留训练 npz（~15-20 MB）+ meta + held-out npz。

### 训练：**60k iter 默认，100k iter 硬顶**

- 0.55 s/iter ⇒ 60k = 9.2 h ≈ **¥53**；100k = 15.3 h ≈ **¥88**。
- 提前停条件：连续两次 EVAL（每 4k iter）in-sample 失败 = 0 且 return ≥ 0.90 → 可在 40-60k 收工。
- allmimic 默认 320k iter 是 BONES 全量口径；我们 330 段库按 LAFAN-40 的对数外推
  （40 段 0.83 h 全过 → 330 段多样性更高，60k iter 是 3000 iter 一阶段基准的 20 倍、
  LAFAN 口径的 ~7 倍），足够且留了 100k 的余量。
- eval cadence、checkpoint 间隔沿用 gm_x1_train.py 现有配置。

### 预算（总硬上限 ¥150）

| 项 | 估算 |
|---|---|
| 一阶段已花（068 等任务） | ~¥10 |
| 二阶段训练 60k iter | ¥53 |
| 余量（延长到 100k、重试、评测复跑） | ≤¥25 |
| **合计最坏情形** | **≤¥88+10 ≈ ¥98 < ¥150** ✓ |

若需超过 ¥150（例如 100k 后仍未收敛要加训），按契约停下询问用户。

## 3. 风险与对策

- **4090D 未被上游验证过**：CUDA-graph/BPTT 编译若失败 → 回退一阶段已验证的配置路径
  （068 就在这张卡上跑通了 3000 iter，风险实际很低）。
- **value clip / 平衡采样**：上游在长跑（>6 h）上发现无 value_clip 会发散；我们 ≤15 h，
  沿用一阶段 config（已含默认值），训练中监控 critic loss，异常即停（契约要求）。
- **数据多样性不足的风险**：330 段若集中在 CMU 少数 subject，policy 可能过拟合到步态风格；
  段数分布按 subject 打散，held-out 用整族隔离。

## 4. 数据下载与处理路径（任务 3/4 的执行计划）

1. 逐文件镜像 `https://hf-mirror.com/datasets/realdream-ai/AMASS/resolve/main/raw/CMU/<sub>/<name>_poses.npz`
   （本机 huggingface.co 不通，hf-mirror.com 可用；写入 `data/amass/CMU_raw/`，gitignored）。
2. SMPL-H → stageii SMPL-X 转换复用 `~/code/X1_29_AMP/data/cmu_select_walk.py` 的 `convert()`
   （透传 import，不改 X1_29_AMP）；窗口选择放宽到各活动各自的稳态判据（跑步用速度带 + 步频）。
3. 批量重定向走一阶段管线（现含 `--arm-mode refine` 手臂修复）+ `tools/x1_retarget_gates.py` 验收门；
   未过门的段剔除并从候选池补齐到 330。
4. 产出 `motions/x1_phase2_train.npz` + `motions/x1_phase2_heldout.npz` + meta（含每段 scorecard）。
