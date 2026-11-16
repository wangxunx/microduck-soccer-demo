# Genesis 上的 football 踢球训练（AMD GPU）

在本机 AMD GPU 上，用 Genesis 训练能看到球的 Microduck 踢球策略。导出的 ONNX 和 manifest 可以直接替换 soccer 游戏里的 `football.onnx` / `football.json`。

`microduck-rl-genesis` 只读引用（`soccer_rl/__init__.py`，可用 `MICRODUCK_GENESIS_REPO` 覆盖路径），那个仓库没有任何改动。

## 文件

| 文件 | 作用 |
|---|---|
| `soccer_rl/football_cfg.py` | **配方**：运行时契约、球、重置分布、奖励、DR，每个值都写了依据 |
| `soccer_rl/football_env.py` | 环境，继承 `MicroduckVelocityEnv`，只覆盖和 football 相关的部分 |
| `soccer_rl/assets/ball_game.xml` | 游戏用的球（100 mm、30 g、`solref .03 .4`） |
| `train_football.py` | PPO 训练（rsl-rl，超参和行走任务相同） |
| `export_football.py` | 导出 ONNX（归一化烘焙进图，并和 torch 输出对比），同时生成游戏 schema 的 manifest |
| `tests/test_contract.py` | 部署契约测试：观测 55–60 和**游戏真实的 JS `footballCommand()`** 逐值比对，外加执行器契约检查 |
| `evaluate_candidate.sh` | 一条命令完成：导出 → S4 双引擎踢球对比 → 游戏自带的评估器（真实比赛）→ 和原作者模型对比 |
| `summarize_game_eval.py` | 汇总游戏评估器的结果，并和基线对比 |

## 用法

```bash
PY=../../microduck-rl-genesis/.venv/bin/python
HIP_VISIBLE_DEVICES=1 $PY tests/test_contract.py
HIP_VISIBLE_DEVICES=1 $PY train_football.py -e smoke -B 64 --max-iterations 5      # 冒烟测试，必须先跑
HIP_VISIBLE_DEVICES=1 $PY train_football.py -e football-v1 -B 4096 --max-iterations 2000
./evaluate_candidate.sh football-v1 1999 1
```

多卡机器上用 `HIP_VISIBLE_DEVICES` 指定一张空闲的卡。注意：HIP 设备编号和 `rocm-smi` 的编号不一定一致，可以用 `torch.cuda.get_device_properties(0).pci_bus_id` 对照 PCI 总线号确认。

## 设计要点（依据见 `../experiments/` 下的 S1–S4）

- **运行时契约**和游戏的 football manifest 完全一致：
  - 61 维观测 / 14 维动作，50 / 200 Hz
  - 观测 48–54 为 0；55–60 是机体 yaw 坐标系下的球位置 xy、球速度 xy、锁定的射门方向 xy，裁剪到 ±3
  - 执行器延迟 4 个物理步（训练时在 3–5 之间 DR），1.75 A 限流，关节速度延迟 1 个控制步
  - 游戏执行 1.2 s 后交给 stand 策略
  - 已由 `test_contract.py` 对照游戏的 JS 函数验证，误差 9.6e-8
- **场景**：全碰撞机器人加游戏用的球；每个物理步之前施加 S3 拟合的滚动和扭转阻力（μ_roll 0.003，w_eps 16 rad/s）。
- **重置分布**：覆盖游戏 AI 的触发条件（球在前方 9–18 cm、侧向 ±8.5 cm、射门方向偏差 ±0.18 rad），再适当放宽；另有 10% 的用例球够不着，正确做法是站稳不动。机器人带一点初速度，球也可能在缓慢滚动。
- **回合长度 1.6 s**，从 0.8 s 起加入"收势"奖励，让 1.2 s 交接给 stand 时姿态是可恢复的。
- **奖励**：
  - 球沿射门方向的速度，以目标速度 1.2 m/s 为上限
  - 超速惩罚、横向偏离惩罚
  - 触球前给脚靠近球的引导奖励
  - 站姿、高度、直立、至少一只脚着地
  - 收势阶段的姿态和速度
  - 和行走任务相同的正则项
  - 机器人位移惩罚，防止追着球走
  - 跌倒时一次性惩罚 −5
- **DR**：继承行走任务的全部 DR；另加球质量 ×0.8–1.25（也间接改变冲量传递，见 S4）、球摩擦 ×0.7–1.3、滚动阻力 ×0.7–1.4、球观测噪声 ±5 mm。
- **验收**：最终以游戏的 MuJoCo 结果为准（S4 发现两个引擎之间的球速比不是常数）。

## 训练记录和结果（2026-11-15）

所有版本都用同一套评估：`evaluate_candidate.sh`，包括 S4 两个引擎各 18 个用例，以及游戏自带的 `evaluate-football.mjs` 全部 11 个场景（单人、2v2、边角）。

| 版本 | 改了什么 | 进球 / 射门 | 踢球引起的跌倒 | 强制归位 | 交接后倾角（S4 中位数） | 结论 |
|---|---|---:|---:|---:|---:|---|
| 原作者 `football.onnx` | – | 61 / 68 | 0 | 0 | 5.6° | 基线 |
| v1 | 初始配方 | – | – | – | – | ❌ 学成了站着不动：站姿奖励约 8.5/步，踢球奖励最多 3/步 |
| v2a | 踢球奖励 10、触球奖励 2、站姿奖励减到约 3.5/步 | 74 / 81 | 4 | 0 | 30.9° | 进球多，但交接后不稳 |
| v2b | 踢球奖励 20、触球奖励 5 | 47 / 62 | 1 | 0 | – | 踢太猛（2.7 m/s）；MuJoCo 里左脚偏 −14° |
| v3 | v2a 基础上：超速惩罚 −10、横向惩罚 −6、转身惩罚 −1 | 59 / 70 | 0 | 0 | 42.6° | 方向准，但交接后大幅摇晃 |
| v4（从头训练） | v3 + 交接放进训练 | 44 / 62 | 4 | 0 | 11.7° | 2000 轮不够 |
| v4ft-v2a | v2a 用 v4 配方微调 1000 轮 | 76 / 83 | 1 | **1** | 6.9° | 进球最多，但有一次没站起来被强制归位 |
| **v4ft-v3** ✅ | v3 用 v4 配方微调 1000 轮 | **68 / 75** | **0** | **0** | 12.5° | **当前最佳** |

当前最佳模型：`exports/football-v4ft-v3-2998.onnx` 和同名 `.json`（manifest 符合游戏的 schema）。

### 关键发现

1. **交接兼容性是最主要的问题。** v3 在自己控制的 1.2 s 内倾角不到 15°，可一交给官方 stand 策略，两个引擎里都会立刻晃到 30–45°，之后能不能站稳带有混沌性，取决于很小的差异。解决办法：`HANDOFF_IN_LOOP=True`，1.2 s 后训练环境改由官方 stand 和 walk 策略控制，规则与游戏一致。两个 ONNX 通过 `soccer_rl/onnx_mlp.py` 转成 torch，在 GPU 上批量运行，误差约 2e-5。
2. **动作越激进，sim2sim 差距越大。** v2b 的左脚偏差只在 MuJoCo 里出现；改球的接触参数（condim、摩擦锥类型、摩擦优先级）都消不掉，力度降下来后偏差就消失了。在两个引擎之间，力度比约为 1.0。
3. **从已经会踢的策略开始微调，比从头训练有效得多。** 带交接后回合长 2.5 s，踢球窗口只占约一半的样本。

### 可以继续改进的
- 力度：游戏里约 1.9 m/s，目标 1.2。
- 方向：MuJoCo 里平均偏 +5°。
- 交接后倾角 12.5°，原作者是 5.6°。
- 训练时机器人从静止开始；游戏里踢球前是在走路的。可以用行走策略先跑一段，再从走动状态开始训练。

## 行走训练（2026-11-15）

| 文件 | 作用 |
|---|---|
| `train_walk.py` | 在 Genesis 上继续训练行走策略（直接使用 microduck-rl-genesis 的 `MicroduckVelocityEnv`），支持 `--resume`、`--curriculum-start-iter`、`--model allcollisions`，以及用 `--set` 覆盖配方参数 |
| `export_walk.py` | 导出单文件 ONNX（归一化烘焙进图，并与 torch 输出对比） |
| `evaluate_walk.sh` | 在游戏的 MuJoCo 里评估：S1 的 14 个场景、从静止开始的指令扫描，以及配合射门模型打整场比赛（游戏评估器的 `WALK_CANDIDATE`） |
| `summarize_walk_eval.py` | 对比多个行走策略 |

| 版本 | 改动（都从 Genesis 行走 checkpoint 2999 继续） | 0.3 m/s 实际速度 | 1.0 rad/s 实际转速 | 比赛进球 / 跌倒 |
|---|---|---:|---:|---:|
| 官方 `BEST_alpha_walking` | – | 0.112 | 0 | 68 / 10 |
| **walk-A** ✅ | 原配方再训 3000 轮 | 0.107 | **1.02** | **71 / 2** |
| walk-B | 速度和转向跟踪的 std 收紧 | 0.096 | 0.40 | 59 / 0 |
| walk-C | B + 全碰撞模型 + 转向范围 ±1.5 | 0.102 | −0.02 | 41 / 1 |
| walk-D2 | 在 walk-A 基础上放宽步幅（姿态 std、平滑惩罚、前进跟踪权重） | **0.246** | 0.89 | 29 / 6 |

- 已加入游戏：`?walk=genesis-v2`（即 walk-A，`exports/walk-A-5998.onnx`）。
- 收紧跟踪的 std 会让转向能力消失。前进速度卡在约 0.1 m/s，限制来自步态相关的奖励设置，放宽后（D2）速度提高约 2.3 倍。
- D2 在比赛里表现差，原因是射门模型只在"标准站姿、静止"状态下训练过，D2 停到球前时的姿态和转动它应付不了。下一步：训练射门时先让行走策略走一段，再接射门。
