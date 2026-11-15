# S4：真实踢球策略的 sim2sim（MuJoCo 游戏 vs Genesis）

问题：同一个踢球策略、同样的初始状态和球，在游戏的 MuJoCo 里和在 Genesis 里，结果是否一致？

## 方法

- 54 个用例：3 种协议 × 左右脚 × 9 个球位。球位是名义位置（0.105, ±0.042）上下左右各 ±1.5 cm。机器人在原点，朝向 0；球用游戏的 100 mm / 30 g 足球。
  - **A**：官方 `kickL`/`kickR` 只跑 0.5 s（游戏默认的窗口），然后是零速度指令的 walk。
  - **B**：官方 `kickL`/`kickR` 跑满 3 s。
  - **C**：作者训练的 `football.onnx`。踢 1.2 s（4 步执行器延迟、1.75 A 限流、关节速度延迟 1 步、球信息放在观测 55–60，射门方向取触发时的朝向），然后 stand 1 s，再 walk。
- **MuJoCo**：直接走游戏真实的 `PhysicsWorld` 踢球路径（`kickRequested`），runtime 不做任何修改。
- **Genesis**：全碰撞机器人 + 游戏的球。按同样的时间表切换策略，每个环境各自切换执行器延迟、限流和关节速度延迟；stand 接管时的种子动作和游戏一致；电池 7.35 V，DR 关闭。球分两种：原生，以及加了 S3 拟合的滚动阻力。

```bash
python3 make_cases.py
cd ../../game && node ../experiments/s4_kick_sim2sim/run_mujoco.mjs ../experiments/s4_kick_sim2sim/results/mujoco.json
cd - && HIP_VISIBLE_DEVICES=1 ../../../microduck-rl-genesis/.venv/bin/python run_genesis.py --out results/genesis.json
../../../microduck-rl-genesis/.venv/bin/python analyze.py   # → results/report.md
```

## 结果（2026-11-14）

| 协议 | 引擎 | 跌倒 | 踢中 | 球峰值速度中位数 | 方向中位数 | 最大倾角 |
|---|---|---:|---:|---:|---:|---:|
| A 官方踢球 | MuJoCo（游戏） | 0 | 11/18 | 0.43 m/s | +1° | 7.0° |
| | Genesis + 滚动阻力 | 0 | 17/18 | 0.70 m/s | −8° | 6.2° |
| C football | MuJoCo（游戏） | 0 | 18/18 | 1.74 m/s | −15° | 15.1° |
| | Genesis 原生 | 1 | 18/18 | 1.51 m/s | −18° | 169°（就是那次跌倒） |
| | Genesis + 滚动阻力 | 0 | 18/18 | 1.51 m/s | −17° | 35.3° |

逐个用例配对比较（Genesis 相对 MuJoCo）：

| 协议 | 踢中/踢空一致 | 球速比（中位数 [最小, 最大]） | 方向差（中位数 / 最大） | 首次触球时间差 |
|---|---:|---|---|---:|
| A/B 官方踢球 | 12/18 | 1.76 [1.17, 3.15] | 4.6° / 7.7° | −0.02 s |
| C football | **18/18** | 0.88 [0.59, 1.22] | 4.4° / 13° | 0.00 s |

## 结论

1. **定性上能迁移。**
   - 触球时机完全一致，方向差中位数在 5° 以内，两种协议下各个球位的强弱排序也一致。
   - 机器人在 MuJoCo 里一次都没摔。Genesis 里 football 的倾角中位数和 MuJoCo 相同（11.8°），但长尾更重（p90 18–21° vs 13°），36 个用例里有 1 个跌倒。
   - 也就是说，Genesis 对激进踢球动作稍微更"苛刻"。对训练来说这是有利的方向：能在 Genesis 里站稳的策略，到 MuJoCo 里一般也没问题。
2. **球速的比例不是常数。**
   - 官方的轻踢约 **1.75 倍**，和 S3 的击球器测试一致：Genesis 更强。
   - 激进的 football 踢法约 **0.88 倍**，Genesis 反而更弱。
   - 所以不能用一个固定系数来标定力度：训练时要对球和接触参数做 DR，力度目标设成一个区间，最终力度以游戏里的评估为准。
3. **擦边球的判定不同。**
   - 官方踢球在网格最远一圈（球再远 1.5 cm）时，Genesis 能轻轻碰到，MuJoCo 踢空。Genesis 的有效踢球距离约远 1 cm。
   - 能看到球的 football 策略 18/18 全部一致，这支持我们训练能看到球的策略。训练时应该对球位做 DR，避免策略依赖只差几毫米的"脚尖擦到"。
4. **滚动阻力补偿是必需的。** 不加的话，Genesis 里球滚得太远（中位数 5.5 m vs 4.05 m）；加了以后剩下的差异主要来自出球速度本身。

## 对 football 训练环境的设计要求

- 机器人用全碰撞模型，球用游戏的足球，加上 S3 的滚动阻力力矩。
- 运行时契约和游戏的 football manifest 完全一致（61 维 / 14 维、观测 55–60、4 步延迟、1.75 A、关节速度延迟 1 步、1.2 s 窗口后交给 stand），这样导出后可以直接替换 `football.onnx`。
- DR：球的质量和摩擦、接触冲量（solref/阻尼）、球位（至少 ±2 cm）。
- 验收：本目录的 S4 脚本做回归测试，加上游戏自带的 `scripts/evaluate-football.mjs`（`FOOTBALL_CANDIDATE=...`）做最终验收。
