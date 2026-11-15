# S1：sim2sim 验证（Genesis 训练的策略 → 游戏里的 MuJoCo）

问题：在 Genesis（AMD GPU）上训练的策略，导出成 ONNX 后放进 soccer 游戏的 MuJoCo 运行时，能不能正常工作？

## 方法

- 5 个行走策略：官方 `BEST_alpha_walking.onnx`（用 mjlab 训练）+ 4 个 Genesis 训练的策略（`velocity`、`velocity-reproduced`、`rough`、`backlash`）。
- 14 个指令场景（`make_scenarios.py` → `scenarios.json`），两个引擎用完全相同的指令时间线：
  - 站立 20 s；前进 0.15 / 0.30；后退 0.30；原地转向 ±1.0 和 +1.5（游戏的最大转速）；弧线 0.2 m/s + 0.5 rad/s
  - 6 条仿游戏操作的随机指令序列，每条 60 s
- 三种运行环境：
  - `mujoco(game)`：游戏原样的 `PhysicsWorld`（MuJoCo 3.11 WASM、全碰撞机器人、BAM、跌倒后由 stand 策略起身），只替换 `runtime.policies.walk`。
  - `genesis[game]`：Genesis 1.2.2，执行器按游戏设置对齐（7.35 V、无压降、无电流限制、无执行器和观测延迟）。剩下的差异只有物理引擎本身，以及碰撞模型（walk 模型只有脚底碰撞，游戏用全碰撞模型）。
  - `genesis[train]`：Genesis 用训练时的执行器设置（随机电压、电压降、1.75 A 限流、15–30 ms 延迟），DR 关闭。
- 指标统一由 `analyze.py` 计算：跌倒次数、机体坐标系下的 vx/wz 跟踪、站立漂移、最大倾角。

## 复现

```bash
python3 make_scenarios.py
cd ../../game   # 资源路径相对 game/ 解析
node ../experiments/s1_sim2sim/run_mujoco.mjs genesis_velocity ../../microduck-rl-genesis/policies/velocity.onnx ../experiments/s1_sim2sim/results/mujoco_genesis_velocity.json
cd ../experiments/s1_sim2sim
HIP_VISIBLE_DEVICES=5 ../../../microduck-rl-genesis/.venv/bin/python run_genesis.py --actuator game  --out results/genesis_game.json
HIP_VISIBLE_DEVICES=6 ../../../microduck-rl-genesis/.venv/bin/python run_genesis.py --actuator train --out results/genesis_train.json
../../../microduck-rl-genesis/.venv/bin/python analyze.py   # → results/report.md
```

完整表格见 `results/report.md`。

## 结论（2026-11-14）

1. **零跌倒**：5 个策略 × 14 个场景 × 3 种运行环境（含 6 × 60 s 随机序列）全部没有跌倒。
2. **两个引擎几乎等价**。同一策略在 `mujoco(game)` 和 `genesis[game]` 上：
   - 固定指令下 vx 平均差 0.002–0.008 m/s，wz 平均差 0.02–0.075 rad/s（最大 0.15）
   - 随机序列的 vx RMSE 基本相同，wz RMSE 在 MuJoCo 里反而小 0.04–0.07
   - 两个方向结果对称：官方（mjlab 训练）策略放进 Genesis，表现也和在 MuJoCo 里一样
3. **差的是策略，不是引擎。** 速度欠跟踪和"起步死区"两个引擎里都一样：
   - 指令 0.3 m/s 实际只走约 0.1 m/s
   - 官方策略从静止出发时，|vx| ≤ 0.2 或 |wz| ≤ 1.0 的指令都不起步；后退要到 −0.4 才动
   - 所以游戏 AI 用的较小指令，对官方 walk 来说基本无效
4. **Genesis 策略的转向比官方 walk 好**：wz 指令 1.0 → 实际 0.77–1.04，官方是 0。指令 +1.5 超出 Genesis 的训练范围，实际 1.0–1.4，没有出现失稳。

## 没有覆盖到的部分

- 只测了平地行走，没有推力扰动、机器人之间的碰撞，也没有球。**踢球是冲击性接触，接触模型的差异会被放大**，需要 S2/S3 以及第一个踢球策略来验证。
- MuJoCo 是确定性的，每个场景只跑一次。
