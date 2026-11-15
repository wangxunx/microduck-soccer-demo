# S3：足球动力学，MuJoCo（游戏）vs Genesis

场景完全复刻游戏里的设置（`scene.xml`，由 `ball_dynamics.py` 生成）：200 Hz Euler 积分，地板摩擦 `1 .005 .0001`，球 r = 0.05 m、30 g、摩擦 `.4 .01 .003`、`solref .03 .4`、`condim 6`，挡板，以及一个 50 g 的"击球器"（水平滑轨上的小球，用来代替脚）。

```bash
PY=../../../microduck-rl-genesis/.venv/bin/python
$PY ball_dynamics.py --engine mujoco --out results/mujoco.json
node crosscheck_wasm.mjs                       # 在游戏自带的 MuJoCo 3.11 WASM 上复核
HIP_VISIBLE_DEVICES=1 $PY ball_dynamics.py --engine genesis --out results/genesis.json
$PY analyze.py                                 # → results/report.md
HIP_VISIBLE_DEVICES=1 $PY calibrate_rollfric.py  # 拟合滚动阻力参数
```

## 结论（2026-11-14）

1. **参考是可靠的**：Python MuJoCo 3.16 和游戏的 MuJoCo 3.11 WASM 在 9 个场景上逐位一致。
2. **滚动阻力：Genesis 原生没有，但可以补上。**
   - 原生 Genesis 里球永远不减速：8 s 后速度不变。MuJoCo 里 1 m/s 的球 4 s 后只剩 0.09 m/s。
   - MuJoCo 的软约束行为是：高速时近似恒定减速（约 0.4 m/s²），低速时近似粘性阻尼（τ ≈ 1.6 s）。
   - 补法：每个物理步给球施加外力矩 `τ = -μ_roll·m·g·ω_h / max(|ω_h|, 16 rad/s)`（再加一个扭转摩擦项）。取 μ_roll = 0.003（就是游戏的值），在 0.3–2 m/s 的滚动和滑动起步场景上，速度 RMSE 为 **0.024–0.034 m/s**（`results/rollfric_calibration.json`）。
3. **击球冲量：Genesis 给球的速度约是 MuJoCo 的 1.9 倍**，两个击球速度下这个比例一致（击球器 0.5 m/s → 0.336 vs 0.180；1.0 m/s → 0.689 vs 0.364）。
   - 原因在 MuJoCo 一侧：软的金字塔摩擦锥把切向冲量和法向力耦合起来了。撞击时地板法向力冲到约 2 N（约 7 倍 mg），地板摩擦随之放大，吞掉了大部分冲量。
   - 把地板设成无摩擦后，MuJoCo 是 0.73，和 Genesis 的 0.69 接近；改用 elliptic 锥，耦合反而更强。
   - 这是游戏引擎的实际行为，所以在 Genesis 里训练要对齐的就是它。
4. 掉落反弹（几乎不弹）和撞墙恢复系数（0.23 vs 0.25）两边一致。

## 对踢球训练的影响

- 训练时必须加上滚动阻力的补偿；否则 Genesis 里的球一踢就滚个不停，奖励信号会严重失真。
- 冲量差约 2 倍，可以先当成"踢球力度的标定系数"处理：训练目标按约 1.9 倍放大，或者对冲量做 DR、奖励以方向为主。但这是用球心高度水平撞击测出来的，**真实的脚从斜下方踢球时比例未必一样**，需要用真实的踢球策略在两个引擎里对比来确认（建议的 S4）。
