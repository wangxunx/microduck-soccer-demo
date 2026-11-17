# Learning Wall 原型

在 AMD GPU 上用 Genesis 实时渲染一大片 Microduck 机器鸭练习射门，回放射门 AI 的真实训练过程：从完全不会，到稳定踢中。

```bash
HIP_VISIBLE_DEVICES=0 ../../microduck-rl-genesis/.venv/bin/python learning_wall.py -o out/learning_wall.mp4   # 约 50 s 视频
HIP_VISIBLE_DEVICES=0 ../../microduck-rl-genesis/.venv/bin/python learning_wall.py --preview --episodes 1     # 只出几张静帧
```

- **内容**：沿着同一条训练链回放：football-v3（从头训练，第 0–1999 轮）→ v4ft-v3（到第 2998 轮）→ v5（到第 4497 轮，即游戏里的 `?football=genesis-v5`）。共 7 个阶段，每个阶段 3 次射门。
- **真实性**：
  - 每个阶段加载当时真实的 checkpoint。训练阶段按"策略均值 + 该 checkpoint 的探索噪声"执行，和训练时机器人的实际表现一致。
  - 最后一个阶段是部署时的行为（确定性动作，不加噪声），画面上标为 "trained AI (deployed)"。
  - 左下角的 GPU 时长和"模拟练习天数"来自训练日志：4497 轮 × 24 步 × 4096 个机器人 × 0.02 s ≈ 102 天，共 3.3 GPU 小时。
  - 右下角是画面中这 64 只机器鸭的实时统计：
    - 踢正：球速沿瞄准方向 > 0.6 m/s，且横向偏差 < 30%
    - 站立：机器人倾斜 < 37°
- **实测**（R9700，1080p，64 只机器人）：渲染约 28 ms/帧（约 35 FPS），因此展台现场实时运行是可行的。

| 阶段 | 踢正 | 站立 |
|---|---:|---:|
| 第 0 轮（未训练） | 2% | 91% |
| 第 50 轮 | 20% | 82% |
| 第 200 轮 | 92% | 96% |
| 第 600 轮 | 98% | 84% |
| 第 1999 轮 | 100% | 99% |
| 第 2998 轮 | 100% | 100% |
| 第 4497 轮（部署行为） | 100% | 100% |

输出在 `out/`（`learning_wall.mp4` 约 156 MB，`learning_wall_share.mp4` 约 41 MB），不提交到仓库。
