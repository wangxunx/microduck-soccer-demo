# Microduck Soccer on AMD GPUs

Training soccer skills for the [Pollen Robotics Microduck](https://github.com/pollen-robotics/microduck)
on **AMD GPUs (ROCm)** with the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator,
and playing them in the [MicroDuck Robot Soccer](https://github.com/wander521/microduck-robot-soccer) browser game.

The upstream training stack (mjlab / MuJoCo Warp) requires NVIDIA GPUs. This project builds on the
Genesis port [microduck-rl-genesis](https://github.com/Macmachi/microduck-rl-genesis) and adds a
ball-aware **football (kick)** task that is trained on a single AMD Radeon AI PRO R9700 and deployed,
unchanged, into the game's MuJoCo runtime.

**Status:** in the game's built-in evaluator (11 scenarios, single run) the AMD-trained kicking policy
`football-v4ft-v3-2998` scored 68 goals with 0 kick-induced falls, versus 61 goals / 0 falls for the
game's original kicking policy. Only the **walking** and **kicking** policies are trained on AMD; the
game's stand-up / recovery and dance policies are the originals.

## Layout

| Path | What |
|---|---|
| [`training/`](training/README.md) | Genesis football environment, recipe, PPO training (rsl-rl), ONNX export, deployment-contract test, one-command evaluation, training history |
| [`experiments/`](experiments/) | Feasibility studies S1–S4: walking sim2sim (Genesis ↔ MuJoCo), Genesis throughput with ball, ball dynamics calibration, kick sim2sim |
| [`game-patches/`](game-patches/) | Our changes to the game: `?walk=` and `?football=` URL switches to select AMD-trained policies |
| [`setup_game.sh`](setup_game.sh) | Recreates `game/` from the pinned upstream commit + our patch + the trained policies |

## Quick start

```bash
# 0. prerequisites: a microduck-rl-genesis checkout next to this repo, with its ROCm venv
#    (see its REPRODUCE_AMD_SINGLE_GPU.md), plus Node.js 22+.
git clone https://github.com/Macmachi/microduck-rl-genesis.git ../microduck-rl-genesis

# 1. the game, with the AMD-trained policies selectable
./setup_game.sh
cd game && npm run dev     # http://localhost:5173/?walk=genesis&football=genesis

# 2. training (one AMD GPU)
cd training
PY=../../microduck-rl-genesis/.venv/bin/python
HIP_VISIBLE_DEVICES=0 $PY tests/test_contract.py
HIP_VISIBLE_DEVICES=0 $PY train_football.py -e smoke -B 64 --max-iterations 5
./evaluate_candidate.sh <exp-name> <checkpoint> 0
```

## License

This repository's code and trained policy weights are licensed under the **Apache License 2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The weights were trained with the Microduck robot model, whose 3D geometry remains under CC BY-NC-SA 4.0 (non-commercial); third-party components below keep their own licenses.

## Credits

- Robot model, MJCF and recipe: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
  (code Apache-2.0; **3D model files CC BY-NC-SA 4.0, non-commercial**). The meshes are used from the
  microduck-rl-genesis checkout and are not redistributed here.
- Genesis port, BAM actuator model: [microduck-rl-genesis](https://github.com/Macmachi/microduck-rl-genesis) (Apache-2.0),
  [Rhoban/bam](https://github.com/Rhoban/bam) (Apache-2.0).
- Simulator and RL: [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) (Apache-2.0),
  [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (BSD-3-Clause).
- Game: [wander521/microduck-robot-soccer](https://github.com/wander521/microduck-robot-soccer). Upstream
  has no license file, so the game is **not vendored**: `setup_game.sh` fetches it at a fixed commit and
  applies `game-patches/`.

---

## 中文简介

在 **AMD GPU（ROCm）** 上用 Genesis 训练 Microduck 机器人的足球技能，并放进 MicroDuck Robot Soccer 浏览器游戏里运行。上游的训练框架只支持 NVIDIA；本项目基于 Genesis 移植版，新增了能看到球的踢球（football）任务，在单张 AMD Radeon AI PRO R9700 上完成训练，导出后无需修改即可在游戏的 MuJoCo 运行时中使用。

在游戏自带评估器中（11 个场景，单次评估），AMD 训练的踢球模型进球 68 个、踢球跌倒 0 次；游戏原有的踢球模型是 61 个、0 次。在 AMD 上训练的只有**行走**和**踢球**两个模型，起身/恢复和舞蹈模型沿用原版。

训练、实验和评估的详细说明见各目录下的 README。

本仓库的代码和训练出的模型权重采用 **Apache License 2.0**（见 LICENSE 和 NOTICE）。权重是用 Microduck 机器人模型训练的，其 3D 几何仍遵守 CC BY-NC-SA 4.0（非商用）。
