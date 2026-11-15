"""S1: run the same walking policies and command timelines in Genesis.

Usage (with the microduck-rl-genesis venv, one GPU):
    HIP_VISIBLE_DEVICES=0 ../../../microduck-rl-genesis/.venv/bin/python run_genesis.py \
        --out results/genesis_gamematched.json [--actuator game|train]

One Genesis env per (policy, scenario), all stepped in one batch. All ONNX
files run through onnxruntime on CPU, exactly as the game does.

--actuator game  (default) matches the game's walking runtime, so the only
                 remaining differences are the physics engine and the robot
                 collision model:
                 fixed 7.35 V battery, no voltage sag, no current limit,
                 no actuator / IMU / joint-velocity delays.
--actuator train keeps the Genesis training actuator (random 6.5-8.2 V, sag,
                 1.75 A limit, 3-6 step bus delay, obs delays), DR still off.

Fall handling mirrors the game: tilted (gz > -0.5) or low (h < 0.025) for
> 0.28 s counts as a fall. The game then runs its stand policy; here the robot
is re-placed upright at its current x/y/yaw after 0.5 s. Recovery frames are
flagged and excluded from tracking metrics in analyze.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GENESIS_REPO = (HERE / "../../../microduck-rl-genesis").resolve()
sys.path.insert(0, str(GENESIS_REPO))

import genesis as gs  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

GAME_BATTERY_VOLTS = 7.35  # game/src/physics/servo.mjs BATTERY_VOLTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--actuator", choices=("game", "train"), default="game")
    ap.add_argument("--policies", default=str(HERE / "policies.json"))
    ap.add_argument("--scenarios", default=None, help="comma list (default: all)")
    args = ap.parse_args()

    spec = json.loads((HERE / "scenarios.json").read_text())
    dt, scenarios = spec["dt"], spec["scenarios"]
    if args.scenarios:
        keep = args.scenarios.split(",")
        scenarios = {k: v for k, v in scenarios.items() if k in keep}
    pol_paths = {k: (HERE / v).resolve() for k, v in json.loads(Path(args.policies).read_text()).items()}

    sessions = {}
    for name, path in pol_paths.items():
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        batch_dim = s.get_inputs()[0].shape[0]
        sessions[name] = (s, s.get_inputs()[0].name, isinstance(batch_dim, str))

    combos = [(p, s) for p in pol_paths for s in scenarios]
    n = len(combos)
    max_steps = max(round(scenarios[s]["duration"] / dt) for _, s in combos)

    gs.init(backend=gs.gpu, logging_level="warning", seed=0)
    from microduck.velocity_env import MicroduckVelocityEnv
    from microduck.bam_actuator import DelayBuffer
    from microduck import constants as C

    t_build = time.time()
    env = MicroduckVelocityEnv(num_envs=n, rough=False, show_viewer=False, play=True, demo=True)
    t_build = time.time() - t_build
    d = env.device

    if args.actuator == "game":
        env.bam.vin_nominal.fill_(GAME_BATTERY_VOLTS)
        env.bam.vin_drop_resistance = None
        env.bam.max_current = 1e9  # game walk path: no firmware current limit
        env.bam._delay = DelayBuffer((n, C.NUM_ACTIONS), 0, 0, 0, device=d)
        obs_dims = {"base_ang_vel": 3, "projected_gravity": 3, "joint_vel": C.NUM_ACTIONS}
        for k, dim in obs_dims.items():
            env.obs_delays[k] = DelayBuffer((n, dim), 0, 0, 0, device=d)

    # The game feeds zeros in the head/body command slots; reset() randomised them.
    env.head_cmd.zero_()
    env.body_cmd.zero_()
    obs = env.place((0.0, 0.0, 0.125), yaw=0.0)["policy"]

    # Per-env command schedule [steps, n, 2] (vx, wz); 0 after a scenario ends.
    sched = np.zeros((max_steps, n, 2), dtype=np.float32)
    for e, (_, sname) in enumerate(combos):
        sc = scenarios[sname]
        steps = round(sc["duration"] / dt)
        seg = 0
        for k in range(steps):
            t = k * dt
            while seg + 1 < len(sc["segments"]) and sc["segments"][seg + 1][0] <= t + 1e-9:
                seg += 1
            sched[k, e] = sc["segments"][seg][1:3]

    groups = {}
    for e, (p, _) in enumerate(combos):
        groups.setdefault(p, []).append(e)

    rec = {k: np.zeros((max_steps, n), dtype=np.float32) for k in ("x", "y", "yaw", "gz", "h", "recovering")}
    fallen_for = np.zeros(n)
    recover_left = np.zeros(n)  # >0: in recovery window
    replace_at = np.full(n, -1.0)
    falls = np.zeros(n, dtype=int)

    t_run = time.time()
    with torch.inference_mode():
        for k in range(max_steps):
            cmd = torch.as_tensor(sched[k], device=d)
            env.twist_cmd[:, 0] = cmd[:, 0]
            env.twist_cmd[:, 1] = 0.0
            env.twist_cmd[:, 2] = cmd[:, 1]
            obs_np = obs.detach().cpu().numpy().astype(np.float32)
            obs_np[:, 48] = sched[k, :, 0]
            obs_np[:, 49] = 0.0
            obs_np[:, 50] = sched[k, :, 1]
            obs_np[:, 51:61] = 0.0
            act = np.zeros((n, C.NUM_ACTIONS), dtype=np.float32)
            for p, idx in groups.items():
                s, iname, dynamic = sessions[p]
                if dynamic:
                    act[idx] = s.run(None, {iname: obs_np[idx]})[0]
                else:
                    for e in idx:
                        act[e] = s.run(None, {iname: obs_np[e:e + 1]})[0][0]
            obs = env.step(torch.as_tensor(act, device=d))[0]["policy"]

            pos = env.base_pos.cpu().numpy()
            quat = env.base_quat.cpu().numpy()
            gz = env.projected_gravity[:, 2].cpu().numpy()
            yaw = np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                             1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
            fallen = (gz > -0.5) | (pos[:, 2] < 0.025)
            fallen_for = np.where(fallen, fallen_for + dt, 0.0)
            new_fall = (fallen_for > 0.28) & (recover_left <= 0)
            for e in np.nonzero(new_fall)[0]:
                falls[e] += 1
                recover_left[e] = 1.0
                replace_at[e] = 0.5
            due = np.nonzero((replace_at >= 0) & (recover_left <= replace_at + 1e-9) & (recover_left > 0))[0]
            if len(due):
                for e in due:
                    obs = env.place((float(pos[e, 0]), float(pos[e, 1]), 0.125), yaw=float(yaw[e]),
                                    env_ids=torch.tensor([e], device=d))["policy"]
                    replace_at[e] = -1.0
                    fallen_for[e] = 0.0
                env.head_cmd.zero_()
                env.body_cmd.zero_()
            rec["recovering"][k] = (recover_left > 0).astype(np.float32)
            recover_left = np.maximum(recover_left - dt, 0.0)
            rec["x"][k], rec["y"][k], rec["yaw"][k] = pos[:, 0], pos[:, 1], yaw
            rec["gz"][k], rec["h"][k] = gz, pos[:, 2]
    t_run = time.time() - t_run

    out = {"engine": "genesis", "engineVersion": gs.__version__, "actuator": args.actuator,
           "dt": dt, "buildSeconds": t_build, "runSeconds": t_run, "policies": {}}
    for e, (p, sname) in enumerate(combos):
        steps = round(scenarios[sname]["duration"] / dt)
        tr = {k: rec[k][:steps, e].tolist() for k in ("x", "y", "yaw", "gz", "h")}
        tr["recovering"] = rec["recovering"][:steps, e].astype(int).tolist()
        tr["vx_cmd"] = sched[:steps, e, 0].tolist()
        tr["wz_cmd"] = sched[:steps, e, 1].tolist()
        tr["t"] = [round((i + 1) * dt, 4) for i in range(steps)]
        out["policies"].setdefault(p, {"policy": p, "policyPath": str(pol_paths[p]), "runs": {}})
        out["policies"][p]["runs"][sname] = {"falls": int(falls[e]), "traj": tr}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"built {n} envs in {t_build:.1f}s, ran {max_steps} control steps in {t_run:.1f}s")
    for p in pol_paths:
        print(p, {s: out["policies"][p]["runs"][s]["falls"] for s in scenarios})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
