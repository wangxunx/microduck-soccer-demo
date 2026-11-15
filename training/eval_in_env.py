"""Evaluate exported football ONNX candidates in the Genesis training env with the
stand/walk handoff in the loop, with and without domain randomisation.

    HIP_VISIBLE_DEVICES=0 ../../microduck-rl-genesis/.venv/bin/python eval_in_env.py \
        exports/football-v3-1999.onnx [more.onnx ...] -B 1024 --episodes 2

Reports per candidate and condition (DR = training conditions, noDR = game-like
fixed 7.35 V / delay 4 / no sensor noise): fall rate inside the 1.2 s window,
fall rate after the handoff, touch rate, mean peak ball speed along aim, mean
lateral speed at that peak.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import soccer_rl  # noqa: E402,F401

import numpy as np  # noqa: E402
import genesis as gs  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from soccer_rl import football_cfg as F  # noqa: E402


def run(env, sess, episodes):
    n = env.num_envs
    env.reset()
    obs = env.obs["policy"]
    stats = {k: [] for k in ("fall_in_window", "fall_after_handoff", "kick_touched", "kick_max_vpar", "kick_max_vperp")}
    for _ in range(episodes * env.max_episode_length):
        act = sess.run(None, {"obs": obs.detach().cpu().numpy().astype(np.float32)})[0]
        td, _, dones, extras = env.step(torch.as_tensor(act, device=env.device))
        obs = td["policy"]
        if dones.any():
            ep = extras.get("episode", {})
            m = int(dones.sum())
            for k in stats:
                if k in ep:
                    stats[k].append((float(ep[k]), m))
    return {k: sum(v * m for v, m in vs) / max(1, sum(m for _, m in vs)) for k, vs in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("-B", "--num-envs", type=int, default=1024)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="recipe override (JSON value)")
    ap.add_argument("--only", choices=("DR", "noDR", "game"), default=None)
    args = ap.parse_args()
    import json
    for item in args.set:
        key, value = item.split("=", 1)
        assert hasattr(F, key), key
        setattr(F, key, json.loads(value))
    F.HANDOFF_IN_LOOP = True
    gs.init(backend=gs.gpu, logging_level="warning", seed=7)
    from soccer_rl.football_env import MicroduckFootballEnv
    envs = {}
    if args.only in (None, "DR"):
        envs["DR"] = MicroduckFootballEnv(num_envs=args.num_envs)
    if args.only in (None, "noDR"):
        envs["noDR"] = MicroduckFootballEnv(num_envs=args.num_envs, play=True, demo=True)
    if args.only in (None, "game"):
        # noDR + the game's exact runtime: 7.35 V, no sag, no IMU obs delay (servo.mjs / world-core.js)
        from microduck.bam_actuator import DelayBuffer
        e = MicroduckFootballEnv(num_envs=args.num_envs, play=True, demo=True)
        e.bam.vin_nominal.fill_(7.35)
        e.bam.vin_drop_resistance = None
        for k in ("base_ang_vel", "projected_gravity"):
            e.obs_delays[k] = DelayBuffer((e.num_envs, 3), 0, 0, 0, device=e.device)
        envs["game"] = e
    print(f"{'candidate':28s} {'cond':5s} {'fall_win':>8s} {'fall_after':>10s} {'touched':>8s} {'vpar':>6s} {'vperp':>6s}")
    for path in args.onnx:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        for cond, env in envs.items():
            s = run(env, sess, args.episodes)
            print(f"{Path(path).stem:28s} {cond:5s} {100*s['fall_in_window']:7.1f}% {100*s['fall_after_handoff']:9.1f}% "
                  f"{100*s['kick_touched']:7.1f}% {s['kick_max_vpar']:6.2f} {s['kick_max_vperp']:6.2f}")


if __name__ == "__main__":
    main()
