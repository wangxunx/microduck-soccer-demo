"""Train the ball-aware football kick policy on Genesis (PPO / rsl_rl).

Always smoke-test first:
    HIP_VISIBLE_DEVICES=1 ../../microduck-rl-genesis/.venv/bin/python train_football.py -e smoke -B 64 --max-iterations 5
Then:
    HIP_VISIBLE_DEVICES=1 ../../microduck-rl-genesis/.venv/bin/python train_football.py -e football-v1 -B 4096 --max-iterations 2000

Logs/checkpoints: training/logs/<exp>/ (tensorboard --logdir logs).
Without --resume an existing logs/<exp> is refused (never silently deleted).
"""
import argparse
import dataclasses
import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import soccer_rl  # noqa: E402,F401  (puts microduck-rl-genesis on sys.path)

import genesis as gs  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from microduck import velocity_cfg as VC  # noqa: E402
from soccer_rl import football_cfg as F  # noqa: E402


def recipe_snapshot() -> dict:
    return {k: v for k, v in vars(F).items() if k.isupper()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="football")
    p.add_argument("-B", "--num-envs", type=int, default=4096)
    p.add_argument("--max-iterations", type=int, default=2000)
    p.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="recipe override, e.g. TOUCH_BONUS=5 or REWARD_WEIGHTS.ball_speed_along_aim=20")
    args = p.parse_args()
    for item in args.set:  # applied before the env reads the recipe; recorded in recipe.json
        key, value = item.split("=", 1)
        value = json.loads(value)
        if "." in key:
            table, sub = key.split(".", 1)
            assert sub in getattr(F, table), f"unknown {table} key {sub}"
            getattr(F, table)[sub] = value
        else:
            assert hasattr(F, key), f"unknown recipe key {key}"
            setattr(F, key, value)
        print(f"recipe override: {key} = {value}")

    log_dir = HERE / "logs" / args.exp_name
    if log_dir.exists() and args.resume is None and args.exp_name != "smoke":
        sys.exit(f"{log_dir} exists; pick another --exp-name or pass --resume")
    if args.exp_name == "smoke" and log_dir.exists():
        import shutil
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    gs.init(backend=gs.gpu, logging_level="warning", seed=args.seed)
    from soccer_rl.football_env import MicroduckFootballEnv

    env = MicroduckFootballEnv(num_envs=args.num_envs)
    train_cfg = json.loads(json.dumps(VC.TRAIN_CFG))  # deep copy; same PPO as the walking task
    train_cfg["run_name"] = args.exp_name
    train_cfg["seed"] = args.seed
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump({"train_cfg": train_cfg, "task": "football", "num_obs": F.OBS_SIZE,
                     "num_privileged_obs": env.cfg["num_privileged_obs"], "recipe": recipe_snapshot()}, f)
    (log_dir / "recipe.json").write_text(json.dumps(recipe_snapshot(), indent=1, default=str))

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=str(gs.device))
    if args.resume:
        runner.load(args.resume)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
