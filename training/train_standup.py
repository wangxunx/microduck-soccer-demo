"""Train the Microduck stand-up / standing (get-up) policy on Genesis, AMD GPU.

Port of upstream microduck_rl's stand-up task (source of the game's BEST_alpha_stand.onnx);
recipe and deviations in soccer_rl/standup_cfg.py.

    HIP_VISIBLE_DEVICES=0 ../../microduck-rl-genesis/.venv/bin/python train_standup.py -e smoke -B 64 --max-iterations 3
    HIP_VISIBLE_DEVICES=0 ../../microduck-rl-genesis/.venv/bin/python train_standup.py -e stand-A -B 4096 --max-iterations 10000

Logs/checkpoints: training/logs/<exp>/. Existing logs/<exp> is refused (except "smoke").
"""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import soccer_rl  # noqa: E402,F401  (puts microduck-rl-genesis on sys.path)

import genesis as gs  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from microduck import constants as MC  # noqa: E402
from microduck import velocity_cfg as VC  # noqa: E402
from soccer_rl import standup_cfg as S  # noqa: E402


def apply_override(item: str) -> None:
    key, value = item.split("=", 1)
    value = json.loads(value)
    if isinstance(value, list):
        value = tuple(value)
    if "." in key:
        table, sub = key.split(".", 1)
        assert sub in getattr(S, table), f"unknown {table} key {sub}"
        getattr(S, table)[sub] = value
    else:
        assert hasattr(S, key), f"unknown standup_cfg key {key}"
        setattr(S, key, value)
    print(f"recipe override: {key} = {value}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", required=True)
    p.add_argument("-B", "--num-envs", type=int, default=4096)
    p.add_argument("--max-iterations", type=int, default=10000)
    p.add_argument("--resume", default=None, help="checkpoint .pt to start from")
    p.add_argument("--curriculum-start-iter", type=int, default=0)
    p.add_argument("--save-interval", type=int, default=250)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="standup_cfg override, e.g. DROP_PROB=0.15 or REWARD_WEIGHTS.gentle_rise=0.01")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    for item in args.set:
        apply_override(item)

    log_dir = HERE / "logs" / args.exp_name
    if log_dir.exists():
        if args.exp_name != "smoke":
            sys.exit(f"{log_dir} exists; pick another --exp-name")
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)

    gs.init(backend=gs.gpu, logging_level="warning", seed=args.seed)
    from soccer_rl.standup_env import MicroduckStandUpEnv

    env = MicroduckStandUpEnv(num_envs=args.num_envs)
    env.common_step_counter = args.curriculum_start_iter * S.NUM_STEPS_PER_ENV

    train_cfg = json.loads(json.dumps(VC.TRAIN_CFG))  # same PPO as upstream MicroduckStandUpRlCfg
    train_cfg["run_name"] = args.exp_name
    train_cfg["seed"] = args.seed
    train_cfg["save_interval"] = args.save_interval
    recipe = {k: v for k, v in vars(S).items() if k.isupper()}
    meta = {"task": "standup", "robot_model": "allcollisions", "resume": args.resume,
            "curriculum_start_iter": args.curriculum_start_iter, "overrides": args.set}
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump({"train_cfg": train_cfg, "num_obs": MC.NUM_OBS,
                     "num_privileged_obs": env.cfg["num_privileged_obs"], "recipe": recipe, **meta}, f)
    (log_dir / "recipe.json").write_text(json.dumps({"meta": meta, "standup_cfg": recipe}, indent=1, default=str))

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=str(gs.device))
    if args.resume:
        runner.load(args.resume)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
