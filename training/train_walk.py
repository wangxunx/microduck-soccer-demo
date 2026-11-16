"""Train / fine-tune the Microduck walking (velocity) policy on Genesis, AMD GPU.

Uses microduck-rl-genesis's MicroduckVelocityEnv unchanged, with optional recipe
overrides (velocity_cfg) and the game's full-collision robot model.

    HIP_VISIBLE_DEVICES=1 ../../microduck-rl-genesis/.venv/bin/python train_walk.py -e walk-B \
        --resume ../../microduck-rl-genesis/logs/microduck-velocity/model_2999.pt --curriculum-start-iter 3000 \
        --max-iterations 3000 --set REWARD_PARAMS.track_linear_velocity_std=0.158

Logs/checkpoints: training/logs/<exp>/. Existing logs/<exp> is refused (never deleted).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import soccer_rl  # noqa: E402,F401  (puts microduck-rl-genesis on sys.path)

import genesis as gs  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import microduck.velocity_env as ve  # noqa: E402
from microduck import constants as MC  # noqa: E402
from microduck import velocity_cfg as VC  # noqa: E402


def apply_override(item: str) -> None:
    key, value = item.split("=", 1)
    value = json.loads(value)
    if isinstance(value, list):
        value = tuple(value)
    if "." in key:
        table, sub = key.split(".", 1)
        assert sub in getattr(VC, table), f"unknown {table} key {sub}"
        getattr(VC, table)[sub] = value
    else:
        assert hasattr(VC, key), f"unknown velocity_cfg key {key}"
        setattr(VC, key, value)
    print(f"recipe override: {key} = {value}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", required=True)
    p.add_argument("-B", "--num-envs", type=int, default=4096)
    p.add_argument("--max-iterations", type=int, default=3000)
    p.add_argument("--resume", default=None, help="checkpoint .pt to start from")
    p.add_argument("--curriculum-start-iter", type=int, default=0,
                   help="start curricula as if this many iterations were done (use when resuming)")
    p.add_argument("--model", choices=("walk", "allcollisions"), default="walk",
                   help="robot collision model; the soccer game uses allcollisions")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help='velocity_cfg override, e.g. REWARD_PARAMS.track_linear_velocity_std=0.158 or CMD_ANG_VEL_Z=[-1.5,1.5]')
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    for item in args.set:
        apply_override(item)

    log_dir = HERE / "logs" / args.exp_name
    if log_dir.exists():
        sys.exit(f"{log_dir} exists; pick another --exp-name")
    log_dir.mkdir(parents=True)

    gs.init(backend=gs.gpu, logging_level="warning", seed=args.seed)
    if args.model == "allcollisions":
        ve.MICRODUCK_WALK_XML = MC.MICRODUCK_ALLCOLLISIONS_XML
    env = ve.MicroduckVelocityEnv(num_envs=args.num_envs)
    env.common_step_counter = args.curriculum_start_iter * VC.NUM_STEPS_PER_ENV
    env.cfg["robot_model"] = args.model

    train_cfg = json.loads(json.dumps(VC.TRAIN_CFG))
    train_cfg["run_name"] = args.exp_name
    train_cfg["seed"] = args.seed
    recipe = {k: v for k, v in vars(VC).items() if k.isupper() and k != "TRAIN_CFG"}
    meta = {"task": "walk", "robot_model": args.model, "resume": args.resume,
            "curriculum_start_iter": args.curriculum_start_iter, "overrides": args.set}
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump({"train_cfg": train_cfg, "num_obs": MC.NUM_OBS,
                     "num_privileged_obs": int(env.get_observations()["privileged"].shape[-1]), **meta}, f)
    (log_dir / "recipe.json").write_text(json.dumps({"meta": meta, "velocity_cfg": recipe}, indent=1, default=str))

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=str(gs.device))
    if args.resume:
        runner.load(args.resume)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
