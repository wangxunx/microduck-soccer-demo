"""S2: Genesis throughput / stability with the full-collision Microduck + soccer ball.

Uses the unmodified training env of microduck-rl-genesis (training mode: DR,
pushes, command resampling, resets on fall) driven by the Genesis-trained
walking ONNX, and optionally:
  --model allcollisions   swap robot_walk.xml (5 collision geoms) for
                          robot_allcollisions.xml (82 mesh collision geoms),
                          which the football task needs (ball vs whole leg)
  --ball                  add the game's 100 mm / 30 g ball (ball_game.xml)
  --self-collision on|off
  --max-pairs N           Genesis collision-pair budget

With --ball, every 2 s the ball is re-placed 9-13 cm in front of each robot
(random lateral offset) so walking robots keep running into it.

    HIP_VISIBLE_DEVICES=2 ../../../microduck-rl-genesis/.venv/bin/python bench.py \
        --model allcollisions --ball -B 4096 --out results/x.json
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

import torch  # noqa: E402

BALL_R = 0.05


def _physical_card() -> str:
    """rocm-smi card id of torch device 0 (HIP index order != rocm-smi order)."""
    import subprocess
    bus = torch.cuda.get_device_properties(0).pci_bus_id
    cards = json.loads(subprocess.run(["rocm-smi", "--showbus", "--json"], capture_output=True, text=True).stdout)
    for card, v in cards.items():
        if int(v["PCI Bus"].split(":")[1], 16) == bus:
            return card
    raise RuntimeError(f"no rocm-smi card with PCI bus {bus}")


def _vram_used(card: str) -> int:
    import subprocess
    out = json.loads(subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json", "-d", card.removeprefix("card")],
                                    capture_output=True, text=True).stdout)
    return int(out[card]["VRAM Total Used Memory (B)"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("walk", "allcollisions"), default="walk")
    ap.add_argument("--ball", action="store_true")
    ap.add_argument("--self-collision", choices=("on", "off"), default="on")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("-B", "--num-envs", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--policy", default=str(GENESIS_REPO / "policies/velocity.onnx"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    card = _physical_card()
    vram0 = _vram_used(card)
    total = torch.cuda.get_device_properties(0).total_memory

    import genesis as gs
    import onnxruntime as ort

    gs.init(backend=gs.gpu, logging_level="warning", seed=0)
    import microduck.velocity_env as ve
    from microduck import constants as C

    if args.model == "allcollisions":
        ve.MICRODUCK_WALK_XML = C.MICRODUCK_ALLCOLLISIONS_XML

    # Override the inline RigidOptions(...) of _build_scene.
    _RO = gs.options.RigidOptions

    def patched_rigid_options(**kw):
        kw["enable_self_collision"] = args.self_collision == "on"
        if args.max_pairs is not None:
            kw["max_collision_pairs"] = args.max_pairs
        print("RigidOptions:", {k: kw[k] for k in ("enable_self_collision", "max_collision_pairs",
                                                    "iterations", "ls_iterations")})
        return _RO(**kw)

    gs.options.RigidOptions = patched_rigid_options
    if args.self_collision == "off":
        C.ENABLE_SELF_COLLISION_PENALTY = False
        ve.C.ENABLE_SELF_COLLISION_PENALTY = False

    extra = [gs.morphs.MJCF(file=str(HERE / "ball_game.xml"))] if args.ball else None

    t0 = time.time()
    env = ve.MicroduckVelocityEnv(num_envs=args.num_envs, rough=False, show_viewer=False,
                                  play=False, extra_morphs=extra)
    build_s = time.time() - t0
    gs.options.RigidOptions = _RO
    n, d = env.num_envs, env.device
    ball = env.decor[0] if args.ball else None
    print(f"built {n} envs ({args.model}, ball={args.ball}, selfcol={args.self_collision}) in {build_s:.1f}s; "
          f"robot links={env.robot.n_links} geoms={env.robot.n_geoms}")

    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    def place_ball():
        pos = env.base_pos
        q = env.base_quat
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        fwd = torch.empty(n, device=d).uniform_(0.09, 0.13)
        lat = torch.empty(n, device=d).uniform_(-0.05, 0.05)
        p = torch.stack([pos[:, 0] + fwd * torch.cos(yaw) - lat * torch.sin(yaw),
                         pos[:, 1] + fwd * torch.sin(yaw) + lat * torch.cos(yaw),
                         torch.full((n,), BALL_R + 0.001, device=d)], dim=-1)
        ball.set_pos(p)
        ball.zero_all_dofs_velocity()

    obs = env.reset()["policy"]
    if ball is not None:
        place_ball()

    stats = dict(step_s=[], infer_s=[], nan_envs=0, solver_err_envs=0, ball_nan=0,
                 ball_penetration_steps=0, ball_fast_steps=0, ball_max_speed=0.0, ball_min_z=1.0,
                 ball_contact_env_frac=[], fell_frac=[], resets=0)
    vram_peak = vram0
    total_steps = args.warmup + args.steps
    with torch.inference_mode():
        for k in range(total_steps):
            ti = time.time()
            act = sess.run(None, {iname: obs.detach().cpu().numpy().astype(np.float32)})[0]
            act_t = torch.as_tensor(act, device=d)
            infer = time.time() - ti
            torch.cuda.synchronize()
            ts = time.time()
            obs_td, _, dones, _ = env.step(act_t)
            torch.cuda.synchronize()
            step = time.time() - ts
            obs = obs_td["policy"]
            if ball is not None and k % 100 == 99:
                place_ball()
            if k < args.warmup:
                continue
            stats["step_s"].append(step)
            stats["infer_s"].append(infer)
            stats["resets"] += int(dones.sum())
            stats["fell_frac"].append(float(env._term_stats["fell_over"]))
            err = env.scene.rigid_solver.get_error_envs_mask()
            stats["solver_err_envs"] += int(err.sum())
            stats["nan_envs"] += int((~torch.isfinite(obs).all(dim=1)).sum())
            if ball is not None:
                bp, bv = ball.get_pos(), ball.get_vel()
                fin = torch.isfinite(bp).all(dim=1) & torch.isfinite(bv).all(dim=1)
                stats["ball_nan"] += int((~fin).sum())
                bz, sp = bp[fin, 2], bv[fin].norm(dim=1)
                stats["ball_penetration_steps"] += int((bz < BALL_R - 0.005).sum())
                stats["ball_fast_steps"] += int((sp > 3.0).sum())
                stats["ball_max_speed"] = max(stats["ball_max_speed"], float(sp.max()))
                stats["ball_min_z"] = min(stats["ball_min_z"], float(bz.min()))
                if k % 10 == 0:
                    c = ball.get_contacts(with_entity=env.robot)
                    if "valid_mask" in c:
                        stats["ball_contact_env_frac"].append(float(c["valid_mask"].any(dim=1).float().mean()))
            if k % 50 == 0:
                vram_peak = max(vram_peak, _vram_used(card))

    step_s = np.array(stats["step_s"])
    env_steps_per_s = n / step_s.mean()
    res = {
        "config": vars(args) | {"out": str(args.out)},
        "num_envs": n, "build_s": build_s,
        "robot_links": env.robot.n_links, "robot_geoms": env.robot.n_geoms,
        "step_ms_mean": 1e3 * step_s.mean(), "step_ms_p95": 1e3 * float(np.percentile(step_s, 95)),
        "env_steps_per_s": env_steps_per_s,
        "ppo_iter_rollout_s_24steps": 24 * step_s.mean(),
        "onnx_cpu_infer_ms_mean": 1e3 * float(np.mean(stats["infer_s"])),
        "physical_gpu": card,
        "vram_used_gb_by_process_approx": (vram_peak - vram0) / 1e9,
        "vram_total_gb": total / 1e9,
        "fall_rate_per_env_step": float(np.mean(stats["fell_frac"])),
        "resets": stats["resets"],
        "solver_error_env_steps": stats["solver_err_envs"],
        "nan_obs_env_steps": stats["nan_envs"],
    }
    if ball is not None:
        res |= {
            "ball_nan_env_steps": stats["ball_nan"],
            "ball_penetration_env_steps(>5mm)": stats["ball_penetration_steps"],
            "ball_fast_env_steps(>3m/s)": stats["ball_fast_steps"],
            "ball_max_speed": stats["ball_max_speed"],
            "ball_min_z": stats["ball_min_z"],
            "ball_robot_contact_env_frac_mean": float(np.mean(stats["ball_contact_env_frac"]))
            if stats["ball_contact_env_frac"] else None,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
