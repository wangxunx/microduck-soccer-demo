"""S4: the same kick cases in Genesis (full-collision robot + game ball).

    HIP_VISIBLE_DEVICES=1 ../../../microduck-rl-genesis/.venv/bin/python run_genesis.py --out results/genesis.json

Reproduces the game's per-control-step policy schedule and runtime contract:
  kick_*  : no actuator delay, undelayed joint_vel, no current limit
  football: 4-physics-step actuator delay, 1-step joint_vel delay, 1.75 A limit,
            ball obs (robot yaw frame, clipped +/-3) in slots 55-60, aim = heading at t=0
  stand   : after football, 1 s; its first `actions` obs is seeded with the
            current joint deviation (world-core.js footballHandoff)
  walk    : zero command
Battery fixed at 7.35 V, no sag (game servo.mjs). DR/noise/pushes off (demo).
Fall (gz > -0.5 or h < 0.025 for > 0.28 s) -> hold 0.22 s -> stand, like the game.
Two ball variants per case: genesis_native, genesis_rollfric (S3 calibrated
rolling/torsional resistance torque, applied before every physics substep).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEN = (HERE / "../../../microduck-rl-genesis").resolve()
GAME_POL = (HERE / "../../game/public/assets/policies").resolve()
BALL_XML = (HERE / "../s2_genesis_ball_bench/ball_game.xml").resolve()
sys.path.insert(0, str(GEN))

import torch  # noqa: E402

R, M, G = 0.05, 0.03, 9.81
MU_ROLL, W_EPS_ROLL, MU_TORS, W_EPS_TORS = 0.003, 16.0, 0.01, 0.5  # S3 calibration
DT_CTRL = 0.02
FOOTBALL_S, HANDOFF_S, KICKLOCK_S = 1.2, 1.0, 0.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--football", default=str(GAME_POL / "football.onnx"))
    ap.add_argument("--protocols", default=None, help="comma list, e.g. C_football")
    args = ap.parse_args()
    spec = json.loads((HERE / "cases.json").read_text())
    cases, duration = spec["cases"], spec["duration"]
    if args.protocols:
        cases = [c for c in cases if c["protocol"] in args.protocols.split(",")]
    variants = ("genesis_native", "genesis_rollfric")
    combos = [(v, c) for v in variants for c in cases]
    n = len(combos)

    import genesis as gs
    import onnxruntime as ort
    gs.init(backend=gs.gpu, logging_level="warning", seed=0)
    import microduck.velocity_env as ve
    from microduck import constants as C
    from microduck.bam_actuator import DelayBuffer

    ve.MICRODUCK_WALK_XML = C.MICRODUCK_ALLCOLLISIONS_XML
    _RO = gs.options.RigidOptions
    gs.options.RigidOptions = lambda **kw: _RO(**(kw | {"max_collision_pairs": 60}))
    env = ve.MicroduckVelocityEnv(num_envs=n, rough=False, show_viewer=False, play=True, demo=True,
                                  extra_morphs=[gs.morphs.MJCF(file=str(BALL_XML))])
    gs.options.RigidOptions = _RO
    d = env.device
    ball = env.decor[0]
    NA = C.NUM_ACTIONS

    # --- game-matched actuator, with per-env switchable delay / current limit
    env.bam.vin_nominal.fill_(7.35)
    env.bam.vin_drop_resistance = None
    env.bam.max_current = torch.full((n, 1), 1e9, device=d)
    env.bam._delay = DelayBuffer((n, NA), 0, 4, 0, device=d)
    env.bam._delay._lag.zero_()
    for k, dim in {"base_ang_vel": 3, "projected_gravity": 3}.items():
        env.obs_delays[k] = DelayBuffer((n, dim), 0, 0, 0, device=d)
    env.obs_delays["joint_vel"] = DelayBuffer((n, NA), 0, 1, 0, device=d)
    env.obs_delays["joint_vel"]._lag.zero_()

    # --- rolling resistance emulation before every physics substep
    rf = torch.tensor([i for i, (v, _) in enumerate(combos) if v == "genesis_rollfric"], device=d)
    solver = env.scene.rigid_solver
    ball_link = ball.base_link_idx if hasattr(ball, "base_link_idx") else ball.link_start
    orig_step = env.scene.step

    def step_with_rollfric(*a, **kw):
        w = ball.get_ang()[rf]
        touching = (ball.get_pos()[rf, 2] < R + 0.002).to(w.dtype).unsqueeze(1)
        N = M * G
        wh = w.clone(); wh[:, 2] = 0.0
        tau = -MU_ROLL * N * wh / torch.clamp(wh.norm(dim=1, keepdim=True), min=W_EPS_ROLL)
        tau[:, 2] = -MU_TORS * N * w[:, 2] / torch.clamp(w[:, 2].abs(), min=W_EPS_TORS)
        solver.apply_links_external_torque((tau * touching).unsqueeze(1), links_idx=[ball_link], envs_idx=rf)
        return orig_step(*a, **kw)

    env.scene.step = step_with_rollfric

    # --- initial state
    env.head_cmd.zero_(); env.body_cmd.zero_(); env.twist_cmd.zero_()
    obs = env.place((0.0, 0.0, 0.125), yaw=0.0)["policy"]
    ball.set_pos(torch.tensor([[c["bx"], c["by"], R] for _, c in combos], dtype=gs.tc_float, device=d))
    ball.zero_all_dofs_velocity()
    env._refresh_state()
    obs = env._compute_observations()["policy"]

    sessions = {k: ort.InferenceSession(str(GAME_POL / f) if not Path(f).is_absolute() else f, providers=["CPUExecutionProvider"]) for k, f in {
        "kick_left": "ball_kick_left.onnx", "kick_right": "ball_kick_right.onnx", "walk": "BEST_alpha_walking.onnx",
        "stand": "BEST_alpha_stand.onnx", "football": args.football}.items()}

    def scheduled(c, t):
        """Policy the game runs at time t if the robot never falls."""
        if c["protocol"] == "C_football":
            return "football" if t < FOOTBALL_S - 1e-9 else ("stand" if t < FOOTBALL_S + HANDOFF_S - 1e-9 else "walk")
        win = 0.5 if c["protocol"] == "A_game_kick" else 3.0
        return f"kick_{c['foot']}" if t < win - 1e-9 else "walk"

    steps = round(duration / DT_CTRL)
    fallen_for = np.zeros(n); recov = [None] * n  # None | ("settle", t0) | "stand"
    falls = np.zeros(n, dtype=int)
    last_act = np.zeros((n, NA), dtype=np.float32)
    prev_pol = [None] * n
    rec = {k: np.zeros((steps, n, 3), dtype=np.float32) for k in ("ball", "bvel", "base")}
    gz_r = np.zeros((steps, n), dtype=np.float32); contact = np.zeros((steps, n), dtype=np.int8)
    pol_r = [[None] * n for _ in range(steps)]
    aim_w = np.array([1.0, 0.0])  # heading at trigger (yaw 0)

    with torch.inference_mode():
        for k in range(steps):
            t = k * DT_CTRL
            pols = []
            for e, (_, c) in enumerate(combos):
                p = scheduled(c, t)
                if recov[e] is not None:
                    p = "settle" if recov[e][0] == "settle" and t - recov[e][1] < 0.22 else "stand"
                    if p == "stand":
                        recov[e] = ("stand", 0)
                pols.append(p)
            fb = np.array([p == "football" for p in pols])
            # actuator contract for the action computed this step
            env.bam._delay._lag[:] = torch.as_tensor(np.where(fb, 4, 0), device=d)
            env.bam.max_current[:, 0] = torch.as_tensor(np.where(fb, 1.75, 1e9), dtype=torch.float32, device=d)

            o = obs.detach().cpu().numpy().astype(np.float32)
            o[:, 48:61] = 0.0
            pos = env.base_pos.cpu().numpy(); quat = env.base_quat.cpu().numpy()
            yaw = np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                             1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
            bp = ball.get_pos().cpu().numpy(); bv = ball.get_vel().cpu().numpy()
            act = last_act.copy()
            for e, p in enumerate(pols):
                if p == "settle":
                    continue  # game: no inference during settle, targets held
                if p == "football":
                    cy, sy = math.cos(yaw[e]), math.sin(yaw[e])
                    dx, dy = bp[e, 0] - pos[e, 0], bp[e, 1] - pos[e, 1]
                    o[e, 55:61] = np.clip([cy * dx + sy * dy, -sy * dx + cy * dy,
                                           cy * bv[e, 0] + sy * bv[e, 1], -sy * bv[e, 0] + cy * bv[e, 1],
                                           cy * aim_w[0] + sy * aim_w[1], -sy * aim_w[0] + cy * aim_w[1]], -3, 3)
                if p == "stand" and prev_pol[e] == "football":
                    o[e, 34:48] = o[e, 6:20]  # footballHandoff: lastAction <- q - default
                act[e] = sessions[p].run(None, {"obs": o[e:e + 1]})[0][0]
            last_act = act
            # joint_vel obs delay for the NEXT step's policy
            nxt = np.array([recov[e] is None and scheduled(c, t + DT_CTRL) == "football"
                            for e, (_, c) in enumerate(combos)])
            env.obs_delays["joint_vel"]._lag[:] = torch.as_tensor(np.where(nxt, 1, 0), device=d)

            obs = env.step(torch.as_tensor(act, device=d))[0]["policy"]
            env.twist_cmd.zero_()

            pos = env.base_pos.cpu().numpy(); gz = env.projected_gravity[:, 2].cpu().numpy()
            fallen = (gz > -0.5) | (pos[:, 2] < 0.025)
            fallen_for = np.where(fallen, fallen_for + DT_CTRL, 0.0)
            for e in np.nonzero((fallen_for > 0.28))[0]:
                if recov[e] is None:
                    recov[e] = ("settle", t + DT_CTRL); falls[e] += 1
            cinfo = ball.get_contacts(with_entity=env.robot)
            if "valid_mask" in cinfo:
                contact[k] = cinfo["valid_mask"].any(dim=1).cpu().numpy()
            rec["ball"][k] = ball.get_pos().cpu().numpy(); rec["bvel"][k] = ball.get_vel()[:, :3].cpu().numpy()
            rec["base"][k] = pos; gz_r[k] = gz
            pol_r[k] = pols; prev_pol = pols

    out = {"engine": "genesis", "engineVersion": gs.__version__, "duration": duration, "football": args.football, "variants": {}}
    for e, (v, c) in enumerate(combos):
        out["variants"].setdefault(v, {})[c["id"]] = c | {"falls": int(falls[e]), "traj": {
            "ball": rec["ball"][:, e].tolist(), "bvel": rec["bvel"][:, e].tolist(), "base": rec["base"][:, e].tolist(),
            "gz": gz_r[:, e].tolist(), "policy": [pol_r[k][e] for k in range(steps)],
            "contact": contact[:, e].astype(int).tolist()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    for v in variants:
        print(v, "falls:", sum(r["falls"] for r in out["variants"][v].values()))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
