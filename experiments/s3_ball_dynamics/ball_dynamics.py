"""S3: soccer-ball dynamics, MuJoCo (game reference) vs Genesis.

    ../../../microduck-rl-genesis/.venv/bin/python ball_dynamics.py --engine mujoco  --out results/mujoco.json
    HIP_VISIBLE_DEVICES=1 ../../../microduck-rl-genesis/.venv/bin/python ball_dynamics.py --engine genesis --out results/genesis.json

The scene replicates game/src/physics/world-core.js: 200 Hz Euler, floor
friction "1 .005 .0001" condim 3, ball r=0.05 m / 30 g, friction ".4 .01 .003",
solref ".03 .4", condim 6; board friction ".5 .01 .001". A 50 g sphere on a
horizontal slide joint ("striker") stands in for a foot hitting the ball.

Genesis only models sliding friction. Variant `genesis_rollfric` adds an
explicit external torque emulating MuJoCo's rolling/torsional friction:
    tau = -mu * N * w / max(|w|, w_eps),   N = m g while touching the floor
with mu_roll = max(.003, .0001) and mu_torsion = max(.01, .005) (MuJoCo's
default max-combination of the two geoms).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R, M, G = 0.05, 0.03, 9.81
DT = 0.005
MU_ROLL, MU_TORS = 0.003, 0.01
W_EPS = 0.5  # rad/s: linear region of the emulated Coulomb torque

SCENE_XML = f"""<mujoco model="s3_ball">
  <option timestep="{DT}" integrator="Euler"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05" friction="1 .005 .0001" condim="3"/>
    <geom name="board" type="box" pos="1.5125 0 0.1" size=".0125 .5 .1" friction=".5 .01 .001"
          contype="4" conaffinity="4"/>
    <body name="ball" pos="0 0 {R}">
      <freejoint name="ball_freejoint"/>
      <geom name="ball_geom" type="sphere" size="{R}" mass="{M}" friction=".4 .01 .003"
            solref=".03 .4" condim="6" contype="5" conaffinity="5"/>
    </body>
    <body name="striker" pos="-0.1 0 {R}">
      <joint name="striker_x" type="slide" axis="1 0 0"/>
      <geom name="striker_geom" type="sphere" size=".02" mass=".05" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def scenarios():
    S = {}
    for v in (0.3, 0.6, 1.0, 2.0):  # rolling without slip along +y: w = (-v/R, 0, 0)
        S[f"roll_{v}"] = dict(pos=(0, 0, R), lin=(0, v, 0), ang=(-v / R, 0, 0), striker=None, T=8.0)
    S["slide_1.0"] = dict(pos=(0, 0, R), lin=(0, 1.0, 0), ang=(0, 0, 0), striker=None, T=8.0)
    S["drop_0.2"] = dict(pos=(0, 0, R + 0.2), lin=(0, 0, 0), ang=(0, 0, 0), striker=None, T=2.0)
    S["wall_1.0"] = dict(pos=(0.9, 0, R), lin=(1.0, 0, 0), ang=(0, 1.0 / R, 0), striker=None, T=2.0)
    for v in (0.5, 1.0):  # striker starts 0.1 m behind the ball centre
        S[f"kick_{v}"] = dict(pos=(0, 0, R), lin=(0, 0, 0), ang=(0, 0, 0), striker=(0.0, v), T=3.0)
    return S


def run_mujoco(S):
    import mujoco
    m = mujoco.MjModel.from_xml_string(SCENE_XML)
    bq = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")]
    bv = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")]
    sq = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "striker_x")]
    sv = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "striker_x")]
    out = {}
    for name, sc in S.items():
        d = mujoco.MjData(m)
        d.qpos[bq:bq + 3] = sc["pos"]
        d.qpos[bq + 3:bq + 7] = (1, 0, 0, 0)
        d.qvel[bv:bv + 3] = sc["lin"]
        d.qvel[bv + 3:bv + 6] = sc["ang"]  # free joint angular velocity is body-frame (= world at identity)
        x0, v0 = sc["striker"] if sc["striker"] else (-5.0, 0.0)
        d.qpos[sq], d.qvel[sv] = x0, v0
        mujoco.mj_forward(m, d)
        tr = {k: [] for k in ("p", "v", "w", "sx", "sv")}
        for _ in range(round(sc["T"] / DT)):
            mujoco.mj_step(m, d)
            q = d.qpos[bq + 3:bq + 7]
            wb = d.qvel[bv + 3:bv + 6]
            rot = np.zeros(9); mujoco.mju_quat2Mat(rot, q)
            tr["p"].append(d.qpos[bq:bq + 3].tolist())
            tr["v"].append(d.qvel[bv:bv + 3].tolist())
            tr["w"].append((rot.reshape(3, 3) @ wb).tolist())  # world frame
            tr["sx"].append(float(d.qpos[sq]) - 0.1)
            tr["sv"].append(float(d.qvel[sv]))
        out[name] = tr
    return {"mujoco_python": {"engine": f"mujoco {mujoco.__version__}", "runs": out}}


def run_genesis(S):
    GEN = (HERE / "../../../microduck-rl-genesis").resolve()
    sys.path.insert(0, str(GEN))
    import genesis as gs
    import torch

    gs.init(backend=gs.gpu, logging_level="warning", seed=0)
    tmp = HERE / "results" / "_scene_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "ball.xml").write_text(f"""<mujoco><worldbody><body name="ball" pos="0 0 {R}">
      <freejoint name="ball_freejoint"/>
      <geom name="ball_geom" type="sphere" size="{R}" mass="{M}" friction=".4 .01 .003" solref=".03 .4" condim="6"/>
    </body></worldbody></mujoco>""")
    (tmp / "striker.xml").write_text(f"""<mujoco><worldbody><body name="striker" pos="-0.1 0 {R}">
      <joint name="striker_x" type="slide" axis="1 0 0"/>
      <geom name="striker_geom" type="sphere" size=".02" mass=".05"/>
    </body></worldbody></mujoco>""")

    names = list(S)
    variants = ("genesis_native", "genesis_rollfric")
    combos = [(v, s) for v in variants for s in names]
    n = len(combos)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1),
        rigid_options=gs.options.RigidOptions(batch_dofs_info=True, iterations=10, ls_iterations=20,
                                              max_collision_pairs=30),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.0))
    scene.add_entity(gs.morphs.Box(pos=(1.5125, 0, 0.1), size=(0.025, 1.0, 0.2), fixed=True),
                     material=gs.materials.Rigid(friction=0.5))
    ball = scene.add_entity(gs.morphs.MJCF(file=str(tmp / "ball.xml")))
    striker = scene.add_entity(gs.morphs.MJCF(file=str(tmp / "striker.xml")))
    scene.build(n_envs=n)
    d = gs.device
    print("ball mass:", float(ball.get_mass()), "striker mass:", float(striker.get_mass()))

    pos = torch.tensor([S[s]["pos"] for _, s in combos], dtype=gs.tc_float, device=d)
    vel = torch.tensor([list(S[s]["lin"]) + list(S[s]["ang"]) for _, s in combos], dtype=gs.tc_float, device=d)
    ball.set_pos(pos)
    ball.set_quat(torch.tensor([[1.0, 0, 0, 0]] * n, dtype=gs.tc_float, device=d))
    ball.set_dofs_velocity(vel)
    sx = torch.tensor([[S[s]["striker"][0] if S[s]["striker"] else -5.0] for _, s in combos],
                      dtype=gs.tc_float, device=d)
    sv = torch.tensor([[S[s]["striker"][1] if S[s]["striker"] else 0.0] for _, s in combos],
                      dtype=gs.tc_float, device=d)
    striker.set_dofs_position(sx)
    striker.set_dofs_velocity(sv)

    rf_envs = torch.tensor([i for i, (v, _) in enumerate(combos) if v == "genesis_rollfric"], device=d)
    solver = scene.sim.rigid_solver
    ball_link = ball.base_link_idx if hasattr(ball, "base_link_idx") else ball.link_start
    steps = round(max(S[s]["T"] for s in names) / DT)
    rec = {k: np.zeros((steps, n, 3), dtype=np.float32) for k in ("p", "v", "w")}
    rs = {k: np.zeros((steps, n), dtype=np.float32) for k in ("sx", "sv")}
    with torch.inference_mode():
        for k in range(steps):
            # Rolling/torsional resistance emulation (rollfric variant only).
            w = ball.get_ang()[rf_envs]
            z = ball.get_pos()[rf_envs, 2]
            touching = (z < R + 0.002).to(w.dtype).unsqueeze(1)
            N = M * G
            wh = w.clone(); wh[:, 2] = 0.0
            tau_r = -MU_ROLL * N * wh / torch.clamp(wh.norm(dim=1, keepdim=True), min=W_EPS)
            tau_t = torch.zeros_like(w)
            tau_t[:, 2] = -MU_TORS * N * w[:, 2] / torch.clamp(w[:, 2].abs(), min=W_EPS)
            tau = (tau_r + tau_t) * touching
            solver.apply_links_external_torque(tau.unsqueeze(1), links_idx=[ball_link], envs_idx=rf_envs)
            scene.step()
            rec["p"][k] = ball.get_pos().cpu().numpy()
            rec["v"][k] = ball.get_vel().cpu().numpy()
            rec["w"][k] = ball.get_ang().cpu().numpy()
            rs["sx"][k] = striker.get_dofs_position().cpu().numpy()[:, 0] - 0.1
            rs["sv"][k] = striker.get_dofs_velocity().cpu().numpy()[:, 0]
    out = {v: {"engine": f"genesis {gs.__version__}", "runs": {}} for v in variants}
    for i, (v, s) in enumerate(combos):
        T = round(S[s]["T"] / DT)
        out[v]["runs"][s] = {"p": rec["p"][:T, i].tolist(), "v": rec["v"][:T, i].tolist(),
                             "w": rec["w"][:T, i].tolist(), "sx": rs["sx"][:T, i].tolist(),
                             "sv": rs["sv"][:T, i].tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("mujoco", "genesis"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    S = scenarios()
    res = run_mujoco(S) if args.engine == "mujoco" else run_genesis(S)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"dt": DT, "scenarios": S, "variants": res}))
    (HERE / "scene.xml").write_text(SCENE_XML)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
