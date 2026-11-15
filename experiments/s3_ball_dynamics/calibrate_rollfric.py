"""Fit the Genesis rolling-resistance emulation to MuJoCo.

Sweeps (mu_roll, w_eps) of  tau = -mu_roll * m g * w_h / max(|w_h|, w_eps)
(torsional term unchanged) over the roll_*/slide scenarios, and reports the
RMSE of horizontal ball speed vs the MuJoCo reference (results/mujoco.json).

    HIP_VISIBLE_DEVICES=1 ../../../microduck-rl-genesis/.venv/bin/python calibrate_rollfric.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ball_dynamics as B  # noqa: E402

MU_ROLLS = (0.002, 0.003, 0.004)
W_EPSS = (4.0, 8.0, 12.0, 16.0, 24.0)
SCN = ("roll_0.3", "roll_0.6", "roll_1.0", "roll_2.0", "slide_1.0")


def main():
    sys.path.insert(0, str((HERE / "../../../microduck-rl-genesis").resolve()))
    import genesis as gs
    import torch

    ref = json.loads((HERE / "results/mujoco.json").read_text())
    S = {k: v for k, v in ref["scenarios"].items() if k in SCN}
    params = list(itertools.product(MU_ROLLS, W_EPSS))
    combos = [(pr, s) for pr in params for s in S]
    n = len(combos)
    gs.init(backend=gs.gpu, logging_level="warning", seed=0)
    parts = HERE / "results" / "_scene_parts"
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=B.DT, substeps=1),
                     rigid_options=gs.options.RigidOptions(batch_dofs_info=True, iterations=10,
                                                           ls_iterations=20, max_collision_pairs=30),
                     show_viewer=False)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.0))
    ball = scene.add_entity(gs.morphs.MJCF(file=str(parts / "ball.xml")))
    scene.build(n_envs=n)
    d = gs.device
    ball.set_pos(torch.tensor([S[s]["pos"] for _, s in combos], dtype=gs.tc_float, device=d))
    ball.set_quat(torch.tensor([[1.0, 0, 0, 0]] * n, dtype=gs.tc_float, device=d))
    ball.set_dofs_velocity(torch.tensor([list(S[s]["lin"]) + list(S[s]["ang"]) for _, s in combos],
                                        dtype=gs.tc_float, device=d))
    mu = torch.tensor([[pr[0]] for pr, _ in combos], dtype=gs.tc_float, device=d)
    weps = torch.tensor([[pr[1]] for pr, _ in combos], dtype=gs.tc_float, device=d)
    solver = scene.sim.rigid_solver
    link = ball.base_link_idx if hasattr(ball, "base_link_idx") else ball.link_start
    steps = round(max(S[s]["T"] for s in S) / B.DT)
    sp = np.zeros((steps, n), dtype=np.float32)
    N = B.M * B.G
    with torch.inference_mode():
        for k in range(steps):
            w = ball.get_ang()
            touching = (ball.get_pos()[:, 2] < B.R + 0.002).to(w.dtype).unsqueeze(1)
            wh = w.clone(); wh[:, 2] = 0.0
            tau = -mu * N * wh / torch.maximum(wh.norm(dim=1, keepdim=True), weps)
            tau[:, 2] = -B.MU_TORS * N * w[:, 2] / torch.clamp(w[:, 2].abs(), min=B.W_EPS)
            solver.apply_links_external_torque((tau * touching).unsqueeze(1), links_idx=[link])
            scene.step()
            sp[k] = ball.get_vel()[:, :2].norm(dim=1).cpu().numpy()
    refsp = {s: np.linalg.norm(np.asarray(ref["variants"]["mujoco_python"]["runs"][s]["v"])[:, :2], axis=1)
             for s in S}
    table = {}
    for i, (pr, s) in enumerate(combos):
        T = len(refsp[s])
        table.setdefault(pr, {})[s] = float(np.sqrt(np.mean((sp[:T, i] - refsp[s]) ** 2)))
    rows = sorted(table.items(), key=lambda kv: np.mean(list(kv[1].values())))
    print("mu_roll  w_eps   " + "  ".join(f"{s:>9s}" for s in S) + "   mean   (speed RMSE m/s vs MuJoCo)")
    for (m_, w_), r in rows:
        print(f"{m_:.3f}   {w_:5.1f}   " + "  ".join(f"{r[s]:9.4f}" for s in S) + f"   {np.mean(list(r.values())):.4f}")
    best = rows[0][0]
    out = {"best": {"mu_roll": best[0], "w_eps": best[1]},
           "table": {f"{k[0]}_{k[1]}": v for k, v in table.items()}}
    (HERE / "results/rollfric_calibration.json").write_text(json.dumps(out, indent=1))
    print("best:", out["best"])


if __name__ == "__main__":
    main()
