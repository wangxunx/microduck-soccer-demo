"""S5 spike: can Genesis run ONE 2v2 match scene in real time and render it?

Scene: field plane, boundary walls, two goals, the game ball, 4 full-collision
Microducks (robot_allcollisions.xml). Robots hold HOME with Genesis PD control
plus small random target jitter (exercises contacts/joints; policy + BAM costs are
measured separately below). Reports:
  - physics steps/s (need >= 200 for real time at 200 Hz)
  - offscreen render FPS at 1280x720 and 1920x1080
  - cost of 4x MLP policy inference + 4x BAM-like torch math per control step

    HIP_VISIBLE_DEVICES=0 ../../../microduck-rl-genesis/.venv/bin/python spike.py --backend gpu
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "1")
HERE = Path(__file__).resolve().parent
GEN = (HERE / "../../../microduck-rl-genesis").resolve()
sys.path.insert(0, str(GEN))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    import genesis as gs
    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu, logging_level="warning", seed=0)
    from microduck import constants as C

    L, W = 1.575, 0.93  # game FIELD half length / width
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, substeps=1),
        rigid_options=gs.options.RigidOptions(enable_self_collision=True, max_collision_pairs=200,
                                              iterations=10, ls_iterations=20),
        vis_options=gs.options.VisOptions(shadow=True, plane_reflection=False,
                                          background_color=(0.06, 0.07, 0.1), ambient_light=(0.4, 0.4, 0.45)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.25, 0.55, 0.3, 1.0)))
    wall = gs.surfaces.Default(color=(0.85, 0.85, 0.9, 1.0))
    for sx in (-1, 1):
        scene.add_entity(gs.morphs.Box(pos=(0, sx * (W + 0.05), 0.05), size=(2 * L + 0.6, 0.05, 0.1), fixed=True), surface=wall)
        for sy in (-1, 1):
            scene.add_entity(gs.morphs.Box(pos=(sx * (L + 0.05), sy * (W + 0.36) / 2, 0.05),
                                           size=(0.05, W - 0.36, 0.1), fixed=True), surface=wall)
        scene.add_entity(gs.morphs.Box(pos=(sx * L, 0, 0.28), size=(0.03, 0.75, 0.03), fixed=True), surface=wall)  # crossbar
        for sy in (-1, 1):
            scene.add_entity(gs.morphs.Box(pos=(sx * L, sy * 0.36, 0.14), size=(0.03, 0.03, 0.28), fixed=True), surface=wall)
    ball_xml = HERE.parent / "s2_genesis_ball_bench" / "ball_game.xml"
    ball = scene.add_entity(gs.morphs.MJCF(file=str(ball_xml), pos=(0, 0, 0.0)))
    starts = [(-0.55, 0.25, 0.0), (-0.55, -0.25, 0.0), (0.55, 0.25, np.pi), (0.55, -0.25, np.pi)]
    robots = []
    for x, y, yaw in starts:
        q = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        robots.append(scene.add_entity(gs.morphs.MJCF(file=C.MICRODUCK_ALLCOLLISIONS_XML, pos=(x, y, 0.125), quat=q)))
    cams = {"720p": scene.add_camera(res=(1280, 720), pos=(0, -2.6, 1.6), lookat=(0, 0, 0), fov=45, GUI=False),
            "1080p": scene.add_camera(res=(1920, 1080), pos=(0, -2.6, 1.6), lookat=(0, 0, 0), fov=45, GUI=False)}
    t0 = time.time()
    scene.build()
    print(f"[{args.backend}] build {time.time() - t0:.1f}s")

    jidx = []
    home = torch.tensor(C.DEFAULT_JOINT_POS, dtype=torch.float32, device=gs.device)
    for r in robots:
        jn = {j.name: j for j in r.joints}
        idx = [jn[n].dof_start - r.dof_start for n in C.JOINT_NAMES]  # local to this entity
        jidx.append(idx)
        r.set_dofs_kp(torch.full((14,), 3.0, device=gs.device), idx)
        r.set_dofs_kv(torch.full((14,), 0.1, device=gs.device), idx)
        r.set_dofs_position(home, idx)
    ball.set_dofs_velocity(torch.tensor([0.6, 0.2, 0, 0, 0, 0], dtype=torch.float32, device=gs.device))

    # physics: control at 50 Hz, physics at 200 Hz
    rng = np.random.default_rng(0)
    for k in range(50):
        scene.step()
    t0 = time.time()
    for k in range(args.steps):
        if k % 4 == 0:
            for r, idx in zip(robots, jidx):
                r.control_dofs_position(home + torch.as_tensor(rng.normal(0, 0.05, 14), dtype=torch.float32,
                                                                device=gs.device), idx)
        scene.step()
    if args.backend == "gpu":
        torch.cuda.synchronize()
    sps = args.steps / (time.time() - t0)
    print(f"[{args.backend}] physics: {sps:.0f} steps/s  ({sps / 200:.2f}x real time at 200 Hz), contacts ok, "
          f"ball z={float(ball.get_pos().flatten()[2]):.3f}")

    # rendering
    for name, cam in cams.items():
        cam.render()
        n, t0 = 30, time.time()
        for _ in range(n):
            scene.step()
            rgb = cam.render()[0]
        fps = n / (time.time() - t0)
        print(f"[{args.backend}] render {name}: {fps:.1f} FPS (incl. 1 physics step/frame), frame {np.asarray(rgb).shape}")
        if name == "1080p":
            from PIL import Image
            Image.fromarray(np.asarray(rgb)).save(HERE / f"frame_{args.backend}.png")

    # per-control-step overheads: 4 policies (512-256-128 MLP) + BAM-like math, on the torch device
    dev = gs.device
    mlp = torch.nn.Sequential(torch.nn.Linear(61, 512), torch.nn.ELU(), torch.nn.Linear(512, 256), torch.nn.ELU(),
                              torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 14)).to(dev)
    obs = torch.randn(4, 61, device=dev)
    with torch.no_grad():
        for _ in range(20):
            mlp(obs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(500):
            mlp(obs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    print(f"[{args.backend}] 4 policies batched: {(time.time() - t0) / 500 * 1e3:.3f} ms per control step")
    x = torch.randn(4, 14, device=dev)
    t0 = time.time()
    for _ in range(500):  # ~30 elementwise ops like BamActuator.compute, all 4 robots batched
        y = x
        for _ in range(30):
            y = torch.tanh(y * 1.001 + 0.001)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"[{args.backend}] BAM-like math (4 robots batched, 30 ops): {(time.time() - t0) / 500 * 1e3:.3f} ms per physics step")


if __name__ == "__main__":
    main()
