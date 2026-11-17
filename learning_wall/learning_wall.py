"""Learning Wall prototype: replay how the Microduck kicking AI learned, rendered live by
Genesis on the AMD GPU.

A grid of Microducks (one Genesis env each, all in one batched scene) practises kicking.
Each stage loads a real training checkpoint of the kick policy lineage
    football-v3 (0-1999, from scratch) -> football-v4ft-v3 (-2998) -> football-v5 (-4497)
and acts like the robots did during training (policy mean + that checkpoint's exploration
noise). The last stage shows the deployed behaviour (deterministic policy, no noise).
Captions show the iteration, cumulative GPU time and simulated practice time from the
training logs, and the live hit rate measured in the rendered scene.

    HIP_VISIBLE_DEVICES=0 ../../microduck-rl-genesis/.venv/bin/python learning_wall.py -o out/learning_wall.mp4
    ... --preview            # a few stills only (layout check)
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("PYGLET_HEADLESS_DEVICE", "0")
HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent / "training"
sys.path.insert(0, str(TRAINING))
import soccer_rl  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# (run, checkpoint, deterministic, label)
STAGES = [
    ("football-v3", 0, False, "untrained"),
    ("football-v3", 50, False, ""),
    ("football-v3", 200, False, ""),
    ("football-v3", 600, False, ""),
    ("football-v3", 1999, False, ""),
    ("football-v4ft-v3", 2998, False, ""),
    ("football-v5", 4497, True, "trained AI (deployed)"),
]
LINEAGE = ["football-v3", "football-v4ft-v3", "football-v5"]
STEPS_PER_ITER, ENVS_TRAIN, DT = 24, 4096, 0.02
FONT_R = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
FONT_B = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"


def gpu_seconds_at():
    """Cumulative wall-clock seconds of training at each iteration, from the training logs."""
    table, offset = {}, 0
    for run in LINEAGE:
        it, last = None, 0
        for line in open(TRAINING / f"logs_{run}.out", errors="ignore"):
            m = re.search(r"Learning iteration (\d+)/", line)
            if m:
                it = int(m.group(1))
            m = re.search(r"Time elapsed: (\d+):(\d+):(\d+)", line)
            if m and it is not None:
                last = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                table[it] = offset + last
        offset += last
    return table


def load_policy(run, ckpt, device):
    from soccer_rl.onnx_mlp import OnnxMlp
    onnx = HERE / "policies" / f"{run}-{ckpt}.onnx"
    if not onnx.exists():
        import subprocess
        onnx.parent.mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(TRAINING / "export_football.py"), "-e", run, "--ckpt", str(ckpt),
                        "-o", str(onnx)], check=True, capture_output=True, cwd=TRAINING)
    sd = torch.load(TRAINING / "logs" / run / f"model_{ckpt}.pt", map_location="cpu", weights_only=False)["actor_state_dict"]
    std = sd["distribution.std_param"].to(device)
    return OnnxMlp(str(onnx)).to(device), std


class Overlay:
    def __init__(self, w, h):
        self.w, self.h = w, h
        s = h / 1080
        self.f_title = ImageFont.truetype(FONT_B, int(40 * s))
        self.f_sub = ImageFont.truetype(FONT_R, int(24 * s))
        self.f_big = ImageFont.truetype(FONT_B, int(64 * s))
        self.f_mid = ImageFont.truetype(FONT_B, int(32 * s))
        self.f_small = ImageFont.truetype(FONT_R, int(22 * s))
        self.s = s

    def draw(self, rgb, st):
        im = Image.fromarray(rgb).convert("RGBA")
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        s, w, h = self.s, self.w, self.h
        d.rectangle([0, 0, w, int(110 * s)], fill=(8, 10, 16, 200))
        d.rectangle([0, 0, w, int(6 * s)], fill=(237, 28, 36, 255))
        d.text((int(40 * s), int(18 * s)), "LEARNING WALL", font=self.f_title, fill=(255, 106, 19, 255))
        d.text((int(400 * s), int(28 * s)), "Robot ducks learning to kick: physics simulation + reinforcement learning on an AMD Radeon GPU",
               font=self.f_sub, fill=(235, 238, 245, 255))
        d.text((int(40 * s), int(70 * s)), "Replay of real training checkpoints  ·  rendered live by Genesis on AMD Radeon AI PRO R9700",
               font=self.f_small, fill=(160, 170, 190, 255))
        # bottom-left: training progress
        x0, y0, bw, bh = int(40 * s), h - int(250 * s), int(1000 * s), int(210 * s)
        d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=int(18 * s), fill=(8, 10, 16, 205))
        d.text((x0 + int(28 * s), y0 + int(16 * s)), "TRAINING ITERATION", font=self.f_small, fill=(160, 170, 190, 255))
        d.text((x0 + int(28 * s), y0 + int(40 * s)), f"{st['iter']:,}", font=self.f_big, fill=(255, 255, 255, 255))
        d.text((x0 + int(300 * s), y0 + int(62 * s)), f"/ {st['iter_max']:,}", font=self.f_mid, fill=(160, 170, 190, 255))
        if st["label"]:
            d.text((x0 + int(470 * s), y0 + int(66 * s)), st["label"], font=self.f_small, fill=(255, 106, 19, 255))
        px, py, pw = x0 + int(28 * s), y0 + int(128 * s), bw - int(56 * s)
        d.rounded_rectangle([px, py, px + pw, py + int(14 * s)], radius=int(7 * s), fill=(46, 54, 71, 255))
        frac = st["iter"] / st["iter_max"]
        d.rounded_rectangle([px, py, px + max(int(14 * s), int(pw * frac)), py + int(14 * s)], radius=int(7 * s), fill=(237, 28, 36, 255))
        d.text((px, py + int(26 * s)),
               f"{st['gpu_h']:.1f} GPU-hours on one Radeon  ·  ~{st['days']:,.0f} days of simulated practice  ·  {ENVS_TRAIN:,} robots in parallel",
               font=self.f_small, fill=(220, 225, 235, 255))
        # bottom-right: live stats of the robots shown
        rx0, rw = w - int(500 * s), int(460 * s)
        d.rounded_rectangle([rx0, y0, rx0 + rw, y0 + bh], radius=int(18 * s), fill=(8, 10, 16, 205))
        d.text((rx0 + int(28 * s), y0 + int(16 * s)), "ROBOTS IN THIS VIEW", font=self.f_small, fill=(160, 170, 190, 255))
        rows = [("on-target kicks", st["hit"]), ("still standing", st["standing"])]
        for k, (name, v) in enumerate(rows):
            yy = y0 + int((48 + 70 * k) * s)
            d.text((rx0 + int(28 * s), yy), "–" if v is None else f"{100 * v:.0f}%", font=self.f_mid,
                   fill=(46, 189, 107, 255) if k == 0 else (255, 255, 255, 255))
            d.text((rx0 + int(150 * s), yy + int(8 * s)), name, font=self.f_small, fill=(220, 225, 235, 255))
        return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path, default=HERE / "out" / "learning_wall.mp4")
    ap.add_argument("-n", "--num-envs", type=int, default=64)
    ap.add_argument("--spacing", type=float, default=0.42)
    ap.add_argument("--episodes", type=int, default=3, help="kick episodes per stage")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--preview", action="store_true", help="render a few stills instead of a video")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    W, H = map(int, args.res.split("x"))
    N = args.num_envs

    import genesis as gs
    gs.init(backend=gs.gpu, logging_level="warning", seed=args.seed)
    import soccer_rl.football_env as fe
    from soccer_rl import football_cfg as F
    F.HANDOFF_IN_LOOP = True  # kick, then the official stand/walk policies take over (like the game)

    # orange render-only ball (Genesis draws collision geoms gray)
    ball_xml = HERE / "out" / "ball_render.xml"
    ball_xml.parent.mkdir(parents=True, exist_ok=True)
    ball_xml.write_text((TRAINING / "soccer_rl/assets/ball_game.xml").read_text().replace(
        "</body>", '  <geom type="sphere" size="0.0505" mass="0" contype="0" conaffinity="0" rgba="1 0.42 0.0 1"/>\n    </body>'))
    fe.BALL_XML = ball_xml
    _VO, _build = gs.options.VisOptions, gs.Scene.build
    gs.options.VisOptions = lambda **kw: _VO(**(kw | {"rendered_envs_idx": list(range(N)), "env_separate_rigid": False}))
    gs.Scene.build = lambda self, n_envs=0, **kw: _build(self, n_envs=n_envs, env_spacing=(args.spacing, args.spacing), **kw)
    env = fe.MicroduckFootballEnv(num_envs=N, play=True, demo=True,
                                  camera_cfg={"res": (W, H), "pos": (3, -2, 1.5), "lookat": (0, 0, 0), "fov": 46})
    gs.options.VisOptions, gs.Scene.build = _VO, _build
    d = env.device
    offs = np.asarray(env.scene.envs_offset)
    centre = offs.mean(axis=0) + offs.max(axis=0)  # rendered grid spans 0..2*extent (see deck/render_genesis_grid.py)
    ext = float(offs.max(axis=0)[:2].max())
    env.cam.set_pose(pos=tuple(centre + np.array([1.25 * ext + 1.0, -0.65 * ext - 0.5, 0.75 * ext + 0.6])),
                     lookat=tuple(centre + np.array([0.15 * ext, 0.05 * ext, 0.0])))

    gpu_s = gpu_seconds_at()
    iter_max = max(gpu_s)
    overlay = Overlay(W, H)
    rng = np.random.default_rng(args.seed)

    def start_episode():
        env.place((0.0, 0.0, 0.125), yaw=0.0)
        side = rng.choice([-1, 1], N) * 0.042 + rng.uniform(-0.02, 0.02, N)
        fwd = 0.11 + rng.uniform(-0.02, 0.03, N)
        env.ball.set_pos(torch.tensor(np.stack([fwd, side, np.full(N, F.BALL_RADIUS)], 1), dtype=torch.float32, device=d))
        env.ball.zero_all_dofs_velocity()
        aim = rng.uniform(-0.2, 0.2, N)
        env.aim_w[:] = torch.tensor(np.stack([np.cos(aim), np.sin(aim)], 1), dtype=torch.float32, device=d)
        env.base_start_xy[:] = 0
        env.base_start_yaw[:] = 0
        env.ball_touched[:] = False
        env.prev_ball_vel[:] = 0
        env.kick_max_vpar[:] = 0
        env.aux_last_action[:] = 0
        env.episode_length_buf[:] = 0
        env._refresh_state()
        return env._compute_observations()["policy"]

    writer = None
    if not args.preview:
        import imageio_ffmpeg
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(str(args.output), (W, H), fps=25, codec="libx264", pix_fmt_in="rgb24",
                                             pix_fmt_out="yuv420p", quality=8, macro_block_size=2, ffmpeg_log_level="warning")
        writer.send(None)
    render_t, frames = 0.0, 0
    hit_hist, stand_hist = [], []
    with torch.inference_mode():
        for si, (run, ckpt, det, label) in enumerate(STAGES):
            policy, std = load_policy(run, ckpt, d)
            st = {"iter": ckpt, "iter_max": iter_max, "label": label, "gpu_h": gpu_s.get(ckpt, 0) / 3600,
                  "days": ckpt * STEPS_PER_ITER * ENVS_TRAIN * DT / 86400, "hit": None, "standing": None}
            hits, stands = [], []
            for ep in range(args.episodes):
                obs = start_episode()
                for k in range(120):  # 1.2 s kick window + stand handoff
                    act = policy(obs)
                    if not det:
                        act = act + std * torch.randn_like(act)
                    obs = env.step(act)[0]["policy"]
                    if k % 2 == 0:
                        t0 = time.time()
                        rgb = np.asarray(env.cam.render()[0])
                        render_t += time.time() - t0
                        frames += 1
                        if k > 60:  # live stats once the kick window is over
                            on = (env.kick_max_vpar > 0.6) & (env.kick_max_vperp < 0.3 * env.kick_max_vpar)
                            st["hit"] = float(on.float().mean())
                            st["standing"] = float((env.projected_gravity[:, 2] < -0.8).float().mean())
                        frame = overlay.draw(rgb, st)
                        if writer is not None:
                            writer.send(np.ascontiguousarray(frame))
                        elif ep == 0 and k == 100:
                            Image.fromarray(frame).save(HERE / "out" / f"preview_{si}_{run}-{ckpt}.png")
                hits.append(st["hit"]); stands.append(st["standing"])
            hit_hist.append(np.mean(hits)); stand_hist.append(np.mean(stands))
            print(f"stage {si}: {run}-{ckpt} {'(deterministic)' if det else f'(noise std {float(std.mean()):.2f})'}: "
                  f"on-target {100 * hit_hist[-1]:.0f}%  standing {100 * stand_hist[-1]:.0f}%")
    if writer is not None:
        writer.close()
        print(f"wrote {args.output}")
    print(f"render: {frames} frames, {1000 * render_t / max(frames, 1):.1f} ms/frame at {W}x{H} with {N} robots "
          f"(≈ {frames / max(render_t, 1e-9):.0f} FPS render only)")


if __name__ == "__main__":
    main()
