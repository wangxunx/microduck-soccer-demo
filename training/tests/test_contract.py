"""Deployment-contract test for the Genesis football env.

Checks, against the soccer game itself:
  1. obs[55:61] == game/src/physics/football-policy.js footballCommand()
     (the real JS function, run under Node) for random robot/ball/aim states;
  2. obs[48:55] == 0 (twist + head slots during a kick);
  3. actuator contract: delay 4 physics steps in demo / within DR in training,
     1.75 A current limit, joint_vel observed with a 1-control-step delay.

    HIP_VISIBLE_DEVICES=1 ../../microduck-rl-genesis/.venv/bin/python tests/test_contract.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
GAME = (TRAINING / "../game").resolve()
sys.path.insert(0, str(TRAINING))
import soccer_rl  # noqa: E402,F401
import genesis as gs  # noqa: E402
import torch  # noqa: E402
from soccer_rl import football_cfg as F  # noqa: E402


def js_football_command(cases):
    script = f"""
import {{ footballCommand }} from "{(GAME / 'src/physics/football-policy.js').as_uri()}";
const cases = JSON.parse(process.argv[1]);
console.log(JSON.stringify(cases.map(c => footballCommand(c.player, c.ball, c.field, c.direction))));
"""
    out = subprocess.run(["node", "--input-type=module", "-e", script, json.dumps(cases)],
                         capture_output=True, text=True, check=True)
    return np.array(json.loads(out.stdout))


def main():
    gs.init(backend=gs.gpu, logging_level="warning", seed=3)
    from soccer_rl.football_env import MicroduckFootballEnv

    n = 16
    env = MicroduckFootballEnv(num_envs=n, play=True, demo=True)
    env.reset()
    zero = torch.zeros((n, 14), device=env.device)
    for _ in range(3):
        env.step(zero)
    # Random yaw spin + push ball around so states are generic (not just the reset layout).
    g = torch.Generator(device="cpu").manual_seed(0)
    bv = torch.zeros((n, 6), device="cpu")
    bv[:, :2] = (torch.rand((n, 2), generator=g, device="cpu") - 0.5) * 2.0
    env.ball.set_dofs_velocity(bv.to(env.device))
    env.aim_w = torch.nn.functional.normalize(torch.randn((n, 2), generator=g, device="cpu"), dim=1).to(env.device)
    obs = env.step(zero)[0]["policy"].cpu().numpy()

    pos, yaw = env.base_pos.cpu().numpy(), env.base_yaw.cpu().numpy()
    bp, bvel, aim = env.ball_pos.cpu().numpy(), env.ball_vel.cpu().numpy(), env.aim_w.cpu().numpy()
    # Game coordinates: x = MuJoCo x, z = -MuJoCo y; yaw is the MuJoCo yaw (poseYaw).
    cases = [dict(player=dict(x=float(pos[i, 0]), z=float(-pos[i, 1]), yaw=float(yaw[i]), team="blue"),
                  ball=dict(x=float(bp[i, 0]), z=float(-bp[i, 1]), vx=float(bvel[i, 0]), vz=float(-bvel[i, 1])),
                  field=dict(halfLength=1.575),
                  direction=dict(x=float(aim[i, 0]), z=float(-aim[i, 1]))) for i in range(n)]
    ref = js_football_command(cases)
    err = np.abs(obs[:, 55:61] - ref).max()
    print(f"[1] obs[55:61] vs game footballCommand(): max |diff| = {err:.2e} over {n} random states")
    assert err < 1e-5, (obs[:, 55:61], ref)
    assert np.abs(obs[:, 55:61]).max() <= F.BALL_OBS_CLIP + 1e-6

    assert np.all(obs[:, 48:55] == 0.0)
    print("[2] obs[48:55] (twist + head) == 0")

    lag = env.bam._delay._lag
    assert torch.all(lag == F.ACTUATOR_DELAY_STEPS), lag
    assert torch.all(env.bam.max_current == F.CURRENT_LIMIT_A)
    jv = env.obs_delays["joint_vel"]
    assert jv.min_lag == jv.max_lag == F.JOINT_VEL_DELAY_STEPS
    print(f"[3] actuator delay {F.ACTUATOR_DELAY_STEPS} physics steps (demo), current limit "
          f"{float(env.bam.max_current.max())} A, joint_vel delay {jv.max_lag} control step")
    print("CONTRACT OK")


if __name__ == "__main__":
    main()
