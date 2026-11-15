"""Football (ball-aware kick) task recipe for Genesis.

Deployment target: the soccer game's `football.onnx` slot
(game/public/assets/policies/football.json, schema microduck-game-football-v1).
Everything under "RUNTIME CONTRACT" must match that manifest and
game/src/physics/{world-core,football-policy}.js exactly, otherwise the
exported ONNX cannot be dropped into the game.

Design inputs (see experiments/):
  S2  full-collision robot + ball is stable in Genesis, ~63-75k env-steps/s.
  S3  Genesis has no rolling/torsional friction -> emulated torque, calibrated
      against the game's MuJoCo: mu_roll 0.003, w_eps 16 rad/s.
  S4  kick outcome transfers qualitatively (timing, direction) but the
      ball-speed ratio Genesis/MuJoCo is NOT constant (0.88x .. 1.75x)
      -> randomise ball/contact, aim for a speed *band*, accept in the game.
"""
from __future__ import annotations

import math

# ── RUNTIME CONTRACT (must match the game's football manifest) ────────────────
OBS_SIZE, ACTION_SIZE = 61, 14
POLICY_HZ, PHYSICS_HZ = 50, 200
WINDOW_S = 1.2                  # game runs the policy 1.2 s, then stand for 1 s
ACTUATOR_DELAY_STEPS = 4        # physics steps (game: fixed 4)
ACTUATOR_DELAY_DR = (3, 5)      # trained around the contract value
JOINT_VEL_DELAY_STEPS = 1       # control steps (velocity env already does 1)
CURRENT_LIMIT_A = 1.75          # BamActuator default max_current
BALL_OBS_SLOT = 55              # obs[55:61] = ball xy, ball vel xy, aim xy (robot yaw frame)
BALL_OBS_CLIP = 3.0
# obs[48:55] (twist + head command) are zero in the game during a kick.

# ── Ball (game ball, world-core.js) ───────────────────────────────────────────
BALL_RADIUS = 0.05
BALL_MASS = 0.03
BALL_MASS_DR = (0.8, 1.25)            # also varies impulse transfer (S4)
BALL_FRICTION_DR = (0.7, 1.3)         # ratio on sliding friction
BALL_ROLL_MU = 0.003                  # S3 calibration
BALL_ROLL_W_EPS = 16.0
BALL_ROLL_MU_DR = (0.7, 1.4)
BALL_TORS_MU, BALL_TORS_W_EPS = 0.01, 0.5
BALL_OBS_NOISE = 0.005                # m and m/s, uniform

# ── Episode / reset distribution ──────────────────────────────────────────────
# The game AI only triggers the kick when (football-ai.js, ballAware branch):
#   0.09 < ball forward < 0.18, |ball side| < 0.085, |aim error| < 0.18 rad,
#   robot speed < 0.12, ball speed < 0.2. Manual F can fire anywhere with aim
#   = heading. Train slightly wider than the AI gate, plus some out-of-reach
#   cases where the only right answer is "stay standing".
EPISODE_LENGTH_S = 1.6          # > WINDOW_S: finish the kick in a recoverable pose (no handoff loop)
SETTLE_START_S = 0.8            # settle rewards switch on (handoff to stand at 1.2 s)

# Handoff in the loop (v4): after WINDOW_S the env hands control to the game's
# official `stand` policy for HANDOFF_STAND_S, then `walk` (zero command), with
# the game's runtime rules for those policies (no actuator delay, no current
# limit, undelayed joint_vel, lastAction seeded with q - HOME). v3 never tilted
# > 45 deg in its own window but fell after the handoff in 8/18 S4 cases.
HANDOFF_IN_LOOP = False
HANDOFF_STAND_S = 1.0
HANDOFF_WALK_S = 0.3
BALL_FWD_RANGE = (0.08, 0.20)
BALL_SIDE_RANGE = (-0.095, 0.095)
AIM_RANGE_RAD = 0.25
FAR_BALL_FRACTION = 0.1         # ball out of reach (fwd 0.22-0.35 or |side| 0.12-0.2)
BALL_INIT_SPEED_MAX = 0.15
ROBOT_INIT_SPEED_MAX = 0.10     # m/s base xy velocity at reset
STAND_Z = 0.115                 # trunk height at HOME (upstream standup/ball_kick)

# ── Kick objective ────────────────────────────────────────────────────────────
# Ball speed band in GENESIS. The game currently gets ~1.7 m/s from the
# author's football.onnx (ball rolls 3-4 m on a 3.15 m pitch). 1.2 m/s target
# here; S4 says the game may see 0.9x-1.8x of it -> re-tune after the game eval.
BALL_TARGET_SPEED = 1.2
TOUCH_SPEED = 0.05              # ball counts as touched once it exceeds this

# Reward weights are RATES (multiplied by dt like mjlab's RewardManager).
# v1 (kick 3/step vs standing stack ~8.5/step) collapsed into "stand still":
# touches peaked at 52% (iter 50) then fell to ~9% as falls became the dominant
# risk. v2: the task dominates (kick 10/step at target, standing stack ~3.5/step)
# and a one-shot TOUCH_BONUS pays early, weak but forward touches.
REWARD_WEIGHTS: dict[str, float] = {
    # task
    "ball_speed_along_aim": 10.0,     # clamp(v·aim / target, 0, 1)
    "ball_overshoot": -2.0,           # (|v| - target)+  [m/s]
    "ball_lateral": -2.0,             # |v x aim| [m/s]
    "foot_ball_proximity": 3.0,       # exp(-d/0.04) of closest foot, until touched
    # stand / balance (whole episode)
    "upright": 1.0,
    "height": 0.5,
    "pose_legs": 0.5,                 # loose std: the swing must stay affordable
    "pose_neck": 0.5,
    "any_foot_grounded": 0.5,         # anti-hop
    # settle (t >= SETTLE_START_S): be a good starting state for `stand`
    "settle_pose": 2.0,
    "settle_joint_vel": -0.002,
    "settle_base_vel": -2.0,
    # regularisers (velocity-env parity)
    "action_rate_l2": -0.1,           # curriculum below
    "body_ang_vel": -0.05,
    "angular_momentum": -0.02,
    "dof_pos_limits": -1.0,
    "self_collisions": -1.0,
    "base_displacement": -1.0,        # don't dribble/walk after the ball: |xy - start|
    "yaw_change": 0.0,                # |heading - heading at reset| [rad]; v2b spun ~120 deg after kicks in the game
}
TERMINATION_PENALTY = -5.0            # one-off, not scaled by dt
TOUCH_BONUS = 2.0                     # one-off at first touch if ball moves along aim > TOUCH_BONUS_MIN_VPAR
TOUCH_BONUS_MIN_VPAR = 0.1

REWARD_PARAMS = {
    "upright_std": math.sqrt(0.05),
    "height_std": 0.04,
    "pose_legs_std": 0.5,
    "pose_neck_std": 0.3,
    "settle_pose_std": 0.25,
    "proximity_scale": 0.04,
}

# Curriculum on action-rate weight (common_step_counter = env steps).
ACTION_RATE_WEIGHT_STAGES = [(0, -0.1), (24 * 600, -0.3), (24 * 1200, -0.6)]

TERMINATION_TILT_DEG = 60.0
