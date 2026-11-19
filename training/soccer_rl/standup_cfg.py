"""Recipe for the Genesis stand-up / standing (get-up) policy.

Port of pollen-robotics/microduck_rl `microduck_standup_env_cfg.py` (the task whose
checkpoint 69u48n8l/model_9750 is the game's BEST_alpha_stand.onnx), restricted to
what the game uses: the policy is called with a zero twist / head / body command, so
body control and the head-pose command curriculum are dropped (all commands stay 0,
the neck is pulled to HOME through head_pose_tracking with a zero command).

Deliberate deviations from upstream (each one is a recipe switch below):
  * robot model: the game's robot_allcollisions.xml (upstream: groundcontact model
    whose non-foot geoms are frictionless condim 1; Genesis ignores condim and the
    game uses normal friction, so we train on the deployment model).
  * extra reset families (game-specific, START_MIX / MIX_STAGES): "drop" = random
    orientation dropped from a small height (falls in a match are not just exact
    face-down / face-up), "handoff" = upright but with a perturbed kick-end posture
    and base velocity (the game hands over from the kick policy to this one).
    Upstream mix = drop 0 / handoff 0.
  * action history seeded with q - HOME at reset (the game sets lastAction this way
    at the kick handoff; after a fall it keeps the walk's last action).
Values not listed here are inherited from the velocity recipe (DR, noise, delays,
PPO), which upstream also shares between the two tasks.
"""
import math

NUM_STEPS_PER_ENV = 24
EPISODE_LENGTH_S = 6.0

# trunk heights (m), measured upstream on the same kinematics
SIT_Z = 0.060
STAND_Z = 0.115

LEG_IDS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
NECK_IDS = [5, 6, 7, 8]

# sit keyframe (14-joint order: 0-4 left leg, 5-8 neck/head, 9-13 right leg)
SITTING_JOINT_OVERRIDES = {1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0,
                           10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0}

# ---- reset families -------------------------------------------------------
FAMILIES = ("face_down", "face_up", "sitting", "standing", "drop", "handoff")
# (step, {family: prob}); probabilities are normalised. Upstream ground_state_mix.
MIX_STAGES = [
    (0,                        {"standing": 0.40, "sitting": 0.40, "face_down": 0.20, "face_up": 0.00}),
    (600 * NUM_STEPS_PER_ENV,  {"standing": 0.25, "sitting": 0.30, "face_down": 0.35, "face_up": 0.10}),
    (1500 * NUM_STEPS_PER_ENV, {"standing": 0.20, "sitting": 0.25, "face_down": 0.30, "face_up": 0.25}),
    (2500 * NUM_STEPS_PER_ENV, {"standing": 0.15, "sitting": 0.20, "face_down": 0.30, "face_up": 0.35}),
]
# Game-specific families added on top of each stage (0 = pure upstream recipe).
GAME_FAMILY_START_STEP = 600 * NUM_STEPS_PER_ENV
DROP_PROB = 0.0
HANDOFF_PROB = 0.0

PRONE_Z = (0.05, 0.09)
FACE_UP_ROLL_MAX = math.radians(90)
SITTING_JOINT_NOISE_STD = 0.12
UPRIGHT_TILT_MAX = math.radians(10)       # sitting + standing
SITTING_Z = (0.05, 0.09)
STANDING_Z = (0.11, 0.12)
DROP_Z = (0.10, 0.16)
DROP_JOINT_NOISE_STD = 0.25
HANDOFF_TILT_MAX = math.radians(15)
HANDOFF_Z = (0.105, 0.12)
HANDOFF_LEG_NOISE_STD = 0.25
HANDOFF_LIN_VEL = 0.15                    # m/s, per axis
HANDOFF_ANG_VEL = 1.0                     # rad/s, per axis
SEED_LAST_ACTION = True

# ---- rewards (weights are rates, multiplied by dt like the velocity env) ----
REWARD_WEIGHTS = {
    "pose_stand_legs": 2.0,
    "head_pose_tracking": 0.75,
    "pose_stand_l1": 1.25,
    "height_stand": 1.0,
    "height_stand_sharp": 1.0,
    "height_stand_l1": 7.5,
    "com_upward_velocity": 0.75,
    "gentle_rise": 0.005,                 # term returns -|a_z| (positive weight, see upstream note)
    "arrival_damping": 0.0,               # curriculum
    "upright_linear": 1.5,
    "upright_sharp": 1.5,
    "standing_composite": 3.75,
    "action_rate_l2": -0.1,               # curriculum
    "joint_torque_rate_l2": 0.0,          # curriculum
    "body_ang_vel": -0.05,
    "angular_momentum": -0.02,
    "dof_pos_limits": -1.0,
    "self_collisions": -1.0,
}
REWARD_PARAMS = {
    "pose_legs_std": 0.5,
    "head_pose_std": 0.5,
    "height_std": 0.04,
    "height_sharp_std": 0.015,
    "upward_max_height": 0.125,
    "arrival_height": (0.09, 0.11),
    "arrival_tilt_deg": (20.0, 45.0),
    "upright_sharp_std": 0.3,
    "composite_height_std": 0.04,
    "composite_upright_std": 0.40,
    "composite_pose_std": 0.40,
}
ACTION_RATE_WEIGHT_STAGES = [
    (0, -0.1), (500 * NUM_STEPS_PER_ENV, -0.2), (750 * NUM_STEPS_PER_ENV, -0.4),
    (1000 * NUM_STEPS_PER_ENV, -0.6), (1250 * NUM_STEPS_PER_ENV, -0.8), (1500 * NUM_STEPS_PER_ENV, -1.0),
]
ARRIVAL_DAMPING_WEIGHT_STAGES = [(0, 0.0), (3000 * NUM_STEPS_PER_ENV, -0.025), (4000 * NUM_STEPS_PER_ENV, -0.05)]
TORQUE_RATE_WEIGHT_STAGES = [(0, 0.0), (3000 * NUM_STEPS_PER_ENV, -1e-3)]
PUSH_RANGE_STAGES = [(0, 0.0), (500 * NUM_STEPS_PER_ENV, 0.08), (1000 * NUM_STEPS_PER_ENV, 0.3)]

# ---- success criterion used for logging and evaluation (game's recovery exit) ----
RECOVERED_GRAVITY_Z = -0.88               # projected gravity z (tilt < ~28 deg)
RECOVERED_HEIGHT = 0.08
MAX_COLLISION_PAIRS = 100
