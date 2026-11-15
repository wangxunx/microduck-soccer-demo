"""Genesis football (ball-aware kick) environment for the Microduck.

Subclass of microduck-rl-genesis's MicroduckVelocityEnv: inherits the BAM
actuator, domain randomisation, IMU/encoder models and their invariants
(batch_dofs_info, non-accumulating DR, armature drawn once). Overrides only
what the football task changes:

  scene     full-collision robot (robot_allcollisions.xml) + the game's ball,
            ball rolling/torsional resistance emulated before every substep (S3)
  actuator  delay drawn in ACTUATOR_DELAY_DR around the contract's 4 steps,
            1.75 A current limit (BamActuator default)
  commands  twist/head/body all zero (game feeds zeros during a kick)
  obs       61-D; slots 55-60 = game's footballCommand(): ball xy, ball vel xy,
            frozen aim xy, all in the robot yaw frame, clipped to +/-3
  reset     robot standing near HOME, ball placed in front (AI kick gate +
            margin, some out-of-reach), aim sampled around heading
  rewards   see football_cfg.REWARD_WEIGHTS

The parent module is imported from the microduck-rl-genesis checkout (path in
soccer_rl/__init__.py); that repository is not modified.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import transform_by_quat

import microduck.velocity_env as ve
from microduck import constants as MC
from microduck import velocity_cfg as VC
from microduck.bam_actuator import DelayBuffer
from microduck.velocity_env import MicroduckVelocityEnv, _rand

from . import football_cfg as F

BALL_XML = Path(__file__).resolve().parent / "assets" / "ball_game.xml"
GAME_POLICIES = Path(__file__).resolve().parents[2] / "game" / "public" / "assets" / "policies"


class MicroduckFootballEnv(MicroduckVelocityEnv):
    def __init__(self, num_envs: int, show_viewer: bool = False, play: bool = False,
                 device=None, camera_cfg=None, demo: bool = False):
        super().__init__(num_envs=num_envs, rough=False, show_viewer=show_viewer, play=play,
                         device=device, camera_cfg=camera_cfg, demo=demo)
        self.cfg["task"] = "Football-Genesis"
        self.cfg["episode_length_s"] = F.EPISODE_LENGTH_S
        self.cfg["reward_weights"] = dict(F.REWARD_WEIGHTS)
        self.cfg["num_privileged_obs"] = int(self.obs["privileged"].shape[-1])

    # ------------------------------------------------------------------ scene
    def _build_scene(self, show_viewer, extra_morphs=None, camera_cfg=None) -> None:
        # The parent builds the robot from the module-level MICRODUCK_WALK_XML and
        # passes RigidOptions inline; swap both for the duration of the build only.
        orig_xml, orig_ro = ve.MICRODUCK_WALK_XML, gs.options.RigidOptions
        ve.MICRODUCK_WALK_XML = MC.MICRODUCK_ALLCOLLISIONS_XML
        gs.options.RigidOptions = lambda **kw: orig_ro(**(kw | {"max_collision_pairs": 60}))
        try:
            super()._build_scene(show_viewer, [gs.morphs.MJCF(file=str(BALL_XML))], camera_cfg)
        finally:
            ve.MICRODUCK_WALK_XML, gs.options.RigidOptions = orig_xml, orig_ro
        self.ball = self.decor[0]
        self.ball_link = getattr(self.ball, "base_link_idx", self.ball.link_start)

        orig_step = self.scene.step

        def step_with_ball_resistance(*a, **kw):
            self._apply_ball_resistance()
            return orig_step(*a, **kw)

        self.scene.step = step_with_ball_resistance

    def _apply_ball_resistance(self) -> None:
        """MuJoCo-like rolling + torsional friction (S3): tau = -mu N w / max(|w|, eps)."""
        w = self.ball.get_ang()
        touching = (self.ball.get_pos()[:, 2] < F.BALL_RADIUS + 0.002).to(w.dtype).unsqueeze(1)
        N = self.ball_mass * 9.81
        wh = w.clone()
        wh[:, 2] = 0.0
        tau = -self.ball_roll_mu * N * wh / torch.clamp(wh.norm(dim=1, keepdim=True), min=F.BALL_ROLL_W_EPS)
        tau[:, 2] = (-F.BALL_TORS_MU * N[:, 0] * w[:, 2]
                     / torch.clamp(w[:, 2].abs(), min=F.BALL_TORS_W_EPS))
        self.scene.rigid_solver.apply_links_external_torque(
            (tau * touching).unsqueeze(1), links_idx=[self.ball_link])

    # ---------------------------------------------------------------- buffers
    def _build_buffers(self) -> None:
        super()._build_buffers()
        n, d = self.num_envs, self.device
        self.handoff = bool(F.HANDOFF_IN_LOOP)
        ep_s = (F.WINDOW_S + F.HANDOFF_STAND_S + F.HANDOFF_WALK_S) if self.handoff else F.EPISODE_LENGTH_S
        self.max_episode_length_s = ep_s
        self.max_episode_length = math.ceil(ep_s / self.dt)
        self.window_steps = round(F.WINDOW_S / self.dt)
        self.stand_end_steps = round((F.WINDOW_S + F.HANDOFF_STAND_S) / self.dt)
        self.fell_phase = torch.zeros(n, dtype=torch.long, device=d)  # 0 none, 1 in window, 2 after handoff
        if self.handoff:
            from .onnx_mlp import OnnxMlp
            self.stand_policy = OnnxMlp(str(GAME_POLICIES / "BEST_alpha_stand.onnx")).to(d)
            self.walk_policy = OnnxMlp(str(GAME_POLICIES / "BEST_alpha_walking.onnx")).to(d)
            self.aux_last_action = torch.zeros((n, MC.NUM_ACTIONS), dtype=gs.tc_float, device=d)
        self.aim_w = torch.zeros((n, 2), dtype=gs.tc_float, device=d)
        self.base_start_xy = torch.zeros((n, 2), dtype=gs.tc_float, device=d)
        self.base_start_yaw = torch.zeros(n, dtype=gs.tc_float, device=d)
        self.ball_touched = torch.zeros(n, dtype=torch.bool, device=d)
        self.prev_ball_vel = torch.zeros((n, 3), dtype=gs.tc_float, device=d)
        self.kick_max_vpar = torch.zeros(n, dtype=gs.tc_float, device=d)
        self.kick_max_vperp = torch.zeros(n, dtype=gs.tc_float, device=d)
        self.ball_mass = torch.full((n, 1), F.BALL_MASS, dtype=gs.tc_float, device=d)
        self.ball_roll_mu = torch.full((n, 1), F.BALL_ROLL_MU, dtype=gs.tc_float, device=d)
        self.reward_names = list(F.REWARD_WEIGHTS)
        self.reward_weights = dict(F.REWARD_WEIGHTS)
        self.episode_sums = {k: torch.zeros(n, dtype=gs.tc_float, device=d) for k in self.reward_names}
        self.episode_sums["termination"] = torch.zeros(n, dtype=gs.tc_float, device=d)
        self.episode_sums["touch_bonus"] = torch.zeros(n, dtype=gs.tc_float, device=d)

    def _build_actuator(self) -> None:
        super()._build_actuator()
        lo, hi = (F.ACTUATOR_DELAY_STEPS, F.ACTUATOR_DELAY_STEPS) if self.demo else F.ACTUATOR_DELAY_DR
        self.bam._delay = DelayBuffer((self.num_envs, MC.NUM_ACTIONS), lo, hi, 0, device=self.device)
        assert self.bam.max_current == F.CURRENT_LIMIT_A, self.bam.max_current
        self.fb_lag = self.bam._delay._lag.clone()
        # per-env current limit so the handoff phase can run without one (game rule)
        self.bam.max_current = torch.full((self.num_envs, 1), F.CURRENT_LIMIT_A, dtype=gs.tc_float,
                                          device=self.device)
        assert VC.OBS_DELAY["joint_vel"][:2] == (F.JOINT_VEL_DELAY_STEPS, F.JOINT_VEL_DELAY_STEPS)

    def _startup_randomization(self) -> None:
        super()._startup_randomization()
        if self.demo:
            return
        n, d = self.num_envs, self.device
        scale = _rand(*F.BALL_MASS_DR, (n, 1), d)
        self.ball.set_mass_shift(F.BALL_MASS * (scale - 1.0), links_idx_local=[0])
        self.ball_mass = F.BALL_MASS * scale
        self.ball.set_friction_ratio(_rand(*F.BALL_FRICTION_DR, (n, 1), d), links_idx_local=[0])
        self.ball_roll_mu = F.BALL_ROLL_MU * _rand(*F.BALL_ROLL_MU_DR, (n, 1), d)

    # ------------------------------------------------------------------- step
    def step(self, actions: torch.Tensor):
        if not self.handoff:
            return super().step(actions)
        k = self.episode_length_buf
        ctrl = k < self.window_steps                       # our policy acts
        if (~ctrl).any():
            first = k == self.window_steps                 # game: lastAction <- q - HOME at handoff
            self.aux_last_action = torch.where(first.unsqueeze(1), self.dof_pos - self.default_dof_pos,
                                               self.aux_last_action)
            aux = self.obs["policy"].clone()
            vel = self.dof_vel_enc
            if not self.demo:
                a = VC.OBS_NOISE["joint_vel"]
                vel = vel + _rand(-a, a, vel.shape, self.device)
            aux[:, 20:34] = vel                            # undelayed joint_vel (footballActive false)
            aux[:, 34:48] = self.aux_last_action
            aux[:, 48:] = 0.0                              # zero command, no ball slots
            stand = k < self.stand_end_steps
            aux_act = torch.where(stand.unsqueeze(1), self.stand_policy(aux), self.walk_policy(aux))
            actions = torch.where(ctrl.unsqueeze(1), actions, aux_act)
            self.aux_last_action = torch.where(ctrl.unsqueeze(1), self.aux_last_action, aux_act)
        self.bam._delay._lag = torch.where(ctrl, self.fb_lag, torch.zeros_like(self.fb_lag))
        self.bam.max_current[:, 0] = torch.where(ctrl, torch.full_like(k, 0, dtype=gs.tc_float) + F.CURRENT_LIMIT_A,
                                                 torch.full_like(k, 0, dtype=gs.tc_float) + 1e9)
        return super().step(actions)

    # --------------------------------------------------------------- commands
    def _resample_due_commands(self) -> None:
        pass  # twist/head/body stay zero: the game feeds zeros during a kick

    def _zero_commands(self, env_ids) -> None:
        self.twist_cmd[env_ids] = 0.0
        self.head_cmd[env_ids] = 0.0
        self.body_cmd[env_ids] = 0.0
        self.is_standing_env[env_ids] = True

    def _apply_curricula(self) -> None:
        super()._apply_curricula()
        self.reward_weights["action_rate_l2"] = VC.stage_value(
            F.ACTION_RATE_WEIGHT_STAGES, self.common_step_counter)

    # ------------------------------------------------------------------ reset
    def reset_idx(self, env_ids) -> None:
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)  # logs + zeroes episode_sums (incl. "termination")
        if hasattr(self, "ball_pos"):
            ep = self.extras["episode"]
            ep["kick_touched"] = self.ball_touched[env_ids].float().mean()
            ep["kick_max_vpar"] = self.kick_max_vpar[env_ids].mean()
            ep["kick_max_vperp"] = self.kick_max_vperp[env_ids].mean()
            ep["fall_in_window"] = (self.fell_phase[env_ids] == 1).float().mean()
            ep["fall_after_handoff"] = (self.fell_phase[env_ids] == 2).float().mean()
        self.fell_phase[env_ids] = 0
        self._zero_commands(env_ids)
        n, d = len(env_ids), self.device

        pos = self.robot.get_pos()[env_ids]
        q = self.robot.get_quat()[env_ids]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

        # small base velocity: the AI fires while still drifting (< 0.12 m/s)
        v = self.robot.get_dofs_velocity(envs_idx=env_ids)
        v[:, :2] = _rand(-F.ROBOT_INIT_SPEED_MAX, F.ROBOT_INIT_SPEED_MAX, (n, 2), d) / math.sqrt(2)
        self.robot.set_dofs_velocity(v[:, :2], dofs_idx_local=[0, 1], envs_idx=env_ids)

        fwd = _rand(*F.BALL_FWD_RANGE, (n,), d)
        side = _rand(*F.BALL_SIDE_RANGE, (n,), d)
        far = torch.rand(n, device=d) < F.FAR_BALL_FRACTION
        far_side = torch.rand(n, device=d) < 0.5
        sign = torch.where(torch.rand(n, device=d) < 0.5, -1.0, 1.0)
        fwd = torch.where(far & ~far_side, _rand(0.22, 0.35, (n,), d), fwd)
        side = torch.where(far & far_side, sign * _rand(0.12, 0.20, (n,), d), side)
        c, s = torch.cos(yaw), torch.sin(yaw)
        bxy = pos[:, :2] + torch.stack([c * fwd - s * side, s * fwd + c * side], dim=-1)
        self.ball.set_pos(torch.cat([bxy, torch.full((n, 1), F.BALL_RADIUS, device=d)], dim=-1),
                          envs_idx=env_ids)
        self.ball.set_quat(torch.tensor([[1.0, 0, 0, 0]], device=d).repeat(n, 1), envs_idx=env_ids)
        ang = _rand(-math.pi, math.pi, (n,), d)
        sp = _rand(0.0, F.BALL_INIT_SPEED_MAX, (n,), d)
        bv = torch.zeros((n, 6), dtype=gs.tc_float, device=d)
        bv[:, 0], bv[:, 1] = sp * torch.cos(ang), sp * torch.sin(ang)
        bv[:, 3], bv[:, 4] = -bv[:, 1] / F.BALL_RADIUS, bv[:, 0] / F.BALL_RADIUS  # rolling, no slip
        self.ball.set_dofs_velocity(bv, envs_idx=env_ids)

        aim = yaw + _rand(-F.AIM_RANGE_RAD, F.AIM_RANGE_RAD, (n,), d)
        self.aim_w[env_ids] = torch.stack([torch.cos(aim), torch.sin(aim)], dim=-1)
        self.base_start_xy[env_ids] = pos[:, :2]
        self.base_start_yaw[env_ids] = yaw
        self.ball_touched[env_ids] = False
        self.prev_ball_vel[env_ids] = bv[:, :3]
        self.kick_max_vpar[env_ids] = 0.0
        self.kick_max_vperp[env_ids] = 0.0

    # ------------------------------------------------------------------ state
    def _refresh_state(self) -> None:
        super()._refresh_state()
        self.ball_pos = self.ball.get_pos()
        self.ball_vel = self.ball.get_vel()
        q = self.base_quat
        self.base_yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                    1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def _yaw_frame(self, xy: torch.Tensor) -> torch.Tensor:
        """World xy -> robot yaw frame (game football-policy.js convention)."""
        c, s = torch.cos(self.base_yaw), torch.sin(self.base_yaw)
        return torch.stack([c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1]], dim=-1)

    def ball_command(self) -> torch.Tensor:
        """obs[55:61], identical to game footballCommand(): no noise, clipped."""
        rel = self._yaw_frame(self.ball_pos[:, :2] - self.base_pos[:, :2])
        vel = self._yaw_frame(self.ball_vel[:, :2])
        aim = self._yaw_frame(self.aim_w)
        return torch.cat([rel, vel, aim], dim=-1).clamp(-F.BALL_OBS_CLIP, F.BALL_OBS_CLIP)

    # ------------------------------------------------------------ observations
    def _compute_observations(self) -> TensorDict:
        ang_vel, gravity = self.base_ang_vel, self.projected_gravity
        if self.imu_misalign_quat is not None:
            ang_vel = transform_by_quat(ang_vel, self.imu_misalign_quat)
            gravity = transform_by_quat(gravity, self.imu_misalign_quat)
        ang_vel = self.obs_delays["base_ang_vel"](ang_vel)
        gravity = self.obs_delays["projected_gravity"](gravity)
        joint_vel = self.obs_delays["joint_vel"](self.dof_vel_enc)
        joint_pos_rel = self.dof_pos_enc - self.default_dof_pos
        actor_joint_pos = joint_pos_rel + self.encoder_bias

        def noisy(x, a):
            return x if self.demo else x + _rand(-a, a, x.shape, self.device)

        ball_cmd = self.ball_command()
        if not self.demo:
            ball_cmd[:, :4] = (ball_cmd[:, :4] + _rand(-F.BALL_OBS_NOISE, F.BALL_OBS_NOISE, (self.num_envs, 4),
                                                       self.device)).clamp(-F.BALL_OBS_CLIP, F.BALL_OBS_CLIP)
        n = self.num_envs
        policy = torch.cat([
            noisy(ang_vel, VC.OBS_NOISE["base_ang_vel"]),
            noisy(gravity, VC.OBS_NOISE["projected_gravity"]),
            noisy(actor_joint_pos, VC.OBS_NOISE["joint_pos"]),
            noisy(joint_vel, VC.OBS_NOISE["joint_vel"]),
            self.actions,
            torch.zeros((n, 7), dtype=gs.tc_float, device=self.device),  # twist(3) + head(4): zero
            ball_cmd,                                                    # body slots 55-60
        ], dim=-1)
        assert policy.shape[-1] == F.OBS_SIZE, policy.shape

        t = (self.episode_length_buf.float() * self.dt / F.EPISODE_LENGTH_S).unsqueeze(1)
        ball_rel3 = torch.cat([self._yaw_frame(self.ball_pos[:, :2] - self.base_pos[:, :2]),
                               self.ball_pos[:, 2:3] - F.BALL_RADIUS], dim=-1)
        ball_vel3 = torch.cat([self._yaw_frame(self.ball_vel[:, :2]), self.ball_vel[:, 2:3]], dim=-1)
        privileged = torch.cat([
            self.base_lin_vel, joint_pos_rel, self.contact.to(gs.tc_float), self.feet_air_time,
            self.foot_height, self.foot_contact_force.reshape(n, -1),
            ball_rel3, ball_vel3, self._yaw_frame(self.aim_w), t, self.ball_touched.float().unsqueeze(1),
        ], dim=-1)
        privileged = torch.nan_to_num(privileged, nan=0.0, posinf=0.0, neginf=0.0)
        self.obs = TensorDict({"policy": policy, "privileged": privileged}, batch_size=[n], device=self.device)
        return self.obs

    # ---------------------------------------------------------------- rewards
    def _compute_rewards(self) -> None:
        P, n = F.REWARD_PARAMS, self.num_envs
        v = self.ball_vel[:, :2]
        vpar = (v * self.aim_w).sum(dim=1)
        vperp = (v[:, 0] * self.aim_w[:, 1] - v[:, 1] * self.aim_w[:, 0]).abs()
        speed = v.norm(dim=1)
        # touched = sudden ball velocity change (free rolling changes < 0.01 m/s per step)
        dv = (self.ball_vel[:, :3] - self.prev_ball_vel).norm(dim=1)
        was_touched = self.ball_touched.clone()
        self.ball_touched |= dv > 0.08
        self.prev_ball_vel = self.ball_vel[:, :3].clone()
        self.kick_max_vpar = torch.maximum(self.kick_max_vpar, vpar)
        self.kick_max_vperp = torch.where(vpar >= self.kick_max_vpar, vperp, self.kick_max_vperp)

        t = self.episode_length_buf.float() * self.dt
        in_window = (t <= F.WINDOW_S + 1e-6) if self.handoff else torch.ones_like(t, dtype=torch.bool)
        settle = ((t >= F.SETTLE_START_S) & in_window).to(gs.tc_float)
        own = in_window.to(gs.tc_float)                    # terms that only make sense for our actions
        leg, neck = self.leg_ids, self.head_ids
        dq = self.dof_pos - self.default_dof_pos
        foot_d = ((self.foot_pos - self.ball_pos.unsqueeze(1)).norm(dim=-1) - F.BALL_RADIUS).clamp(min=0).min(dim=1).values

        terms = {
            "ball_speed_along_aim": lambda: (vpar / F.BALL_TARGET_SPEED).clamp(0.0, 1.0),
            "ball_overshoot": lambda: (speed - F.BALL_TARGET_SPEED).clamp(min=0.0),
            "ball_lateral": lambda: vperp,
            "foot_ball_proximity": lambda: torch.exp(-foot_d / P["proximity_scale"]) * (~was_touched).float(),
            "upright": self._rew_upright,
            "height": lambda: torch.exp(-torch.square(self.base_pos[:, 2] - F.STAND_Z) / P["height_std"] ** 2),
            "pose_legs": lambda: torch.exp(-torch.mean(torch.square(dq[:, leg]), dim=1) / P["pose_legs_std"] ** 2),
            "pose_neck": lambda: torch.exp(-torch.mean(torch.square(dq[:, neck]), dim=1) / P["pose_neck_std"] ** 2),
            "any_foot_grounded": lambda: self.contact.any(dim=1).to(gs.tc_float),
            "settle_pose": lambda: settle * torch.exp(-torch.mean(torch.square(dq), dim=1) / P["settle_pose_std"] ** 2),
            "settle_joint_vel": lambda: settle * torch.sum(torch.square(self.dof_vel), dim=1),
            "settle_base_vel": lambda: settle * torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1),
            "action_rate_l2": lambda: own * self._rew_action_rate(),
            "body_ang_vel": self._rew_body_ang_vel,
            "angular_momentum": self._rew_angular_momentum,
            "dof_pos_limits": self._rew_dof_pos_limits,
            "self_collisions": lambda: self.self_collision_count,
            "base_displacement": lambda: (self.base_pos[:, :2] - self.base_start_xy).norm(dim=1),
            "yaw_change": lambda: torch.atan2(torch.sin(self.base_yaw - self.base_start_yaw),
                                              torch.cos(self.base_yaw - self.base_start_yaw)).abs(),
        }
        self.rew_buf.zero_()
        for name, fn in terms.items():
            w = self.reward_weights[name]
            if w == 0.0:
                continue
            value = torch.nan_to_num(fn() * w * self.dt, nan=0.0, posinf=0.0, neginf=0.0)
            self.rew_buf += value
            self.episode_sums[name] += value
        new_touch = self.ball_touched & ~was_touched & (vpar > F.TOUCH_BONUS_MIN_VPAR)
        bonus = new_touch.to(gs.tc_float) * F.TOUCH_BONUS
        self.rew_buf += bonus
        self.episode_sums["touch_bonus"] += bonus

    # ------------------------------------------------------------ termination
    def _check_termination(self) -> None:
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        cos_tilt = -self.projected_gravity[:, 2].clamp(-1.0, 1.0)
        fell = cos_tilt < math.cos(math.radians(F.TERMINATION_TILT_DEG))
        nan_state = self.scene.rigid_solver.get_error_envs_mask()
        nan_state = nan_state | ~torch.isfinite(self.base_pos).all(dim=1) | ~torch.isfinite(self.dof_pos).all(dim=1)
        nan_state = nan_state | ~torch.isfinite(self.ball_pos).all(dim=1) | ~torch.isfinite(self.ball_vel).all(dim=1)
        if self.demo:
            self.reset_buf = nan_state | self.time_out_buf
        else:
            self.reset_buf = self.time_out_buf | fell | nan_state
            after = (self.episode_length_buf > self.window_steps) if self.handoff else torch.zeros_like(fell)
            self.fell_phase = torch.where(fell, torch.where(after, 2, 1), self.fell_phase)
            pen = fell.to(gs.tc_float) * F.TERMINATION_PENALTY
            self.rew_buf += pen
            self.episode_sums["termination"] += pen
        self._term_stats = {"fell_over": fell.to(gs.tc_float).mean(),
                            "nan_state": nan_state.to(gs.tc_float).mean()}
