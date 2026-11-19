"""Genesis stand-up / standing (get-up) environment for the Microduck.

Subclass of microduck-rl-genesis's MicroduckVelocityEnv (BAM actuator, DR, IMU /
encoder models, 61-D actor obs) with the upstream stand-up task on top; recipe and
deviations in standup_cfg.py. What changes vs the velocity env:

  scene     the game's full-collision robot (robot_allcollisions.xml)
  commands  twist / head / body all zero (the game calls the stand policy that way)
  reset     random ground state: face-down, face-up (+ roll noise), sitting keyframe,
            standing; optionally the game families "drop" and "handoff"
  rewards   upstream stand-up terms (standup_cfg.REWARD_WEIGHTS)
  done      time-out (6 s) or invalid state only: a fall is the task, not a failure
  critic    + trunk height and family one-hot (privileged only; actor obs unchanged)

Logged per family: "recovered" = the game's recovery exit (projected gravity z <
-0.88 and trunk z > 0.08 held for 0.6 s), its time, and "upright at the end".
"""
from __future__ import annotations

import math

import torch
from tensordict import TensorDict

import genesis as gs

import microduck.velocity_env as ve
from microduck import constants as MC
from microduck import velocity_cfg as VC
from microduck.velocity_env import MicroduckVelocityEnv, _rand

from . import standup_cfg as S

HOLD_STEPS = 30  # 0.6 s at 50 Hz, the game's stand -> walk exit


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def _yaw_pitch_roll_quat(yaw, pitch, roll):
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    return torch.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dim=-1)


class MicroduckStandUpEnv(MicroduckVelocityEnv):
    def __init__(self, num_envs: int, show_viewer: bool = False, play: bool = False,
                 device=None, camera_cfg=None, demo: bool = False):
        self.forced_family = None  # evaluation: force one family for every reset
        super().__init__(num_envs=num_envs, rough=False, show_viewer=show_viewer, play=play,
                         device=device, camera_cfg=camera_cfg, demo=demo)
        self.cfg["task"] = "StandUp-Genesis"
        self.cfg["episode_length_s"] = S.EPISODE_LENGTH_S
        self.cfg["reward_weights"] = dict(S.REWARD_WEIGHTS)
        self.cfg["num_privileged_obs"] = int(self.obs["privileged"].shape[-1])

    # ------------------------------------------------------------------ scene
    def _build_scene(self, show_viewer, extra_morphs=None, camera_cfg=None) -> None:
        orig_xml, orig_ro = ve.MICRODUCK_WALK_XML, gs.options.RigidOptions
        ve.MICRODUCK_WALK_XML = MC.MICRODUCK_ALLCOLLISIONS_XML
        gs.options.RigidOptions = lambda **kw: orig_ro(**(kw | {"max_collision_pairs": S.MAX_COLLISION_PAIRS}))
        try:
            super()._build_scene(show_viewer, extra_morphs, camera_cfg)
        finally:
            ve.MICRODUCK_WALK_XML, gs.options.RigidOptions = orig_xml, orig_ro

    def _build_buffers(self) -> None:
        super()._build_buffers()
        n, d = self.num_envs, self.device
        self.max_episode_length_s = S.EPISODE_LENGTH_S
        self.max_episode_length = math.ceil(S.EPISODE_LENGTH_S / self.dt)
        self.reward_names = list(S.REWARD_WEIGHTS)
        self.reward_weights = dict(S.REWARD_WEIGHTS)
        self.episode_sums = {k: torch.zeros(n, dtype=gs.tc_float, device=d) for k in self.reward_names}
        self.family = torch.zeros(n, dtype=torch.long, device=d)
        self.upright_count = torch.zeros(n, dtype=torch.long, device=d)
        self.rec_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self.max_tilt_after_rec = torch.zeros(n, dtype=gs.tc_float, device=d)
        self.prev_vz = torch.zeros(n, dtype=gs.tc_float, device=d)
        self.last_tau = torch.zeros((n, MC.NUM_ACTIONS), dtype=gs.tc_float, device=d)
        self.prev_tau = torch.zeros_like(self.last_tau)
        self.leg_t = torch.tensor(S.LEG_IDS, dtype=torch.long, device=d)

    def _build_actuator(self) -> None:
        super()._build_actuator()
        compute = self.bam.compute

        def compute_and_keep(target):
            tau = compute(target)
            self.last_tau = tau
            return tau

        self.bam.compute = compute_and_keep

    # --------------------------------------------------------------- commands
    def _resample_due_commands(self) -> None:
        pass

    def _zero_commands(self, env_ids) -> None:
        self.twist_cmd[env_ids] = 0.0
        self.head_cmd[env_ids] = 0.0
        self.body_cmd[env_ids] = 0.0
        self.is_standing_env[env_ids] = True

    def _maybe_push(self) -> None:
        if not VC.ENABLE_VELOCITY_PUSHES or self.demo:
            return
        r = VC.stage_value(S.PUSH_RANGE_STAGES, self.common_step_counter)
        due = (self.episode_length_buf >= self.push_at).nonzero().flatten()
        if len(due) == 0:
            return
        if r > 0:
            vel = self.robot.get_vel()
            vel[due, 0] += _rand(-r, r, (len(due),), self.device)
            vel[due, 1] += _rand(-r, r, (len(due),), self.device)
            self.robot.set_dofs_velocity(vel[due], dofs_idx_local=[0, 1, 2], envs_idx=due)
        lo, hi = (0.5, 1.0) if self.play else VC.VELOCITY_PUSH_INTERVAL_S
        self.push_at[due] = self.episode_length_buf[due] + (
            torch.rand(len(due), device=self.device) * (hi - lo) + lo).div(self.dt).long()

    def _apply_curricula(self) -> None:
        super()._apply_curricula()  # CoM / head-CoM ranges (same stages as upstream stand-up)
        s = self.common_step_counter
        self.reward_weights["action_rate_l2"] = VC.stage_value(S.ACTION_RATE_WEIGHT_STAGES, s)
        self.reward_weights["arrival_damping"] = VC.stage_value(S.ARRIVAL_DAMPING_WEIGHT_STAGES, s)
        self.reward_weights["joint_torque_rate_l2"] = VC.stage_value(S.TORQUE_RATE_WEIGHT_STAGES, s)

    def family_probs(self) -> torch.Tensor:
        mix = dict(VC.stage_value(S.MIX_STAGES, self.common_step_counter))
        if self.common_step_counter >= S.GAME_FAMILY_START_STEP:
            mix["drop"] = S.DROP_PROB
            mix["handoff"] = S.HANDOFF_PROB
        p = torch.tensor([mix.get(f, 0.0) for f in S.FAMILIES], dtype=gs.tc_float, device=self.device)
        return p / p.sum()

    # ------------------------------------------------------------------ reset
    def reset_idx(self, env_ids) -> None:
        if len(env_ids) == 0:
            return
        logs = self._family_logs(env_ids) if hasattr(self, "base_pos") else {}
        super().reset_idx(env_ids)
        self.extras["episode"].update(logs)
        self._zero_commands(env_ids)
        n, d = len(env_ids), self.device

        if self.forced_family is not None:
            fam = torch.full((n,), S.FAMILIES.index(self.forced_family), dtype=torch.long, device=d)
        else:
            fam = torch.multinomial(self.family_probs(), n, replacement=True)
        self.family[env_ids] = fam
        is_ = {f: fam == i for i, f in enumerate(S.FAMILIES)}

        yaw = _rand(-math.pi, math.pi, (n,), d)
        zero = torch.zeros_like(yaw)
        s = 2.0 ** -0.5
        cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
        face_down = torch.stack([s * cy, -s * sy, s * cy, s * sy], dim=1)
        face_up = torch.stack([s * cy, s * sy, -s * cy, s * sy], dim=1)
        th = _rand(-S.FACE_UP_ROLL_MAX, S.FACE_UP_ROLL_MAX, (n,), d)
        face_up = _quat_mul(face_up, torch.stack([torch.cos(th / 2), zero, zero, torch.sin(th / 2)], dim=1))
        t = S.UPRIGHT_TILT_MAX
        upright = _yaw_pitch_roll_quat(yaw, _rand(-t, t, (n,), d), _rand(-t, t, (n,), d))
        t = S.HANDOFF_TILT_MAX
        handoff = _yaw_pitch_roll_quat(yaw, _rand(-t, t, (n,), d), _rand(-t, t, (n,), d))
        drop = torch.randn((n, 4), device=d)
        drop = drop / drop.norm(dim=1, keepdim=True)

        quat = face_down.clone()
        quat[is_["face_up"]] = face_up[is_["face_up"]]
        up = is_["sitting"] | is_["standing"]
        quat[up] = upright[up]
        quat[is_["handoff"]] = handoff[is_["handoff"]]
        quat[is_["drop"]] = drop[is_["drop"]]

        z = _rand(*S.PRONE_Z, (n,), d)
        for f, rng in (("sitting", S.SITTING_Z), ("standing", S.STANDING_Z), ("drop", S.DROP_Z),
                       ("handoff", S.HANDOFF_Z)):
            z = torch.where(is_[f], _rand(*rng, (n,), d), z)
        pos = self.robot.get_pos()[env_ids]
        pos[:, 2] = self.env_origins[env_ids, 2] + z

        q = self.default_dof_pos.unsqueeze(0).repeat(n, 1)
        sit = is_["sitting"]
        for j, a in S.SITTING_JOINT_OVERRIDES.items():
            q[sit, j] = a
        q += torch.randn_like(q) * S.SITTING_JOINT_NOISE_STD * sit.unsqueeze(1)
        q += torch.randn_like(q) * S.DROP_JOINT_NOISE_STD * is_["drop"].unsqueeze(1)
        legn = torch.zeros_like(q)
        legn[:, self.leg_t] = torch.randn((n, len(S.LEG_IDS)), device=d) * S.HANDOFF_LEG_NOISE_STD
        q += legn * is_["handoff"].unsqueeze(1)
        q = torch.maximum(torch.minimum(q, self.soft_limit_hi), self.soft_limit_lo)

        self.robot.set_pos(pos, envs_idx=env_ids)
        self.robot.set_quat(quat, envs_idx=env_ids)
        self.robot.zero_all_dofs_velocity(envs_idx=env_ids)
        self.robot.set_dofs_position(q, self.motors_dof_idx, envs_idx=env_ids)
        ho = is_["handoff"]
        if ho.any():
            v = torch.zeros((n, 6), dtype=gs.tc_float, device=d)
            v[:, :2] = _rand(-S.HANDOFF_LIN_VEL, S.HANDOFF_LIN_VEL, (n, 2), d)
            v[:, 3:] = _rand(-S.HANDOFF_ANG_VEL, S.HANDOFF_ANG_VEL, (n, 3), d)
            ids = env_ids[ho]
            self.robot.set_dofs_velocity(v[ho], dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=ids)

        seed = (q - self.default_dof_pos) if S.SEED_LAST_ACTION else torch.zeros_like(q)
        self.actions[env_ids] = seed
        self.last_actions[env_ids] = seed
        self.upright_count[env_ids] = 0
        self.rec_step[env_ids] = -1
        self.max_tilt_after_rec[env_ids] = 0.0
        self.prev_vz[env_ids] = 0.0

    def _family_logs(self, env_ids) -> dict:
        out = {}
        fam = self.family[env_ids]
        rec = self.rec_step[env_ids]
        up_now = self._upright_now()[env_ids]
        for i, f in enumerate(S.FAMILIES):
            m = fam == i
            if not m.any():
                continue
            ok = rec[m] >= 0
            out[f"rec_{f}"] = ok.float().mean()
            out[f"up_end_{f}"] = up_now[m].float().mean()
            if ok.any():
                out[f"rec_time_{f}"] = rec[m][ok].float().mean() * self.dt
        return out

    # ------------------------------------------------------------------- step
    def _upright_now(self) -> torch.Tensor:
        z = self.base_pos[:, 2] - self.env_origins[:, 2]
        return (self.projected_gravity[:, 2] < S.RECOVERED_GRAVITY_Z) & (z > S.RECOVERED_HEIGHT)

    def _compute_rewards(self) -> None:
        up = self._upright_now()
        self.upright_count = torch.where(up, self.upright_count + 1, torch.zeros_like(self.upright_count))
        newly = (self.upright_count >= HOLD_STEPS) & (self.rec_step < 0)
        self.rec_step = torch.where(newly, self.episode_length_buf - HOLD_STEPS, self.rec_step)

        P = S.REWARD_PARAMS
        z = torch.nan_to_num(self.base_pos[:, 2] - self.env_origins[:, 2], nan=0.0)
        qx, qy = self.base_quat[:, 1], self.base_quat[:, 2]
        tilt_sq = 2.0 * (qx * qx + qy * qy)
        cos_tilt = 1.0 - tilt_sq
        dq_legs = self.dof_pos[:, self.leg_t] - self.default_dof_pos[self.leg_t]
        vz = torch.nan_to_num(self.robot.get_vel()[:, 2], nan=0.0)
        fresh = self.episode_length_buf <= 1
        a_z = torch.where(fresh, torch.zeros_like(vz), (vz - self.prev_vz) / self.dt)
        self.prev_vz = vz
        tau_rate = torch.where(fresh.unsqueeze(1), torch.zeros_like(self.last_tau), self.last_tau - self.prev_tau)
        self.prev_tau = self.last_tau.clone()

        def smooth(x):
            x = x.clamp(0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)

        def arrival():
            lo, hi = P["arrival_height"]
            tf, tz = P["arrival_tilt_deg"]
            tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
            gate = smooth((z - lo) / (hi - lo)) * smooth((tz - tilt_deg) / (tz - tf))
            w = self._links_ang[:, self.trunk_idx, :2]
            return torch.sum(w * w, dim=1) * gate

        def composite():
            h = torch.exp(-((z - S.STAND_Z) / P["composite_height_std"]) ** 2)
            u = torch.exp(-tilt_sq / P["composite_upright_std"] ** 2)
            p = torch.exp(-(dq_legs ** 2).mean(dim=1) / P["composite_pose_std"] ** 2)
            return h * u * p

        terms = {
            "pose_stand_legs": lambda: torch.exp(-(dq_legs / P["pose_legs_std"]) ** 2).mean(dim=1),
            "head_pose_tracking": self._rew_head_pose_tracking,
            "pose_stand_l1": lambda: -dq_legs.abs().mean(dim=1),
            "height_stand": lambda: torch.exp(-((z - S.STAND_Z) / P["height_std"]) ** 2),
            "height_stand_sharp": lambda: torch.exp(-((z - S.STAND_Z) / P["height_sharp_std"]) ** 2),
            "height_stand_l1": lambda: -(z - S.STAND_Z).abs(),
            "com_upward_velocity": lambda: vz.clamp(min=0.0) * (z < P["upward_max_height"]).to(gs.tc_float),
            "gentle_rise": lambda: -a_z.abs(),
            "arrival_damping": arrival,
            "upright_linear": lambda: cos_tilt,
            "upright_sharp": lambda: torch.exp(-tilt_sq / P["upright_sharp_std"] ** 2)
            * smooth((z - S.SIT_Z) / (S.STAND_Z - S.SIT_Z)),
            "standing_composite": composite,
            "action_rate_l2": self._rew_action_rate,
            "joint_torque_rate_l2": lambda: torch.sum(tau_rate * tau_rate, dim=1),
            "body_ang_vel": self._rew_body_ang_vel,
            "angular_momentum": self._rew_angular_momentum,
            "dof_pos_limits": self._rew_dof_pos_limits,
            "self_collisions": lambda: self.self_collision_count,
        }
        self.rew_buf.zero_()
        for name, fn in terms.items():
            w = self.reward_weights[name]
            if w == 0.0:
                continue
            value = torch.nan_to_num(fn() * w * self.dt, nan=0.0, posinf=0.0, neginf=0.0)
            self.rew_buf += value
            self.episode_sums[name] += value

    def _check_termination(self) -> None:
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        nan_state = self.scene.rigid_solver.get_error_envs_mask()
        nan_state = nan_state | ~torch.isfinite(self.base_pos).all(dim=1)
        nan_state = nan_state | ~torch.isfinite(self.dof_pos).all(dim=1)
        self.reset_buf = nan_state if self.demo else (self.time_out_buf | nan_state)
        self._term_stats = {"nan_state": nan_state.to(gs.tc_float).mean()}

    def _compute_observations(self) -> TensorDict:
        obs = super()._compute_observations()
        z = (self.base_pos[:, 2] - self.env_origins[:, 2]).unsqueeze(1)
        onehot = torch.nn.functional.one_hot(self.family, len(S.FAMILIES)).to(gs.tc_float)
        obs["privileged"] = torch.cat([obs["privileged"], torch.nan_to_num(z), onehot], dim=-1)
        self.obs = obs
        return obs
