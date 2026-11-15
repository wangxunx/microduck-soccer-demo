"""Export a football checkpoint to a game-ready ONNX + manifest.

    ../../microduck-rl-genesis/.venv/bin/python export_football.py -e football-v1 --ckpt 1999 \
        -o exports/football-v1-1999.onnx

Writes <out>.onnx (single file, observation normaliser baked in, verified
against the torch actor) and <out>.json in the game's
microduck-game-football-v1 manifest schema, so the pair can replace
game/public/assets/policies/football.{onnx,json} or be passed to the game's
evaluator via FOOTBALL_CANDIDATE.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import soccer_rl  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from soccer_rl import football_cfg as F  # noqa: E402


def load_genesis_exporter():
    spec = importlib.util.spec_from_file_location("genesis_export_onnx", soccer_rl.GENESIS_REPO / "export_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ExportedPolicy


class _StubEnv:
    """Just enough of the env interface for OnPolicyRunner to rebuild the networks."""

    def __init__(self, num_privileged):
        self.num_envs, self.num_actions, self.max_episode_length = 1, F.ACTION_SIZE, 80
        self.device, self.cfg = "cpu", {}
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self._obs = TensorDict({"policy": torch.zeros(1, F.OBS_SIZE),
                                "privileged": torch.zeros(1, num_privileged)}, batch_size=[1])

    def get_observations(self):
        return self._obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", required=True)
    p.add_argument("--ckpt", type=int, default=-1)
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    torch.set_default_device("cpu")
    log_dir = HERE / "logs" / args.exp_name
    saved = pickle.loads((log_dir / "cfgs.pkl").read_bytes())
    ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    ckpt = ckpts[-1] if args.ckpt < 0 else log_dir / f"model_{args.ckpt}.pt"
    it = int(ckpt.stem.split("_")[1])

    runner = OnPolicyRunner(_StubEnv(saved["num_privileged_obs"]), saved["train_cfg"], None, device="cpu")
    runner.load(str(ckpt))
    actor = getattr(runner.alg, "_raw_actor", None) or runner.alg.actor
    exported = load_genesis_exporter()(actor).eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, F.OBS_SIZE)
    torch.onnx.export(exported, dummy, str(out), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17)
    import onnx
    onnx.save_model(onnx.load(str(out)), str(out), save_as_external_data=False)
    if os.path.exists(str(out) + ".data"):
        os.remove(str(out) + ".data")

    import onnxruntime as ort
    torch.manual_seed(0)
    probe = torch.cat([dummy, torch.randn(64, F.OBS_SIZE)], dim=0)
    with torch.no_grad():
        ref = exported(probe).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = np.concatenate([sess.run(None, {"obs": probe[i:i + 1].numpy()})[0] for i in range(len(probe))])
    err = float(np.abs(got - ref).max())
    assert err < 1e-4, err

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "schema": "microduck-game-football-v1",
        "id": f"genesis-{args.exp_name}-{it}",
        "enabled": True,
        "file": out.name,
        "sha256": sha,
        "windowSeconds": F.WINDOW_S,
        "currentLimitAmps": F.CURRENT_LIMIT_A,
        "source": {"trainer": "Genesis (AMD ROCm) via microduck-soccer-demo/training",
                   "parent": "https://github.com/pollen-robotics/microduck_rl",
                   "engine": "genesis + microduck-rl-genesis BAM actuator"},
        "training": {"task": "Football-Genesis", "checkpointIteration": it,
                     "seed": saved["train_cfg"].get("seed"), "status": "candidate"},
        "runtime": {"observationSize": F.OBS_SIZE, "actionSize": F.ACTION_SIZE, "policyHz": F.POLICY_HZ,
                    "physicsHz": F.PHYSICS_HZ, "actuatorDelayPhysicsSteps": F.ACTUATOR_DELAY_STEPS,
                    "jointVelocityDelayControlSteps": F.JOINT_VEL_DELAY_STEPS,
                    "bodyCommand": "yaw-relative ball x/y, yaw-relative ball velocity x/y, frozen shot direction x/y",
                    "handoff": "trained stand policy for 1 second, then walk"},
        "verification": {"onnxVsTorchMaxAbs": err},
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, sha256 {sha[:12]}…), ONNX vs torch max |diff| {err:.2e}")
    print(f"wrote {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
