"""Export a walking checkpoint (train_walk.py run) to a single-file ONNX for the game.

    ../../microduck-rl-genesis/.venv/bin/python export_walk.py -e walk-B --ckpt 5998 -o exports/walk-B-5998.onnx

The observation normaliser is baked into the graph (same exporter as microduck-rl-genesis);
the ONNX is verified against the torch actor. Writes <out>.json with provenance.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import soccer_rl  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from export_football import _StubEnv, load_genesis_exporter  # noqa: E402

OBS, ACT = 61, 14


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
    dummy = torch.zeros(1, OBS)
    torch.onnx.export(exported, dummy, str(out), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17)
    import onnx
    onnx.save_model(onnx.load(str(out)), str(out), save_as_external_data=False)
    if os.path.exists(str(out) + ".data"):
        os.remove(str(out) + ".data")

    import onnxruntime as ort
    torch.manual_seed(0)
    probe = torch.cat([dummy, torch.randn(64, OBS)], dim=0)
    with torch.no_grad():
        ref = exported(probe).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"obs": probe.numpy()})[0]
    err = float(np.abs(got - ref).max())
    assert err < 1e-4, err
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {"id": f"genesis-{args.exp_name}-{it}", "task": "walk (velocity tracking)", "file": out.name, "sha256": sha,
            "checkpointIteration": it, "robotModel": saved.get("robot_model"), "resumedFrom": saved.get("resume"),
            "recipeOverrides": saved.get("overrides"), "contract": "61D obs / 14D action / 50 Hz, same as BEST_alpha_walking.onnx",
            "onnxVsTorchMaxAbs": err}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} (sha256 {sha[:12]}…), ONNX vs torch max |diff| {err:.2e}")


if __name__ == "__main__":
    main()
