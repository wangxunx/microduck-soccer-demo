"""Load the game's exported MLP policies (Sub/Div normaliser -> Gemm/Elu x3 -> Gemm)
as batched torch modules, so the official `stand` / `walk` ONNX can run for
thousands of envs on the GPU (the files themselves have a fixed batch of 1).

Strict: any graph that is not exactly that pattern is rejected rather than
approximated. `verify()` compares against onnxruntime.
"""
from __future__ import annotations

import numpy as np
import onnx
import torch
from onnx import numpy_helper


class OnnxMlp(torch.nn.Module):
    def __init__(self, path: str):
        super().__init__()
        m = onnx.load(path)
        init = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in m.graph.initializer}
        nodes = list(m.graph.node)
        ops = [n.op_type for n in nodes]
        if ops != ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]:
            raise ValueError(f"{path}: unsupported graph {ops}")
        self.register_buffer("mean", init[nodes[0].input[1]].float())
        self.register_buffer("div", init[nodes[1].input[1]].float())
        layers = []
        for n in nodes[2:]:
            if n.op_type == "Gemm":
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
                if attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0 or attrs.get("transA", 0):
                    raise ValueError(f"{path}: unsupported Gemm attrs {attrs}")
                w, b = init[n.input[1]].float(), init[n.input[2]].float()
                if not attrs.get("transB", 0):
                    w = w.t()
                lin = torch.nn.Linear(w.shape[1], w.shape[0])
                lin.weight.data.copy_(w)
                lin.bias.data.copy_(b)
                layers.append(lin)
            else:
                alpha = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}.get("alpha", 1.0)
                layers.append(torch.nn.ELU(alpha=alpha))
        self.mlp = torch.nn.Sequential(*layers)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / self.div)


def verify(path: str, n: int = 256, seed: int = 0) -> float:
    import onnxruntime as ort
    torch_mod = OnnxMlp(path).cpu()
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, 61)).astype(np.float32)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: obs[i:i + 1]})[0] for i in range(n)])
    got = torch_mod(torch.from_numpy(obs).cpu()).numpy()
    return float(np.abs(ref - got).max())
