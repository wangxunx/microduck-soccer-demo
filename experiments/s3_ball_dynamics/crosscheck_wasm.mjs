// Cross-check: run scene.xml in the game's own MuJoCo build (WASM, @mujoco/mujoco)
// to confirm the Python MuJoCo reference matches the engine the game ships.
//   node experiments/s3_ball_dynamics/crosscheck_wasm.mjs   (from repo root)
import { readFile, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const { default: loadMujoco } = await import(
  pathToFileURL(resolve(here, "../../game/node_modules/@mujoco/mujoco/mujoco.js")).href);
const mj = await loadMujoco();
const xml = await readFile(resolve(here, "scene.xml"), "utf8");
const ref = JSON.parse(await readFile(resolve(here, "results/mujoco.json"), "utf8"));
const model = mj.MjModel.from_xml_string(xml);
const bq = Number(model.jnt("ball_freejoint").qposadr), bv = Number(model.jnt("ball_freejoint").dofadr);
const sq = Number(model.jnt("striker_x").qposadr), sv = Number(model.jnt("striker_x").dofadr);
const out = { engine: `mujoco ${mj.mj_versionString()} (wasm)`, runs: {} };
for (const [name, sc] of Object.entries(ref.scenarios)) {
  const data = new mj.MjData(model);
  data.qpos.set([...sc.pos, 1, 0, 0, 0], bq);
  data.qvel.set([...sc.lin, ...sc.ang], bv);
  const [x0, v0] = sc.striker || [-5, 0];
  data.qpos[sq] = x0; data.qvel[sv] = v0;
  mj.mj_forward(model, data);
  const tr = { p: [], v: [] };
  for (let k = 0; k < Math.round(sc.T / ref.dt); k++) {
    mj.mj_step(model, data);
    tr.p.push(Array.from(data.qpos.slice(bq, bq + 3)));
    tr.v.push(Array.from(data.qvel.slice(bv, bv + 3)));
  }
  data.delete();
  // Max deviation from the Python reference.
  const r = ref.variants.mujoco_python.runs[name];
  let dp = 0, dv = 0;
  tr.p.forEach((p, k) => { dp = Math.max(dp, Math.hypot(...p.map((x, i) => x - r.p[k][i]))); });
  tr.v.forEach((v, k) => { dv = Math.max(dv, Math.hypot(...v.map((x, i) => x - r.v[k][i]))); });
  out.runs[name] = { max_pos_dev_m: dp, max_vel_dev_mps: dv };
  console.log(`${name.padEnd(10)} max |Δpos| ${dp.toExponential(2)} m   max |Δvel| ${dv.toExponential(2)} m/s`);
}
await writeFile(resolve(here, "results/crosscheck_wasm.json"), JSON.stringify(out, null, 1));
console.log("engine:", out.engine);
