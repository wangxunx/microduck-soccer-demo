// S1: run a walking policy inside the soccer game's real MuJoCo runtime.
//
// Usage (cwd MUST be the game directory: assets resolve from ./public/assets):
//   cd game && node ../experiments/s1_sim2sim/run_mujoco.mjs <policyName> <policy.onnx> <out.json>
//
// The game's PhysicsWorld is used unmodified (BAM servo model, full-collision
// robot, 200 Hz physics / 50 Hz policy, automatic stand-policy fall recovery).
// Only runtime.policies.walk is swapped for the candidate.
import { readFile, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const game = resolve(process.cwd());
const imp = (p) => import(pathToFileURL(resolve(game, p)).href);
const { createRoom, addPlayer } = await imp("src/simulation.js");
const { loadPhysicsRuntime, PhysicsWorld } = await imp("src/physics/world.mjs");
const { projectedGravity } = await imp("src/physics/constants.js");

const [name, policyPath, outPath] = process.argv.slice(2);
if (!name || !policyPath || !outPath) {
  console.error("usage: run_mujoco.mjs <policyName> <policy.onnx> <out.json>");
  process.exit(2);
}
const { dt, scenarios } = JSON.parse(
  await readFile(resolve(here, process.env.S1_SCENARIO_FILE || "scenarios.json"), "utf8"),
);
const only = process.env.S1_SCENARIOS?.split(",");

const runtime = await loadPhysicsRuntime();
runtime.policies.walk = await runtime.ort.InferenceSession.create(
  resolve(policyPath),
  { executionProviders: ["cpu"], intraOpNumThreads: 1, interOpNumThreads: 1 },
);
// Huge field: no boards/goals near the robot. The ball is parked far away.
const FIELD = {
  halfLength: 40, halfWidth: 40, goalHalfWidth: 0.36, goalDepth: 0.24,
  goalHeight: 0.28, runoff: 0.34, ballRadius: 0.05,
};
const yawOf = (q) =>
  Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));

const out = { engine: "mujoco", engineVersion: runtime.engineVersion, policy: name,
  policyPath: resolve(policyPath), dt, runs: {} };
for (const [sname, sc] of Object.entries(scenarios)) {
  if (only && !only.includes(sname)) continue;
  const room = createRoom();
  room.ai = false;
  room.balls[0] = { id: 0, x: 30, z: 30, y: 0.05, vx: 0, vz: 0, cooldown: 0 };
  const p = addPlayer(room, "duck", "duck", "blue");
  p.x = 0; p.z = 0; p.yaw = 0;
  const world = new PhysicsWorld(runtime, room, FIELD);
  const slot = world.slots.get(p.id);
  const steps = Math.round(sc.duration / dt);
  const tr = { t: [], x: [], y: [], yaw: [], gz: [], h: [], vx_cmd: [], wz_cmd: [],
    recovering: [] };
  let seg = 0, falls = 0, wasRecovering = false;
  const t0 = performance.now();
  for (let k = 0; k < steps; k++) {
    const t = k * dt;
    while (seg + 1 < sc.segments.length && sc.segments[seg + 1][0] <= t + 1e-9) seg++;
    const [, vx, wz] = sc.segments[seg];
    // velocityCommand(forward, turn) = [forward*0.3, 0, turn*1.5] in walk mode.
    p.forward = vx / 0.3;
    p.turn = wz / 1.5;
    await world.step();
    const q = world.data.qpos, b = slot.baseQ;
    const quat = Array.from(q.slice(b + 3, b + 7));
    const rec = !!slot.recovery;
    if (rec && !wasRecovering) falls++;
    wasRecovering = rec;
    tr.t.push(+(t + dt).toFixed(4));
    tr.x.push(q[b]); tr.y.push(q[b + 1]); tr.yaw.push(yawOf(quat));
    tr.gz.push(projectedGravity(quat)[2]); tr.h.push(q[b + 2]);
    tr.vx_cmd.push(vx); tr.wz_cmd.push(wz); tr.recovering.push(rec ? 1 : 0);
  }
  world.dispose();
  out.runs[sname] = { falls, wallSeconds: (performance.now() - t0) / 1000, traj: tr };
  console.log(`[mujoco] ${name} ${sname}: falls=${falls} ` +
    `(${((performance.now() - t0) / 1000).toFixed(1)} s wall)`);
}
await writeFile(resolve(outPath), JSON.stringify(out));
console.log(`wrote ${resolve(outPath)}`);
