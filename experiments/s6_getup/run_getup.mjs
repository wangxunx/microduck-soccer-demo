// S6: get-up / recovery through the soccer game's real recovery path (PhysicsWorld).
//
//   cd game && node ../experiments/s6_getup/run_getup.mjs ../experiments/s6_getup/results/official.json
//   STAND_CANDIDATE=../training/exports/stand-A-9999.onnx node ../experiments/s6_getup/run_getup.mjs out.json
//   (WALK_CANDIDATE=... also replaces the walking policy that takes over after the recovery.)
//
// Each case puts one duck in a pose (lying, sitting, random drop, or standing with a shove)
// and lets the game do what it does in a match: fallen > 0.28 s -> hold 0.22 s -> `stand`
// policy -> back to walking (zero command) once upright (gravity z < -0.88, h > 0.08) for
// 0.6 s; forced reset after 7 s of failed recovery. Nothing is scripted except the start.
import { writeFile, mkdir } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { pathToFileURL } from "node:url";

const game = resolve(process.cwd());
const imp = (p) => import(pathToFileURL(resolve(game, p)).href);
const { createRoom, addPlayer } = await imp("src/simulation.js");
const { loadPhysicsRuntime, PhysicsWorld } = await imp("src/physics/world.mjs");
const { projectedGravity, CONTROL_DT, JOINT_NAMES, DEFAULT_POSE } = await imp("src/physics/constants.js");

const outPath = resolve(process.argv[2] || "getup.json");
const DURATION = Number(process.env.GETUP_DURATION || 10);
const runtime = await loadPhysicsRuntime();
const session = (f) => runtime.ort.InferenceSession.create(resolve(f),
  { executionProviders: ["cpu"], intraOpNumThreads: 1, interOpNumThreads: 1 });
if (process.env.STAND_CANDIDATE) runtime.policies.stand = await session(process.env.STAND_CANDIDATE);
if (process.env.WALK_CANDIDATE) runtime.policies.walk = await session(process.env.WALK_CANDIDATE);
const FIELD = { halfLength: 40, halfWidth: 40, goalHalfWidth: 0.36, goalDepth: 0.24,
  goalHeight: 0.28, runoff: 0.34, ballRadius: 0.05 };

// ---- cases ---------------------------------------------------------------------------
let seed = 12345;
const rand = () => { // mulberry32, deterministic across runs
  seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const mul = (a, b) => [
  a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
const axis = (x, y, z, ang) => [Math.cos(ang / 2), x * Math.sin(ang / 2), y * Math.sin(ang / 2), z * Math.sin(ang / 2)];
const S = Math.SQRT1_2;
const SIT = { left_hip_roll: 0, left_hip_pitch: -0.4079, left_knee: 1.35, left_ankle: 0,
  right_hip_roll: 0, right_hip_pitch: 0.4079, right_knee: -1.35, right_ankle: 0 };
const yaws = [0, Math.PI / 2, Math.PI, -Math.PI / 2];
const cases = [];
for (const [i, yaw] of yaws.entries()) {
  const qy = axis(0, 0, 1, yaw);
  cases.push({ id: `face_down_${i}`, family: "face_down", quat: mul(qy, [S, 0, S, 0]), z: 0.09 });
  cases.push({ id: `face_up_${i}`, family: "face_up", quat: mul(qy, [S, 0, -S, 0]), z: 0.09 });
  cases.push({ id: `side_left_${i}`, family: "side", quat: mul(qy, [S, S, 0, 0]), z: 0.09 });
  cases.push({ id: `side_right_${i}`, family: "side", quat: mul(qy, [S, -S, 0, 0]), z: 0.09 });
  cases.push({ id: `sitting_${i}`, family: "sitting", quat: qy, z: 0.075, joints: SIT });
  for (const [k, [vx, vy]] of [[1.2, 0], [-1.2, 0], [0, 1.2], [0, -1.2]].entries())
    cases.push({ id: `shove_${i}_${k}`, family: "shove", quat: qy, z: 0.125, vel: [vx, vy] });
}
for (let i = 0; i < 24; i++) { // uniform random orientation (Shoemake), dropped
  const [u1, u2, u3] = [rand(), rand(), rand()];
  const q = [Math.sqrt(u1) * Math.cos(2 * Math.PI * u3), Math.sqrt(1 - u1) * Math.sin(2 * Math.PI * u2),
    Math.sqrt(1 - u1) * Math.cos(2 * Math.PI * u2), Math.sqrt(u1) * Math.sin(2 * Math.PI * u3)];
  cases.push({ id: `drop_${i}`, family: "drop", quat: q, z: 0.15 });
}
const select = process.env.GETUP_CASE;

// ---- run ----------------------------------------------------------------------------
const tiltDeg = (gz) => (Math.acos(Math.max(-1, Math.min(1, -gz))) * 180) / Math.PI;
const out = { engine: "mujoco(game)", engineVersion: runtime.engineVersion, duration: DURATION,
  stand: process.env.STAND_CANDIDATE || "BEST_alpha_stand.onnx",
  walk: process.env.WALK_CANDIDATE || "game default", runs: {} };
for (const c of cases) {
  if (select && c.id !== select) continue;
  const room = createRoom();
  room.ai = false;
  room.balls[0] = { id: 0, x: 5, z: 5, y: 0.05, vx: 0, vz: 0, cooldown: 0 };
  const p = addPlayer(room, "duck", "duck", "blue");
  Object.assign(p, { x: 0, z: 0, yaw: 0, auto: false, forward: 0, turn: 0 });
  const world = new PhysicsWorld(runtime, room, FIELD);
  const slot = world.slots.get(p.id);
  let forcedResets = 0;
  const reset = world.reset.bind(world);
  world.reset = (...a) => { forcedResets++; return reset(...a); };
  const q = world.data.qpos, v = world.data.qvel, b = slot.baseQ;
  q.set([0, 0, c.z, ...c.quat], b);
  v.fill(0, slot.baseV, slot.baseV + 6);
  if (c.vel) { v[slot.baseV] = c.vel[0]; v[slot.baseV + 1] = c.vel[1]; }
  slot.qpos.forEach((a, j) => (q[a] = c.joints?.[JOINT_NAMES[j]] ?? DEFAULT_POSE[j]));
  slot.dof.forEach((a) => (v[a] = 0));
  slot.targets.set(slot.qpos.map((a) => q[a]));
  runtime.mj.mj_forward(world.model, world.data);

  const gz = [], h = [], phase = [];
  let entries = 0, standStart = null, firstExit = null, firstUpright = null, prev = null;
  for (let k = 0; k < Math.round(DURATION / CONTROL_DT); k++) {
    await world.step();
    const quat = Array.from(q.slice(b + 3, b + 7));
    const r = slot.recovery;
    if (r && !prev) entries++;
    if (r === "stand" && standStart === null) standStart = (k + 1) * CONTROL_DT;
    if (!r && prev === "stand" && firstExit === null) firstExit = (k + 1) * CONTROL_DT;
    prev = r;
    const g = projectedGravity(quat)[2];
    if (standStart !== null && firstUpright === null && g < -0.88 && q[b + 2] > 0.08) firstUpright = (k + 1) * CONTROL_DT;
    gz.push(projectedGravity(quat)[2]);
    h.push(q[b + 2]);
    phase.push(r || "walk");
  }
  world.dispose();
  const after = firstExit === null ? [] : gz.slice(Math.round(firstExit / CONTROL_DT), Math.round((firstExit + 2) / CONTROL_DT));
  const res = {
    ...c, forcedResets, recoveryEntries: entries,
    // the stand policy started at standStart (null: never needed one), handed back at firstExit
    standStart, firstExit,
    standToUpright: firstUpright !== null ? firstUpright - standStart : null,
    standToExit: firstExit !== null && standStart !== null ? firstExit - standStart : null,
    recovered: firstExit !== null && forcedResets === 0,
    maxTiltAfterExit: after.length ? Math.max(...after.map(tiltDeg)) : null,
    uprightAtEnd: gz.at(-1) < -0.88 && h.at(-1) > 0.08,
    traj: { gz, h, phase },
  };
  out.runs[c.id] = res;
  console.log(`${c.id.padEnd(16)} entries=${entries} resets=${forcedResets} stand->upright=${
    res.standToUpright?.toFixed(2) ?? "-"} exit=${res.standToExit?.toFixed(2) ?? "-"} tiltAfter=${res.maxTiltAfterExit?.toFixed(1) ?? "-"} uprightEnd=${res.uprightAtEnd}`);
}
await mkdir(dirname(outPath), { recursive: true });
await writeFile(outPath, JSON.stringify(out));
console.log("wrote", outPath);
