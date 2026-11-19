// S4: kick policies through the soccer game's real kick path (PhysicsWorld).
//   cd game && node ../experiments/s4_kick_sim2sim/run_mujoco.mjs ../experiments/s4_kick_sim2sim/results/mujoco.json
import { readFile, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const game = resolve(process.cwd());
const imp = (p) => import(pathToFileURL(resolve(game, p)).href);
const { createRoom, addPlayer } = await imp("src/simulation.js");
const { loadPhysicsRuntime, PhysicsWorld } = await imp("src/physics/world.mjs");
const { projectedGravity, CONTROL_DT } = await imp("src/physics/constants.js");

const outPath = resolve(process.argv[2] || resolve(here, "results/mujoco.json"));
const { duration, cases } = JSON.parse(await readFile(resolve(here, "cases.json"), "utf8"));
const runtime = await loadPhysicsRuntime();
// Optional candidate football policy (+ its manifest next to it), like the game's evaluate-football.mjs.
if (process.env.FOOTBALL_CANDIDATE) {
  const onnx = resolve(process.env.FOOTBALL_CANDIDATE);
  runtime.policies.football = await runtime.ort.InferenceSession.create(onnx,
    { executionProviders: ["cpu"], intraOpNumThreads: 1, interOpNumThreads: 1 });
  const manifest = JSON.parse(await readFile(onnx.replace(/\.onnx$/, ".json"), "utf8"));
  runtime.footballInfo = { ...manifest, id: manifest.id || onnx };
  console.log("candidate football:", runtime.footballInfo.id);
}
for (const [env, key] of [["STAND_CANDIDATE", "stand"], ["WALK_CANDIDATE", "walk"]])
  if (process.env[env]) {
    runtime.policies[key] = await runtime.ort.InferenceSession.create(resolve(process.env[env]),
      { executionProviders: ["cpu"], intraOpNumThreads: 1, interOpNumThreads: 1 });
    console.log(`candidate ${key}:`, process.env[env]);
  }
const protocols = process.env.S4_PROTOCOLS?.split(",");
const orig = { football: runtime.policies.football, info: runtime.footballInfo };
const FIELD = { halfLength: 40, halfWidth: 40, goalHalfWidth: 0.36, goalDepth: 0.24,
  goalHeight: 0.28, runoff: 0.34, ballRadius: 0.05 };

function configure(protocol) {
  if (protocol === "C_football") {
    runtime.policies.football = orig.football;
    runtime.footballInfo = orig.info;
  } else {
    runtime.policies.football = null;
    runtime.footballInfo = protocol === "B_full_kick"
      ? { windowSeconds: 3.0, currentLimitAmps: null } : null; // null -> game default 0.5 s
  }
}

const out = { engine: "mujoco(game)", engineVersion: runtime.engineVersion, duration,
  football: runtime.footballInfo?.id || null, runs: {} };
for (const c of cases) {
  if (protocols && !protocols.includes(c.protocol)) continue;
  configure(c.protocol);
  const room = createRoom();
  room.ai = false;
  room.balls[0] = { id: 0, x: c.bx, z: -c.by, y: 0.05, vx: 0, vz: 0, cooldown: 0 };
  const p = addPlayer(room, "duck", "duck", "blue");
  p.x = 0; p.z = 0; p.yaw = 0;
  const world = new PhysicsWorld(runtime, room, FIELD);
  const slot = world.slots.get(p.id);
  const ballGeom = runtime.mj.mj_name2id(world.model, runtime.mj.mjtObj.mjOBJ_GEOM.value, "ball_geom");
  p.kickRequested = true; p.kickFoot = c.foot; p.kickSource = "manual";
  const tr = { ball: [], bvel: [], base: [], gz: [], policy: [], contact: [] };
  let falls = 0, wasRec = false;
  for (let k = 0; k < Math.round(duration / CONTROL_DT); k++) {
    // Policy that will run THIS control step (mirrors world-core control()).
    const pol = slot.recovery ? "stand" : slot.footballHandoff > 0 ? "stand"
      : slot.kickTime > 0 || p.kickRequested
        ? (c.protocol === "C_football" ? "football" : `kick_${c.foot}`) : "walk";
    await world.step();
    const q = world.data.qpos, b = slot.baseQ;
    const ball = world.readBall();
    const quat = Array.from(q.slice(b + 3, b + 7));
    let contact = 0;
    for (let i = 0; i < world.data.ncon; i++) {
      const con = world.data.contact.get ? world.data.contact.get(i) : world.data.contact[i];
      const g1 = con.geom1, g2 = con.geom2;
      if ((g1 === ballGeom && world.geomPlayers.has(g2)) || (g2 === ballGeom && world.geomPlayers.has(g1))) contact = 1;
    }
    const rec = !!slot.recovery;
    if (rec && !wasRec) falls++;
    wasRec = rec;
    tr.ball.push([ball.x, -ball.z, ball.y]);
    tr.bvel.push([ball.vx, -ball.vz, ball.vy]);
    tr.base.push([q[b], q[b + 1], q[b + 2]]);
    tr.gz.push(projectedGravity(quat)[2]);
    tr.policy.push(pol);
    tr.contact.push(contact);
  }
  world.dispose();
  out.runs[c.id] = { ...c, falls, traj: tr };
  const d = tr.ball.at(-1), s = Math.hypot(d[0] - c.bx, d[1] - c.by);
  console.log(`${c.id.padEnd(34)} falls=${falls} ball moved ${(100 * s).toFixed(1)} cm`);
}
await writeFile(outPath, JSON.stringify(out));
console.log("wrote", outPath);
