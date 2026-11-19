# S6 — get-up / recovery in the game's MuJoCo

Tests a `stand` (get-up) policy through the soccer game's **real recovery path** (`PhysicsWorld`):
fallen for more than 0.28 s → hold 0.22 s → `stand` policy → back to walking (zero command) once
upright (projected gravity z < −0.88, trunk height > 0.08 m) for 0.6 s; forced reset after 7 s.
Only the start pose is scripted.

60 cases: face-down, face-up, lying on the left / right side and sitting, each at 4 headings; 24
random orientations dropped from 0.15 m (fixed seed); 16 upright ducks shoved at 1.2 m/s.

```bash
cd game
node ../experiments/s6_getup/run_getup.mjs ../experiments/s6_getup/results/official.json
STAND_CANDIDATE=../training/exports/stand-B-9999.onnx WALK_CANDIDATE=../training/exports/walk-A-5998.onnx \
  node ../experiments/s6_getup/run_getup.mjs ../experiments/s6_getup/results/stand-B-walkA.json
python3 ../experiments/s6_getup/summarize.py ../experiments/s6_getup/results/*.json
```

Per case: recovered (handed back to walking without a forced reset), time from the start of the
`stand` policy to the first upright frame, relapses (fell again after the hand-back), maximum tilt
in the 2 s after the hand-back. `training/evaluate_stand.sh` runs this together with the post-kick
handoff and full matches; results are in `training/MODELS.md` §2.4.

Finding: the walking policy keeps control during the first 0.28 s on the ground, so the pose the
`stand` policy starts from depends on the walk. The original `BEST_alpha_stand` recovers 59/59
after the game's default walk but needs 3 forced resets after walk-A; `stand-B-9999` (trained on
AMD with random-drop starts) recovers in every case after both.
