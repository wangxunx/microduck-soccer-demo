# S4 kick sim2sim: MuJoCo(game) vs Genesis

## Per protocol summary (18 cases each = 2 feet x 9 ball positions)

| protocol | engine | falls | hits | median peak ball speed (m/s) | median dir (deg) | median dist 4 s (m) | max tilt (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| C_football | mujoco(game) | 0 | 18/18 | 2.51 | -3.35 | 5.17 | 43.2 |
| C_football | genesis_native | 0 | 18/18 | 2.40 | -1.53 | 7.42 | 38.8 |
| C_football | genesis_rollfric | 0 | 18/18 | 2.40 | -0.88 | 4.87 | 38.3 |

## Football handoff: max tilt in the window / within 0.6 s after handing over to `stand` (deg)

| engine | window median | window max | after-handoff median | after-handoff p90 | after-handoff max | cases > 30 deg |
|---|---:|---:|---:|---:|---:|---:|
| mujoco(game) | 39.6 | 43.2 | 30.9 | 32.1 | 33.2 | 15/18 |
| genesis_native | 36.2 | 38.8 | 31.7 | 36.0 | 38.0 | 14/18 |
| genesis_rollfric | 36.8 | 38.3 | 34.9 | 36.7 | 37.8 | 16/18 |

## Paired comparison vs MuJoCo(game)

| protocol | engine | hit/miss agreement | peak-speed ratio gen/mj (median [min, max], both hit) | |Δdir| deg (median / max) | Δt first move s (median) |
|---|---|---:|---|---|---:|
| C_football | genesis_native | 18/18 | 0.95 [0.88, 1.02] | 1.6 / 6.2 | -0.02 |
| C_football | genesis_rollfric | 18/18 | 0.95 [0.88, 1.02] | 2.0 / 7.2 | -0.02 |

## Per case (peak ball speed m/s @ dir deg; F = fall; · = ball not hit)

| case | mujoco(game) | genesis_native | genesis_rollfric |
|---|---|---|---|
| C_football|left|-0.015|-0.015 | 2.48 @ -1 | 2.18 @ -0 | 2.18 @ +1 |
| C_football|left|-0.015|+0.000 | 2.48 @ -1 | 2.19 @ -2 | 2.19 @ -1 |
| C_football|left|-0.015|+0.015 | 2.48 @ -4 | 2.42 @ +2 | 2.41 @ +3 |
| C_football|left|+0.000|-0.015 | 2.47 @ -2 | 2.27 @ +0 | 2.26 @ +1 |
| C_football|left|+0.000|+0.000 | 2.48 @ -4 | 2.31 @ -1 | 2.31 @ -0 |
| C_football|left|+0.000|+0.015 | 2.50 @ -3 | 2.32 @ -0 | 2.32 @ +1 |
| C_football|left|+0.015|-0.015 | 2.50 @ -3 | 2.43 @ -5 | 2.43 @ -5 |
| C_football|left|+0.015|+0.000 | 2.51 @ -3 | 2.38 @ -0 | 2.37 @ +0 |
| C_football|left|+0.015|+0.015 | 2.43 @ -4 | 2.21 @ -3 | 2.21 @ -3 |
| C_football|right|-0.015|-0.015 | 2.65 @ -7 | 2.57 @ -6 | 2.56 @ -5 |
| C_football|right|-0.015|+0.000 | 2.61 @ -5 | 2.61 @ -4 | 2.60 @ -3 |
| C_football|right|-0.015|+0.015 | 2.50 @ -4 | 2.32 @ +1 | 2.33 @ +3 |
| C_football|right|+0.000|-0.015 | 2.65 @ -3 | 2.58 @ -5 | 2.57 @ -5 |
| C_football|right|+0.000|+0.000 | 2.59 @ -2 | 2.52 @ -4 | 2.49 @ -3 |
| C_football|right|+0.000|+0.015 | 2.61 @ -1 | 2.35 @ -1 | 2.39 @ +0 |
| C_football|right|+0.015|-0.015 | 2.64 @ -3 | 2.52 @ -3 | 2.52 @ -3 |
| C_football|right|+0.015|+0.000 | 2.58 @ -4 | 2.54 @ -1 | 2.54 @ -1 |
| C_football|right|+0.015|+0.015 | 2.52 @ -0 | 2.57 @ -2 | 2.57 @ -2 |
