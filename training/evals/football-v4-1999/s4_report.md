# S4 kick sim2sim: MuJoCo(game) vs Genesis

## Per protocol summary (18 cases each = 2 feet x 9 ball positions)

| protocol | engine | falls | hits | median peak ball speed (m/s) | median dir (deg) | median dist 4 s (m) | max tilt (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| C_football | mujoco(game) | 0 | 18/18 | 2.49 | -1.53 | 5.01 | 39.4 |
| C_football | genesis_native | 0 | 18/18 | 2.46 | -2.04 | 7.37 | 41.0 |
| C_football | genesis_rollfric | 0 | 18/18 | 2.47 | -1.39 | 4.75 | 40.7 |

## Football handoff: max tilt in the window / within 0.6 s after handing over to `stand` (deg)

| engine | window median | window max | after-handoff median | after-handoff p90 | after-handoff max | cases > 30 deg |
|---|---:|---:|---:|---:|---:|---:|
| mujoco(game) | 36.1 | 39.4 | 11.7 | 12.8 | 15.7 | 0/18 |
| genesis_native | 36.0 | 41.0 | 10.7 | 12.0 | 17.3 | 0/18 |
| genesis_rollfric | 35.8 | 40.7 | 11.3 | 12.5 | 24.4 | 0/18 |

## Paired comparison vs MuJoCo(game)

| protocol | engine | hit/miss agreement | peak-speed ratio gen/mj (median [min, max], both hit) | |Δdir| deg (median / max) | Δt first move s (median) |
|---|---|---:|---|---|---:|
| C_football | genesis_native | 18/18 | 0.99 [0.88, 1.05] | 2.3 / 5.8 | -0.02 |
| C_football | genesis_rollfric | 18/18 | 0.99 [0.88, 1.05] | 1.8 / 6.2 | -0.02 |

## Per case (peak ball speed m/s @ dir deg; F = fall; · = ball not hit)

| case | mujoco(game) | genesis_native | genesis_rollfric |
|---|---|---|---|
| C_football|left|-0.015|-0.015 | 2.46 @ +1 | 2.58 @ -2 | 2.58 @ -1 |
| C_football|left|-0.015|+0.000 | 2.49 @ -1 | 2.28 @ -2 | 2.27 @ -1 |
| C_football|left|-0.015|+0.015 | 2.51 @ +1 | 2.21 @ -5 | 2.20 @ -4 |
| C_football|left|+0.000|-0.015 | 2.55 @ +1 | 2.34 @ -3 | 2.35 @ -2 |
| C_football|left|+0.000|+0.000 | 2.57 @ -2 | 2.39 @ -3 | 2.36 @ -2 |
| C_football|left|+0.000|+0.015 | 2.54 @ -1 | 2.50 @ -3 | 2.51 @ -2 |
| C_football|left|+0.015|-0.015 | 2.55 @ -1 | 2.53 @ -2 | 2.55 @ -1 |
| C_football|left|+0.015|+0.000 | 2.60 @ -1 | 2.51 @ -3 | 2.51 @ -3 |
| C_football|left|+0.015|+0.015 | 2.47 @ -2 | 2.55 @ -3 | 2.57 @ -2 |
| C_football|right|-0.015|-0.015 | 2.49 @ -7 | 2.49 @ -1 | 2.50 @ -0 |
| C_football|right|-0.015|+0.000 | 2.39 @ -7 | 2.45 @ -2 | 2.46 @ -1 |
| C_football|right|-0.015|+0.015 | 2.40 @ -2 | 2.36 @ -3 | 2.37 @ -2 |
| C_football|right|+0.000|-0.015 | 2.44 @ -7 | 2.50 @ -1 | 2.50 @ -1 |
| C_football|right|+0.000|+0.000 | 2.49 @ -5 | 2.35 @ +1 | 2.35 @ +1 |
| C_football|right|+0.000|+0.015 | 2.45 @ -1 | 2.46 @ -1 | 2.46 @ -0 |
| C_football|right|+0.015|-0.015 | 2.47 @ -7 | 2.46 @ -3 | 2.48 @ -2 |
| C_football|right|+0.015|+0.000 | 2.48 @ -3 | 2.48 @ -2 | 2.49 @ -1 |
| C_football|right|+0.015|+0.015 | 2.56 @ -1 | 2.44 @ +1 | 2.44 @ +1 |
