# S4 kick sim2sim: MuJoCo(game) vs Genesis

## Per protocol summary (18 cases each = 2 feet x 9 ball positions)

| protocol | engine | falls | hits | median peak ball speed (m/s) | median dir (deg) | median dist 4 s (m) | max tilt (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| C_football | mujoco(game) | 0 | 18/18 | 2.27 | -6.65 | 4.42 | 41.9 |
| C_football | genesis_native | 0 | 18/18 | 2.12 | -5.85 | 6.55 | 40.3 |
| C_football | genesis_rollfric | 0 | 18/18 | 2.10 | -4.26 | 3.95 | 46.3 |

## Football handoff: max tilt in the window / within 0.6 s after handing over to `stand` (deg)

| engine | window median | window max | after-handoff median | after-handoff p90 | after-handoff max | cases > 30 deg |
|---|---:|---:|---:|---:|---:|---:|
| mujoco(game) | 36.7 | 41.9 | 6.9 | 14.6 | 18.2 | 0/18 |
| genesis_native | 38.2 | 40.3 | 9.1 | 15.5 | 19.1 | 0/18 |
| genesis_rollfric | 38.4 | 46.3 | 7.6 | 11.0 | 14.7 | 0/18 |

## Paired comparison vs MuJoCo(game)

| protocol | engine | hit/miss agreement | peak-speed ratio gen/mj (median [min, max], both hit) | |Δdir| deg (median / max) | Δt first move s (median) |
|---|---|---:|---|---|---:|
| C_football | genesis_native | 18/18 | 0.91 [0.78, 1.15] | 5.3 / 13.3 | -0.03 |
| C_football | genesis_rollfric | 18/18 | 0.90 [0.77, 1.15] | 5.0 / 14.5 | -0.03 |

## Per case (peak ball speed m/s @ dir deg; F = fall; · = ball not hit)

| case | mujoco(game) | genesis_native | genesis_rollfric |
|---|---|---|---|
| C_football|left|-0.015|-0.015 | 2.44 @ -7 | 2.24 @ -2 | 2.19 @ +0 |
| C_football|left|-0.015|+0.000 | 2.32 @ -3 | 2.17 @ -3 | 2.11 @ +1 |
| C_football|left|-0.015|+0.015 | 2.27 @ -3 | 1.99 @ -7 | 2.01 @ -6 |
| C_football|left|+0.000|-0.015 | 2.29 @ -19 | 1.99 @ -6 | 2.05 @ -5 |
| C_football|left|+0.000|+0.000 | 2.22 @ -12 | 2.15 @ -7 | 2.19 @ -4 |
| C_football|left|+0.000|+0.015 | 2.28 @ -10 | 2.04 @ -4 | 2.03 @ -3 |
| C_football|left|+0.015|-0.015 | 2.39 @ -10 | 2.16 @ -4 | 2.15 @ -3 |
| C_football|left|+0.015|+0.000 | 2.43 @ -10 | 2.10 @ -9 | 2.10 @ -8 |
| C_football|left|+0.015|+0.015 | 2.27 @ -13 | 2.09 @ -6 | 2.08 @ -6 |
| C_football|right|-0.015|-0.015 | 2.02 @ -2 | 2.32 @ -10 | 2.32 @ -8 |
| C_football|right|-0.015|+0.000 | 2.15 @ -8 | 2.26 @ -7 | 2.26 @ -5 |
| C_football|right|-0.015|+0.015 | 2.26 @ -9 | 1.77 @ -2 | 1.73 @ +3 |
| C_football|right|+0.000|-0.015 | 2.12 @ +2 | 2.20 @ -6 | 2.21 @ -5 |
| C_football|right|+0.000|+0.000 | 2.19 @ -4 | 2.33 @ -7 | 2.30 @ -6 |
| C_football|right|+0.000|+0.015 | 2.36 @ -5 | 1.99 @ -4 | 2.00 @ -3 |
| C_football|right|+0.015|-0.015 | 2.22 @ +1 | 2.36 @ -5 | 2.35 @ -4 |
| C_football|right|+0.015|+0.000 | 2.31 @ -6 | 1.93 @ -6 | 1.93 @ -5 |
| C_football|right|+0.015|+0.015 | 2.48 @ -2 | 1.98 @ -5 | 1.98 @ -3 |
