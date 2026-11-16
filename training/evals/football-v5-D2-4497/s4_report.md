# S4 kick sim2sim: MuJoCo(game) vs Genesis

## Per protocol summary (18 cases each = 2 feet x 9 ball positions)

| protocol | engine | falls | hits | median peak ball speed (m/s) | median dir (deg) | median dist 4 s (m) | max tilt (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| C_football | mujoco(game) | 0 | 18/18 | 1.76 | 5.68 | 3.87 | 26.8 |
| C_football | genesis_native | 0 | 18/18 | 1.66 | 0.38 | 6.17 | 25.0 |
| C_football | genesis_rollfric | 0 | 18/18 | 1.68 | -1.20 | 3.47 | 25.0 |

## Football handoff: max tilt in the window / within 0.6 s after handing over to `stand` (deg)

| engine | window median | window max | after-handoff median | after-handoff p90 | after-handoff max | cases > 30 deg |
|---|---:|---:|---:|---:|---:|---:|
| mujoco(game) | 19.9 | 26.8 | 12.6 | 12.9 | 13.1 | 0/18 |
| genesis_native | 18.3 | 25.0 | 13.4 | 13.5 | 13.6 | 0/18 |
| genesis_rollfric | 17.6 | 25.0 | 13.3 | 13.4 | 13.6 | 0/18 |

## Paired comparison vs MuJoCo(game)

| protocol | engine | hit/miss agreement | peak-speed ratio gen/mj (median [min, max], both hit) | |Δdir| deg (median / max) | Δt first move s (median) |
|---|---|---:|---|---|---:|
| C_football | genesis_native | 18/18 | 0.93 [0.84, 1.12] | 3.4 / 16.1 | +0.00 |
| C_football | genesis_rollfric | 18/18 | 0.94 [0.85, 1.12] | 6.4 / 13.8 | +0.00 |

## Per case (peak ball speed m/s @ dir deg; F = fall; · = ball not hit)

| case | mujoco(game) | genesis_native | genesis_rollfric |
|---|---|---|---|
| C_football|left|-0.015|-0.015 | 1.56 @ +8 | 1.53 @ +0 | 1.53 @ -3 |
| C_football|left|-0.015|+0.000 | 1.70 @ +5 | 1.51 @ +1 | 1.50 @ -1 |
| C_football|left|-0.015|+0.015 | 1.72 @ +4 | 1.47 @ +0 | 1.47 @ -3 |
| C_football|left|+0.000|-0.015 | 1.62 @ +9 | 1.50 @ +2 | 1.51 @ -1 |
| C_football|left|+0.000|+0.000 | 1.76 @ +6 | 1.63 @ +3 | 1.64 @ +0 |
| C_football|left|+0.000|+0.015 | 1.78 @ +5 | 1.54 @ +2 | 1.56 @ -2 |
| C_football|left|+0.015|-0.015 | 1.76 @ +11 | 1.59 @ +4 | 1.59 @ +2 |
| C_football|left|+0.015|+0.000 | 1.83 @ +8 | 1.68 @ +5 | 1.71 @ +2 |
| C_football|left|+0.015|+0.015 | 1.86 @ +8 | 1.78 @ +3 | 1.77 @ +1 |
| C_football|right|-0.015|-0.015 | 1.73 @ -7 | 1.74 @ -8 | 1.76 @ -6 |
| C_football|right|-0.015|+0.000 | 1.65 @ -5 | 1.60 @ -5 | 1.61 @ -3 |
| C_football|right|-0.015|+0.015 | 1.78 @ +13 | 1.49 @ -3 | 1.51 @ -1 |
| C_football|right|+0.000|-0.015 | 2.00 @ -4 | 1.78 @ -7 | 1.80 @ -4 |
| C_football|right|+0.000|+0.000 | 1.65 @ -3 | 1.86 @ -2 | 1.85 @ -2 |
| C_football|right|+0.000|+0.015 | 1.68 @ +12 | 1.70 @ -3 | 1.71 @ -1 |
| C_football|right|+0.015|-0.015 | 2.09 @ +7 | 1.95 @ -3 | 1.97 @ -1 |
| C_football|right|+0.015|+0.000 | 1.96 @ -0 | 1.95 @ -0 | 1.98 @ +1 |
| C_football|right|+0.015|+0.015 | 1.96 @ +3 | 1.98 @ +3 | 1.98 @ +2 |
