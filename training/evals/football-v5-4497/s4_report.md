# S4 kick sim2sim: MuJoCo(game) vs Genesis

## Per protocol summary (18 cases each = 2 feet x 9 ball positions)

| protocol | engine | falls | hits | median peak ball speed (m/s) | median dir (deg) | median dist 4 s (m) | max tilt (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| C_football | mujoco(game) | 0 | 18/18 | 1.83 | 5.20 | 4.10 | 28.3 |
| C_football | genesis_native | 0 | 18/18 | 1.69 | 2.21 | 6.26 | 22.6 |
| C_football | genesis_rollfric | 0 | 18/18 | 1.69 | 0.77 | 3.54 | 22.6 |

## Football handoff: max tilt in the window / within 0.6 s after handing over to `stand` (deg)

| engine | window median | window max | after-handoff median | after-handoff p90 | after-handoff max | cases > 30 deg |
|---|---:|---:|---:|---:|---:|---:|
| mujoco(game) | 18.7 | 28.3 | 12.1 | 12.3 | 12.6 | 0/18 |
| genesis_native | 18.3 | 22.6 | 10.4 | 11.5 | 11.9 | 0/18 |
| genesis_rollfric | 17.8 | 22.6 | 10.5 | 11.6 | 12.9 | 0/18 |

## Paired comparison vs MuJoCo(game)

| protocol | engine | hit/miss agreement | peak-speed ratio gen/mj (median [min, max], both hit) | |Δdir| deg (median / max) | Δt first move s (median) |
|---|---|---:|---|---|---:|
| C_football | genesis_native | 18/18 | 0.90 [0.80, 1.01] | 2.9 / 7.7 | +0.00 |
| C_football | genesis_rollfric | 18/18 | 0.91 [0.82, 1.00] | 3.3 / 10.7 | +0.00 |

## Per case (peak ball speed m/s @ dir deg; F = fall; · = ball not hit)

| case | mujoco(game) | genesis_native | genesis_rollfric |
|---|---|---|---|
| C_football|left|-0.015|-0.015 | 1.65 @ +8 | 1.45 @ +6 | 1.46 @ +4 |
| C_football|left|-0.015|+0.000 | 1.72 @ +5 | 1.49 @ +8 | 1.49 @ +5 |
| C_football|left|-0.015|+0.015 | 1.79 @ +8 | 1.52 @ +0 | 1.52 @ -2 |
| C_football|left|+0.000|-0.015 | 1.67 @ +5 | 1.57 @ +6 | 1.60 @ +3 |
| C_football|left|+0.000|+0.000 | 1.81 @ +8 | 1.57 @ +4 | 1.61 @ +1 |
| C_football|left|+0.000|+0.015 | 1.83 @ +10 | 1.63 @ +2 | 1.62 @ -1 |
| C_football|left|+0.015|-0.015 | 1.95 @ +6 | 1.56 @ +6 | 1.60 @ +2 |
| C_football|left|+0.015|+0.000 | 1.81 @ +5 | 1.60 @ +5 | 1.64 @ +2 |
| C_football|left|+0.015|+0.015 | 1.84 @ +6 | 1.81 @ +3 | 1.81 @ +0 |
| C_football|right|-0.015|-0.015 | 1.87 @ +1 | 1.70 @ -3 | 1.72 @ -1 |
| C_football|right|-0.015|+0.000 | 1.86 @ -3 | 1.76 @ -2 | 1.76 @ -1 |
| C_football|right|-0.015|+0.015 | 1.68 @ -5 | 1.68 @ +3 | 1.67 @ +3 |
| C_football|right|+0.000|-0.015 | 2.03 @ +5 | 1.85 @ +1 | 1.86 @ +4 |
| C_football|right|+0.000|+0.000 | 1.98 @ -0 | 1.73 @ -0 | 1.72 @ -1 |
| C_football|right|+0.000|+0.015 | 1.73 @ +0 | 1.76 @ +2 | 1.74 @ +1 |
| C_football|right|+0.015|-0.015 | 1.97 @ +7 | 1.87 @ +1 | 1.86 @ +3 |
| C_football|right|+0.015|+0.000 | 1.95 @ +1 | 1.80 @ -0 | 1.79 @ -1 |
| C_football|right|+0.015|+0.015 | 2.01 @ +6 | 1.81 @ +2 | 1.82 @ +0 |
