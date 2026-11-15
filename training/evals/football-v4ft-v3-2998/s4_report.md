# S4 kick sim2sim: MuJoCo(game) vs Genesis

## Per protocol summary (18 cases each = 2 feet x 9 ball positions)

| protocol | engine | falls | hits | median peak ball speed (m/s) | median dir (deg) | median dist 4 s (m) | max tilt (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| C_football | mujoco(game) | 0 | 18/18 | 1.90 | 5.12 | 4.24 | 56.2 |
| C_football | genesis_native | 0 | 18/18 | 1.80 | 2.25 | 6.67 | 26.4 |
| C_football | genesis_rollfric | 0 | 18/18 | 1.82 | 1.18 | 3.89 | 26.4 |

## Football handoff: max tilt in the window / within 0.6 s after handing over to `stand` (deg)

| engine | window median | window max | after-handoff median | after-handoff p90 | after-handoff max | cases > 30 deg |
|---|---:|---:|---:|---:|---:|---:|
| mujoco(game) | 24.4 | 56.2 | 12.5 | 13.3 | 13.5 | 0/18 |
| genesis_native | 18.9 | 26.4 | 13.6 | 14.3 | 14.5 | 0/18 |
| genesis_rollfric | 18.7 | 26.4 | 13.1 | 13.8 | 14.8 | 0/18 |

## Paired comparison vs MuJoCo(game)

| protocol | engine | hit/miss agreement | peak-speed ratio gen/mj (median [min, max], both hit) | |Δdir| deg (median / max) | Δt first move s (median) |
|---|---|---:|---|---|---:|
| C_football | genesis_native | 18/18 | 0.98 [0.86, 1.28] | 5.5 / 11.9 | +0.00 |
| C_football | genesis_rollfric | 18/18 | 0.98 [0.87, 1.30] | 4.0 / 14.8 | +0.00 |

## Per case (peak ball speed m/s @ dir deg; F = fall; · = ball not hit)

| case | mujoco(game) | genesis_native | genesis_rollfric |
|---|---|---|---|
| C_football|left|-0.015|-0.015 | 1.28 @ +5 | 1.43 @ +9 | 1.48 @ +6 |
| C_football|left|-0.015|+0.000 | 1.54 @ +3 | 1.67 @ +2 | 1.69 @ -1 |
| C_football|left|-0.015|+0.015 | 1.70 @ -2 | 1.72 @ +3 | 1.72 @ -1 |
| C_football|left|+0.000|-0.015 | 1.42 @ +14 | 1.63 @ +9 | 1.67 @ +5 |
| C_football|left|+0.000|+0.000 | 1.46 @ +6 | 1.87 @ +6 | 1.87 @ +2 |
| C_football|left|+0.000|+0.015 | 1.88 @ -3 | 1.79 @ +4 | 1.82 @ +1 |
| C_football|left|+0.015|-0.015 | 1.37 @ +18 | 1.68 @ +6 | 1.78 @ +3 |
| C_football|left|+0.015|+0.000 | 1.64 @ +13 | 1.98 @ +5 | 1.99 @ +1 |
| C_football|left|+0.015|+0.015 | 2.06 @ -2 | 1.91 @ +4 | 1.92 @ +0 |
| C_football|right|-0.015|-0.015 | 1.96 @ -2 | 1.69 @ -7 | 1.71 @ -5 |
| C_football|right|-0.015|+0.000 | 1.98 @ +3 | 1.80 @ +2 | 1.82 @ +3 |
| C_football|right|-0.015|+0.015 | 2.05 @ +4 | 1.78 @ +2 | 1.78 @ +1 |
| C_football|right|+0.000|-0.015 | 1.98 @ +1 | 1.81 @ -2 | 1.86 @ +1 |
| C_football|right|+0.000|+0.000 | 2.05 @ +8 | 1.86 @ -3 | 1.89 @ -1 |
| C_football|right|+0.000|+0.015 | 1.96 @ +9 | 1.97 @ +2 | 1.96 @ +1 |
| C_football|right|+0.015|-0.015 | 2.08 @ +5 | 1.79 @ -3 | 1.83 @ +0 |
| C_football|right|+0.015|+0.000 | 1.93 @ +7 | 1.85 @ -2 | 1.87 @ +1 |
| C_football|right|+0.015|+0.015 | 1.69 @ +7 | 1.90 @ +2 | 1.88 @ +2 |
