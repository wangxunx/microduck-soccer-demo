# S4 kick sim2sim: MuJoCo(game) vs Genesis

## Per protocol summary (18 cases each = 2 feet x 9 ball positions)

| protocol | engine | falls | hits | median peak ball speed (m/s) | median dir (deg) | median dist 4 s (m) | max tilt (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| C_football | mujoco(game) | 3 | 18/18 | 1.96 | 4.84 | 4.33 | 129.4 |
| C_football | genesis_native | 4 | 18/18 | 1.92 | 0.86 | 6.83 | 133.1 |
| C_football | genesis_rollfric | 3 | 18/18 | 1.92 | -0.07 | 4.16 | 127.6 |

## Football handoff: max tilt in the window / within 0.6 s after handing over to `stand` (deg)

| engine | window median | window max | after-handoff median | after-handoff p90 | after-handoff max | cases > 30 deg |
|---|---:|---:|---:|---:|---:|---:|
| mujoco(game) | 15.4 | 21.0 | 42.6 | 47.5 | 49.3 | 17/18 |
| genesis_native | 17.1 | 22.9 | 49.6 | 57.5 | 120.9 | 17/18 |
| genesis_rollfric | 12.8 | 14.8 | 44.8 | 48.8 | 49.8 | 18/18 |

## Paired comparison vs MuJoCo(game)

| protocol | engine | hit/miss agreement | peak-speed ratio gen/mj (median [min, max], both hit) | |Δdir| deg (median / max) | Δt first move s (median) |
|---|---|---:|---|---|---:|
| C_football | genesis_native | 18/18 | 0.95 [0.91, 1.12] | 3.7 / 6.9 | +0.00 |
| C_football | genesis_rollfric | 18/18 | 0.96 [0.90, 1.16] | 5.0 / 8.2 | +0.00 |

## Per case (peak ball speed m/s @ dir deg; F = fall; · = ball not hit)

| case | mujoco(game) | genesis_native | genesis_rollfric |
|---|---|---|---|
| C_football|left|-0.015|-0.015 | 1.98 @ +5 F | 1.89 @ +1 | 1.88 @ -0 |
| C_football|left|-0.015|+0.000 | 2.03 @ +7 | 1.88 @ +2 | 1.91 @ +0 |
| C_football|left|-0.015|+0.015 | 2.03 @ +8 | 1.85 @ +2 | 1.86 @ -0 F |
| C_football|left|+0.000|-0.015 | 1.94 @ +6 | 1.93 @ +1 F | 1.96 @ -0 |
| C_football|left|+0.000|+0.000 | 1.95 @ +7 | 1.85 @ +3 | 1.88 @ +1 F |
| C_football|left|+0.000|+0.015 | 2.02 @ +7 | 1.86 @ +4 | 1.89 @ +2 |
| C_football|left|+0.015|-0.015 | 1.93 @ +5 | 1.95 @ +0 | 1.92 @ -0 |
| C_football|left|+0.015|+0.000 | 1.93 @ +5 | 2.00 @ +2 | 2.03 @ +0 |
| C_football|left|+0.015|+0.015 | 1.87 @ +5 F | 1.99 @ +2 | 2.04 @ -0 |
| C_football|right|-0.015|-0.015 | 1.77 @ -3 | 1.97 @ -2 | 2.00 @ -3 |
| C_football|right|-0.015|+0.000 | 1.79 @ -2 | 1.87 @ -1 F | 1.85 @ -3 |
| C_football|right|-0.015|+0.015 | 1.75 @ +4 | 1.96 @ +4 | 2.03 @ +1 |
| C_football|right|+0.000|-0.015 | 2.08 @ +4 | 1.90 @ -2 | 1.87 @ +1 |
| C_football|right|+0.000|+0.000 | 1.91 @ -7 F | 1.93 @ -2 | 1.92 @ +0 |
| C_football|right|+0.000|+0.015 | 1.97 @ -2 | 1.84 @ -2 | 1.86 @ -1 |
| C_football|right|+0.015|-0.015 | 2.06 @ +3 | 1.96 @ -2 F | 1.93 @ +1 |
| C_football|right|+0.015|+0.000 | 2.13 @ +5 | 1.98 @ -2 F | 1.98 @ -0 |
| C_football|right|+0.015|+0.015 | 2.09 @ -2 | 1.92 @ -1 | 1.91 @ -0 F |
