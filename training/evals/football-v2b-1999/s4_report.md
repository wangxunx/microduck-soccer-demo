# S4 kick sim2sim: MuJoCo(game) vs Genesis

## Per protocol summary (18 cases each = 2 feet x 9 ball positions)

| protocol | engine | falls | hits | median peak ball speed (m/s) | median dir (deg) | median dist 4 s (m) | max tilt (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| C_football | mujoco(game) | 1 | 18/18 | 2.68 | -6.57 | 5.76 | 78.1 |
| C_football | genesis_native | 0 | 18/18 | 2.66 | -0.82 | 8.29 | 66.7 |
| C_football | genesis_rollfric | 0 | 18/18 | 2.68 | -0.24 | 5.73 | 64.9 |

## Paired comparison vs MuJoCo(game)

| protocol | engine | hit/miss agreement | peak-speed ratio gen/mj (median [min, max], both hit) | |Δdir| deg (median / max) | Δt first move s (median) |
|---|---|---:|---|---|---:|
| C_football | genesis_native | 18/18 | 1.00 [0.90, 1.14] | 5.5 / 24.6 | -0.02 |
| C_football | genesis_rollfric | 18/18 | 1.00 [0.86, 1.14] | 6.2 / 27.9 | -0.02 |

## Per case (peak ball speed m/s @ dir deg; F = fall; · = ball not hit)

| case | mujoco(game) | genesis_native | genesis_rollfric |
|---|---|---|---|
| C_football|left|-0.015|-0.015 | 2.69 @ -3 | 2.56 @ +3 | 2.41 @ +6 |
| C_football|left|-0.015|+0.000 | 2.55 @ -11 | 2.50 @ +2 | 2.43 @ +7 |
| C_football|left|-0.015|+0.015 | 2.37 @ -7 | 2.70 @ -1 | 2.71 @ +2 |
| C_football|left|+0.000|-0.015 | 2.60 @ -12 | 2.42 @ +3 | 2.45 @ +6 |
| C_football|left|+0.000|+0.000 | 2.48 @ -19 | 2.48 @ +6 | 2.14 @ +9 |
| C_football|left|+0.000|+0.015 | 2.58 @ -14 | 2.64 @ -5 | 2.64 @ -4 |
| C_football|left|+0.015|-0.015 | 2.70 @ -17 F | 2.54 @ -4 | 2.54 @ -4 |
| C_football|left|+0.015|+0.000 | 2.58 @ -16 | 2.65 @ -11 | 2.67 @ -10 |
| C_football|left|+0.015|+0.015 | 2.64 @ -16 | 2.37 @ -0 | 2.37 @ +0 |
| C_football|right|-0.015|-0.015 | 2.67 @ -6 | 2.73 @ -0 | 2.72 @ +0 |
| C_football|right|-0.015|+0.000 | 2.72 @ -2 | 2.77 @ +3 | 2.76 @ +4 |
| C_football|right|-0.015|+0.015 | 2.74 @ +0 | 2.77 @ +3 | 2.74 @ +3 |
| C_football|right|+0.000|-0.015 | 2.80 @ -3 | 2.66 @ -4 | 2.67 @ -4 |
| C_football|right|+0.000|+0.000 | 2.78 @ -4 | 2.83 @ -1 | 2.83 @ -1 |
| C_football|right|+0.000|+0.015 | 2.67 @ -11 | 2.59 @ -10 | 2.72 @ -8 |
| C_football|right|+0.015|-0.015 | 2.79 @ -1 | 2.82 @ -6 | 2.82 @ -6 |
| C_football|right|+0.015|+0.000 | 2.84 @ -2 | 2.83 @ -4 | 2.83 @ -4 |
| C_football|right|+0.015|+0.015 | 2.88 @ -4 | 2.69 @ -9 | 2.68 @ -9 |
