# S4 kick sim2sim: MuJoCo(game) vs Genesis

## Per protocol summary (18 cases each = 2 feet x 9 ball positions)

| protocol | engine | falls | hits | median peak ball speed (m/s) | median dir (deg) | median dist 4 s (m) | max tilt (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| A_game_kick | mujoco(game) | 0 | 11/18 | 0.43 | 1.44 | 0.58 | 7.0 |
| A_game_kick | genesis_native | 0 | 17/18 | 0.69 | -6.48 | 2.48 | 6.2 |
| A_game_kick | genesis_rollfric | 0 | 17/18 | 0.70 | -7.98 | 1.10 | 6.2 |
| B_full_kick | mujoco(game) | 0 | 11/18 | 0.43 | 1.44 | 0.58 | 7.0 |
| B_full_kick | genesis_native | 0 | 17/18 | 0.69 | -6.48 | 2.48 | 6.2 |
| B_full_kick | genesis_rollfric | 0 | 17/18 | 0.70 | -7.98 | 1.10 | 6.2 |
| C_football | mujoco(game) | 0 | 18/18 | 1.74 | -15.45 | 4.05 | 15.1 |
| C_football | genesis_native | 1 | 18/18 | 1.51 | -18.09 | 5.50 | 169.4 |
| C_football | genesis_rollfric | 0 | 18/18 | 1.51 | -16.67 | 2.94 | 35.3 |

## Paired comparison vs MuJoCo(game)

| protocol | engine | hit/miss agreement | peak-speed ratio gen/mj (median [min, max], both hit) | |Δdir| deg (median / max) | Δt first move s (median) |
|---|---|---:|---|---|---:|
| A_game_kick | genesis_native | 12/18 | 1.75 [1.17, 3.14] | 4.2 / 7.3 | -0.02 |
| A_game_kick | genesis_rollfric | 12/18 | 1.76 [1.17, 3.15] | 4.6 / 7.7 | -0.02 |
| B_full_kick | genesis_native | 12/18 | 1.75 [1.17, 3.14] | 4.2 / 7.3 | -0.02 |
| B_full_kick | genesis_rollfric | 12/18 | 1.76 [1.17, 3.15] | 4.6 / 7.7 | -0.02 |
| C_football | genesis_native | 18/18 | 0.86 [0.41, 1.15] | 4.0 / 16.5 | +0.00 |
| C_football | genesis_rollfric | 18/18 | 0.88 [0.59, 1.22] | 4.4 / 13.0 | +0.00 |

## Per case (peak ball speed m/s @ dir deg; F = fall; · = ball not hit)

| case | mujoco(game) | genesis_native | genesis_rollfric |
|---|---|---|---|
| A_game_kick|left|-0.015|-0.015 | 0.43 @ -15 | 0.83 @ -11 | 0.84 @ -11 |
| A_game_kick|left|-0.015|+0.000 | 0.64 @ +3 | 1.02 @ +8 | 1.02 @ +8 |
| A_game_kick|left|-0.015|+0.015 | 0.69 @ +19 | 1.02 @ +22 | 1.02 @ +22 |
| A_game_kick|left|+0.000|-0.015 | · | 0.62 @ -11 | 0.62 @ -11 |
| A_game_kick|left|+0.000|+0.000 | 0.25 @ +1 | 0.78 @ +4 | 0.78 @ +4 |
| A_game_kick|left|+0.000|+0.015 | 0.30 @ +14 | 0.79 @ +18 | 0.79 @ +19 |
| A_game_kick|left|+0.015|-0.015 | · | 0.21 @ -11 | 0.22 @ -9 |
| A_game_kick|left|+0.015|+0.000 | · | 0.37 @ +1 | 0.37 @ +3 |
| A_game_kick|left|+0.015|+0.015 | · | 0.40 @ +13 | 0.40 @ +15 |
| A_game_kick|right|-0.015|-0.015 | 0.79 @ -23 | 0.92 @ -18 | 0.92 @ -18 |
| A_game_kick|right|-0.015|+0.000 | 0.77 @ -9 | 1.00 @ -16 | 1.01 @ -17 |
| A_game_kick|right|-0.015|+0.015 | 0.62 @ +7 | 0.84 @ +1 | 0.85 @ +1 |
| A_game_kick|right|+0.000|-0.015 | 0.38 @ -18 | 0.67 @ -19 | 0.67 @ -20 |
| A_game_kick|right|+0.000|+0.000 | 0.32 @ -4 | 0.69 @ -10 | 0.70 @ -10 |
| A_game_kick|right|+0.000|+0.015 | 0.22 @ +7 | 0.56 @ +3 | 0.56 @ +2 |
| A_game_kick|right|+0.015|-0.015 | · | 0.14 @ -22 | 0.15 @ -23 |
| A_game_kick|right|+0.015|+0.000 | · | 0.12 @ -6 | 0.12 @ -8 |
| A_game_kick|right|+0.015|+0.015 | · | · | · |
| B_full_kick|left|-0.015|-0.015 | 0.43 @ -15 | 0.83 @ -11 | 0.84 @ -11 |
| B_full_kick|left|-0.015|+0.000 | 0.64 @ +3 | 1.02 @ +8 | 1.02 @ +8 |
| B_full_kick|left|-0.015|+0.015 | 0.69 @ +19 | 1.02 @ +22 | 1.02 @ +22 |
| B_full_kick|left|+0.000|-0.015 | · | 0.62 @ -11 | 0.62 @ -11 |
| B_full_kick|left|+0.000|+0.000 | 0.25 @ +1 | 0.78 @ +4 | 0.78 @ +4 |
| B_full_kick|left|+0.000|+0.015 | 0.30 @ +14 | 0.79 @ +18 | 0.79 @ +19 |
| B_full_kick|left|+0.015|-0.015 | · | 0.21 @ -11 | 0.22 @ -9 |
| B_full_kick|left|+0.015|+0.000 | · | 0.37 @ +1 | 0.37 @ +3 |
| B_full_kick|left|+0.015|+0.015 | · | 0.40 @ +13 | 0.40 @ +15 |
| B_full_kick|right|-0.015|-0.015 | 0.79 @ -23 | 0.92 @ -18 | 0.92 @ -18 |
| B_full_kick|right|-0.015|+0.000 | 0.77 @ -9 | 1.00 @ -16 | 1.01 @ -17 |
| B_full_kick|right|-0.015|+0.015 | 0.62 @ +7 | 0.84 @ +1 | 0.85 @ +1 |
| B_full_kick|right|+0.000|-0.015 | 0.38 @ -18 | 0.67 @ -19 | 0.67 @ -20 |
| B_full_kick|right|+0.000|+0.000 | 0.32 @ -4 | 0.69 @ -10 | 0.70 @ -10 |
| B_full_kick|right|+0.000|+0.015 | 0.22 @ +7 | 0.56 @ +3 | 0.56 @ +2 |
| B_full_kick|right|+0.015|-0.015 | · | 0.14 @ -22 | 0.15 @ -23 |
| B_full_kick|right|+0.015|+0.000 | · | 0.12 @ -6 | 0.12 @ -8 |
| B_full_kick|right|+0.015|+0.015 | · | · | · |
| C_football|left|-0.015|-0.015 | 1.50 @ -10 | 1.73 @ -6 | 1.83 @ -5 |
| C_football|left|-0.015|+0.000 | 1.50 @ -5 | 0.61 @ +12 F | 0.89 @ +8 |
| C_football|left|-0.015|+0.015 | 1.52 @ +2 | 1.22 @ +12 | 1.46 @ +12 |
| C_football|left|+0.000|-0.015 | 1.67 @ -11 | 1.36 @ -15 | 1.37 @ -13 |
| C_football|left|+0.000|+0.000 | 1.70 @ -6 | 1.63 @ -3 | 1.64 @ -1 |
| C_football|left|+0.000|+0.015 | 1.74 @ -2 | 1.45 @ -0 | 1.44 @ +1 |
| C_football|left|+0.015|-0.015 | 1.82 @ -10 | 1.41 @ -15 | 1.40 @ -14 |
| C_football|left|+0.015|+0.000 | 1.83 @ -7 | 1.45 @ -13 | 1.46 @ -11 |
| C_football|left|+0.015|+0.015 | 1.88 @ -2 | 1.47 @ -8 | 1.48 @ -7 |
| C_football|right|-0.015|-0.015 | 1.66 @ -42 | 1.48 @ -39 | 1.48 @ -37 |
| C_football|right|-0.015|+0.000 | 1.69 @ -35 | 1.54 @ -30 | 1.54 @ -28 |
| C_football|right|-0.015|+0.015 | 1.71 @ -25 | 1.35 @ -28 | 1.36 @ -27 |
| C_football|right|+0.000|-0.015 | 1.76 @ -37 | 1.60 @ -36 | 1.60 @ -35 |
| C_football|right|+0.000|+0.000 | 1.73 @ -33 | 1.60 @ -29 | 1.60 @ -27 |
| C_football|right|+0.000|+0.015 | 1.84 @ -22 | 1.55 @ -26 | 1.55 @ -24 |
| C_football|right|+0.015|-0.015 | 1.86 @ -32 | 1.72 @ -34 | 1.72 @ -32 |
| C_football|right|+0.015|+0.000 | 1.83 @ -21 | 1.66 @ -28 | 1.66 @ -26 |
| C_football|right|+0.015|+0.015 | 1.88 @ -20 | 1.65 @ -21 | 1.64 @ -19 |
