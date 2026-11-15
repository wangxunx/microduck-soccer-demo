# S1 sim2sim results

## Falls per scenario (total over all scenarios)

| policy | mujoco(game) | genesis[game] | genesis[train] |
|---|---:|---:|---:|
| official_walk | 0 | 0 | 0 |
| genesis_backlash | 0 | 0 | 0 |
| genesis_rough | 0 | 0 | 0 |
| genesis_velocity | 0 | 0 | 0 |
| genesis_velocity_repro | 0 | 0 | 0 |

## Fixed-command tracking: measured vx (m/s) / wz (rad/s), t ≥ 3 s

| scenario | cmd | policy | mujoco(game) | genesis[game] | genesis[train] |
|---|---|---|---|---|---|
| fwd_0.15 | 0.15 / 0.0 | official_walk | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / -0.00 |
| fwd_0.15 | 0.15 / 0.0 | genesis_backlash | 0.06 / -0.05 | 0.07 / -0.04 | 0.09 / -0.08 |
| fwd_0.15 | 0.15 / 0.0 | genesis_rough | 0.07 / -0.10 | 0.07 / -0.21 | 0.09 / -0.10 |
| fwd_0.15 | 0.15 / 0.0 | genesis_velocity | 0.06 / -0.05 | 0.06 / -0.18 | 0.06 / -0.08 |
| fwd_0.15 | 0.15 / 0.0 | genesis_velocity_repro | 0.05 / -0.01 | -0.00 / 0.00 | 0.07 / -0.02 |
| fwd_0.30 | 0.30 / 0.0 | official_walk | 0.11 / -0.00 | 0.11 / -0.01 | 0.14 / 0.00 |
| fwd_0.30 | 0.30 / 0.0 | genesis_backlash | 0.14 / -0.03 | 0.12 / -0.06 | 0.15 / -0.12 |
| fwd_0.30 | 0.30 / 0.0 | genesis_rough | 0.14 / -0.07 | 0.14 / -0.18 | 0.18 / -0.08 |
| fwd_0.30 | 0.30 / 0.0 | genesis_velocity | 0.10 / -0.09 | 0.10 / -0.16 | 0.12 / -0.06 |
| fwd_0.30 | 0.30 / 0.0 | genesis_velocity_repro | 0.09 / 0.09 | 0.09 / 0.02 | 0.11 / 0.02 |
| back_0.30 | -0.30 / 0.0 | official_walk | -0.00 / 0.00 | -0.00 / -0.00 | -0.00 / -0.00 |
| back_0.30 | -0.30 / 0.0 | genesis_backlash | -0.15 / 0.03 | -0.14 / 0.15 | -0.16 / 0.06 |
| back_0.30 | -0.30 / 0.0 | genesis_rough | -0.17 / -0.10 | -0.16 / -0.19 | -0.18 / 0.12 |
| back_0.30 | -0.30 / 0.0 | genesis_velocity | -0.12 / -0.36 | -0.11 / -0.30 | -0.09 / -0.09 |
| back_0.30 | -0.30 / 0.0 | genesis_velocity_repro | -0.10 / -0.00 | -0.10 / 0.02 | -0.08 / 0.05 |
| turn_+1.0 | 0.00 / 1.0 | official_walk | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| turn_+1.0 | 0.00 / 1.0 | genesis_backlash | -0.00 / 0.64 | -0.01 / 0.68 | -0.01 / 0.75 |
| turn_+1.0 | 0.00 / 1.0 | genesis_rough | 0.01 / 0.92 | 0.00 / 0.94 | 0.02 / 0.86 |
| turn_+1.0 | 0.00 / 1.0 | genesis_velocity | 0.00 / 0.77 | 0.00 / 0.71 | 0.01 / 0.76 |
| turn_+1.0 | 0.00 / 1.0 | genesis_velocity_repro | 0.00 / 0.86 | -0.00 / 0.84 | 0.03 / 0.76 |
| turn_-1.0 | 0.00 / -1.0 | official_walk | -0.00 / -0.00 | -0.00 / -0.00 | -0.00 / -0.00 |
| turn_-1.0 | 0.00 / -1.0 | genesis_backlash | 0.01 / -0.73 | 0.00 / -0.68 | 0.02 / -0.80 |
| turn_-1.0 | 0.00 / -1.0 | genesis_rough | 0.01 / -0.98 | 0.01 / -1.07 | 0.03 / -1.07 |
| turn_-1.0 | 0.00 / -1.0 | genesis_velocity | 0.01 / -1.04 | 0.01 / -1.09 | 0.03 / -1.03 |
| turn_-1.0 | 0.00 / -1.0 | genesis_velocity_repro | 0.00 / -0.93 | 0.00 / -0.93 | 0.02 / -0.89 |
| turn_+1.5 | 0.00 / 1.5 | official_walk | 0.01 / 0.74 | 0.00 / 0.63 | 0.01 / 0.76 |
| turn_+1.5 | 0.00 / 1.5 | genesis_backlash | -0.01 / 1.01 | -0.01 / 1.06 | -0.01 / 1.14 |
| turn_+1.5 | 0.00 / 1.5 | genesis_rough | 0.00 / 1.38 | -0.00 / 1.35 | 0.01 / 1.32 |
| turn_+1.5 | 0.00 / 1.5 | genesis_velocity | -0.01 / 1.15 | -0.01 / 1.12 | -0.00 / 1.23 |
| turn_+1.5 | 0.00 / 1.5 | genesis_velocity_repro | -0.00 / 1.27 | -0.01 / 1.21 | 0.01 / 1.25 |
| arc_0.2_0.5 | 0.20 / 0.5 | official_walk | 0.00 / 0.00 | 0.00 / 0.00 | 0.10 / 0.22 |
| arc_0.2_0.5 | 0.20 / 0.5 | genesis_backlash | 0.08 / 0.35 | 0.08 / 0.20 | 0.11 / 0.29 |
| arc_0.2_0.5 | 0.20 / 0.5 | genesis_rough | 0.09 / 0.31 | 0.09 / 0.23 | 0.12 / 0.31 |
| arc_0.2_0.5 | 0.20 / 0.5 | genesis_velocity | 0.07 / 0.27 | 0.06 / 0.24 | 0.07 / 0.38 |
| arc_0.2_0.5 | 0.20 / 0.5 | genesis_velocity_repro | 0.06 / 0.45 | 0.06 / 0.32 | 0.08 / 0.35 |

## Standing still 20 s: drift (m) / yaw drift (deg) / max tilt (deg)

| policy | mujoco(game) | genesis[game] | genesis[train] |
|---|---|---|---|
| official_walk | 0.003 / 0.7 / 0.9 | 0.001 / 0.3 / 0.7 | 0.002 / 0.2 / 1.3 |
| genesis_backlash | 0.003 / 1.3 / 1.7 | 0.003 / 2.2 / 1.3 | 0.003 / 2.2 / 1.8 |
| genesis_rough | 0.003 / 0.6 / 1.1 | 0.002 / 1.1 / 1.1 | 0.002 / 1.1 / 1.6 |
| genesis_velocity | 0.002 / 0.7 / 1.5 | 0.002 / 1.2 / 1.3 | 0.002 / 1.1 / 1.6 |
| genesis_velocity_repro | 0.003 / 0.6 / 1.5 | 0.004 / 1.2 / 1.8 | 0.004 / 1.4 / 1.9 |

## Game-like random command sequences (6 × 60 s): mean vx RMSE / wz RMSE / falls / max tilt

| policy | mujoco(game) | genesis[game] | genesis[train] |
|---|---|---|---|
| official_walk | 0.142 / 0.375 / 0 / 5.5 | 0.143 / 0.416 / 0 / 5.1 | 0.128 / 0.330 / 0 / 7.8 |
| genesis_backlash | 0.115 / 0.182 / 0 / 4.5 | 0.126 / 0.222 / 0 / 4.8 | 0.105 / 0.162 / 0 / 6.6 |
| genesis_rough | 0.109 / 0.133 / 0 / 4.3 | 0.110 / 0.202 / 0 / 3.9 | 0.095 / 0.129 / 0 / 6.5 |
| genesis_velocity | 0.140 / 0.226 / 0 / 4.8 | 0.143 / 0.273 / 0 / 5.5 | 0.138 / 0.168 / 0 / 6.3 |
| genesis_velocity_repro | 0.151 / 0.121 / 0 / 3.6 | 0.151 / 0.163 / 0 / 3.7 | 0.145 / 0.134 / 0 / 5.8 |
