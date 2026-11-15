# S3 ball dynamics: MuJoCo (game reference) vs Genesis

MuJoCo reference = game engine (Python 3.16 and game WASM 3.11 bit-identical, see crosscheck_wasm.json).


### roll_0.3

| metric | mujoco_python | genesis_native | genesis_rollfric |
|---|---:|---:|---:|
| v@0.5s | 0.220 | 0.300 | 0.090 |
| v@1s | 0.156 | 0.300 | 0.000 |
| v@2s | 0.064 | 0.300 | 0.000 |
| v@4s | 0.009 | 0.300 | 0.000 |
| stop_s | 3.190 | – | 0.670 |
| dist_8s_m | 0.392 | 2.397 | 0.105 |
| slip@0.5s | 0.000 | 0.001 | 0.001 |

### roll_0.6

| metric | mujoco_python | genesis_native | genesis_rollfric |
|---|---:|---:|---:|
| v@0.5s | 0.462 | 0.600 | 0.389 |
| v@1s | 0.351 | 0.600 | 0.179 |
| v@2s | 0.189 | 0.600 | 0.000 |
| v@4s | 0.031 | 0.600 | 0.000 |
| stop_s | 4.465 | – | 1.380 |
| dist_8s_m | 0.944 | 4.794 | 0.424 |
| slip@0.5s | 0.000 | 0.001 | 0.000 |

### roll_1.0

| metric | mujoco_python | genesis_native | genesis_rollfric |
|---|---:|---:|---:|
| v@0.5s | 0.803 | 0.999 | 0.789 |
| v@1s | 0.626 | 0.999 | 0.579 |
| v@2s | 0.368 | 0.999 | 0.158 |
| v@4s | 0.092 | 0.999 | 0.000 |
| stop_s | 5.550 | – | 2.330 |
| dist_8s_m | 1.800 | 7.990 | 1.181 |
| slip@0.5s | 0.000 | 0.002 | 0.001 |

### roll_2.0

| metric | mujoco_python | genesis_native | genesis_rollfric |
|---|---:|---:|---:|
| v@0.5s | 1.812 | 1.999 | 1.789 |
| v@1s | 1.603 | 1.999 | 1.578 |
| v@2s | 1.196 | 1.999 | 1.157 |
| v@4s | 0.478 | 1.999 | 0.316 |
| stop_s | – | – | 4.705 |
| dist_8s_m | 5.508 | 15.981 | 4.734 |
| slip@0.5s | 0.041 | 0.004 | 0.003 |

### slide_1.0

| metric | mujoco_python | genesis_native | genesis_rollfric |
|---|---:|---:|---:|
| v@0.5s | 0.562 | 0.713 | 0.507 |
| v@1s | 0.431 | 0.713 | 0.297 |
| v@2s | 0.241 | 0.713 | 0.000 |
| v@4s | 0.045 | 0.713 | 0.000 |
| stop_s | 4.835 | – | 1.660 |
| dist_8s_m | 1.191 | 5.707 | 0.610 |
| slip@0.5s | 0.000 | 0.001 | 0.000 |

### drop_0.2

| metric | mujoco_python | genesis_native | genesis_rollfric |
|---|---:|---:|---:|
| bounce_apex_mm | 0.022 | -0.060 | -0.060 |
| settle_z_mm | 49.869 | 49.797 | 49.797 |

### wall_1.0

| metric | mujoco_python | genesis_native | genesis_rollfric |
|---|---:|---:|---:|
| v_in | 0.755 | 0.999 | 0.735 |
| v_out | 0.171 | 0.253 | 0.175 |
| restitution | 0.226 | 0.253 | 0.238 |

### kick_0.5

| metric | mujoco_python | genesis_native | genesis_rollfric |
|---|---:|---:|---:|
| ball_peak_v | 0.180 | 0.336 | 0.335 |
| ball_v@1s | 0.070 | 0.289 | 0.028 |
| ball_v@2s | 0.026 | 0.289 | 0.000 |
| ball_dist_3s_m | 0.164 | 0.849 | 0.112 |
| striker_v_after | -0.012 | 0.118 | 0.118 |
| ball_lift_mm | -0.121 | -0.200 | -0.200 |

### kick_1.0

| metric | mujoco_python | genesis_native | genesis_rollfric |
|---|---:|---:|---:|
| ball_peak_v | 0.364 | 0.689 | 0.686 |
| ball_v@1s | 0.178 | 0.607 | 0.206 |
| ball_v@2s | 0.077 | 0.607 | 0.063 |
| ball_dist_3s_m | 0.411 | 1.407 | 0.472 |
| striker_v_after | -0.018 | 0.243 | 0.243 |
| ball_lift_mm | -0.091 | 0.765 | -0.200 |
