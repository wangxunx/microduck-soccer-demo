"""S4 cases: protocol x kicking foot x ball placement (robot at origin, yaw 0).

Ball = game ball (r 0.05 m, 30 g). Nominal centre (0.105, +/-0.042): the
upstream BallKick nominal (0.09, +/-0.042) shifted +1.5 cm so the ball's rear
surface keeps the same toe clearance with the bigger game ball. Grid +/-1.5 cm
= upstream BALL_POS_NOISE_XY.

Protocols (mirroring game/src/physics/world-core.js):
  A_game_kick  official kickL/kickR 0.5 s (game default window), then walk with
               zero command (0.4 s kick lock, then idle)
  B_full_kick  official kickL/kickR for 3.0 s, then walk zero command
  C_football   football.onnx 1.2 s (4-step actuator delay, 1.75 A limit,
               1-step joint-velocity delay, ball obs in slots 55-60, aim =
               heading at trigger), then stand 1 s, then walk zero command
"""
import json
from pathlib import Path

DUR = 4.0
cases = []
for proto in ("A_game_kick", "B_full_kick", "C_football"):
    for foot, side in (("left", 1.0), ("right", -1.0)):
        for dx in (-0.015, 0.0, 0.015):
            for dy in (-0.015, 0.0, 0.015):
                cases.append(dict(id=f"{proto}|{foot}|{dx:+.3f}|{dy:+.3f}", protocol=proto, foot=foot,
                                  bx=round(0.105 + dx, 4), by=round(side * 0.042 + dy, 4)))
Path(__file__).with_name("cases.json").write_text(json.dumps({"duration": DUR, "cases": cases}, indent=1))
print(len(cases), "cases")
