# Sources and assumptions

Retrieved/consulted 2026-10-18.

## OpenArm

- [OpenArm 1.1 release](https://github.com/enactic/openarm/releases/tag/1.1):
  hardware reliability, calibration, wiring/covers, and camera mount updates.
  No joint-axis redesign is listed. This supports using the v1-family model as a
  starting point; it does not substitute for checking v1.1 CAD.
- [Official robot description](https://github.com/enactic/openarm_description),
  pinned revision `206eb8a4cf2e5e564536b4f5366ef369843a4aa7`.
- [Pinned v1 URDF](https://github.com/enactic/openarm_description/blob/206eb8a4cf2e5e564536b4f5366ef369843a4aa7/assets/robot/openarm_v1.0/urdf/example/v1.urdf):
  copied unchanged to `references/openarm_v1.urdf`; provides left-arm axes, origin
  rotations and provisional joint limits. Copyright 2025 Enactic, Inc.; Apache-2.0
  license copied to `references/OPENARM_LICENSE.txt`.
- [Hardware 1.1.0 release](https://github.com/enactic/openarm_hardware/releases/tag/1.1.0):
  future source for CAD reconciliation. CAD has not been imported into this model.

## RobStride

These manufacturer pages supply the numeric properties in the `motors` section of `configs/arm.yaml`:

| Model | Mass (kg) | Rated torque (N·m) | Peak (N·m) | Rated speed (rpm) | No-load (rpm) | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| [03](https://robstride.com/products/robStride03) | 0.880 | 20 | 60 | 100 | 195 | 9 |
| [02](https://robstride.com/products/robStride02) | 0.405 | 6 | 17 | 100 | 410 | 7.75 |
| [06](https://robstride.com/products/robStride06) | 0.621 | 11 | 36 | 100 | 480 | 9 |

These are nominal 48 V data. Standard 02 is assumed, not 02 IP67. No-load speed is
not a loaded speed guarantee. The dimensions used to approximate cylindrical
inertias are 106×106×56, 78.5×78.5×45.5, and 88×88×49 mm respectively; cylinder
inertias are our approximation, not vendor-specified inertias.

No full manufacturer torque-speed curve, allowable peak-duration model, thermal
model, rotor inertia, bearing-load limits, or gearbox life calculation has been
verified. The configured proxies are identified as assumptions in all reports.

## Software references

- [MuJoCo visualization](https://mujoco.readthedocs.io/en/stable/programming/visualization.html):
  mjv_updateScene, mjr_render, camera controls, and joint torque figures.
- [Dear ImGui Bundle](https://github.com/pthom/imgui_bundle):
  native controls in the same GLFW window as MuJoCo, using its OpenGL2 backend.
- [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html):
  explicit inertias, position actuators, force limits, joint torque sensors.
- [MuJoCo data reference](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html):
  qfrc_actuator is generalized joint actuation force (Nm for our hinges).
- SciPy supplies rigid rotations; Matplotlib supplies saved plots.

## User-supplied design

The user supplied a seven-span sketch with dimensions 100, 100, 125, 100, 70, 75,
70 (interpreted as millimeters), corrected actuator assignment 03/03/06/06/02/02/02, and the
OpenArm v1.1 joint arrangement. The original image remains in the user's Downloads;
it is not required to run this repository. Payload, task, tube section, bracket
mass, gripper mass/COM, gravity mounting, and custom translation directions are
explicit preliminary assumptions. See `configs/arm.yaml` and `docs/MODEL.md`.

