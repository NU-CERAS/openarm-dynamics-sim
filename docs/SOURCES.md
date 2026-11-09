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
| [00](https://robstride.com/products/robStride00) | 0.310 | 5 | 14 | 100 | 315 | 10 |

RobStride 00 specifications checked on 2026-10-19: 48 V nominal, 57 × 57 × 51 mm.
The user corrected J5/J6/J7 to model 00. Both the generated arm and imported CAD
arm now use the 00 ratings there. The 02 catalog entry is retained for reference.

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
70 (interpreted as millimeters), corrected actuator assignment 03/03/06/06/00/00/00, and the
OpenArm v1.1 joint arrangement. The original image remains in the user's Downloads;
it is not required to run this repository. Payload, task, tube section, bracket
mass, gripper mass/COM, gravity mounting, and custom translation directions are
explicit preliminary assumptions. See `configs/arm.yaml` and `docs/MODEL.md`.

The replacement `robots/moving_asm/urdf/moving_asm.urdf` includes the user's
seven-link Onshape CAD and local mesh files. The real J7 drives `part_7` on the
final `rs00_1` housing, which remains fixed to `part_6`. All exported joint origins,
axes, mesh geometry, and structural link inertial data are retained. `part_7`
weighs 0.427991 kg and uses its exported COM and full inertia tensor. This replaces
the prior model's provisional box; no additional end-effector mass is added.
The reported tip is initially the `part_7` link origin, configurable through
`urdf.tip_offset_m` when a measured tool-tip location is available.

All three imported RS00 housing masses were corrected from the CAD's 0.000127908
kg to the catalog 0.310 kg; their six inertia components were scaled by the mass
ratio with the CAD COM retained. This is a uniform-density approximation, not a
measured motor inertia. Both RS06 housing masses were subsequently corrected
from 0.000238549 kg to the catalog 0.621 kg, using the same proportional inertia
scaling and retaining the CAD COM. Other CAD link properties are retained. The
compiled replacement arm totals 7.446340 kg, including the real seventh link.

`robots/right_arm_assembly/urdf/right_arm_assembly.urdf` is a second Onshape export:
the same joint chain as `moving_asm`, rotated 90° about Z, with a `base`/`screwon`/`drumstick`
tool instead of `part_7`. The export had no `<limit>` elements; effort (RobStride peak) and
velocity (no-load speed) were added per motor (03/03/06/06/00/00/00, identified by each
joint's parent housing). Hardware limits were mapped from `moving_asm`: J2 −20..200°,
J4 ±135°, J6 −90..50°, J7 ±70° (J7's axis is reversed in this export; the range is
symmetric, and with this tool it is collision-free, unlike `part_7`'s −56°). RS00/RS06
housing masses and inertias received the same catalog correction as above. The tool links
had near-zero export masses (no material assigned); the user's measured 0.151 lb (`base`),
0.065 lb (`screwon`) and 0.043 lb (`drumstick`) replace them, with CAD COM retained and
inertia scaled by the mass ratio. The compiled arm totals 5.4116 kg. Its `.sim.yaml` copies `moving_asm`'s drive data, gains and floor.

