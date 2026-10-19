# OpenArm MuJoCo experiments

A small MuJoCo tool for jogging seven joints, saving poses A/B, running a timed
move, and examining motor torque and motion tracking. MuJoCo performs the physics;
there is no custom integrator, dashboard, or automatic design optimizer.

## Start

```bash
cd '/Users/bradylin/Documents/Projects/Club & Lab/openarm-dynamics-sim'
./scripts/setup.sh       # first run, or after pulling dependency changes
./scripts/start.sh
```

Python 3.13 is the default (`brew install python@3.13` on macOS if needed).
The start script launches the single native window on the OS main thread, including macOS.
No web server is required.

## Use the window

The left panel contains the controls, the center shows MuJoCo's renderer, and a
fixed right column shows seven live torque charts, J1 at the top through J7 at the
bottom. All seven stay visible without scrolling. Physics remains entirely in MuJoCo. The
small native control panel uses Dear ImGui because the packaged Python viewer does
not expose custom buttons. There is no browser dashboard or custom physics solver.

Each chart shows applied output torque in Nm over the same ten-second time window.
Yellow dotted lines mark positive/negative nominal (`rated_torque_nm`) ratings;
red dotted lines mark positive/negative maximum (`peak_torque_nm`) ratings. Yellow
bands cover magnitudes between nominal and maximum; red bands cover magnitudes
beyond maximum. Current values and trace endpoints change color when exceeded.
The charts have independent vertical scales that always include both limit pairs.
These are the configured motor ratings, before derating or speed-envelope limits;
the actual actuator caps still apply, so a capped actuator may never enter red.
Reload YAML after changing a motor assignment or rating to update the charts.

| Control | Action |
|---|---|
| **Joint target sliders** | Jog each joint in degrees; Ctrl-click to type an angle |
| **Jog speed (deg/s)** | Maximum target slew rate during manual jogging |
| **Zero arm** | Set all seven targets to 0° and move at jog speed |
| **Randomize joints** | Pick each target within its joint limits and move at jog speed |
| **Link lengths / weights** | Edit all seven lengths and structural masses, end-effector mass, and payload in one dialog |
| **Save A position / Save B position** | Capture actual angles and immediately save that pose to YAML |
| **Go to A / Go to B** | Smoothly move from the current pose to the stored pose, under motor torque limits |
| **Move duration (s)** | Duration of Go-to moves and the A → B sweep segment |
| **Run A → B sweep** | Approach A, settle, sweep to B, hold, then save the load report |
| **Pause / Resume** | Pause or resume physics without advancing trajectory time |
| **Stop / hold** | Cancel motion and command the current pose; interrupted sweeps save partial results |
| **Advanced: per-joint P / D** | Select J1–J7 and edit that joint's P and D gains live |
| **Save controls** | Persist poses, duration, jog speed, and all P/D settings |
| **Reload YAML** | Rebuild from the file and initialize at saved A |
| **Edit all parameters** | Open the YAML for geometry, masses, motor ratings, and other parameters |
| **Open last report** | Open the last recorded sweep's torque/tracking plots |

Save A/B captures **actual angles**, not slider targets. Let the arm settle first.
Zero and Randomize update targets without teleporting the arm or changing saved
poses. They respect Pause and are disabled during a Go-to or sweep. Random poses
are limited by joint angles; they are not checked for collisions or reachability
under the available motor torque.
Those buttons save only their respective pose, preserving other edits on disk.
Other live control changes remain in memory until **Save controls**. Reloading
replaces in-memory settings with the file. Do not save controls over intentional
external changes to those same control fields; reload those changes first.

Go-to moves use the selected duration and then wait `settle_s` before reporting
whether the destination was reached. They do not teleport the arm or start a
recorded A/B experiment. Their torque is visible in the live graph. Jog speed
applies to manual jogging, not timed Go-to/sweep motion.

During a Go-to or sweep, pose capture, targets, gains, duration, and reload are
locked; **Pause** and **Stop / hold** remain available. The smooth quintic trajectory
has zero endpoint velocity and acceleration. The sweep's initial approach uses
`approach_s`, followed by `settle_s`; its final hold uses `hold_s`. A sweep aborts if
the arm cannot reach A within the configured tracking tolerance.

P is in Nm/rad; D is in Nm·s/rad. These are simulation PD gains, not RobStride
firmware gains. D may be zero. Gravity compensation remains optional and consumes
the same motor torque budget. Keep gains fixed when comparing linkage designs.
Sweep reports include the live gains in both their YAML and generated XML.

Camera: drag to rotate, right-drag to pan, wheel to zoom; Shift changes the drag axis.
Tab navigates buttons/fields and Space activates the focused button. Outside a
focused control, the original shortcuts remain: 1–7 then Up/Down for 2° jogging,
A/B to save poses, Enter to sweep, [/] for duration, Space to pause, X to stop,
F5 to reload, F8 to save controls, E to edit YAML, and P for the report.

## Parameters to tune

For quick changes, click **Link lengths / weights** in the window. Lengths are in
millimeters; masses are in kilograms. **Auto mass** estimates hollow aluminum
tubes plus brackets using the YAML dimensions; it is not a solid-link estimate.
Uncheck it to enter a measured or CAD structural mass (motor weight is separate).
For solid aluminum, mass is volume times density: using 2700 kg/m³, a
100 × 40 × 40 mm solid bar is 0.432 kg, while a 100 × 30 × 8 mm solid plate is
0.0648 kg, before brackets, holes, or machining. A manual mass alone does not
replace the estimated COM/inertia; use CAD overrides when available.

The same editor has **Actuator masses (kg)** for J1–J7 independently, even when
several joints share a motor model. These update motor housing mass and inertia;
they do not change torque ratings, dimensions, controller gains, or rotor armature.
**Use catalog motor masses** resets all seven to the selected motor models' masses.
Overrides are saved as each joint's `motor_mass_override_kg`; `null` uses the catalog.
J1's housing is fixed to the base; J2–J7 housings load their upstream links.

**Apply** rebuilds the simulation temporarily; **Apply + save** also writes these
geometry/mass fields to YAML. Applying keeps the current joint angles, clears
velocity and chart history, and pauses physics. Close the dialog and press
**Resume** or run a sweep. A/B poses and controller gains are preserved. Existing
CAD COM/inertia overrides remain fixed and must be updated separately in YAML.
**Save controls** does not save geometry; use **Apply + save** for these edits.

Motor assignment: **J1–J2: RobStride 03; J3–J4: RobStride 06; J5–J7: RobStride 02**.
Each assignment selects that model's mass, housing dimensions, torque ratings,
and speed ratings. Controller gains remain independently adjustable per joint.

Everything is in **`configs/arm.yaml`**. Edit, save, then click **Reload YAML** to apply.

| Section | Parameters |
|---|---|
| `links` (seven entries) | Length, tube outer width/wall, density, bracket mass/radius; independent structural mass, COM, and inertia overrides |
| `joints` (seven entries) | Motor model, local axis, mounting rotation, angle limits, position/damping gains, friction, viscous damping, reflected rotor inertia |
| `motors` | Housing mass/dimensions, rated/peak output torque, rated/no-load speed for each RobStride model |
| `tool`, `payload` | Mass, box dimensions, COM placement |
| `sweep` | A/B angles in degrees, duration, approach/settle/hold times, tracking tolerance, jog speed |
| `drive` | Torque derating, assumed torque-speed envelope, gravity compensation |
| `simulation` | MuJoCo timestep, primitive contact enable, floor height |
| Root fields | Base placement/rotation and gravity |

For example, change the third link's `length_mm: 125` to `length_mm: 150`.
Its mass, COM, inertia, downstream joint placement, and geometry are rebuilt.
To use a measured structural mass, set that entry's `mass_override_kg: 0.25`.
This replaces the **tube + bracket** mass; motor housings, tool, and payload remain
separate. Set `com_override_m` and `inertia_override_kg_m2` when you have CAD data;
otherwise inertia is scaled from the estimated structure. A fixed mass override
stays fixed as length changes. Reset overrides to `null` to restore estimates.

Initial structure: 40 × 40 × 2 mm aluminum square tubes (2700 kg/m³), 100 g brackets,
400 g gripper, 500 g example payload. Structural masses in J1→J2 through J7→tip order:
**182, 182, 203, 182, 157, 162, 157 g**, excluding motors/tool/payload.
Lengths are **100, 100, 125, 100, 70, 75, 70 mm**. The last length is the J7→tool adapter.

## Read the results

Every sweep saves `results/<timestamp>/` with:

- `report.html` and `loads.png`: all seven torque histories and actual/requested angles.
- `loads.csv`: every physics step, with phase, torque, angle, speed, limits, mechanical
  power, tool-tip position, and contact count.
- `summary.json`: peak/RMS torque, maximum tracking error, saturation, speed, power,
  time above the rated proxy, and completion status. Summary covers A → B + final hold.
- `config.yaml` and `model.xml`: the parameter snapshot and generated MuJoCo model.

**Low measured torque is useful only if the arm follows the move.** Check tracking
error and saturation alongside the torque curves. The report measures applied
motor torque, not the unlimited torque required for perfect trajectory tracking.
A “tracking within tolerance” result does not establish continuous-duty capability.
Motor speed envelopes are estimates, and RMS torque alone is not a thermal model.

The exported XML contains geometry and static actuator settings. Replaying the
experiment requires the Python runner, which supplies trajectories, gravity
compensation, and the changing torque-speed caps. Use its `config.yaml` to reproduce
an experiment; opening XML alone does not reproduce that controller.

## Run without the viewer

```bash
.venv/bin/python -m arm_design.cli sweep --duration 3
.venv/bin/python -m arm_design.cli sweep --config results/<timestamp>/config.yaml
.venv/bin/python -m arm_design.cli export --out results/model
.venv/bin/python -m pytest -q
```

The headless sweep exits with code 2 when it does not complete within the tracking
tolerance. The `arm-sim` entry point supports the same commands. Use `start.sh` to launch
the native window; the custom window does not need the `mjpython` trampoline.

## Comparing lengths

Save a baseline sweep, change one link or parameter, reload, and repeat. Compare
reports and actual tool-tip paths. **Identical A/B joint angles with different link
lengths produce different Cartesian tasks.** Re-jog A/B to the same required tool
poses if your requirement is a fixed reach or endpoint. Intermediate paths can still
differ. This minimal tool does not include Cartesian IK, trajectory optimization,
FEA, or an automated length optimizer.

See [model assumptions](docs/MODEL.md) and [sources](docs/SOURCES.md). Earlier
Pinocchio/Streamlit simulation code was removed; old results folders were preserved
as historical artifacts and are not current-format configurations.
