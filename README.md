# OpenArm MuJoCo

## Start

Install Python 3.13 (`brew install python@3.13` on macOS), then run from the project folder:

```bash
./scripts/setup.sh   # first time only
./scripts/start.sh
```

For later launches, just run `./scripts/start.sh`.

`setup.sh` creates `.venv` (set `PYTHON_BIN` to use another interpreter than
`python3.13`) and does an editable install with the dev extras, pinned by
`requirements-lock.txt`. `start.sh [--urdf robot.urdf ...]` runs the GUI viewer, the
same as `arm-sim view`; extra arguments pass straight through.

**Windows:** both scripts are bash and detect `.venv/Scripts/python.exe`, so they run
from Git Bash. If `python3.13` is not on your PATH there, use `PYTHON_BIN` (for
example `PYTHON_BIN=python ./scripts/setup.sh`). To do it by hand in PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -c requirements-lock.txt -e ".[dev]"
.venv\Scripts\arm-sim view            # the viewer
.venv\Scripts\arm-sim sweep --out results\x
```

The command examples below use the POSIX `.venv/bin/arm-sim`; on Windows write
`.venv\Scripts\arm-sim` (Git Bash accepts `.venv/Scripts/arm-sim`). `arm-sim --help`
lists every flag.

## Command overview

Every command takes `--urdf robot.urdf` (imported robot) or `--config file.yaml`
(default `configs/arm.yaml`, the built-in arm); the two are mutually exclusive.
`--package-root` (repeatable) requires `--urdf`. `--out` sets the output folder
(default `results/`, gitignored).

| Command | Use it to | Writes under `--out` |
| --- | --- | --- |
| `view` (default) | jog, save poses A/B, run sweeps, record loads (GUI) | timestamped sweep folders, `fea-*` |
| `sweep` | run the saved A -> B (or B -> A) move without a window | `<stamp>/` with `report.html`, CSV, PNG, YAML and MJCF snapshots |
| `export` | write the generated MJCF | `arm.xml` |
| `campaign` | find worst-case dynamic loads over many sweeps | `fea-*/`, `campaign-progress.csv` |
| `watch` | plot a running or finished campaign's convergence | nothing |
| `bounds` | static gravity, blocked-tip and (optional) clamped-tip loads, in seconds | `bounds-*/` |
| `fea-cases` | print a ready-to-enter FEA setup for one part | `fea_setup_<part>.csv` in each export |

Tests: `.venv/bin/pytest` (or `tests/test_urdf.py::test_name` for one). The suite
takes about 1-4 minutes because campaign tests run real sweeps. A stray
`MUJOCO_LOG.TXT` in the repo root is harmless test output; delete it.

## Import your own URDF

Place your exported robot in **`robots/my_robot/`**, keeping the URDF and its mesh
folders together. For example:

```text
robots/
  my_robot/
    robot.urdf
    meshes/
      base.stl
      arm.stl
```

Then run from this project folder:

```bash
./scripts/start.sh --urdf robots/my_robot/robot.urdf
```

The mesh paths inside the URDF must match that layout (for example,
`meshes/arm.stl`). A URDF in an `urdf/` subfolder can use `../meshes/arm.stl`.
You can also use an absolute URDF path anywhere on your computer; quote paths
containing spaces. The folder must be writable for the settings file.

For ROS `package://my_robot_description/meshes/...` paths, keep the package
folder's name and layout. Packages beside the URDF or its parent folders are
found automatically. If packages live elsewhere, specify the directory containing
them (or the package directory itself):

```bash
./scripts/start.sh --urdf robots/my_robot/robot.urdf --package-root /path/to/ros_ws/src
```

Repeat `--package-root` for multiple locations. `ROS_PACKAGE_PATH` is also supported.
The importer resolves local relative, absolute, and `file://` mesh paths.

- Supports **any number of revolute/continuous joints**, plus fixed links, in a
  fixed-base robot. Sliders, torque charts, and A/B reports adapt to the joint count.
  Prismatic, floating, planar, and mimic joints currently produce an explicit error.
- Supply a plain URDF, not Xacro: expand it first with
  `xacro robot.urdf.xacro > robot.urdf`. Meshes must be MuJoCo-compatible STL, OBJ,
  or MSH files; convert DAE meshes before importing.
- Supply valid mass/inertia for moving links and positive `effort` and `velocity`
  joint limits. URDF geometry/inertia are used directly; no OpenArm motor or tube
  masses are added. The root link is anchored at the origin by default.
- First launch creates **`robot.sim.yaml` beside `robot.urdf`**. This stores poses,
  gains, gravity, base placement, and simulation settings. Subsequent launches keep
  your saved controls. A and B initially match; save different poses before sweeping.
- **Edit source URDF** opens the robot geometry. **Edit all parameters** opens its
  `.sim.yaml`. **Reload YAML** reloads both. If you change the joint names/order/type,
  rename or remove the generated `.sim.yaml` and launch with `--urdf` again.
- Continuous joints use a configurable ±180° jogging window, with no physical
  joint stop. Torque ceilings come from URDF effort limits. The nominal chart line
  initially equals that ceiling: URDF does not provide a continuous motor rating.
  The torque-speed envelope is disabled; velocity limits are reported, not enforced
  as hard speed constraints. Tune the generated P/D gains for your robot.
- Contacts default to off. Set `simulation.contacts: true` to use the URDF collision
  geometry. `urdf.tip_link` selects the link tracked in reports; it defaults to the
  last leaf link listed in the URDF. `urdf.tip_offset_m` sets the tracked point on it.

Check an import or run saved poses without opening a window:

```bash
# Check an import without opening a window:
.venv/bin/arm-sim export --urdf robots/my_robot/robot.urdf --out results/import-check
# Run saved A/B poses without a window:
.venv/bin/arm-sim sweep --urdf robots/my_robot/robot.urdf
```

Exported MJCF files reference meshes by absolute path, so retain the source mesh
files when reopening exports on this computer. Running `./scripts/start.sh` without
`--urdf` still starts the configured OpenArm model.

### Your imported CAD arm

The replacement Onshape export in `robots/moving_asm` has seven real CAD links
and seven controlled joints, including `part_7` driven by J7. Start it with:

```bash
./scripts/start.sh --urdf robots/moving_asm/urdf/moving_asm.urdf
```

| Imported joints | Drive model | Rated / peak torque (Nm) | Rated / no-load speed (rpm) |
| --- | --- | --- | --- |
| revolute_1–2 | RobStride 03 | 20 / 60 | 100 / 195 |
| revolute_3–4 | RobStride 06 | 11 / 36 | 100 / 480 |
| revolute_5–7 | RobStride 00 | 5 / 14 | 100 / 315 |

The wrist drives use the [RobStride 00 manufacturer ratings](https://robstride.com/products/robStride00).
The saved `moving_asm.sim.yaml` includes the original
80% torque derating, speed-dependent torque envelope, gravity compensation, and
P/D gains. URDF effort limits contain the peak torques; velocity limits contain
the no-load speeds in rad/s. Mesh references use the sibling `meshes/` directory.
Keep this curated `.sim.yaml` to retain the rated/peak distinction and drive settings.
J1, J3 and J5 are continuous. J2 (−20° to 200°), J4 (±135°), J6 (−90° to 50°)
and J7 (−56° to 70°) are revolute with the real hardware limits. J7 rotates the real `part_7` CAD link on the
`rs00_1` actuator; its housing stays fixed to `part_6`. The exported joint origins,
axes, link geometry, and `part_7` mass/inertia are retained. The seventh link weighs
**0.427991 kg**. No placeholder tool is added. The report's tracked point is the
`part_7` link origin; set `urdf.tip_offset_m` to a measured tool-tip offset if needed.

The three RS00 housings use the catalog mass of 0.310 kg each, and both RS06
housings use 0.621 kg each. Their CAD COMs are retained and inertias are scaled
proportionally to mass (a uniform-density approximation). Other CAD inertias/masses
are retained. Total simulated mass, including the real seventh link, is **7.446340 kg**.
The original generated arm in `configs/arm.yaml` also uses 00 drives at J5–J7.

> **Self-collision:** links never collide in the physics (the CAD meshes are
> visual-only and contacts are off). Within the joint limits, about 4% of random
> poses put one part inside another, mostly `part_7` against `part_5` at certain
> wrist angle combinations. Per-joint limits cannot exclude these; the campaign's
> self-collision check (below) rejects them.

### Right-arm drumstick CAD

`robots/right_arm_assembly/urdf/right_arm_assembly.urdf` imports the same way
(`./scripts/start.sh --urdf robots/right_arm_assembly/urdf/right_arm_assembly.urdf`).
Its `.sim.yaml` excludes the `rs*` actuators from load rows; `base`, `screwon` and
`drumstick` report as the single part `base`, and a massless wrapper part `body_1`
carries the mount reaction (`body_1 <- world`). See "Loads for FEA" below.

### Run both arm models

The original arm's structural link masses now match the CAD URDF's `part_1` through
`part_7`, in joint order:

| Link | Mass (kg, excluding motors) |
| --- | --- |
| 1 | 0.758773 |
| 2 | 0.604800 |
| 3 | 0.738336 |
| 4 | 0.354697 |
| 5 | 0.163622 |
| 6 | 0.258995 |
| 7 | 0.427991 |

These are saved mass overrides, not a live sync with future URDF edits. The original
arm retains its sketch geometry, estimated COMs, and mass-scaled tube/bracket
inertias. Its separate 0.4 kg tool and motor masses are unchanged; the URDF uses its
own CAD inertial properties and has no separate placeholder tool. Matching link
masses alone does not make the models dynamically equivalent.

Run either viewer from this project folder (separate output folders keep reports apart):

```bash
# Original seven-DOF arm with the matched link masses:
./scripts/start.sh --out results/original

# Imported CAD arm:
./scripts/start.sh --urdf robots/moving_asm/urdf/moving_asm.urdf --out results/cad
```

In each viewer, save poses A and B, set **Move duration**, then click
**Run A -> B sweep** and **Open last report**. Each model keeps its own saved poses;
their joint frames differ, so equal joint angles do not necessarily give equal
physical poses.

To run the saved motions without opening a window:

```bash
.venv/bin/arm-sim sweep --out results/original
.venv/bin/arm-sim sweep --urdf robots/moving_asm/urdf/moving_asm.urdf --out results/cad
```

## Use

1. Move the joint sliders to pose A, let the arm settle, then click **Save A position**. Repeat for B.
2. Set **Move duration**, then click **Run A -> B sweep**.
3. Click **Open last report** to inspect torque and tracking error. Staying below the red torque limit alone does not mean the move succeeded.

- **Go to A / B:** move to a saved pose.
- **Jog speed:** change slider movement speed. **Zero arm / Randomize joints:** set new targets.
- **Link lengths / weights:** edit lengths (mm), link/actuator masses, end-effector mass, and payload (kg). Uncheck **Auto mass** to enter a link's mass excluding motors.
- **Apply + save:** rebuild and save geometry/masses, then close the editor and press **Resume**. **Apply** makes temporary changes.
- **Advanced P / D:** adjust each joint's gains. **Save controls** saves gains, speed, and duration.
- **Pause / Resume / Stop:** control motion.
- **Run B -> A sweep:** approach saved B, settle and verify the start pose,
  sweep to A over **Move duration**, then hold A. Both sweep directions save
  torque/tracking reports with the direction identified.
- **Reload YAML:** load external edits to `configs/arm.yaml`.

Charts: one per joint, top to bottom (scroll for larger robots). Yellow = nominal torque; red = maximum torque.

Camera: drag to rotate, right-drag to pan, scroll to zoom.

Keyboard (when no text box has focus): **1-9** select a joint, **Up/Down** jog it
2 degrees, **A / B** save the current pose as A / B, **Enter** run the A -> B sweep,
**[ / ]** change move duration by 0.5 s, **Space** pause/resume, **X** stop,
**P** open the last report, **E** edit the YAML, **F5** reload, **F8** save settings,
**Esc** quit.

Headless equivalent of the sweep buttons (saves the same report folder; exits with
status 2 if tracking exceeds tolerance):

```bash
.venv/bin/arm-sim sweep --out results/x [--duration 2] [--direction b-to-a]
```

It writes a timestamped folder under `--out` with `report.html`, the sampled CSV,
a PNG of the charts, the config snapshot and the MJCF; the last line printed is the
`report.html` path. `--duration` overrides `sweep.duration_s` for that run only.
`.venv/bin/arm-sim export --out results/x` writes only the generated MJCF as
`results/x/arm.xml`.

## Loads for FEA

### How loads are reported

All load tools (window recorder, `campaign`, `bounds`) use the same convention.
Loads are reported **per structural part**, not per MuJoCo body:

- **Actuators get no rows.** Bodies whose names match a glob in `fea.exclude_bodies`
  (in the config YAML; `['rs*']`, the RobStride housings, in both checked-in
  `.sim.yaml` files; empty by default, which makes every body a part) are actuators.
  They receive no load rows, but a link's mates with them are kept: the load a link
  carries from `rs03output` is a load on the link.
- **Fixed-joined parts merge.** Structural bodies joined with no joint become one
  part automatically. In the right arm, `base + screwon + drumstick` is the single
  part `base`, with combined mass, COM and inertia, and its loads are in `base`'s
  frame. The right-arm import also reports a massless wrapper part `body_1`;
  `body_1 <- world` is the mount reaction.
- **Rows** are labelled `<part> <- <other>`, one per mate: the proximal mate (toward
  the base) first, e.g. `l2 <- rs03output`, then distal ones, e.g. `l2 <- rs06_1`.
  Each row has Fx, Fy, Fz, Mx, My, Mz **acting on the part**, at the mate point, in
  the part's root-body frame (N, Nm, m), plus the magnitudes `force_N` and
  `moment_Nm`.
- **`<part> delta`** is the sum of all the part's mate wrenches, with moments moved
  to the proximal mate point: F = sum F_i, M = sum [M_i + (p_i - p_prox) x F_i]. It is
  what the part's own mass absorbs, m(a_COM - g) plus rotational inertia terms,
  minus any external contact load; at rest its force is minus the weight. Parts with
  a single mate (the merged `base`, `part_7`) and massless parts have no delta row.
  **Do not apply delta as an FEA load**: body loads already carry it. It is there to
  read.

These are observed rigid-body loads, not stress or deflection. Each wrench is a
simultaneous snapshot, so never combine extrema from different samples into one
load case.

### Which result governs what

- **`bounds` blocked** governs static strength (tip blocked, motors at peak torque,
  capped by `--max-contact-force-n`). With `--max-contact-moment-nm`, **clamped**
  adds a gripped tip that also reacts a moment; it mainly raises the distal links.
- **`bounds` gravity** is the everyday baseline and the fatigue **mean** load.
- **`campaign`** shows whether dynamics matter (for the CAD arm they do not: about
  1.5-3x gravity, below a 100 N tip contact), gives a conservative fatigue **range**,
  and motor/speed limits. Do not chase campaign convergence; rerun it when payload,
  geometry, gains or motors change. The CAD arm has no payload or gripper modelled yet.

### FEA data collection (window)

Expand **FEA data collection tools** in the left panel. Click **Start recording**,
then jog, use Go to A/B, or run either sweep. Click **Stop recording** to inspect
the results; this stops data capture, not arm motion. Pause/Resume suspends/resumes
sampling with simulation time. Data is captured every physics step (1,000 Hz with
the saved 1 ms timestep), independently of the live torque charts.

Select a **Part** and **Connection** (a mate, or `delta (sum of mates)`) to inspect
signed minimum/maximum Fx/Fy/Fz and Mx/My/Mz, plus force and moment magnitudes.
Values use the part's root-body axes. Click a peak to pause motion and preview its
recorded pose without changing the live simulation state. **Select load case** keeps
that complete timestamp for export. **Return to live pose** exits inspection; Resume
also exits inspection and resumes the live arm.

Exports are saved under the output folder in a timestamped `fea-*` directory:

- **Export peak summary:** `peaks.csv` with extrema and their timestamps.
- **Export all peak load cases / selected load cases:** `load_cases.json` contains
  complete simultaneous states and all rows; `load_cases.csv` contains their forces,
  moments and application points. With no selection, all distinct extrema timestamps
  are exported. With selections, those selected timestamps are exported.
- **Export full recording + load cases:** also includes `history.jsonl` (one complete
  state per line) and `interface_loads.csv` (one row per part mate or delta per sample).

In the CSVs `body` is the part and `other_body` is the mate (`delta` on delta rows).
In `peaks.csv`, `min_sample`/`max_sample` refer to each sample's `sample_index`
field, not its position in `load_cases.json`; look samples up by `sample_index`.

Every export also includes `metadata.json` (`schema_version` 2: units, frames, mass
and inertia, row definitions, recording checks, limitations, plus `parts` with each
part's combined mass/COM/inertia and `excluded_bodies`), a configuration snapshot
(`config.yaml`), the exact compiled `model.mjb`, and `source_model.xml`. The compiled
model preserves live controller gains. Source XML is the model's original build
source; use the compiled model/configuration snapshot for the exact recorded settings.

Moments are about the listed application points. Gravity and physical COM/angular
accelerations are supplied for consistent inertial loading. Contact/perturbation
resultants are exported separately at each body's COM; they do not specify contact
pressure distributions. CAD surface mapping, supports, materials and mesh checks
remain FEA tasks.

The imported CAD arms keep each `part_*` link separate from its RobStride housing
(the housing is excluded from rows, its mates stay). The original generated model
aggregates each structure with its attached motor housing and any tool/payload; its
exported loads describe that assembly, not an isolated machined linkage.

Changing geometry or reloading ends the current recording and retains its data.
Controller gain edits are disabled while recording. A capacity limit (displayed
in the panel, sized by model complexity) stops recording without dropping earlier
samples. Export before **New recording (replace)** if you want to retain the old
recording. Closing the window saves an unexported recording automatically.

### Worst-case loads across sweeps (`campaign`)

To find the largest part loads that any A -> B sweep can produce, run a campaign.
It runs the saved A/B sweep in both directions, then random A/B poses within the
joint limits with log-uniform move durations. Next it refines the worst sweep found
for the force and moment magnitude of **every reported row** (each mate and delta):

```bash
.venv/bin/arm-sim campaign --samples 200 --iterations 20 --min-duration 0.1 --max-duration 5
.venv/bin/arm-sim campaign --urdf robots/moving_asm/urdf/moving_asm.urdf --out results/cad-campaign
```

Options: `--samples` random sweeps (default 200), `--iterations` refinement sweeps per
row and magnitude (default 20), `--min-duration/--max-duration` (0.1 / 5 s), `--seed`
(repeatable runs), `--workers` (spawned worker processes, default CPU count),
`--include-failed`, `--max-overlap-mm`, `--allow-self-collision`, `--live`. It prints
a table of each row's maximum force and moment and the sweeps that set them, then the
output folder. The run takes minutes; the sweep count grows with the number of rows.

Each sweep starts at rest at A, settles, sweeps and holds. Only the sweep and hold
are measured. By default a sweep counts only if it tracks within
`sweep.tracking_tolerance_deg`. Commands the motors cannot follow end in
joint-stop impacts, whose loads depend on MuJoCo's constraint softness, not on the
drives. `--include-failed` counts them anyway; leave it off for design loads.

The campaign rejects self-colliding motion. A copy of the model with every
geom colliding (CAD meshes as convex hulls, plus the floor) checks each random
A -> B path before simulating it, then the simulated motion every 5 ms, which
catches overshoot. Colliding paths are redrawn and colliding sweeps are stopped
and never counted. Body pairs touching in the zero pose are treated as designed
overlaps and ignored. `--max-overlap-mm` (default 0) allows small overlaps; hulls are
conservative, so 1-2 mm accepts near-misses they misjudge. `--allow-self-collision`
turns the check off. Contacts never act in the physics, so loads are unchanged.

The result is an `fea-*` folder in the same format as the window's export:
`peaks.csv` (each row's extrema plus `min_sweep`/`max_sweep`), `load_cases.json/csv`
(the complete simultaneous state at every extremum), `metadata.json`, `model.mjb`,
`sweeps.csv` (every sweep's A/B poses, duration, tracking result and any collision),
`campaign.json` (settings) and `convergence.csv`. These are the largest loads the
search **found**, not a proven global maximum. Jogging and Go to A/B moves are not
searched.

**Convergence.** While it runs, the campaign appends to `<out>/campaign-progress.csv`
(one line per 10 counted sweeps plus a final one), copied into the export as
`convergence.csv`. Columns: `counted,sweeps,elapsed_s,stage`, then for each row and
objective `<label>|force_N` and `<label>|force_N|Fx_N` ... `|Mz_Nm` (the full wrench
of the sample that set that peak), and likewise for `moment_Nm`.

Add `--live` to the campaign, or run `arm-sim watch` on its own, to open a plot
window (it blocks until closed):

```bash
.venv/bin/arm-sim campaign --samples 200 --iterations 20 --live
.venv/bin/arm-sim watch [--results <campaign-progress.csv, convergence.csv or folder>]
```

`watch` opens the newest progress or convergence CSV under `--results` (default
`--out`), running or finished. Left: peak force, peak moment and largest-rise panels,
one colour per part (solid = proximal mate, dashed = distal, dotted = delta). Right:
two tables with the Fx..Mz record state at each peak force and peak moment. When the
peaks stop rising, more sampling is not buying you anything.

**Campaign replay (window).** To watch the sweeps behind the peaks, open the viewer
for the same robot (`./scripts/start.sh`, with `--urdf` if the campaign used it) and
expand **Campaign replay**. **Load latest campaign** finds the newest campaign for
this robot anywhere under the output folder; **Load folder** opens the `fea-*` path
typed below it. Choose a load (`force_N`, `moment_Nm`, Fx...Mz) to list each mate's
and delta's campaign minimum and maximum. Click a value to replay the sweep that
produced it. The replay starts at rest at A, exactly as in the campaign, and with
**Pause at peak** it stops on the step where that peak occurred; Resume finishes
the sweep and saves its usual report. Replays never change your saved A/B or move
duration. Loading warns if the model, drive or gain settings differ from the
campaign's, because the replay uses the live model.

### Static bounding loads (`bounds`)

The campaign measures loads during successful sweeps. Structural design must also
cover events it excludes on purpose. `bounds` computes two static load cases (three
with `--max-contact-moment-nm`) that need no random sweeps. **Always pass
`--max-contact-force-n`:**

```bash
.venv/bin/arm-sim bounds --max-contact-force-n 200
.venv/bin/arm-sim bounds --urdf robots/moving_asm/urdf/moving_asm.urdf --max-contact-force-n 200 --out results/cad-bounds
.venv/bin/arm-sim bounds --urdf robots/moving_asm/urdf/moving_asm.urdf --max-contact-force-n 100 --max-contact-moment-nm 10
```

- **gravity:** the arm holding a pose against gravity alone, searched for the worst
  pose of every row.
- **blocked:** an obstacle stops the tool tip while the motors push. The obstacle
  force is tangent to one joint's rotation and as large as every motor can resist
  at full peak torque (`--derated` uses the YAML derating), capped at
  `--max-contact-force-n`. `--contacts all` also blocks every moving body origin
  (a pinch along the arm), and `--min-lever-m` (default 0.05 m) skips points
  closer than that to the blocked joint's axis.
- **clamped** (only with `--max-contact-moment-nm`, which needs
  `--max-contact-force-n` too): the tip grips a fixed object, so it reacts a force
  and a moment in any direction. The search covers load directions as well as
  poses; each load is the largest along its direction that every motor can hold
  (several may sit at their caps at once), with |F| and |M| within the two caps. It
  is static, not "all motors at max". Results scale with both caps: take the moment
  cap from the wrist motors' torque or the gripper's slip limit. Always at the tip,
  even with `--contacts all`.

Set `--max-contact-force-n` to a design value: a requirement, collision-detection
threshold or tip-force rating. Without it, aligned poses (like a straight arm
pressing on a table, or a toggle clamp) let small torques resist huge forces, so
the blocked loads have no real ceiling and grow with more sampling (the command
prints a warning). Blocked loads scale almost linearly with the cap (about cap +
weight in force, cap x reach in moment). For reference, the CAD arm's motors resist
a median of about 200 N at the tip across random poses and directions, and about
125 N in its weakest direction at full reach.

All cases reject self-colliding poses (`--max-overlap-mm`, `--allow-self-collision`)
and poses a motor cannot hold. They start with `--samples` random poses (default
2000; clamped also scores 12 random load directions per pose), then refine each
row's worst force and moment over `--iterations` poses (default 300; clamped
perturbs the load direction too); `--workers` and `--seed` apply. The run takes
seconds. Results go to `bounds-*/gravity/`, `bounds-*/blocked/` and, if enabled,
`bounds-*/clamped/`, each in the usual load-case format (`peaks.csv`,
`load_cases.*`, `metadata.json`, `model.mjb`; blocked and clamped samples also list
the contact body, point and force, clamped ones the contact moment too).
`bounds-*/summary.csv` compares them and `bounds.json` records the settings and
assumptions. Combine these with the campaign's dynamic peaks and apply your safety
factor. Velocities and accelerations are zero in every case.

### Setting up a part in FEA (`fea-cases`)

`fea-cases` turns any load-case export (a campaign `fea-*` folder, a `bounds-*`
folder, or one of its `gravity`/`blocked`/`clamped` subfolders) into ready-to-enter FEA setups
for one structural part:

```bash
.venv/bin/arm-sim fea-cases --results results/cad-bounds/bounds-<stamp> --part part_3
```

`--part` is a structural part name from the export's `metadata.json` `parts` list
(for the right arm, `base` is the merged `base + screwon + drumstick`; `screwon`,
`drumstick` and `rs*` actuators are rejected with a list of valid parts). Exports
from before the per-part change (schema 1) are refused: rerun the campaign or bounds.

For each worst case of that part's mates (peak force and moment magnitudes;
`--all-extremes` adds every Fx...Mz extreme), it prints:

- **FIX:** the mate to constrain, by default the proximal one (toward the base;
  `--fixed <other body>` picks another, e.g. `rs06_1`). Its load is listed only as
  a check: FEA computes that reaction.
- **APPLY:** a remote force and moment from each other mate, about the listed
  point, plus any contact force (and, clamped, moment) acting on this part.
- **BODY:** the part's mass and the acceleration field to apply (gravity minus the
  COM acceleration), plus angular velocity and acceleration for moving cases. This
  is where the delta load enters; do not also apply the delta row.
- **Force balance residual:** should be about zero; it confirms the case and frames
  are consistent.

Everything is in the part's root-body frame (the URDF/CAD part frame), in N, Nm and
m. Each case is one instant, so never combine rows from different samples. The same
table is saved as `fea_setup_<part>.csv` in each export folder.
