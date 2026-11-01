# OpenArm MuJoCo

## Start

Install Python 3.13 (`brew install python@3.13` on macOS), then run from the project folder:

```bash
./scripts/setup.sh   # first time only
./scripts/start.sh
```

For later launches, just run `./scripts/start.sh`.

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
All seven joints are continuous. J7 rotates the real `part_7` CAD link on the
`rs00_1` actuator; its housing stays fixed to `part_6`. The exported joint origins,
axes, link geometry, and `part_7` mass/inertia are retained. The seventh link weighs
**0.427991 kg**. No placeholder tool is added. The report's tracked point is the
`part_7` link origin; set `urdf.tip_offset_m` to a measured tool-tip offset if needed.

The three RS00 housings use the catalog mass of 0.310 kg each, and both RS06
housings use 0.621 kg each. Their CAD COMs are retained and inertias are scaled
proportionally to mass (a uniform-density approximation). Other CAD inertias/masses
are retained. Total simulated mass, including the real seventh link, is **7.446340 kg**.
The original generated arm in `configs/arm.yaml` also uses 00 drives at J5–J7.

> **TODO: set the CAD arm's real joint limits.** The Onshape mates have no
> limits, so every joint is exported as continuous and the simulator allows
> ±180° on all seven. Links never collide in the simulation (the CAD meshes
> are visual-only and contacts are off). A convex-hull check found that about
> 79% of random poses in that range put one part inside another. Campaign
> results for this arm therefore include motions the real arm cannot make;
> do not use them for FEA until the limits are set.
>
> To find the limits, check each joint for whichever stops it first:
>
> - **CAD:** rotate each joint in the Onshape assembly until parts interfere,
>   with neighboring joints at typical angles (some limits depend on pose).
> - **Hardware:** hard stops, cable wind-up (no slip rings, so roughly ±180°
>   or less), and the table or base mount.
> - **Motors:** any RobStride firmware position limits.
> - **Starting point:** the OpenArm v1 limits in `configs/arm.yaml` (J1 −200
>   to 80°, J2 −190 to 10°, J3 ±90°, J4 0 to 140°, J5 ±90°, J6 ±45°, J7 ±90°).
>   Verify them against this CAD; zero positions may differ.
>
> Enter them as `limits_deg` for each joint in
> `robots/moving_asm/urdf/moving_asm.sim.yaml`, or set Onshape mate limits and
> re-export. Then rerun the campaign.

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

### FEA data collection

Expand **FEA data collection tools** in the left panel. Click **Start recording**,
then jog, use Go to A/B, or run either sweep. Click **Stop recording** to inspect
the results; this stops data capture, not arm motion. Pause/Resume suspends/resumes
sampling with simulation time. Data is captured every physics step (1,000 Hz with
the saved 1 ms timestep), independently of the live torque charts.

Select a **Link / body** and **Connection** to inspect signed minimum/maximum
Fx/Fy/Fz and Mx/My/Mz, plus force and moment magnitudes. Values use the receiving
body's local axes. Click a peak to pause motion and preview its recorded pose
without changing the live simulation state. **Select load case** keeps that
complete timestamp for export. **Return to live pose** exits inspection; Resume
also exits inspection and resumes the live arm.

Exports are saved under the output folder in a timestamped `fea-*` directory:

- **Peak summary:** `peaks.csv` with extrema and their timestamps.
- **Load cases:** `load_cases.json` contains complete simultaneous states and
  all interface loads; `load_cases.csv` contains their local forces, moments,
  and application points. With no selection, all distinct extrema timestamps
  are exported. With selections, those selected timestamps are exported.
- **Full recording:** also includes `history.jsonl` (one complete state per line)
  and `interface_loads.csv` (one receiving-body interface per sample).

Every export includes `metadata.json` with units, frames, mass/inertia, interface
definitions, recording checks and limitations; a configuration snapshot; the exact
compiled `model.mjb`; and `source_model.xml`. The compiled model preserves live
controller gains. Source XML is the model's original build source; use the compiled
model/configuration snapshot for the exact recorded settings.

**These are observed rigid-body loads, not stress or deflection results.** Each
interface is represented on both receiving bodies with opposite forces; use only
the loads acting on the body you are analyzing. Moments are about the listed
application points. Gravity and physical COM/angular accelerations are supplied
for consistent inertial loading. Contact/perturbation resultants are exported
separately at each body's COM; they do not specify contact pressure distributions.
CAD surface mapping, supports, materials and mesh checks remain FEA tasks.

The imported CAD arm preserves fixed motor attachments and individual `part_*`
bodies. The original generated model aggregates each structure with its attached
motor housing and any tool/payload; its exported body loads describe that assembly,
not an isolated machined linkage. Do not combine component extrema from different
times into one load case.

Changing geometry or reloading ends the current recording and retains its data.
Controller gain edits are disabled while recording. A capacity limit (displayed
in the panel, sized by model complexity) stops recording without dropping earlier
samples. Export before **New recording (replace)** if you want to retain the old
recording. Closing the window saves an unexported recording automatically.

For a reverse headless torque sweep:

```bash
.venv/bin/arm-sim sweep --direction b-to-a
```

### Worst-case loads across sweeps

To find the largest interface forces and moments that any A -> B sweep can
produce, run a campaign. It runs the saved A/B sweep in both directions, then
random A/B poses within the joint limits with log-uniform move durations. Next it
refines the worst sweep found for each link's force and moment magnitude:

```bash
.venv/bin/arm-sim campaign --samples 200 --iterations 20 --min-duration 0.1 --max-duration 5
.venv/bin/arm-sim campaign --urdf robots/moving_asm/urdf/moving_asm.urdf --out results/cad-campaign
```

Each sweep starts at rest at A, settles, sweeps and holds. Only the sweep and hold
are measured. By default a sweep counts only if it tracks within
`sweep.tracking_tolerance_deg`. Commands the motors cannot follow end in
joint-stop impacts, whose loads depend on MuJoCo's constraint softness, not on the
drives. `--include-failed` counts them anyway. Sweeps run in parallel on all CPU
cores (`--workers`); `--seed` makes a campaign repeatable. A sweep takes a few
seconds of computer time per worker, so the default campaign (about 520 sweeps
for the OpenArm model) takes several minutes.

The result is an `fea-*` folder in the same format as the window's load-case
export. `peaks.csv` gives each interface's extrema, plus the sweep
(`min_sweep`/`max_sweep`) and load case that produced them. `load_cases.*` holds
the complete simultaneous state at every extremum. `sweeps.csv` lists every
sweep's A/B poses, duration and tracking result; `campaign.json` lists the
settings. These are the largest loads the search **found**, not a proven
global maximum. Run more samples or iterations until the peaks stop increasing.
Jogging and Go to A/B moves are not searched.

To watch the sweeps behind the peaks, open the viewer for the same robot
(`./scripts/start.sh`, with `--urdf` if the campaign used it) and expand
**Campaign replay**. **Load latest campaign** finds the newest campaign for this
robot anywhere under the output folder; **Load folder** opens the `fea-*` path
typed below it. Choose a load (`force_N`, `moment_Nm`, Fx...Mz) to list each
connection's campaign minimum and maximum. Click a value to replay the sweep that
produced it. The replay starts at rest at A, exactly as in the campaign, and with
**Pause at peak** it stops on the step where that peak occurred; Resume finishes
the sweep and saves its usual report. **Both sides** also lists each connection's
opposite receiving-body view. Replays never change your saved A/B or move
duration. Loading warns if the model, drive or gain settings differ from the
campaign's, because the replay uses the live model.
