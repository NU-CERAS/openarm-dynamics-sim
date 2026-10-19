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
- **Reload YAML:** load external edits to `configs/arm.yaml`.

Charts: one per joint, top to bottom (scroll for larger robots). Yellow = nominal torque; red = maximum torque.

Camera: drag to rotate, right-drag to pan, scroll to zoom.
