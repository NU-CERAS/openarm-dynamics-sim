# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MuJoCo-based dynamics sim for a 7-DOF OpenArm-style arm with RobStride actuators: joint jogging, timed A/B pose sweeps with torque/tracking reports, URDF import of arbitrary fixed-base rotary robots, and an FEA load recorder. Package `arm_design` lives under `src/`; entry point is `arm-sim` (`arm_design.cli:main`).

## Commands

```bash
./scripts/setup.sh                      # create .venv (python3.13, override with PYTHON_BIN) + editable install with dev extras, pinned by requirements-lock.txt
./scripts/start.sh [--urdf path.urdf]   # GUI viewer (= python -m arm_design.cli view)

.venv/bin/pytest                        # all tests
.venv/bin/pytest tests/test_urdf.py::test_name   # single test

# Headless (no window)
.venv/bin/arm-sim sweep [--urdf ...] [--direction b-to-a] [--duration S] --out results/x   # exits 2 if tracking fails
.venv/bin/arm-sim export [--urdf ...] --out results/x   # writes generated MJCF arm.xml
.venv/bin/arm-sim campaign [--urdf ...] --samples N --iterations K [--workers W] [--include-failed] [--live]   # worst-case loads over many sweeps; --live plots peaks every 10 counted sweeps
.venv/bin/arm-sim watch [--results <campaign-progress.csv, convergence.csv or folder>]   # live/finished campaign convergence plot
.venv/bin/arm-sim bounds [--urdf ...] --max-contact-force-n F   # static gravity + blocked-tip bounding cases (seconds)
.venv/bin/arm-sim fea-cases --results <fea-* or bounds-*> --part part_3   # per-part FEA setup: fixed face, remote loads, body loads
```

`--config` and `--urdf` are mutually exclusive; `--package-root` (repeatable) requires `--urdf`. `results/` is gitignored output.

## Architecture

**Config is the single source of truth.** Everything is driven by a YAML dict (`configs/arm.yaml` for the built-in arm, or `<robot>.sim.yaml` for imports). `config.validate` checks it; `config.save_controls`/`save_sweep` write poses, gains and geometry back to it from the UI. Joint angles in YAML are degrees; units are in field names (`_mm`, `_kg`, `_deg`, `_nm`).

**Two model builders, one `Arm` dataclass.** `model.build_arm(c)` dispatches on `c['model_type']`:
- Default: generates MJCF from scratch — analytical hollow-tube + bracket inertia (`structure`), motor housings as cylinders on the parent side of each joint, tool/payload boxes on J7. Axes and origin rotations come from the pinned OpenArm v1 URDF in `references/`.
- `urdf`: `urdf.py` parses the URDF, resolves mesh paths (relative, absolute, `file://`, `package://` via package roots / `ROS_PACKAGE_PATH`), and compiles through MuJoCo's native URDF compiler. `prepare_urdf_config` generates the `.sim.yaml` beside the URDF on first launch; `build_urdf_arm` refuses to build if joint names/types/limits drift from the saved YAML or peak torque exceeds URDF effort. Only revolute/continuous/fixed joints are supported.

Both return `Arm(cfg, model, xml, structural_masses)`; downstream code is model-agnostic and adapts to any joint count.

**MuJoCo owns all physics.** `simulation.py` only wraps `mj_step`: joint PD + velocity feedforward + optional gravity feedforward, where gravity comp is added *inside* each actuator's force clamp. `torque_limit` applies 0.8 derating and the optional speed envelope. `Sweep` runs approach → settle → smooth A↔B move → hold; `run_headless` is the CLI path.

**Layering for the GUI.** `experiment.Experiment` holds all window-independent state/actions (load/reload, goto, sweeps, saving, FEA recording) so buttons and keyboard shortcuts share logic. `viewer.py` is a single GLFW window with MuJoCo rendering + an ImGui (`imgui_bundle`) panel; `torque_charts.py` draws per-joint charts. `Experiment.load` builds the new arm before swapping, so invalid YAML leaves the running sim intact.

**Outputs.** `reporting.save_result` writes a timestamped folder (CSV, YAML snapshot, MJCF, PNG, `report.html`). `fea.FEARecording` samples every physics step via `mj_rnePostConstraint`, transforms interface wrenches into each receiving body's local frame, and exports `fea-*` directories (peaks, load cases, optional full history, `model.mjb`, metadata). These are rigid-body loads, not stresses. `campaign.py` runs many recorded A -> B sweeps (random, then (1+1)-ES refinement per interface force/moment magnitude) in spawn-based worker processes. It keeps only the extrema samples (`Peaks`) and exports them through `CampaignRecording`, an `FEARecording` subclass, in the same `fea-*` format. Only sweeps within tracking tolerance count by default. `collision.CollisionChecker` (a recompiled contact-enabled copy of the model; convex hulls) rejects self-colliding planned paths and stops sweeps whose simulated motion collides. `bounds.py` adds static cases (worst gravity pose; tip blocked with motors at peak torque, capped by `--max-contact-force-n`), exported via `FEARecording.export(..., name=)` into `bounds-*/gravity|blocked/`. The viewer's **Campaign replay** panel (`campaign.load_results`, `Experiment.replay_sweep`) reruns a campaign sweep from rest at A at time zero, so it reproduces recorded samples exactly and can pause on the peak's step.

## Gotchas

- **Linux rendering:** the imgui-bundle Linux wheel vendors a private libglvnd, so its native renderer can't see GLFW's context. `viewer.py` uses the PyOpenGL `ProgrammablePipelineRenderer` on Linux and forces PyOpenGL to GLX (it picks EGL under WSLg). Don't revert this.
- `robots/**/*.sim.yaml` is gitignored except `robots/moving_asm/urdf/moving_asm.sim.yaml`, which is hand-curated (rated vs. peak torque, speed envelope, gains) — don't regenerate it.
- Link masses in `configs/arm.yaml` intentionally match `part_1..part_7` in `moving_asm.urdf`; `tests/test_mujoco.py` asserts this.
- Actuator gear is 1: torques are output-shaft Nm; RobStride gear ratios are metadata only.
- `.sim.yaml` files store `urdf.path` relative to themselves (`config.portable`) and keep their leading comments (`config.header`) on every save, so committed settings work from any checkout. `load_config` resolves the path back to absolute in memory.
- Contacts are off by default (`simulation.contacts`) so coarse geometry doesn't support the arm.
- The CAD arm's meshes import as visual-only (`contype 0`), so `simulation.contacts: true` does not make its links collide. MuJoCo builds body collision filters and mesh convex hulls at compile time: to check collisions, recompile with `contype`/`conaffinity` set (as `CollisionChecker` does); editing `model.geom_contype` afterwards silently does nothing.
- CAD arm joint limits are the real hardware limits: J1/J3/J5 continuous, J2 −20..200°, J4 ±135°, J6 −90..50°, J7 −56..70°. J7's hardware limit is −70°, but `part_7`'s hull overlaps `part_5` below −56° with the other joints at zero. About 4% of random poses still self-collide (wrist combinations, `part_7` folding back to the shoulder); only the collision check removes those.
- Don't re-export `moving_asm.urdf` from Onshape just to change limits: it carries hand edits (RS00/RS06 catalog housing masses with scaled inertia, effort = RobStride peak torque, velocity = no-load speed). Edit the `<joint>`/`<limit>` elements directly and update the matching `type`/`limits_deg` in `moving_asm.sim.yaml`, or `build_urdf_arm` rejects the mismatch. Continuous joints have no physical stop in MuJoCo; revolute ones do.
- Campaign counts only sweeps that track within `tracking_tolerance_deg`: untrackable commands end in joint-stop impacts whose loads depend on MuJoCo constraint softness (a 0.22 s sweep once gave a bogus 1650 N). Keep that default.
- The uncapped `bounds` blocked case has no ceiling: near aligned poses (toggle-clamp geometry) small torques resist unbounded forces, so results grow with sampling. Always pass `--max-contact-force-n`; blocked loads scale almost linearly with it (≈ cap + weight in force, ≈ cap × reach in moment).
- Design roles: `bounds` blocked governs static strength; `bounds` gravity is the everyday baseline and the fatigue mean load; the campaign shows dynamics don't govern (CAD arm: about 1.5–3× gravity, below a 100 N tip contact) and gives a conservative fatigue range and motor/speed limits. Don't chase campaign convergence; rerun it when payload, geometry, gains or motors change. The CAD arm has no payload or gripper modelled yet.
- In exports, `peaks.csv` `min_sample`/`max_sample` refer to each sample's `sample_index` field, not its position in `load_cases.json` (tied samples may be unreferenced). Look samples up by `sample_index`.
- The full test suite takes about 1–4 minutes (campaign tests run real sweeps). A stray `MUJOCO_LOG.TXT` (instability warnings from deliberately violent test sweeps) can appear in the repo root; delete it, never commit it.
- Modeling assumptions are documented in `docs/MODEL.md`; data provenance (OpenArm pinned revision, RobStride specs) in `docs/SOURCES.md`. Update them when changing model assumptions or motor data.

## Style

Code is deliberately compact: short names, dense one-liners in places, few comments (module docstrings state each file's responsibility). Match that rather than expanding it.
