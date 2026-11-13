# Model assumptions

MuJoCo integrates a fixed-base seven-revolute-joint rigid arm, using implicitfast
at a default 1 ms timestep. Position actuators have unit transmission gear, so
actuator force and generalized hinge torque are both output-shaft Nm. Published
RobStride gearbox ratios are metadata and are not multiplied into output torque.

The local axes, origin rotations, and provisional limits are retained from the
pinned OpenArm v1 left-arm URDF in `references/`. The base has a -90° X rotation;
J2 also has a -90° X origin rotation. These transformations matter as much as the
local axes. Custom translations are seven sketch spans along local +Z. This is a
v1.1-style model, not a verified import of the team's v1.1 CAD.

Structural inertia comes from analytical hollow square tubes and spherical bracket
approximations, aggregated by the parallel-axis theorem. Visual tube boxes are
solid-looking but never supply the structural inertia. A link override replaces
the tube+bracket structure only. If COM alone is overridden, the estimated inertia
is interpreted about that new COM; supply the CAD inertia too for faithful dynamics.

Each entire motor housing is represented as a cylinder on the parent side of its
joint. J1's housing is fixed to the base. Actual motor rotors/gear internals are not
separately modeled; joint armature defaults to zero and can be set to a known
reflected inertia. Gripper and payload are separate boxes attached to J7. Payload
COM is offset from the tool tip, in J7 coordinates.

The controller is joint position PD with desired-velocity feedforward and optional
gravity feedforward. Gravity compensation is added INSIDE each actuator's force
clamp, so it cannot supply unlimited external support. It uses the simulated mass
model perfectly; real firmware, delays, friction uncertainty, and calibration can
produce different tracking. No custom physics integration is performed.

`arm-sim bounds` searches static holding poses and records a gravity-only case plus
blocked-tip force cases, capped by `--max-contact-force-n`. Supplying
`--max-contact-moment-nm` also searches clamped-tip wrench cases: a fixed grasp may react
any force and moment direction, bounded by both caps, while each motor stays within its
peak torque cap. This is a sampled pose and load-direction search, so it finds the largest
loads encountered, not a proven maximum. The clamped comparison contains the blocked-tip
cases when using the default `--contacts tip`; with `--contacts all`, blocked cases can
also act on moving-body origins, which a tip clamp does not reproduce. The moment cap must
come from the gripper's slip limit or the wrist motors' capability; it is an input
assumption, not a value inferred by the model. These cases assume zero velocity and
acceleration and do not cover a constrained dynamic transient.

Peak/rated output torques are derated by 0.8 by default. If `speed_envelope` is true,
limits are constant up to rated speed, then linearly fall to zero at no-load speed.
This symmetric envelope is a screening assumption, not a measured RobStride curve
or a guarantee of conservative braking behavior. Peak caps are enforced, while
rated caps are reported. There is no verified peak-duration limit or thermal model.
Time above the rated proxy is reported without inventing an allowable duration.
Power is mechanical torque × speed, not electrical input power.

Contacts are OFF by default to keep coarse housing/tube geometry from supplying
unintended support during motor-load experiments. They can be enabled in YAML;
then MuJoCo uses approximate primitive collision geometry, including a floor.
Parent/child contact exclusions follow MuJoCo defaults. This is not CAD clearance
validation. With contacts enabled, contact forces can support the arm and reduce
motor torque. Inspect contact counts and the scene before comparing designs.

Applied torque, speed, target, angle, and tip position are recorded at the same
pre-integration timestamp. Approach/settle data is saved but excluded from the
summary's sweep+hold window. Acceleration is set by the smooth trajectory duration;
changing duration changes both speed and inertial load. Each run snapshots its
parameters. The window allows live P/D edits between runs. Those values are captured in the
run snapshot and generated XML; use Save controls to persist them to the main YAML.
Geometry and inertial changes require editing YAML and reloading.

The optional FEA recorder calls MuJoCo's `mj_rnePostConstraint` after the same
pre-integration forward pass. It translates parent/child interaction wrenches from
the subtree COM reference to each connection point, then rotates them into each
receiving part's local frame. Hinge connections use their joint anchor; fixed
connections use the child body origin. Actuator torque is already represented in the
interaction loads and must not be added to those loads a second time.

Loads are reported per structural part, not per MuJoCo body. Bodies whose names match
a glob in `fea.exclude_bodies` (the CAD arms use `rs*`, the RobStride housings) are
actuators: they receive no load rows, but a link's mates with them are kept, since
those are the loads the link carries. With the key absent or empty every body is a
part. A structural body attached with no joint to another structural body merges into
it (the right arm's `base` + `screwon` + `drumstick`): the merged part has no internal
rows, its loads and mate points are in its root body's frame (`base`), and metadata
`parts` gives its combined mass, COM and inertia about the COM in that frame. The world
is never merged into; the mate to it is the base reaction.

Each part has one row per mate, a wrench `[Fx, Fy, Fz, Mx, My, Mz]` acting ON the part
at the mate point: first its proximal mate (toward the base), then its distal ones.
Where two parts mate directly with no actuator
between them, as in the built-in generated arm, each part has its own row for that mate.
The campaign refines the force and moment magnitude of every row and logs the full
wrench of each record-setting sample (see `convergence.csv`).

Body COM velocity and physical acceleration are recorded in world axes, with
angular components first. MuJoCo's RNE gravity offset is removed from acceleration
for moving bodies; bodies welded to the world have zero physical acceleration.
External contact/perturbation resultants are recorded about each body's COM in
local axes, separately from interface loads and gravity. They preserve the net
wrench but not contact pressure distributions. These kinematic arrays stay per
MuJoCo body. Fixed URDF motor bodies remain distinct bodies (so a link's mass excludes
its actuators); the generated model's aggregate mass/inertia does not isolate motor
mount forces from its structural link.

The FEA recorder retains all phases, including approach and settling, and is
independent of the sweep report's narrower summary window. The A/B definitions
are never swapped in configuration: reverse sweeps select B as their start and A
as their destination, with the same preparation, trajectory and hold behavior.

No structural stress, deflection, fatigue, thermal, firmware, or hardware-control
prediction is implied by the joint torque measurements or FEA load exports.


## Dynamic restrained-hand enumeration

`restrained` is distinct from static `bounds clamped`: a six-dimensional site weld fixes
the initial hand transform to world while allowing the arm's internal freedom to move.
Each independent trial ramps one combination of direct actuator torques from rest; no
position PD or gravity compensation is applied. Full output-shaft peak ratings and the
configured speed envelope bound torque. The numerical weld uses solimp 0.999/0.9999,
torquescale 1 m and configurable solref time constant (default 0.005 s, damping ratio 1).
It represents neither measured compliance nor a force/moment slip threshold. Reactions
are extracted through RNE and exported at the tip, including the independent moment.

Trial duration, ramp and solver parameters must be recorded and varied for sensitivity
studies. Excess weld drift, detected self/floor collision, joint-limit contact and
instability reject a trial. Accepted peaks do not include joint-stop impacts. Rejected
prefixes are retained for diagnosis. Arbitrary torque combinations are integrated,
not falsely declared static equilibrium. Each sampled pose defines its own fixed grasp;
no common-grasp inverse-kinematics search or extra elbow fixture is implied.

Campaigns now seed exclusively random sweeps. Exact full-step histogram counts are
retained separately for random/refinement and accepted/rejected trials. Time weighting
and one-peak-per-trial weighting answer different questions; neither identifies a real
operating duty cycle without representative input motion probabilities. Force and moment
components are part-local and snapshots remain simultaneous for FEA.


Aggressive free-motion campaigns optionally enable `drive.trajectory_feedforward`:
MuJoCo inverse dynamics computes the acceleration/velocity/gravity feedforward
at the actual joint state and the quintic reference acceleration. That demand and
PD feedback pass through the same actuator torque caps; `qfrc_applied` stays zero.
This tests a model-based controller with exact model knowledge, not measured hardware
tracking performance. The ordinary controller remains gravity-feedforward PD.
Duration search does not alter motor ratings or prove global time optimality. Both
motion directions start independently at rest; joint-stop/collision loads are excluded.

Aggressive screening uses at most a 0.5 ms timestep. Record-setting sweeps are rerun
at 0.25 ms and 0.125 ms (or finer if the initial step was already smaller), with
acceptance and peak changes saved separately. Finite-step screening values are not
silently replaced by reruns; inspect `timestep-validation.json` for final design use.
