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
receiving body's local frame. Hinge connections use their joint anchor; fixed
connections use the child body origin. Equal/opposite interface views are exported
for the two receiving bodies. Actuator torque is already represented in the
interaction loads and must not be added to those loads a second time.

Body COM velocity and physical acceleration are recorded in world axes, with
angular components first. MuJoCo's RNE gravity offset is removed from acceleration
for moving bodies; bodies welded to the world have zero physical acceleration.
External contact/perturbation resultants are recorded about each body's COM in
local axes, separately from interface loads and gravity. They preserve the net
wrench but not contact pressure distributions. Fixed URDF motor bodies remain
distinct; the generated model's aggregate mass/inertia does not isolate motor
mount forces from its structural link.

The FEA recorder retains all phases, including approach and settling, and is
independent of the sweep report's narrower summary window. The A/B definitions
are never swapped in configuration: reverse sweeps select B as their start and A
as their destination, with the same preparation, trajectory and hold behavior.

No structural stress, deflection, fatigue, thermal, firmware, or hardware-control
prediction is implied by the joint torque measurements or FEA load exports.
