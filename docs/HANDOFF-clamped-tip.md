# Handoff: clamped-tip case for `arm-sim bounds`

Status as of 2026-11-12, branch `feature/load-campaign`. Nothing below is committed; all changes are in the working tree.

## Goal

`arm-sim bounds` had two static load cases: `gravity` (worst holding pose) and `blocked` (one pure force at the tip, along the tangent of one joint's rotation, grown until the first motor reaches peak torque or `--max-contact-force-n`).

The user wants a third case, `clamped`: the tip grips a fixed object, so it can react a force and a moment in any direction and several motors can be at their caps at once. It is for sizing the links, mainly the distal ones, where a pure tip force has almost no lever arm.

It is a static case. It is not "all motors at max": on a 7-joint arm a clamped tip leaves one freedom (elbow swing), so seven independent torques are not in equilibrium. The case is the largest tip force and moment the motors can statically hold, with caps on both.

## Design

With tip load `(F, M)` on the arm at the `tip` site:

    motor torque = g - jacp.T @ F - jacr.T @ M        (g = qfrc_bias at rest)

Constraints: every `|torque_i| <= cap_i`, `|F| <= max_force_n`, `|M| <= max_moment_nm`.

Each mate load is affine in `(F, M)` for a fixed pose and its magnitude is convex, so the maximum is on the boundary of the feasible set. That set is convex and contains zero, so every boundary point lies on a ray from the origin. The search is therefore a ray search:

- A clamped case is `('clamped', pf_x, pf_y, pf_z, pm_x, pm_y, pm_z)`: two 3-vectors, each inside the unit ball (`unit_balls`).
- Base load `(max_force_n * pf, max_moment_nm * pm)`, scaled by the largest `s` that keeps every motor within its cap and `s*|pf| <= 1`, `s*|pm| <= 1` (`Statics.clamping_load`, sharing `Statics.scale` with the blocked case).
- `_search` scores `CLAMPED_DIRECTIONS = 12` random directions per accepted pose (`pf`, `pm` random unit vectors), from a separate RNG stream so gravity and blocked results for a seed do not change when clamped is enabled.
- `_refine` perturbs the pose and the direction together for clamped targets.
- Every blocked tip force is also a clamped case with zero moment (`Statics.as_clamped`). Blocked tip hits seed the clamped bests during the search, and the refined blocked peaks are re-added after refinement, so clamped should never report less than blocked at any row.

The case runs only when `--max-contact-moment-nm` is given, and it requires `--max-contact-force-n`. Without the flag, outputs are meant to be identical to before.

## What is in the working tree

| File | Change |
|---|---|
| `src/arm_design/bounds.py` | `Statics` takes `max_moment_nm`, has per-run `kinds`, `scale`, `clamping_load`, `as_clamped`, `contact` (returns `(body, point, force, moment)` for either case type); `loads` and `capture` handle a moment; rows record `contact_moment_on_arm_world_Nm`; `_search`, `_refine`, `bounds()` and `summary.csv` use the run's kinds. |
| `src/arm_design/cli.py` | `--max-contact-moment-nm`; the printed table has one column pair per kind that ran. |
| `src/arm_design/fea_cases.py` | The contact carries the moment (rotated into the part frame); the report line and CSV row include it. |
| `tests/test_bounds.py` | About 76 new lines of clamped tests. |
| `CLAUDE.md`, `README.md` | Commands, outputs and gotchas updated for clamped. |
| `docs/MODEL.md` | Not updated. Check whether it describes the bounds assumptions and add clamped if so. |

## Verification state

See "Final state from the implementing agent" at the end of this file if it is present. If that section is missing, treat everything as unverified: the implementing agent was interrupted once and had not reported test results when this file was first written.

Known at the time of writing:

- **The test suite has not been confirmed passing with the current code.**
- **`results/bounds-20261113T021010-382822Z`** is a real right-arm run at 100 N / 10 Nm (2000 poses, 300 iterations). It may predate the final code. In it, clamped moments are lower than blocked at several proximal mates (for example `l2 <- rs03output`: blocked 76.80 Nm, clamped 75.12 Nm), which is a search shortfall, since clamped contains blocked. The "re-add refined blocked peaks" step in `bounds()` looks like the fix for this; a fresh run is needed to confirm clamped >= blocked at every row.
- In the same run the distal mates behave as expected: moments rise by roughly the moment cap (for example `base <- rs00_2`: blocked 24.67 Nm, clamped 34.62 Nm) and forces are unchanged (cap + outboard weight). Note that on the right arm the part named `base` is the distal end of the chain.
- Most clamped peaks in that run sit at 100.0 N with 9.43 Nm, not 10.00 Nm, so the search had not reached the corner where both caps are active for those rows. Expect a little more moment to be available.
- `src/arm_design/bounds.py` line 98 has a mis-encoded character in a comment (`Â±cap`, should be `±cap`). Harmless, fix it.
- Clamped forces come out about 1e-7 N below blocked at the same row (scaling round-off). Tests comparing the two need a tolerance.

## How to check and finish

On Windows the venv tools are under `.venv\Scripts\`.

    .venv\Scripts\pytest tests\test_bounds.py
    .venv\Scripts\pytest
    .venv\Scripts\arm-sim bounds --urdf robots\right_arm_assembly\urdf\right_arm_assembly.urdf --max-contact-force-n 100 --max-contact-moment-nm 10
    .venv\Scripts\arm-sim fea-cases --results results\bounds-<stamp>\clamped --part base

Remaining work, in order:

1. Get the full suite passing (it takes 1-4 minutes). Delete any stray `MUJOCO_LOG.TXT` in the repo root afterwards.
2. Rerun the right-arm case on the final code and confirm clamped >= blocked at every row for both force and moment (within round-off).
3. Confirm a run without `--max-contact-moment-nm` still writes only `gravity/` and `blocked/` with the same numbers as before for a given seed. `results/bounds-20261113T011004-306401Z` is a two-case run at 100 N to compare against, but check which code produced it.
4. Fix the comment encoding on line 98 of `bounds.py`.
5. Update `docs/MODEL.md` if it covers the bounds cases.
6. Optional improvement: also seed the search with directions where both caps are active and the moment is aligned to add to the blocked force's moment at each mate, so the both-caps corner is found without relying on refinement.

## Open decisions for the user

- **The moment cap.** 10 Nm is a placeholder. It should come from the wrist motors' peak torques or the gripper's slip limit. Clamped results scale with it.
- **Whether to use clamped for design at all.** It applies only if the arm grips fixed objects. It mainly raises the distal links' moments; near the base it is close to blocked at the same force cap.
- Neither arm model has a gripper or payload yet, so the tip point and the wrist loads will shift when those are added.
- Not covered by any case: the dynamic transient with every motor saturated and the tip constrained (elbow swinging into a joint stop). That would need its own constrained dynamic simulation.

## Conventions to keep

Read `CLAUDE.md` first. Code is deliberately compact; match it. Do not commit unless the user asks.
