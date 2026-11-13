"""Bounding static load cases sweeps never reach: the worst gravity pose, the arm blocked at a point
while its motors push at up to peak torque, and optionally the tip clamped (any force and moment,
several motors at their caps). MuJoCo's RNE does all the mechanics."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import csv
import json
import multiprocessing
from pathlib import Path
import time

import mujoco
import numpy as np

from .collision import CollisionChecker
from .fea import FEARecording, METRICS
from .model import build_arm
from .simulation import rating

# Default nearest block to a joint axis; force grows as torque/lever, unbounded at the axis.
MIN_LEVER_M = .05
OBJECTIVES = (METRICS.index('force_N'), METRICS.index('moment_Nm'))
KINDS = ('gravity', 'blocked')
# Random clamped tip load directions scored per accepted pose.
CLAMPED_DIRECTIONS = 12
_statics = None


def unit_balls(p):
    """Project both 3-vector halves of a clamped direction into the unit ball."""
    p = np.asarray(p, float).reshape(2, 3)
    return (p/np.maximum(np.linalg.norm(p, axis=1), 1)[:, None]).ravel()


class Statics:
    """Part mate loads at rest (zero velocity and acceleration), optionally with a blocking or clamping load."""
    def __init__(self, arm, derating=1., max_overlap_mm=0., min_lever_m=MIN_LEVER_M, contacts='tip', max_force_n=None,
                 max_moment_nm=None):
        self.m, self.d = arm.model, mujoco.MjData(arm.model)
        self.min_lever_m = min_lever_m
        # Near toggle-like alignments, small torques resist huge forces; a design contact force caps them.
        self.max_force_n = np.inf if max_force_n is None else max_force_n
        if max_moment_nm is not None and (max_force_n is None or not max_moment_nm > 0):
            raise ValueError('The clamped case needs a positive moment cap and a contact force cap.')
        self.max_moment_nm = max_moment_nm
        self.kinds = KINDS + ('clamped',)*(max_moment_nm is not None)
        self.rec = FEARecording(arm, self.d)
        self.cap = rating(arm.cfg, 'peak_torque_nm')*derating
        self.limits = np.deg2rad([j['limits_deg'] for j in arm.cfg['joints']])
        self.checker = None if max_overlap_mm is None else CollisionChecker(arm, max_overlap_mm)
        m = self.m
        tip = m.site('tip')
        # Candidate contacts: the tool tip, plus every moving body's origin for 'all'.
        if contacts not in ('tip', 'all'):
            raise ValueError("Contacts must be 'tip' or 'all'.")
        self.points = [('tip', int(m.site_bodyid[tip.id]), tip.id)]
        if contacts == 'all':
            self.points += [(self.rec.names[b], b, None) for b in range(1, m.nbody) if m.body_weldid[b]]
        # (joint, point, direction): a joint can only be blocked through a body it moves.
        self.cases = [(j, k, sign) for j in range(m.njnt) for k, (_, body, _) in enumerate(self.points)
                      if self.moves(j, body) for sign in (1, -1)]
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def moves(self, joint, body):
        while body:
            if body == self.m.jnt_bodyid[joint]:
                return True
            body = self.m.body_parentid[body]
        return False

    def pose(self, q):
        """Gravity torque holding q at rest; None if q self-collides or a motor cannot hold it."""
        m, d = self.m, self.d
        d.qpos[:] = q
        d.qvel[:] = d.qacc[:] = d.xfrc_applied[:] = 0
        mujoco.mj_fwdPosition(m, d)
        mujoco.mj_fwdVelocity(m, d)
        g = d.qfrc_bias.copy()
        if np.any(np.abs(g) > self.cap) or (self.checker and self.checker.contacts(q)):
            return None
        return g

    def blocking_force(self, g, joint, k, sign):
        """Largest obstacle force at point k, tangent to the joint's rotation, every motor can resist (capped)."""
        m, d = self.m, self.d
        _, body, site = self.points[k]
        p = (d.site_xpos[site] if site is not None else d.xpos[body]).copy()
        lever = np.cross(d.xaxis[joint], p-d.xanchor[joint])
        r = np.linalg.norm(lever)
        if r < self.min_lever_m:
            return None
        u = sign*lever/r
        mujoco.mj_jac(m, d, self.jacp, self.jacr, p, body)
        return body, p, self.scale(g, self.jacp.T@u, self.max_force_n)*u

    def scale(self, g, a, limit):
        # Statics: motor torque = g - s*a. Grow s until the first motor reaches ±cap.
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(a > 1e-12, (g+self.cap)/a, np.where(a < -1e-12, (g-self.cap)/a, np.inf))
        return min(s.min(), limit)

    def clamping_load(self, g, p):
        """Largest tip load along p = (pf, pm) the motors can hold, with |F| and |M| within their caps."""
        m, d = self.m, self.d
        p = np.asarray(p, float)
        pf, pm = p[:3], p[3:]
        nf, nm = np.linalg.norm(pf), np.linalg.norm(pm)
        if max(nf, nm) < 1e-9:
            return None
        _, body, site = self.points[0]
        point = d.site_xpos[site].copy()
        mujoco.mj_jac(m, d, self.jacp, self.jacr, point, body)
        f0, m0 = self.max_force_n*pf, self.max_moment_nm*pm
        s = self.scale(g, self.jacp.T@f0 + self.jacr.T@m0, 1/max(nf, nm))
        return body, point, s*f0, s*m0

    def as_clamped(self, force):
        """A blocked tip force as the clamped case with no moment (a clamp reacts any capped tip force)."""
        return ('clamped', *(force/self.max_force_n).tolist(), 0., 0., 0.)

    def contact(self, g, case):
        """(body, point, force, moment) on the arm for a blocked or clamped case; None if skipped."""
        if case[0] == 'clamped':
            return self.clamping_load(g, case[1:])
        hit = self.blocking_force(g, *case)
        return None if hit is None else (*hit, np.zeros(3))

    def loads(self, body=None, point=None, force=None, moment=None):
        m, d = self.m, self.d
        d.xfrc_applied[:] = 0
        if force is not None:
            # xfrc_applied acts at the body COM; carry the force's moment about it.
            torque = np.cross(point-d.xipos[body], force) + (0 if moment is None else moment)
            d.xfrc_applied[body] = np.concatenate((force, torque))
        mujoco.mj_rnePostConstraint(m, d)
        loads, _ = self.rec.interface_loads(m, d)
        return np.column_stack((loads, np.linalg.norm(loads[:, :3], axis=1), np.linalg.norm(loads[:, 3:], axis=1)))

    def evaluate(self, q, case=None):
        g = self.pose(q)
        if g is None:
            return None
        if case is None:
            return self.loads()
        hit = self.contact(g, case)
        return None if hit is None else self.loads(*hit)

    def describe(self, case):
        if case is None:
            return 'gravity'
        if case[0] == 'clamped':
            return f'clamped at {self.points[0][0]}'
        joint, k, sign = case
        return f"blocked at {self.points[k][0]} against {self.m.joint(joint).name}{'+' if sign > 0 else '-'}"

    def capture(self, recording, q, case=None):
        """Append the complete state for one load case to an FEA recording."""
        g = self.pose(q)
        torque, phase = g, self.describe(case)
        if case is not None:
            body, p, force, moment = self.contact(g, case)
            torque = g - self.jacp.T@force - self.jacr.T@moment
            self.loads(body, p, force, moment)
            phase += f': {np.linalg.norm(force):.1f} N' + (f', {np.linalg.norm(moment):.2f} Nm' if case[0] == 'clamped' else '')
        recording.capture(self.m, self.d, dict(q=np.asarray(q, float), velocity=np.zeros(self.m.nv), target=np.asarray(q, float),
                                               torque=torque, cap=self.cap.copy()), phase)
        if case is not None:
            recording.rows[-1].update(contact_body=self.rec.names[body], contact_point_world_m=p,
                                      contact_force_on_arm_world_N=force)
            if case[0] == 'clamped':
                recording.rows[-1]['contact_moment_on_arm_world_Nm'] = moment


class Best:
    """Per interface/metric extrema with the (pose, case) behind each."""
    def __init__(self, shape):
        self.minimum = np.full(shape, np.inf)
        self.maximum = np.full(shape, -np.inf)
        self.args = {}

    def _update(self, low, high, source):
        for side, mine, theirs, better in (('min', self.minimum, low, np.less),
                                           ('max', self.maximum, high, np.greater)):
            mask = better(theirs, mine)
            mine[mask] = theirs[mask]
            for cell in zip(*np.nonzero(mask)):
                cell = tuple(map(int, cell))
                self.args[(side, cell)] = source(side, cell)

    def add(self, values, q, case):
        self._update(values, values, lambda side, cell: (q, case))

    def merge(self, other):
        self._update(other.minimum, other.maximum, lambda side, cell: other.args[(side, cell)])


def _init(cfg, *options):
    global _statics
    _statics = Statics(build_arm(deepcopy(cfg)), *options)


def _search(job):
    """Random rest poses; every pose is scored for gravity, every blocking case and random clamped directions."""
    s = _statics
    rng = np.random.default_rng(job['seed'])
    # A separate stream, so enabling clamped leaves the gravity and blocked results unchanged.
    clamp_rng = np.random.default_rng([*job['seed'], 1])
    shape = (len(s.rec.interfaces), len(METRICS))
    best = {kind: Best(shape) for kind in s.kinds}
    rejected = 0
    for _ in range(job['poses']):
        q = rng.uniform(s.limits[:, 0], s.limits[:, 1])
        g = s.pose(q)
        if g is None:
            rejected += 1
            continue
        best['gravity'].add(s.loads(), q, None)
        for case in s.cases:
            hit = s.blocking_force(g, *case)
            if hit is not None:
                values = s.loads(*hit)
                best['blocked'].add(values, q, case)
                # Random force directions rarely hit blocked's tangents; seed clamped with them.
                if 'clamped' in s.kinds and case[1] == 0:
                    best['clamped'].add(values, q, s.as_clamped(hit[2]))
        if 'clamped' in s.kinds:
            p = clamp_rng.normal(size=(CLAMPED_DIRECTIONS, 2, 3))
            for p in (p/np.linalg.norm(p, axis=2, keepdims=True)).reshape(-1, 6):
                hit = s.clamping_load(g, p)
                if hit is not None:
                    best['clamped'].add(s.loads(*hit), q, ('clamped', *p.tolist()))
    return best, rejected


def _refine(job):
    """(1+1) evolution strategy on one target, over the pose with its blocking case fixed, or over
    the pose and the clamped load direction together."""
    s = _statics
    q, case = job['start']
    lo, hi = s.limits[:, 0], s.limits[:, 1]
    rng = np.random.default_rng(job['seed'])
    best = Best((len(s.rec.interfaces), len(METRICS)))
    value, sigma = job['score'], .05
    for _ in range(job['iterations']):
        trial = np.clip(q + rng.normal(0, sigma, q.shape)*(hi-lo), lo, hi)
        trial_case = case
        if job['kind'] == 'clamped':
            trial_case = ('clamped', *unit_balls(np.array(case[1:]) + rng.normal(0, 2*sigma, 6)).tolist())
        values = s.evaluate(trial, trial_case)
        if values is not None:
            best.add(values, trial, trial_case)
        if values is not None and values[job['interface'], job['metric']] > value:
            value, q, case, sigma = values[job['interface'], job['metric']], trial, trial_case, min(sigma*1.5, .3)
        else:
            sigma = max(sigma*.9, .002)
    return job['kind'], best


def bounds(cfg, out, poses=2000, iterations=300, workers=None, seed=0, derating=1., max_overlap_mm=0.,
           min_lever_m=MIN_LEVER_M, contacts='tip', max_force_n=None, max_moment_nm=None, log=print):
    """max_overlap_mm=None disables the self-collision check; max_moment_nm enables the clamped case."""
    if poses < 1 or iterations < 0:
        raise ValueError('Need at least one pose and nonnegative iterations.')
    if not 0 < derating <= 1:
        raise ValueError('Derating must be in (0, 1].')
    if max_force_n is not None and not max_force_n > 0:
        raise ValueError('Maximum contact force must be positive.')
    if max_moment_nm is not None and not max_moment_nm > 0:
        raise ValueError('Maximum contact moment must be positive.')
    if max_moment_nm is not None and max_force_n is None:
        raise ValueError('The clamped case (--max-contact-moment-nm) also needs --max-contact-force-n.')
    if not min_lever_m > 0:
        raise ValueError('Minimum lever must be positive.')
    if max_overlap_mm is not None and not max_overlap_mm >= 0:
        raise ValueError('Allowed overlap must be a nonnegative number of millimetres.')
    workers = workers or multiprocessing.cpu_count()
    options = (derating, max_overlap_mm, min_lever_m, contacts, max_force_n, max_moment_nm)
    statics = Statics(build_arm(deepcopy(cfg)), *options)
    kinds = statics.kinds
    shape = (len(statics.rec.interfaces), len(METRICS))
    best = {kind: Best(shape) for kind in kinds}
    started = time.monotonic()
    chunks = [dict(seed=[seed, i], poses=n) for i, n in enumerate(np.diff(np.linspace(0, poses, workers*4+1).astype(int))) if n]
    targets = [(kind, i, j) for kind in kinds for i in range(shape[0]) for j in OBJECTIVES]
    rejected = 0
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(workers, context, _init, (cfg, *options)) as pool:
        for found, n in pool.map(_search, chunks):
            rejected += n
            for kind in kinds:
                best[kind].merge(found[kind])
        log(f'{poses} random poses ({rejected} colliding or unholdable), {time.monotonic()-started:.0f} s', flush=True)
        if not np.isfinite(best['gravity'].maximum).all():
            raise RuntimeError('No valid pose found; check the joint limits and motor ratings.')
        jobs = [dict(kind=kind, interface=i, metric=j, iterations=iterations, seed=[seed, 10**6, n],
                     start=best[kind].args[('max', (i, j))], score=best[kind].maximum[i, j])
                for n, (kind, i, j) in enumerate(targets) if np.isfinite(best[kind].maximum[i, j])]
        for kind, found in pool.map(_refine, jobs):
            best[kind].merge(found)
    if 'clamped' in kinds:
        # Refined blocked tip peaks are clamped cases too, so clamped never reports less at any row.
        for q, case in list(best['blocked'].args.values()):
            if case[1] == 0:
                clamped = statics.as_clamped(statics.blocking_force(statics.pose(q), *case)[2])
                best['clamped'].add(statics.evaluate(q, clamped), q, clamped)
    log(f'Refined {len(jobs)} targets, {time.monotonic()-started:.0f} s', flush=True)
    folder = Path(out).resolve()/('bounds-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ'))
    recordings = {}
    for kind in kinds:
        arm = build_arm(deepcopy(cfg))
        rec = FEARecording(arm, arm.data())
        unique = {}
        for q, case in best[kind].args.values():
            unique[(tuple(q), case)] = (q, case)
        for q, case in unique.values():
            statics.capture(rec, q, case)
        rec.stop(f'{kind} bound: {len(unique)} load cases.')
        rec.export(folder, 'cases', kind)
        recordings[kind] = rec
    settings = dict(poses=poses, refine_iterations=iterations, seed=seed, rejected_poses=rejected,
                    torque_cap_nm=dict(zip([j['name'] for j in cfg['joints']], statics.cap.tolist())),
                    derating=derating, max_overlap_mm=max_overlap_mm, min_lever_m=min_lever_m, contacts=contacts, max_contact_force_n=max_force_n,
                    max_contact_moment_nm=max_moment_nm, contact_points=[p[0] for p in statics.points],
                    assumptions=[
                        'Static: zero joint velocity and acceleration. Dynamic loads come from arm-sim campaign.',
                        'gravity: the arm holding a pose against gravity alone.',
                        'blocked: an obstacle pushes on one contact point (the tip, or with contacts=all also a moving body origin), tangent to '
                        'one joint\'s rotation, with the largest force that keeps every motor within its torque cap, '
                        'capped at max_contact_force_n. Uncapped, aligned poses let small torques resist huge forces.',
                        *(['clamped: the tip grips a fixed object that reacts any force and moment; the largest load along each searched '
                           'direction that keeps every motor within its torque cap (several may saturate), with |F| <= max_contact_force_n '
                           'and |M| <= max_contact_moment_nm. Results scale with both caps.'] if 'clamped' in kinds else []),
                        f'Contacts closer than {min_lever_m} m to the blocked joint axis are skipped; '
                        'force grows as torque/lever, so this sets the proximal blocked loads.',
                        'Poses that self-collide (convex hulls) or that a motor cannot hold are rejected.',
                        'A search over sampled poses: the largest loads found, not a proven maximum.'])
    (folder/'bounds.json').write_text(json.dumps(settings, indent=2))
    loaded = [k for k in ('blocked', 'clamped') if k in kinds]
    with (folder/'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['interface', *(f'{kind}_{METRICS[j]}' for kind in kinds for j in OBJECTIVES), *(f'{k}_force_case' for k in loaded)])
        for i, interface in enumerate(statics.rec.interfaces):
            cases = [recordings[k].rows[recordings[k].max_index[i, OBJECTIVES[0]]]['phase'] for k in loaded]
            writer.writerow([interface['label'], *(recordings[kind].maximum[i, j] for kind in kinds for j in OBJECTIVES), *cases])
    return folder, recordings
