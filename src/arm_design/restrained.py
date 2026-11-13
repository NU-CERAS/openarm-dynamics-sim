"""Fixed-hand dynamic torque enumeration; the elbow is not artificially locked."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product
import multiprocessing
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from .analysis import RunStore, distribution, result_from_recording
from .campaign import Peaks, CampaignRecording
from .collision import CollisionChecker, describe
from .fea import FEARecording
from .model import Arm, build_arm, values
from .simulation import torque_limit
from .live_preview import LivePublisher

_worker = None


def combinations(n):
    return product((-1, 0, 1), repeat=n)


def build_restrained(cfg, time_constant=.005):
    cfg = deepcopy(cfg)
    cfg['drive']['gravity_compensation'] = False
    cfg['drive']['torque_derating'] = 1.
    arm = build_arm(cfg)
    root = ET.fromstring(arm.xml)
    actuators = root.find('actuator')
    actuators.clear()
    for j in cfg['joints']:
        cap = cfg['motors'][j['motor']]['peak_torque_nm']
        ET.SubElement(actuators, 'motor', joint=j['name'], gear='1',
                      ctrllimited='false', forcerange=values([-cap, cap]))
    ET.SubElement(root.find('worldbody'), 'site', name='clamp_target', pos='0 0 0', size='.008', rgba='1 .5 .1 1')
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    ET.SubElement(equality, 'weld', name='tip_clamp', site1='tip', site2='clamp_target',
                  solref=values([time_constant, 1]), solimp='.999 .9999 .001', torquescale='1')
    xml = ET.tostring(root, encoding='unicode')
    return Arm(cfg, mujoco.MjModel.from_xml_string(xml), xml, arm.structural_masses)


def set_pose(arm, q):
    m = arm.model
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    tip, anchor = m.site('tip').id, m.site('clamp_target').id
    position, rotation = d.site_xpos[tip].copy(), d.site_xmat[tip].copy()
    m.site_sameframe[anchor] = 0  # Target moves between trials; disable compiler identity shortcut.
    m.site_pos[anchor] = position
    mujoco.mju_mat2Quat(m.site_quat[anchor], rotation)
    mujoco.mj_forward(m, d)
    return d, position, rotation.reshape(3, 3)


def trial(arm, q, signs, duration=.5, ramp=.05, checker=None, full_history=False,
          position_tolerance=.005, angle_tolerance=.05, preview=None):
    m, c = arm.model, arm.cfg
    d, target_p, target_R = set_pose(arm, q)
    rec = FEARecording(arm, d)
    signs = np.asarray(signs)
    reason, collision, limit_hit = '', '', False
    max_position, max_angle = 0., 0.
    tip = m.site('tip').id
    body = int(m.site_bodyid[tip])
    if preview:
        preview.begin('restrained',float(duration))
    for k in range(int(np.ceil(duration/m.opt.timestep))):
        if preview:
            preview.sample(d,'restrained')
        cap = torque_limit(c, d.qvel)
        u = min(float(d.time)/ramp, 1.)
        command = signs*cap*(u*u*(3-2*u))
        m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1] = -cap, cap
        d.ctrl[:] = command
        mujoco.mj_forward(m, d)
        position_error = np.linalg.norm(d.site_xpos[tip]-target_p)
        relative = target_R.T@d.site_xmat[tip].reshape(3, 3)
        angle_error = np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))
        max_position, max_angle = max(max_position, position_error), max(max_angle, angle_error)
        if position_error > position_tolerance or angle_error > angle_tolerance:
            reason = 'Clamp drift exceeded tolerance.'
            break
        limit_hit = bool(np.any(d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)))
        if limit_hit:
            reason = 'Joint-limit impact; excluded from accepted load distribution.'
            break
        if checker and k % 5 == 0:
            collision = describe(checker.contacts(d.qpos))
            if collision:
                reason = 'Self/floor collision: '+collision
                break
        try:
            rec.capture(m, d, dict(q=d.qpos.copy(), velocity=d.qvel.copy(), target=np.asarray(q),
                                   torque=d.qfrc_actuator.copy(), cap=cap.copy()), 'restrained')
        except RuntimeError as exc:
            reason = str(exc)
            break
        s = rec.rows[-1]
        R = s['body_rotation_world'][body-1]
        ext = s['external_resultant_at_com_local'][body-1]
        force, moment_com = R@ext[:3], R@ext[3:]
        point = d.site_xpos[tip].copy()
        moment = moment_com + np.cross(d.xipos[body]-point, force)
        s.update(contact_body=m.body(body).name, contact_point_world_m=point,
                 contact_force_on_arm_world_N=force, contact_moment_on_arm_world_Nm=moment,
                 commanded_torque_Nm=command.copy(), clamp_position_world_m=target_p,
                 clamp_rotation_world=target_R, clamp_position_error_m=float(position_error),
                 clamp_angle_error_rad=float(angle_error))
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):
            reason = 'Numerical instability.'
            break
        if not rec.active:
            reason = rec.reason
            break
    rec.stop(reason or 'Trial complete.')
    result = result_from_recording(rec, q_deg=np.rad2deg(q), signs=signs, duration_s=duration,
                                  collision=collision, joint_limit_hit=limit_hit,
                                  max_clamp_position_error_m=float(max_position), max_clamp_angle_error_rad=float(max_angle),
                                  clamp_position_world_m=target_p, clamp_rotation_world=target_R,
                                  distribution=distribution(rec.rows, full_history))
    result.update(reason=reason, completed=not reason, counted=bool(rec.rows) and not reason,
                  max_tracking_error_deg=0., tracking_ok=not reason)
    return result


def _init(cfg, options, preview_path=None, preview_owner=None):
    global _worker
    arm = build_restrained(cfg, options['time_constant'])
    checker = CollisionChecker(build_arm(cfg))
    _worker = arm, checker, options, LivePublisher(preview_path,preview_owner)


def _run(job):
    arm, checker, opts, preview = _worker
    pose_id, q, signs = job
    result = trial(arm, q, signs, opts['duration'], opts['ramp'], checker, opts['full_history'],
                   opts['position_tolerance'], opts['angle_tolerance'],preview=preview)
    return dict(result, pose_id=pose_id)


class RestrainedRecording(CampaignRecording):
    def metadata(self):
        meta = FEARecording.metadata(self)
        meta.update(samples_meaning='Extrema snapshots from torque enumeration, not an A/B campaign.',
                    duration_s=None, restrained=self.settings,
                    recorded_samples=sum(s['samples'] for s in self.sweeps))
        meta['limitations'] += self.settings['assumptions'] + [
            'Reproduction requires the snapshot clamp position/orientation, torque ramp and initial pose.',
            'Only accepted trials contribute global peaks. Rejected trial prefixes remain separately inspectable.',
            'A parameterized dynamic screening experiment, not a proven absolute hardware maximum.']
        return meta


def restrained(cfg, out, poses=3, seed=0, workers=1, duration=.5, ramp=.05,
               time_constant=.005, position_tolerance=.005, angle_tolerance=.05,
               poses_deg=None, full_history=False, max_trials=None, log=print):
    if not (np.isfinite([duration, ramp, time_constant, position_tolerance, angle_tolerance]).all()
            and .01 <= duration <= 120 and 0 < ramp <= duration and time_constant >= 2*cfg['simulation']['step_s']
            and position_tolerance > 0 and angle_tolerance > 0):
        raise ValueError('Invalid duration/ramp, clamp tolerance or solver time constant (must be >= 2 timesteps).')
    if poses < 1 or workers < 1 or (max_trials is not None and max_trials < 1):
        raise ValueError('Pose, worker and optional trial counts must be positive.')
    base = build_arm(deepcopy(cfg))
    checker = CollisionChecker(base)
    limits = np.deg2rad([j['limits_deg'] for j in cfg['joints']])
    rng = np.random.default_rng(seed)
    if poses_deg is not None:
        qs = np.deg2rad(np.asarray(poses_deg, float))
        if qs.ndim != 2 or qs.shape[1] != base.model.nq or not len(qs) or not np.isfinite(qs).all():
            raise ValueError('Pose file must contain a nonempty list of joint-angle lists in degrees.')
        if np.any(qs < limits[:, 0]) or np.any(qs > limits[:, 1]) or any(checker.contacts(q) for q in qs):
            raise ValueError('A supplied pose exceeds limits or collides.')
    else:
        qs = []
        for _ in range(poses*1000):
            if (Path(out)/'CANCEL').exists():
                raise RuntimeError('Stopped during pose preparation.')
            q = rng.uniform(limits[:, 0], limits[:, 1])
            if not checker.contacts(q):
                qs.append(q)
            if len(qs) == poses:
                break
        if len(qs) != poses:
            raise ValueError('Could not find enough collision-free clamp poses.')
    options = dict(duration=duration, ramp=ramp, time_constant=time_constant,
                   position_tolerance=position_tolerance, angle_tolerance=angle_tolerance, full_history=full_history)
    arm = build_restrained(cfg, time_constant)
    n = arm.model.nv
    expected = len(qs)*3**n
    settings = dict(**options, seed=seed, poses_deg=np.rad2deg(qs).tolist(), expected_trials=expected,
                    max_trials=max_trials, workers=workers, restraint='6D tip weld to world; internal motion allowed',
                    assumptions=['Full peak output torque at zero speed; configured speed envelope applies.',
                      'Zero command has no PD or gravity compensation. Gravity and passive joint forces remain.',
                      'Independent trials start at rest; each pose has its own fixed grasp.',
                      'Soft numerical weld, not a measured gripper. No force/moment slip cap.',
                      'Joint-stop impacts, collisions, excessive clamp drift and unstable trials are rejected.',
                      'Durations and clamp parameters are screening assumptions, not hardware peak-duration ratings.'])
    store = RunStore(out, arm, 'restrained', settings)
    peaks = Peaks((len(store.meta['interfaces']), 8))
    summaries = []
    jobs = ((i, q, s) for i, q in enumerate(qs) for s in combinations(n))
    from itertools import islice
    jobs = iter(islice(jobs, max_trials)) if max_trials else iter(jobs)
    context = multiprocessing.get_context('spawn')
    try:
        with ProcessPoolExecutor(workers, context, _init, (cfg, options,str(store.path),context.Value('i',0))) as pool:
            while not store.cancelled:
                batch = list(islice(jobs, workers))
                if not batch:
                    break
                for result in pool.map(_run, batch):
                    i = len(summaries)
                    store.add(i, result, 'enumeration')
                    peaks.add(i, result)
                    summaries.append({k: v for k, v in result.items() if k not in
                        ('distribution', 'extrema', 'minimum', 'maximum', 'min_index', 'max_index')})
                    log(f'{i+1}/{min(expected, max_trials or expected)} torque trials; {store.state["counted"]} accepted', flush=True)
        export = None
        if peaks.samples:
            rec = RestrainedRecording(arm, peaks, summaries, settings)
            # This is an extrema recorder, not a replayable A/B campaign.
            export = rec.export(store.path, 'cases')
        status = 'cancelled' if store.cancelled else 'partial' if len(summaries) < expected else 'complete'
        store.finish(status, export)
        return store.path
    except BaseException as exc:
        store.finish('failed', error=str(exc))
        raise
