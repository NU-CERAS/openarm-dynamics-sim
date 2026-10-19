"""Trajectory commands and measurements around MuJoCo's mj_step, not a physics solver."""
from copy import deepcopy
import mujoco
import numpy as np
from .model import build_arm


def smooth_move(a, b, t, duration):
    u = np.clip(t / duration, 0, 1)
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = (30*u**2 - 60*u**3 + 30*u**4) / duration
    return a + (b-a)*s, (b-a)*ds


def rating(c, name):
    return np.array([c['motors'][j['motor']][name] for j in c['joints']])


def torque_limit(c, qvel, rated=False):
    limit = rating(c, 'rated_torque_nm' if rated else 'peak_torque_nm') * c['drive']['torque_derating']
    if c['drive']['speed_envelope']:
        # Assumed symmetric envelope, not a measured RobStride curve.
        speed = np.abs(qvel)*30/np.pi
        rated_rpm, unloaded_rpm = rating(c, 'rated_speed_rpm'), rating(c, 'no_load_speed_rpm')
        limit *= np.clip((unloaded_rpm-speed)/(unloaded_rpm-rated_rpm), 0, 1)
    return limit


def control(arm, d, target, velocity=None):
    """Output-side position PD + optional gravity feedforward, all force-limited."""
    m, c = arm.model, arm.cfg
    velocity = np.zeros(m.nv) if velocity is None else velocity
    # Compute gravity at the current position with zero velocity on separate data.
    # qfrc_applied is never used: compensation consumes the same motor torque budget.
    if c['drive']['gravity_compensation']:
        if not hasattr(arm, '_gravity_data'):
            arm._gravity_data = mujoco.MjData(m)
        gd = arm._gravity_data
        gd.qpos[:] = d.qpos
        mujoco.mj_fwdPosition(m, gd)
        mujoco.mj_fwdVelocity(m, gd)
        gravity = gd.qfrc_bias.copy()
    else:
        gravity = np.zeros(m.nv)
    kd = np.array([j['kd_nm_s_rad'] for j in c['joints']])
    m.actuator_biasprm[:, 0] = gravity + kd*velocity
    cap = torque_limit(c, d.qvel)
    m.actuator_forcerange[:, 0] = -cap
    m.actuator_forcerange[:, 1] = cap
    d.ctrl[:] = target
    return cap


def step(arm, d, target, velocity=None):
    cap = control(arm, d, target, velocity)
    # Log all quantities at the same PRE-step timestamp. mj_step's acceleration/
    # torque fields otherwise describe the old state but qpos describes the new one.
    mujoco.mj_forward(arm.model, d)
    row = dict(time_s=float(d.time), q=d.qpos.copy(), velocity=d.qvel.copy(),
               target=np.asarray(target).copy(), torque=d.qfrc_actuator.copy(),
               cap=cap.copy(), rated_cap=torque_limit(arm.cfg, d.qvel, rated=True),
               tip=d.site('tip').xpos.copy(), contacts=int(d.ncon))
    mujoco.mj_step(arm.model, d)
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(w.number for w in d.warning):
        raise RuntimeError('MuJoCo reported an unstable/invalid simulation. Reduce gains or timestep and inspect the model.')
    return row


class Sweep:
    """Approach A, settle, timed A->B, then hold B; never teleport during a run."""
    def __init__(self, arm, data):
        self.cfg = deepcopy(arm.cfg)
        self.start_time = float(data.time)
        self.start_q = data.qpos.copy()
        self.a = np.deg2rad(self.cfg['sweep']['a_deg'])
        self.b = np.deg2rad(self.cfg['sweep']['b_deg'])
        self.rows = []
        self.complete = False
        self.reason = ''
        self.checked_start = False
        s = self.cfg['sweep']
        self.sweep_start = s['approach_s'] + s['settle_s']
        self.sweep_end = self.sweep_start + s['duration_s']
        self.end = self.sweep_end + s['hold_s']

    def tick(self, arm, data):
        t = float(data.time) - self.start_time
        s = self.cfg['sweep']
        if t + 1e-9 >= self.end:
            self.complete = True
            return False
        if t < s['approach_s']:
            q, v = smooth_move(self.start_q, self.a, t, s['approach_s'])
            phase = 'approach'
        elif t + 1e-9 < self.sweep_start:
            q, v, phase = self.a, np.zeros_like(self.a), 'settle'
        elif t + 1e-9 < self.sweep_end:
            if not self.checked_start:
                self.checked_start = True
                err = np.max(np.abs(np.rad2deg(data.qpos-self.a)))
                if err > s['tracking_tolerance_deg']:
                    self.reason = f'Could not reach A: {err:.2f} deg error. Sweep not executed.'
                    return False
            q, v = smooth_move(self.a, self.b, t-self.sweep_start, s['duration_s'])
            phase = 'sweep'
        else:
            q, v, phase = self.b, np.zeros_like(self.b), 'hold'
        row = step(arm, data, q, v)
        row['time_s'] = t
        row['phase'] = phase
        self.rows.append(row)
        return True


def run_headless(cfg):
    arm = build_arm(cfg)
    data = arm.data()
    run = Sweep(arm, data)
    while run.tick(arm, data):
        pass
    return arm, run
