"""Window-independent experiment actions, shared by buttons and shortcuts."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import mujoco
import numpy as np
from .config import load_config, save_controls, merge_geometry
from .model import build_arm
from .simulation import Sweep, smooth_move, step
from .reporting import save_result, summary
from .fea import FEARecording
from .campaign import latest_results, load_results


class Experiment:
    def __init__(self, path, out):
        self.path, self.out = Path(path), Path(out)
        self.last_report = None
        self.fea = None
        self.campaign = None
        self.message = 'Jog to a pose, then click Save A or Save B.'
        self.load()

    def load(self):
        # Build first: invalid YAML leaves the running experiment intact.
        arm = build_arm(load_config(self.path))
        if self.fea and self.fea.active:
            self.fea.stop('Model reloaded; recording ended.')
        self.arm, self.cfg = arm, arm.cfg
        self.data = arm.data()
        self.target = self.data.qpos.copy()
        self.command = self.target.copy()
        self.run = None
        self.replay = None
        self.goto = None
        self.paused = False
        self.history = deque(maxlen=600)

    @property
    def busy(self):
        return self.run is not None or self.goto is not None

    @property
    def state(self):
        if self.paused:
            return 'PAUSED'
        if self.goto:
            return 'GO TO '+self.goto['name']
        if self.run:
            return self.run.rows[-1]['phase'].upper() if self.run.rows else 'APPROACH'
        return 'JOG'

    def idle_required(self):
        if self.busy:
            raise ValueError('Stop or finish the current move first.')

    def geometry_settings(self):
        if self.cfg.get('model_type') == 'urdf':
            return None
        return dict(links=[{k:link[k] for k in ('length_mm','mass_override_kg')}
                           for link in self.cfg['links']],
                    tool_mass_kg=self.cfg['tool']['mass_kg'],
                    payload_mass_kg=self.cfg['payload']['mass_kg'],
                    motor_mass_overrides_kg={j['name']:j.get('motor_mass_override_kg')
                                             for j in self.cfg['joints']})

    def apply_geometry(self, geometry, save=False):
        self.idle_required()
        config = deepcopy(self.cfg)
        merge_geometry(config,geometry)
        # Construct and validate before replacing the working model or saving.
        arm = build_arm(config)
        data = arm.data()
        data.qpos[:] = self.data.qpos
        data.ctrl[:] = data.qpos
        mujoco.mj_forward(arm.model,data)
        if save:
            save_controls(self.path,geometry=geometry)
        if self.fea and self.fea.active:
            self.fea.stop('Geometry/mass changed; recording ended.')
        self.arm,self.cfg,self.data = arm,config,data
        self.target = data.qpos.copy()
        self.command = self.target.copy()
        self.history.clear()
        self.paused = True
        self.message = ('Applied and saved' if save else 'Applied (not saved)') + ' lengths and masses. Paused at current angles; Resume to simulate.'

    def set_duration(self, value):
        self.idle_required()
        if not np.isfinite(value) or not .1 <= value <= 120:
            raise ValueError('Move duration must be 0.1 to 120 seconds.')
        self.cfg['sweep']['duration_s'] = float(value)
        if self.fea:
            self.fea.event(self.data.time, 'move_duration_s', float(value))

    def set_jog_speed(self, value):
        self.idle_required()
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Jog speed must be positive.')
        self.cfg['sweep']['jog_speed_deg_s'] = float(value)
        if self.fea:
            self.fea.event(self.data.time, 'jog_speed_deg_s', float(value))

    def set_target(self, index, degrees):
        self.idle_required()
        lo, hi = self.cfg['joints'][index]['limits_deg']
        if not np.isfinite(degrees) or not lo <= degrees <= hi:
            raise ValueError('Target exceeds joint limits.')
        self.target[index] = np.deg2rad(degrees)

    def set_gains(self, index, kp, kd):
        self.idle_required()
        if self.fea and self.fea.active:
            raise ValueError('Stop FEA recording before changing controller gains.')
        if not np.isfinite([kp,kd]).all() or kp <= 0 or kd < 0:
            raise ValueError('P must be positive and D must be nonnegative.')
        j = self.cfg['joints'][index]
        j['kp_nm_rad'], j['kd_nm_s_rad'] = float(kp), float(kd)
        m = self.arm.model
        m.actuator_gainprm[index, 0] = kp
        m.actuator_biasprm[index, 1:3] = [-kp, -kd]

    def zero_arm(self):
        self.idle_required()
        if any(not lo <= 0 <= hi for lo,hi in (j['limits_deg'] for j in self.cfg['joints'])):
            raise ValueError('Cannot zero: zero degrees is outside a joint limit.')
        self.target[:] = 0
        self.message = 'All targets set to 0 degrees. Moving at jog speed; resume if paused.'

    def randomize_joints(self):
        self.idle_required()
        limits = np.array([j['limits_deg'] for j in self.cfg['joints']])
        self.target[:] = np.deg2rad(np.random.default_rng().uniform(limits[:,0],limits[:,1]))
        self.message = 'Random targets set within joint limits. Moving at jog speed; resume if paused.'

    def save_pose(self, name):
        self.idle_required()
        if name not in ('A','B'):
            raise ValueError('Pose must be A or B.')
        q = np.rad2deg(self.data.qpos)
        limits = np.array([j['limits_deg'] for j in self.cfg['joints']])
        if np.any(q < limits[:,0]-.01) or np.any(q > limits[:,1]+.01):
            raise ValueError('Actual pose exceeds the joint limits.')
        q = np.clip(q, limits[:,0], limits[:,1]).tolist()
        key = name.lower()+'_deg'
        # Persist first: don't claim success if the file cannot be written.
        save_controls(self.path, {key:q})
        self.cfg['sweep'][key] = q
        self.message = f'Saved actual pose {name} to {self.path.name}.'

    def save_settings(self):
        self.idle_required()
        gains = {j['name']:{k:j[k] for k in ('kp_nm_rad','kd_nm_s_rad')} for j in self.cfg['joints']}
        save_controls(self.path, self.cfg['sweep'].copy(), gains)
        self.message = 'Saved poses, speeds, duration, and all P/D gains.'

    def go_to(self, name):
        self.idle_required()
        if name not in ('A','B'):
            raise ValueError('Pose must be A or B.')
        self.goto = dict(name=name, start=float(self.data.time), q0=self.data.qpos.copy(),
                         q1=np.deg2rad(self.cfg['sweep'][name.lower()+'_deg']),
                         duration=self.cfg['sweep']['duration_s'])
        self.paused = False
        self.message = f'Moving smoothly to {name}; torque limits remain active.'

    def start_sweep(self, direction='A -> B'):
        self.idle_required()
        self.run = Sweep(self.arm,self.data,direction)
        self.paused = False
        self.message = f'Approaching {self.run.start_name}, then sweeping to {self.run.end_name} and holding.'

    def start_fea(self):
        if self.fea and self.fea.active:
            raise ValueError('FEA recording is already active.')
        self.fea = FEARecording(self.arm, self.data)
        self.message = 'FEA recording started. Jog, Go to A/B, or run a sweep; Resume if paused.'

    def stop_fea(self):
        if self.fea and self.fea.active:
            self.fea.stop()
            self.message = f'FEA recording stopped: {len(self.fea.rows)} samples. Motion is unchanged.'

    def export_fea(self, kind='all'):
        if self.fea is None:
            raise ValueError('Record FEA data first.')
        folder = self.fea.export(self.out, kind)
        self.message = f'FEA export saved: {folder}'
        print(self.message, flush=True)

    def load_campaign(self, folder=None):
        self.idle_required()
        names = [j['name'] for j in self.cfg['joints']]
        result = load_results(folder or latest_results(self.out, names))
        if [j['name'] for j in result['cfg']['joints']] != names:
            raise ValueError('That campaign was run on a different robot; launch the viewer with its model.')
        self.campaign = result
        s = result['settings']
        self.message = f"Loaded campaign {result['folder'].name}: {s['counted']} of {s['sweeps']} sweeps counted. Click a peak to replay its sweep."
        # Replays use the live model and gains, so a different config gives different loads.
        if {k: v for k, v in result['cfg'].items() if k != 'sweep'} != {k: v for k, v in self.cfg.items() if k != 'sweep'}:
            self.message += ' WARNING: model, drive or gain settings differ from that campaign; replayed loads will differ.'

    def replay_sweep(self, sweep_id, pause_at=None, label=''):
        """Rerun a campaign sweep exactly as the campaign did: at rest at A from time zero."""
        self.idle_required()
        if self.campaign is None:
            raise ValueError('Load a campaign first.')
        if self.fea and self.fea.active:
            raise ValueError('Stop FEA recording before replaying a campaign sweep.')
        s = self.campaign['sweeps'][sweep_id]
        saved = self.cfg['sweep']
        # Sweep snapshots the config, so the saved A/B and duration are never changed.
        self.cfg['sweep'] = dict(saved, a_deg=s['a_deg'], b_deg=s['b_deg'],
                                 duration_s=s['duration_s'], approach_s=.1)
        try:
            data = self.arm.data()
            run = Sweep(self.arm, data)
        finally:
            self.cfg['sweep'] = saved
        self.data, self.run = data, run
        self.target = data.qpos.copy()
        self.command = self.target.copy()
        self.goto = None
        self.history.clear()
        self.replay = dict(sweep=sweep_id, pause_at=pause_at, label=label)
        self.paused = False
        self.message = f"Replaying campaign sweep {sweep_id} ({s['duration_s']:.2f} s A -> B)" + (f': {label}' if label else '') + '.'

    def finish_sweep(self):
        replay, self.replay = self.replay, None
        folder = save_result(self.arm,self.run,self.out)
        self.last_report = folder/'report.html'
        result = summary(self.run)
        self.message = ('Sweep finished; tracking within tolerance.' if result['tracking_ok'] else self.run.reason or 'Sweep finished; tracking exceeded tolerance.')
        self.message += ' Open the report to inspect motor loads.'
        if replay:
            self.message = f"Campaign sweep {replay['sweep']} replay finished. " + self.message
        print(self.message+' '+str(self.last_report),flush=True)
        self.run = None
        self.target = self.data.qpos.copy()
        self.command = self.target.copy()

    def stop(self, reason='Stopped by user.'):
        if self.fea:
            self.fea.event(self.data.time, 'stop_hold', reason)
        if self.run:
            self.run.reason = reason
            self.finish_sweep()
        self.goto = None
        self.target = self.data.qpos.copy()
        self.command = self.target.copy()
        self.message = reason+' Holding the current pose.'

    def tick(self):
        if self.paused:
            return
        try:
            if self.run:
                r = self.replay
                # Pause BEFORE stepping: this pre-step state is exactly the recorded peak sample.
                if r and r['pause_at'] is not None and self.data.time + 1e-9 >= r['pause_at']:
                    r['pause_at'] = None
                    self.paused = True
                    mujoco.mj_forward(self.arm.model, self.data)
                    self.message = f"Paused at the peak (t={self.data.time:.3f} s): {r['label']}. Resume to finish the sweep."
                    return
                if not self.run.tick(self.arm,self.data,recorder=self.fea):
                    self.finish_sweep()
            elif self.goto:
                g = self.goto
                t = float(self.data.time)-g['start']
                q,v = smooth_move(g['q0'],g['q1'],t,g['duration'])
                if t >= g['duration']+self.cfg['sweep']['settle_s']:
                    error = float(np.max(np.abs(np.rad2deg(self.data.qpos-g['q1']))))
                    self.target = g['q1'].copy()
                    self.command = self.target.copy()
                    self.message = (f'Reached {g["name"]}.' if error <= self.cfg['sweep']['tracking_tolerance_deg'] else f'Could not reach {g["name"]}: {error:.2f} deg error. Inspect motor limits.')
                    self.goto = None
                else:
                    step(self.arm,self.data,q,v,recorder=self.fea,phase='go to '+g['name'])
                    self.command = q.copy()
            else:
                delta = np.deg2rad(self.cfg['sweep']['jog_speed_deg_s'])*self.arm.model.opt.timestep
                self.command += np.clip(self.target-self.command,-delta,delta)
                step(self.arm,self.data,self.command,recorder=self.fea)
        except RuntimeError as exc:
            if self.fea and self.fea.active:
                self.fea.stop('Simulation error: '+str(exc))
            self.stop(str(exc))
            self.paused = True
            raise

    def sample_history(self):
        # Cap display sampling at 50 Hz so a high-refresh monitor cannot evict
        # half of the ten-second window from the bounded plot buffer.
        if self.history and self.data.time-self.history[-1][0] < .02-1e-9:
            return
        self.history.append((float(self.data.time),self.data.qfrc_actuator.copy()))
        while self.history and self.history[0][0] < self.data.time-10:
            self.history.popleft()
