"""Time-aligned rigid-body load histories; structural stresses belong in FEA."""
from copy import copy, deepcopy
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

import mujoco
import numpy as np
import yaml


METRICS = ('Fx_N', 'Fy_N', 'Fz_N', 'Mx_Nm', 'My_Nm', 'Mz_Nm', 'force_N', 'moment_Nm')


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value)}')


def cross(a, b):
    # np.cross's axis handling dominated per-step recording cost for 3-vectors.
    return np.array((a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]))


def local_wrench(wrench, reference, point, rotation):
    """MuJoCo world [moment, force] about reference -> local [force, moment]."""
    force = wrench[3:]
    moment = wrench[:3] + cross(reference-point, force)
    return np.concatenate((rotation.T@force, rotation.T@moment))


class FEARecording:
    def __init__(self, arm, data):
        # Preserve the actual compiled model, including live gain edits.
        self.model = copy(arm.model)
        self.cfg = deepcopy(arm.cfg)
        self.source_xml = arm.xml
        self.start_time = float(data.time)
        self.active = True
        self.reason = ''
        # Optional phase names (e.g. sweep, hold) to capture; None records every step.
        self.phases = None
        self.rows = []
        self.last_export = None
        self.selected_samples = set()
        self.events = []
        m = self.model
        self.body_ids = list(range(1, m.nbody))
        self.names = {i: m.body(i).name or f'body_{i}' for i in range(m.nbody)}
        self.names[0] = 'world'
        self.interfaces = []
        for child in self.body_ids:
            parent = int(m.body_parentid[child])
            self.interfaces.append(dict(body_id=child, other_body_id=parent,
                                        child_id=child, sign=1,
                                        label=f'{self.names[child]} <- {self.names[parent]}'))
            if parent:
                self.interfaces.append(dict(body_id=parent, other_body_id=child,
                                            child_id=child, sign=-1,
                                            label=f'{self.names[parent]} <- {self.names[child]}'))
        shape = (len(self.interfaces), len(METRICS))
        self.minimum = np.full(shape, np.inf)
        self.maximum = np.full(shape, -np.inf)
        self.min_index = np.zeros(shape, dtype=int)
        self.max_index = np.zeros(shape, dtype=int)
        # Explicit capacity; stop without discarding old samples when full.
        estimate = 4096 + 8*(len(self.interfaces)*9 + len(self.body_ids)*37 + m.nv*5)
        self.max_samples = min(120000, max(1, 256*1024*1024//estimate))
        self.max_tracking_error_deg = 0.
        self.saturated_samples = 0
        self.contact_samples = 0

    @property
    def duration(self):
        return len(self.rows)*self.model.opt.timestep

    def stop(self, reason='Stopped by user.'):
        self.active = False
        self.reason = reason

    def event(self, time, name, value):
        if self.active:
            self.events.append(dict(time_s=float(time)-self.start_time, name=name, value=value))

    def capture(self, m, d, row, phase):
        if not self.active or (self.phases and phase.rsplit(': ', 1)[-1] not in self.phases):
            return
        mujoco.mj_rnePostConstraint(m, d)
        rotations = d.xmat.reshape((-1, 3, 3))
        loads, points = [], []
        for interface in self.interfaces:
            body, child = interface['body_id'], interface['child_id']
            # A hinge's anchor need not equal the child body origin.
            joint = int(m.body_jntadr[child])
            point = d.xanchor[joint] if m.body_jntnum[child] else d.xpos[child]
            reference = d.subtree_com[m.body_rootid[child]]
            loads.append(local_wrench(interface['sign']*d.cfrc_int[child],
                                      reference, point, rotations[body]))
            points.append(rotations[body].T@(point-d.xpos[body]))
        loads, points = np.asarray(loads), np.asarray(points)
        velocities, accelerations, external = [], [], []
        for body in self.body_ids:
            velocity, acceleration = np.zeros(6), np.zeros(6)
            # mjOBJ_BODY uses the body's COM, with world-oriented output.
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_BODY, body, acceleration, 0)
            # RNE includes -gravity in its acceleration. Export physical a_COM.
            # MuJoCo returns zero directly for bodies welded to the world.
            if m.body_weldid[body] and not (m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)):
                acceleration[3:] += m.opt.gravity
            velocities.append(velocity)
            accelerations.append(acceleration)
            external.append(local_wrench(d.cfrc_ext[body],
                            d.subtree_com[m.body_rootid[body]], d.xipos[body], rotations[body]))
        sample = dict(
            sample_index=len(self.rows), time_s=float(d.time)-self.start_time,
            simulation_time_s=float(d.time), phase=phase,
            q_rad=row['q'].copy(), qvel_rad_s=row['velocity'].copy(),
            qacc_rad_s2=d.qacc.copy(), target_rad=row['target'].copy(),
            actuator_torque_Nm=row['torque'].copy(), torque_cap_Nm=row['cap'].copy(),
            loads_local=loads, application_points_local_m=points,
            body_origin_world_m=d.xpos[1:].copy(), body_rotation_world=rotations[1:].copy(),
            com_world_m=d.xipos[1:].copy(),
            com_velocity_world=np.asarray(velocities),
            com_acceleration_world=np.asarray(accelerations),
            external_resultant_at_com_local=np.asarray(external),
            contacts=int(d.ncon))
        if not all(np.isfinite(v).all() for v in sample.values() if isinstance(v, np.ndarray)):
            self.stop('Invalid/nonfinite sample rejected; recording ended.')
            raise RuntimeError(self.reason)
        values = np.column_stack((loads, np.linalg.norm(loads[:, :3], axis=1),
                                  np.linalg.norm(loads[:, 3:], axis=1)))
        idx = len(self.rows)
        self.min_index = np.where(values < self.minimum, idx, self.min_index)
        self.max_index = np.where(values > self.maximum, idx, self.max_index)
        self.minimum = np.minimum(self.minimum, values)
        self.maximum = np.maximum(self.maximum, values)
        self.rows.append(sample)
        self.max_tracking_error_deg = max(self.max_tracking_error_deg,
            float(np.max(np.abs(np.rad2deg(row['q']-row['target'])), initial=0)))
        self.saturated_samples += int(np.any((row['cap'] > 1e-8) &
                                            (np.abs(row['torque']) >= .99*row['cap'])))
        self.contact_samples += int(d.ncon > 0)
        if len(self.rows) >= self.max_samples:
            self.stop('Recording capacity reached; all collected samples retained. Export before starting a new recording.')

    def peaks(self):
        result = []
        if not self.rows:
            return result
        for i, interface in enumerate(self.interfaces):
            for j, metric in enumerate(METRICS):
                low, high = int(self.min_index[i, j]), int(self.max_index[i, j])
                result.append(dict(interface=interface['label'], body=self.names[interface['body_id']],
                    metric=metric, minimum=float(self.minimum[i, j]), maximum=float(self.maximum[i, j]),
                    min_sample=low, max_sample=high, min_time_s=self.rows[low]['time_s'],
                    max_time_s=self.rows[high]['time_s']))
        return result

    def metadata(self):
        m = self.model
        return dict(schema_version=1, mujoco_version=mujoco.__version__,
            start_simulation_time_s=self.start_time, samples=len(self.rows),
            sample_interval_s=float(m.opt.timestep), duration_s=self.duration,
            stop_reason=self.reason, gravity_world_m_s2=m.opt.gravity.tolist(),
            frame='Each receiving body local frame (URDF link frame for imported bodies).',
            load_order=['Fx_N', 'Fy_N', 'Fz_N', 'Mx_Nm', 'My_Nm', 'Mz_Nm'],
            kinematics_order=['angular_xyz', 'linear_xyz'],
            convention='Loads act ON body_id FROM other_body_id; moments are about the listed application point.',
            external_loads='external_resultant_at_com_local includes contact/perturbation resultants, excludes gravity. Apply once; do not add duplicate contact loads.',
            inertia='com_acceleration_world is physical acceleration (gravity removed from RNE output). Use inertial body loads consistently for dynamic snapshots.',
            limitations=[
                'Rigid-body load export, not a structural integrity or fatigue verdict.',
                'Opposite sides of each interface are alternative receiving-body views; do not apply both to one body.',
                'Contact resultants do not retain contact pressure distributions; map to actual contact surfaces in FEA.',
                'Original generated model bodies aggregate structure, motor housings and any tool/payload. These are assembly loads, not isolated structural-part loads.',
                'CAD face mapping, support selection, mesh validation and material properties must be supplied in the FEA tool.',
                'Component extrema occur at different times; use complete snapshots as load cases.',
            ],
            max_tracking_error_deg=self.max_tracking_error_deg,
            saturated_samples=self.saturated_samples, contact_samples=self.contact_samples,
            contacts_enabled=self.cfg['simulation']['contacts'],
            events=self.events,
            bodies=[dict(body_id=i, name=self.names[i], parent_id=int(m.body_parentid[i]),
                         mass_kg=float(m.body_mass[i]), com_local_m=m.body_ipos[i].tolist(),
                         principal_inertia_kg_m2=m.body_inertia[i].tolist(),
                         inertial_frame_quaternion_wxyz=m.body_iquat[i].tolist()) for i in self.body_ids],
            interfaces=self.interfaces, joints=[j['name'] for j in self.cfg['joints']])

    def export(self, out, kind='all'):
        if self.active:
            raise ValueError('Stop recording before exporting.')
        if not self.rows:
            raise ValueError('No FEA samples to export.')
        if kind not in ('all', 'peaks', 'cases'):
            raise ValueError('Unknown FEA export type.')
        folder = Path(out).resolve() / ('fea-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ'))
        folder.mkdir(parents=True)
        (folder/'metadata.json').write_text(json.dumps(self.metadata(), indent=2))
        (folder/'config.yaml').write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        mujoco.mj_saveModel(self.model, str(folder/'model.mjb'), None)
        # Original source plus exact compiled model: later URDF edits cannot
        # change the mass properties or live gains in this recording.
        (folder/'source_model.xml').write_text(self.source_xml)
        peaks = self.peaks()
        with (folder/'peaks.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(peaks[0]))
            writer.writeheader()
            writer.writerows(peaks)
        if kind in ('all', 'cases'):
            indices = sorted(self.selected_samples or
                (set(self.min_index.ravel().tolist()) | set(self.max_index.ravel().tolist())))
            with (folder/'load_cases.json').open('w') as f:
                json.dump(dict(selection='user selected' if self.selected_samples else 'all extrema timestamps',
                               samples=[self.rows[i] for i in indices]), f, default=jsonable)
            self._write_loads(folder/'load_cases.csv', indices)
        if kind == 'all':
            # Stream instead of making a second in-memory copy of the history.
            with (folder/'history.jsonl').open('w') as f:
                for row in self.rows:
                    f.write(json.dumps(row, default=jsonable)+'\n')
            self._write_loads(folder/'interface_loads.csv', range(len(self.rows)))
        self.last_export = folder
        return folder

    def _write_loads(self, path, indices):
        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_index', 'time_s', 'phase', 'body', 'other_body',
                             'point_x_m', 'point_y_m', 'point_z_m', *METRICS[:6]])
            for index in indices:
                row = self.rows[index]
                for i, interface in enumerate(self.interfaces):
                    writer.writerow([index, row['time_s'], row['phase'],
                        self.names[interface['body_id']], self.names[interface['other_body_id']],
                        *row['application_points_local_m'][i], *row['loads_local'][i]])
