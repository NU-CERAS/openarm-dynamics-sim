"""Turn an FEA load-case export into per-part FEA setups: one fixed face, applied loads, body loads."""
import csv
import json
from pathlib import Path

import numpy as np

from .fea import METRICS

COMPONENTS = METRICS[:6]


def exports(folder):
    """The load-case export folders in a campaign fea-*, bounds-*, or bounds-*/<kind> path."""
    folder = Path(folder).expanduser().resolve()
    if (folder/'peaks.csv').exists():
        return [folder]
    found = sorted(p.parent for p in folder.glob('*/peaks.csv'))
    if not found:
        raise ValueError(f'No load-case export (peaks.csv) in {folder}.')
    return found


def part_cases(folder, part, fixed=None, all_extremes=False):
    """Setups for every selected load case of one part in one export folder."""
    meta = json.loads((folder/'metadata.json').read_text())
    if 'parts' not in meta:
        raise ValueError(f'{folder} predates per-part exports (schema 2); rerun the campaign or bounds.')
    parts = {p['name']: p for p in meta['parts']}
    if part not in parts:
        raise ValueError(f'No structural part {part!r}. Parts: {", ".join(parts)}')
    body = parts[part]
    names = {b['body_id']: b['name'] for b in meta['bodies']} | {0: 'world'}
    rows_of = [(i, f) for i, f in enumerate(meta['interfaces']) if f['body_id'] == body['body_id']]
    mates = [(i, f) for i, f in rows_of if f['kind'] == 'mate']
    delta = next((i for i, f in rows_of if f['kind'] == 'delta'), None)
    others = [names[f['other_body_id']] for _, f in mates]
    # Fix the proximal side (toward the base) unless told otherwise.
    fixed = fixed or others[0]
    if fixed not in others:
        raise ValueError(f'{part} connects to {", ".join(others)}; cannot fix {fixed!r}.')
    labels = {f['label'] for _, f in rows_of}
    metrics = METRICS if all_extremes else ('force_N', 'moment_Nm')
    reasons = {}
    with (folder/'peaks.csv').open() as f:
        for peak in csv.DictReader(f):
            if peak['interface'] not in labels or peak['metric'] not in metrics:
                continue
            # Magnitudes only have meaningful maxima.
            for side in (('max',) if peak['metric'] in METRICS[6:] else ('min', 'max')):
                value = float(peak['maximum' if side == 'max' else 'minimum'])
                reasons.setdefault(int(peak[side+'_sample']), []).append(
                    f"{side} {peak['metric']} {value:.4g} at {peak['interface']}")
    samples = {s['sample_index']: s for s in json.loads((folder/'load_cases.json').read_text())['samples']}
    gravity_world = np.array(meta['gravity_world_m_s2'])
    k = body['body_id']-1
    # A merged part moves as one rigid body: combined mass, mass-weighted COM acceleration.
    members = [(b['body_id']-1, b['name'], b['mass_kg']) for b in meta['bodies'] if b['name'] in body['bodies']]
    mass = body['mass_kg']
    result = []
    for index in sorted(reasons):
        s = samples[index]
        R = np.array(s['body_rotation_world'][k])
        origin = np.array(s['body_origin_world_m'][k])
        loads = np.array(s['loads_local'])
        points = np.array(s['application_points_local_m'])
        rows = [dict(other=names[f['other_body_id']], point=points[i], load=loads[i]) for i, f in mates]
        gravity = R.T@gravity_world
        accelerations = np.array(s['com_acceleration_world'])
        acceleration = R.T@(sum(m*accelerations[b, 3:] for b, _, m in members)/mass if mass else accelerations[k, 3:])
        angular_velocity = R.T@np.array(s['com_velocity_world'][k][:3])
        angular_acceleration = R.T@accelerations[k, :3]
        contact = None
        if s.get('contact_body') in body['bodies']:
            contact = dict(point=R.T@(np.array(s['contact_point_world_m'])-origin),
                           force=R.T@np.array(s['contact_force_on_arm_world_N']))
            external = contact['force']
        else:
            # Each member's resultant is in its own frame; sum them in the part frame.
            external = R.T@sum(np.array(s['body_rotation_world'][b])@np.array(s['external_resultant_at_com_local'][b][:3])
                               for b, _, _ in members)
            if np.linalg.norm(external) > 1e-9:
                contact = dict(point=np.array(body['com_local_m']), force=external, at_com=True)
        # Newton on the rigid part: connections + external + weight = m a.
        residual = sum(r['load'][:3] for r in rows) + external + mass*(gravity-acceleration)
        result.append(dict(sample=index, phase=s['phase'], reasons=reasons[index], fixed=fixed,
                           applied=[r for r in rows if r['other'] != fixed],
                           reaction=next(r for r in rows if r['other'] == fixed),
                           delta=None if delta is None else dict(other=others[0], point=points[delta], load=loads[delta]),
                           bodies=body['bodies'], mass_kg=mass, com_m=np.array(body['com_local_m']),
                           gravity=gravity, acceleration=acceleration, effective_acceleration=gravity-acceleration,
                           angular_velocity=angular_velocity, angular_acceleration=angular_acceleration,
                           contact=contact, residual_n=float(np.linalg.norm(residual))))
    return result


def vec(v, digits=4):
    return '(' + ', '.join(f'{x:.{digits}g}' for x in v) + ')'


def report(folder, part, cases):
    lines = [f'=== {part}: {folder.parent.name}/{folder.name} ({len(cases)} load cases)',
             'Frame: the part\'s own link frame (URDF/CAD). Forces N, moments Nm, points m.']
    if cases and len(cases[0]['bodies']) > 1:
        lines.append(f"One solid part: {' + '.join(cases[0]['bodies'])}, in the {part} frame; combined mass and COM.")
    for c in cases:
        lines += ['', f"Case {c['sample']}: {c['phase']}", '  Selected for: '+'; '.join(c['reasons']),
                  f"  FIX: the {c['fixed']} mounting face (its reaction is computed by FEA; check: "
                  f"F {vec(c['reaction']['load'][:3])}, M {vec(c['reaction']['load'][3:])} at {vec(c['reaction']['point'])})"]
        for r in c['applied']:
            lines.append(f"  APPLY remote load from {r['other']} at {vec(r['point'])}: "
                         f"F {vec(r['load'][:3])}, M {vec(r['load'][3:])}")
        if c['contact']:
            where = 'at the COM (resultant)' if c['contact'].get('at_com') else 'at'
            lines.append(f"  APPLY contact force {where} {vec(c['contact']['point'])}: F {vec(c['contact']['force'])}")
        if c['delta']:
            lines.append(f"  DELTA (sum of mate loads about the {c['delta']['other']} mate; do not apply, body loads carry it): "
                         f"F {vec(c['delta']['load'][:3])}, M {vec(c['delta']['load'][3:])}")
        lines.append(f"  BODY: mass {c['mass_kg']:.4g} kg, COM {vec(c['com_m'])}; acceleration field (gravity - COM acceleration) "
                     f"{vec(c['effective_acceleration'])} m/s^2")
        if np.linalg.norm(c['angular_velocity']) > 1e-9 or np.linalg.norm(c['angular_acceleration']) > 1e-9:
            lines.append(f"        angular velocity {vec(c['angular_velocity'])} rad/s, "
                         f"angular acceleration {vec(c['angular_acceleration'])} rad/s^2")
        lines.append(f"  Force balance residual: {c['residual_n']:.2e} N")
    return '\n'.join(lines)


def write_csv(path, cases):
    with Path(path).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'phase', 'selected_for', 'role', 'from', 'point_x_m', 'point_y_m', 'point_z_m',
                         *COMPONENTS, 'accel_x_m_s2', 'accel_y_m_s2', 'accel_z_m_s2', 'mass_kg'])
        for c in cases:
            common = [c['sample'], c['phase'], '; '.join(c['reasons'])]
            body = [*c['effective_acceleration'], c['mass_kg']]
            writer.writerow(common+['fixed (reaction check)', c['fixed'], *c['reaction']['point'],
                                    *c['reaction']['load'], *body])
            for r in c['applied']:
                writer.writerow(common+['apply', r['other'], *r['point'], *r['load'], *body])
            if c['contact']:
                writer.writerow(common+['apply contact', 'environment', *c['contact']['point'],
                                        *c['contact']['force'], 0, 0, 0, *body])
