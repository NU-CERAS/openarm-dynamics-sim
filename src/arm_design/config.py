"""The editable YAML is the single source of model and experiment parameters."""
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs' / 'arm.yaml'


def vector(value, n, label):
    v = np.asarray(value, float)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ValueError(f'{label}: expected {n} finite numbers')
    return v


def positive(value, label, zero=False):
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{label} must be {"nonnegative" if zero else "positive"}')


def validate(c):
    for k in ('base_xyz_m', 'base_rpy_deg', 'gravity_m_s2'):
        vector(c[k], 3, k)
    if len(c['joints']) != 7 or len(c['links']) != 7:
        raise ValueError('Exactly seven joints and seven links are required')
    if len({j['name'] for j in c['joints']}) != 7:
        raise ValueError('Joint names must be unique')
    for j, link in zip(c['joints'], c['links']):
        for obj, key in ((j, 'axis'), (link, 'direction')):
            if not np.isclose(np.linalg.norm(vector(obj[key], 3, key)), 1):
                raise ValueError(f'{key} must be a unit vector')
        vector(j['origin_rpy_deg'], 3, 'origin_rpy_deg')
        lo, hi = vector(j['limits_deg'], 2, 'limits_deg')
        if lo >= hi:
            raise ValueError('Joint lower limit must be less than upper limit')
        positive(j['kp_nm_rad'], 'kp_nm_rad')
        positive(j['kd_nm_s_rad'], 'kd_nm_s_rad', zero=True)
        for k in ('friction_nm', 'damping_nm_s_rad', 'armature_kg_m2'):
            positive(j[k], k, zero=True)
        if j.get('motor_mass_override_kg') is not None:
            positive(j['motor_mass_override_kg'], 'motor_mass_override_kg')
        motor = c['motors'][j['motor']]
        for k in ('mass_kg', 'diameter_m', 'depth_m', 'rated_torque_nm', 'peak_torque_nm', 'rated_speed_rpm', 'no_load_speed_rpm'):
            positive(motor[k], k)
        if motor['rated_torque_nm'] > motor['peak_torque_nm'] or motor['rated_speed_rpm'] >= motor['no_load_speed_rpm']:
            raise ValueError('Inconsistent motor ratings')
        for k in ('length_mm', 'outer_mm', 'wall_mm', 'density_kg_m3', 'bracket_radius_m'):
            positive(link[k], k)
        positive(link['bracket_mass_kg'], 'bracket_mass_kg', zero=True)
        if 2 * link['wall_mm'] >= link['outer_mm']:
            raise ValueError('Tube wall must be less than half its outer dimension')
        if link['mass_override_kg'] is not None:
            positive(link['mass_override_kg'], 'mass_override_kg')
        if link['com_override_m'] is not None:
            vector(link['com_override_m'], 3, 'com_override_m')
        if link['inertia_override_kg_m2'] is not None:
            xx, yy, zz, xy, xz, yz = vector(link['inertia_override_kg_m2'], 6, 'inertia_override_kg_m2')
            eig = np.linalg.eigvalsh([[xx,xy,xz], [xy,yy,yz], [xz,yz,zz]])
            if eig[0] <= 0 or eig[2] > eig[0] + eig[1] + 1e-12:
                raise ValueError('Inertia must be positive and satisfy the triangle inequality')
    for k in ('tool', 'payload'):
        positive(c[k]['mass_kg'], k + '.mass_kg', zero=True)
        if np.any(vector(c[k]['box_size_m'], 3, k + '.box_size_m') <= 0):
            raise ValueError('Box sizes must be positive')
    vector(c['payload']['com_offset_from_tip_m'], 3, 'payload offset')
    if not 0 <= c['tool']['com_fraction_of_last_span'] <= 1:
        raise ValueError('Tool COM fraction must be between 0 and 1')
    if not 0 < c['drive']['torque_derating'] <= 1:
        raise ValueError('Torque derating must be in (0, 1]')
    for k in ('speed_envelope', 'gravity_compensation'):
        if not isinstance(c['drive'][k], bool):
            raise ValueError(f'drive.{k} must be true or false')
    if not isinstance(c['simulation']['contacts'], bool):
        raise ValueError('simulation.contacts must be true or false')
    if not np.isfinite(c['simulation']['floor_z_m']):
        raise ValueError('floor_z_m must be finite')
    if not 0 < c['simulation']['step_s'] <= .005:
        raise ValueError('MuJoCo timestep must be in (0, 0.005] seconds')
    s = c['sweep']
    bounds = np.array([j['limits_deg'] for j in c['joints']])
    for k in ('a_deg', 'b_deg'):
        q = vector(s[k], 7, k)
        if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
            raise ValueError(f'{k} exceeds a joint limit')
    for k in ('duration_s', 'approach_s', 'settle_s', 'hold_s'):
        if not .1 <= s[k] <= 120:
            raise ValueError(f'{k} must be between 0.1 and 120 seconds')
    positive(s['tracking_tolerance_deg'], 'tracking_tolerance_deg')
    positive(s['jog_speed_deg_s'], 'jog_speed_deg_s')


def load_config(path=DEFAULT_CONFIG):
    c = yaml.safe_load(Path(path).read_text())
    validate(c)
    return c


def save_sweep(path, sweep):
    save_controls(path, sweep)


def merge_geometry(c, geometry):
    if len(geometry['links']) != 7:
        raise ValueError('Exactly seven link settings are required')
    for link, edit in zip(c['links'],geometry['links']):
        for key in ('length_mm','mass_override_kg'):
            link[key] = edit[key]
    c['tool']['mass_kg'] = geometry['tool_mass_kg']
    c['payload']['mass_kg'] = geometry['payload_mass_kg']
    if 'motor_mass_overrides_kg' in geometry:
        overrides = geometry['motor_mass_overrides_kg']
        if set(overrides) != {j['name'] for j in c['joints']}:
            raise ValueError('Joint names changed; reload before saving actuator masses')
        for joint in c['joints']:
            joint['motor_mass_override_kg'] = overrides[joint['name']]


def save_controls(path, sweep_updates=None, gains=None, geometry=None):
    """Merge only requested controls into the latest file; keep geometry edits."""
    path = Path(path)
    c = load_config(path)
    if sweep_updates:
        c['sweep'].update(sweep_updates)
    if gains:
        if set(gains) != {j['name'] for j in c['joints']}:
            raise ValueError('Joint names changed on disk; reload before saving gains')
        for j in c['joints']:
            j.update(gains[j['name']])
    if geometry is not None:
        merge_geometry(c,geometry)
    validate(c)
    # Keep the explanatory header; safe_dump preserves values, not inline comments.
    header = []
    for line in path.read_text().splitlines():
        if not line.startswith('#'):
            break
        header.append(line)
    text = '\n'.join(header) + '\n' + yaml.safe_dump(c, sort_keys=False)
    tmp = path.with_suffix('.yaml.tmp')
    tmp.write_text(text)
    tmp.replace(path)
