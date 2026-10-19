"""Generate an MJCF model; MuJoCo owns all dynamics and integration."""
from dataclasses import dataclass
from xml.etree import ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .config import validate


def rpy(deg):
    return Rotation.from_euler('xyz', deg, degrees=True).as_matrix()


def quat(rotation):
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def align_z(z):
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    helper = np.array([1., 0, 0]) if abs(z[0]) < .9 else np.array([0., 1, 0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def values(x):
    return ' '.join(f'{v:.12g}' for v in np.asarray(x).ravel())


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) if np.isscalar(v) else values(v) for k,v in attrs.items()})


def box_inertia(mass, size):
    x, y, z = size
    return np.diag([y*y+z*z, x*x+z*z, x*x+y*y]) * mass / 12


def combine(parts):
    """Combine (mass, COM, inertia-about-COM) in one common frame."""
    mass = sum(p[0] for p in parts)
    com = sum(m * np.asarray(c) for m,c,_ in parts) / mass
    inertia = np.zeros((3, 3))
    for m,c,I in parts:
        d = np.asarray(c) - com
        inertia += I + m * (np.dot(d,d) * np.eye(3) - np.outer(d,d))
    return mass, com, inertia


def structure(link):
    L, b, t = np.array([link['length_mm'],link['outer_mm'],link['wall_mm']]) / 1000
    inner = b - 2*t
    m = link['density_kg_m3'] * (b*b-inner*inner) * L
    I = np.diag([m*(L*L+b*b+inner*inner)/12]*2 + [m*(b*b+inner*inner)/6])
    R = align_z(link['direction'])
    span = np.array(link['direction']) * L
    bm = link['bracket_mass_kg']
    mass, com, inertia = combine([(m, span/2, R@I@R.T), (bm, np.zeros(3), np.eye(3)*.4*bm*link['bracket_radius_m']**2)])
    if link['mass_override_kg'] is not None:
        inertia *= link['mass_override_kg'] / mass
        mass = link['mass_override_kg']
    if link['com_override_m'] is not None:
        com = np.array(link['com_override_m'], float)
    if link['inertia_override_kg_m2'] is not None:
        xx, yy, zz, xy, xz, yz = link['inertia_override_kg_m2']
        inertia = np.array([[xx,xy,xz], [xy,yy,yz], [xz,yz,zz]])
    return mass, com, inertia


def motor_part(c, i, position, rotation):
    j = c['joints'][i]
    m = c['motors'][j['motor']]
    mass = j.get('motor_mass_override_kg')
    if mass is None:
        mass = m['mass_kg']
    radius, depth = m['diameter_m']/2, m['depth_m']
    R = rotation @ align_z(j['axis'])
    I = np.diag([mass*(3*radius**2+depth**2)/12]*2 + [mass*radius**2/2])
    return (mass, position, R@I@R.T), R, radius, depth


@dataclass
class Arm:
    cfg: dict
    model: object
    xml: str
    structural_masses: np.ndarray

    def data(self):
        d = mujoco.MjData(self.model)
        d.qpos[:] = np.deg2rad(self.cfg['sweep']['a_deg'])
        d.ctrl[:] = d.qpos
        mujoco.mj_forward(self.model, d)
        return d


def build_arm(c):
    validate(c)
    if c.get('model_type') == 'urdf':
        from .urdf import build_urdf_arm
        return build_urdf_arm(c)
    root = ET.Element('mujoco', model=c['name'])
    element(root, 'compiler', angle='radian', autolimits='true')
    element(root, 'option', timestep=c['simulation']['step_s'], gravity=c['gravity_m_s2'], integrator='implicitfast')
    defaults = element(root, 'default')
    element(defaults, 'geom', contype=1 if c['simulation']['contacts'] else 0, conaffinity=1 if c['simulation']['contacts'] else 0, friction=[.6,.005,.0001])
    world = element(root, 'worldbody')
    element(world, 'light', pos=[0,-1,2], dir=[0,1,-2])
    element(world, 'geom', name='floor', type='plane', pos=[0,0,c['simulation']['floor_z_m']], size=[2,2,.05], rgba=[.18,.2,.23,1])
    base = element(world, 'body', name='base', pos=c['base_xyz_m'], quat=quat(rpy(c['base_rpy_deg'])))
    motor, R, radius, depth = motor_part(c, 0, np.zeros(3), rpy(c['joints'][0]['origin_rpy_deg']))
    element(base, 'geom', name='motor_J1', type='cylinder', size=[radius,depth/2], quat=quat(R), mass=motor[0], rgba=[.2,.2,.22,1])
    parent = base
    spans = [np.asarray(l['direction'])*l['length_mm']/1000 for l in c['links']]
    masses = []
    for i,(j,link) in enumerate(zip(c['joints'], c['links'])):
        body = element(parent, 'body', name=f'link{i+1}', pos=np.zeros(3) if i == 0 else spans[i-1], quat=quat(rpy(j['origin_rpy_deg'])))
        element(body, 'joint', name=j['name'], type='hinge', axis=j['axis'], range=np.deg2rad(j['limits_deg']), damping=j['damping_nm_s_rad'], frictionloss=j['friction_nm'], armature=j['armature_kg_m2'])
        structural = structure(link)
        masses.append(structural[0])
        parts = [structural]
        # The housing of J(i+1) belongs to the parent link, not its own output.
        if i < 6:
            motor, R, radius, depth = motor_part(c, i+1, spans[i], rpy(c['joints'][i+1]['origin_rpy_deg']))
            parts.append(motor)
            element(body, 'geom', name=f'motor_J{i+2}', type='cylinder', pos=spans[i], size=[radius,depth/2], quat=quat(R), rgba=[.2,.2,.22,1])
        else:
            for name, obj, pos in [('tool', c['tool'], spans[i]*c['tool']['com_fraction_of_last_span']), ('payload', c['payload'], spans[i]+np.asarray(c['payload']['com_offset_from_tip_m']))]:
                if obj['mass_kg']:
                    parts.append((obj['mass_kg'], pos, box_inertia(obj['mass_kg'], obj['box_size_m'])))
                    element(body, 'geom', name=name, type='box', pos=pos, size=np.asarray(obj['box_size_m'])/2, rgba=[.9,.55,.15,1] if name=='payload' else [.3,.3,.3,1])
            element(body, 'site', name='tip', pos=spans[i], size=.008, rgba=[1,.2,.2,1])
        mass, com, I = combine(parts)
        element(body, 'inertial', pos=com, mass=mass, fullinertia=[I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]])
        element(body, 'geom', name=f'tube{i+1}', type='box', pos=spans[i]/2, quat=quat(align_z(link['direction'])), size=[link['outer_mm']/2000]*2 + [link['length_mm']/2000], rgba=[.3,.62,.82,1])
        parent = body
    actuators = element(root, 'actuator')
    for j in c['joints']:
        peak = c['drive']['torque_derating']*c['motors'][j['motor']]['peak_torque_nm']
        element(actuators, 'position', name=j['name']+'_target_rad', joint=j['name'], kp=j['kp_nm_rad'], kv=j['kd_nm_s_rad'], ctrlrange=np.deg2rad(j['limits_deg']), forcerange=[-peak, peak], gear=1)
    sensors = element(root, 'sensor')
    for j in c['joints']:
        element(sensors, 'jointactuatorfrc', name=j['name']+'_torque_Nm', joint=j['name'])
    xml = ET.tostring(root, encoding='unicode')
    return Arm(c, mujoco.MjModel.from_xml_string(xml), xml, np.array(masses))
