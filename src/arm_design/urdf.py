"""Import fixed-base rotary URDF robots through MuJoCo's native compiler."""
import os
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import yaml

from .config import load_config, positive, vector


def resolve_asset(filename, urdf_path, package_roots=(), meshdir=''):
    """Resolve local assets without stripping their directory components."""
    if not filename.startswith('file://'):
        filename = unquote(filename)
    if filename.startswith('package://'):
        package, sep, relative = filename[len('package://'):].partition('/')
        if not sep or not package or not relative:
            raise ValueError(f'Invalid package mesh URI: {filename}')
        roots = [Path(p).expanduser().resolve() for p in package_roots]
        roots += [Path(p).expanduser().resolve() for p in os.environ.get('ROS_PACKAGE_PATH', '').split(os.pathsep) if p]
        roots += list(urdf_path.parents)
        candidates = []
        for root in roots:
            if root.name == package:
                candidates.append(root / relative)
            candidates.append(root / package / relative)
    elif filename.startswith('file://'):
        uri = urlparse(filename)
        if uri.netloc not in ('', 'localhost'):
            raise ValueError(f'Only local file:// mesh paths are supported: {filename}')
        candidates = [Path(unquote(uri.path))]
    elif '://' in filename:
        raise ValueError(f'Download the mesh locally and update its URDF path: {filename}')
    else:
        asset = Path(filename).expanduser()
        candidates = [asset] if asset.is_absolute() else [urdf_path.parent / meshdir / asset]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    hint = ' Use --package-root /path/to/packages (or the package folder itself).' if filename.startswith('package://') else ''
    raise ValueError(f'Mesh not found: {filename} (URDF: {urdf_path}). Keep the exported mesh folders alongside the URDF.{hint}')


def read_robot(path, package_roots=()):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'URDF not found: {path}')
    text = path.read_text()
    if path.suffix.lower() == '.xacro' or '${' in text or '$(' in text:
        raise ValueError('Expand Xacro into a plain .urdf first: xacro robot.urdf.xacro > robot.urdf')
    try:
        robot = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f'Invalid URDF XML in {path}: {exc}') from exc
    if robot.tag != 'robot':
        raise ValueError('Expected a URDF <robot> root element.')
    links = [link.get('name') for link in robot.findall('link')]
    joints = robot.findall('joint')
    names = [j.get('name') for j in joints]
    if not links or any(not name for name in links) or len(set(links)) != len(links):
        raise ValueError('URDF links must have unique, nonempty names.')
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError('URDF joints must have unique, nonempty names.')
    definitions = {}
    children = set()
    parents = set()
    descendants = {link: [] for link in links}
    for joint in joints:
        name, kind = joint.get('name'), joint.get('type')
        if kind not in ('fixed', 'revolute', 'continuous'):
            raise ValueError(f'Joint {name!r} has unsupported type {kind!r}. This viewer supports fixed-base robots with fixed, revolute, and continuous joints.')
        if joint.find('mimic') is not None:
            raise ValueError(f'Joint {name!r} uses mimic coupling, which this importer does not yet support.')
        parent, child = joint.find('parent'), joint.find('child')
        if parent is None or child is None or parent.get('link') not in links or child.get('link') not in links:
            raise ValueError(f'Joint {name!r} must reference existing parent and child links.')
        if child.get('link') in children:
            raise ValueError(f'Link {child.get("link")!r} has more than one parent joint.')
        parents.add(parent.get('link'))
        children.add(child.get('link'))
        descendants[parent.get('link')].append(child.get('link'))
        if kind == 'fixed':
            continue
        limit = joint.find('limit')
        if limit is None or not all(key in limit.attrib for key in ('effort', 'velocity')):
            raise ValueError(f'Joint {name!r} needs positive effort and velocity limits in the URDF.')
        effort, velocity = float(limit.get('effort')), float(limit.get('velocity'))
        positive(effort, name + '.effort')
        positive(velocity, name + '.velocity')
        if kind == 'revolute':
            if not all(key in limit.attrib for key in ('lower', 'upper')):
                raise ValueError(f'Joint {name!r} needs lower and upper limits (radians).')
            bounds = vector([limit.get('lower'), limit.get('upper')], 2, name + '.limits')
            if bounds[0] >= bounds[1]:
                raise ValueError(f'Joint {name!r} lower limit must be less than its upper limit.')
            bounds = np.rad2deg(bounds).tolist()
        else:
            bounds = [-180., 180.]  # A finite jogging window; the physics joint remains unlimited.
            # Some exporters leave lower/upper on continuous joints. MuJoCo
            # otherwise interprets those as physical stops despite the type.
            limit.attrib.pop('lower', None)
            limit.attrib.pop('upper', None)
        definitions[name] = dict(type=kind, limits_deg=bounds, effort=effort, velocity=velocity)
    roots = set(links) - children
    if len(roots) != 1:
        raise ValueError('The URDF must have one connected robot tree and one fixed root link.')
    reached = set()
    pending = list(roots)
    while pending:
        link = pending.pop()
        if link in reached:
            raise ValueError('The URDF joint tree contains a cycle.')
        reached.add(link)
        pending.extend(descendants[link])
    if reached != set(links):
        raise ValueError('The URDF contains disconnected links or a joint cycle.')
    if not definitions:
        raise ValueError('The URDF needs at least one revolute or continuous joint to simulate.')

    extension = robot.find('mujoco')
    if extension is None:
        extension = ET.SubElement(robot, 'mujoco')
    compiler = extension.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(extension, 'compiler')
    meshdir = compiler.get('meshdir', '')
    aliases = {}
    mesh_keys = {}
    for mesh in robot.findall('.//mesh'):
        filename = mesh.get('filename')
        if not filename:
            raise ValueError('URDF mesh is missing its filename.')
        resolved = resolve_asset(filename, path, package_roots, meshdir)
        if resolved.suffix.lower() not in ('.stl', '.obj', '.msh'):
            raise ValueError(f'Unsupported mesh format: {resolved.name}. Export/convert the mesh to STL or OBJ and update the URDF filename.')
        # The native URDF reader identifies meshes by basename, so distinct
        # directories containing e.g. part.stl can otherwise silently alias.
        # Give each path/scale pair a unique parse-time name, then restore the
        # actual file path on the spec before compilation (no copied files).
        key = (str(resolved), mesh.get('scale', '1 1 1'))
        if key not in mesh_keys:
            alias = f'urdf_asset_{len(aliases)}_{resolved.suffix.lower()}'
            mesh_keys[key] = alias
            aliases[alias] = str(resolved)
        mesh.set('filename', mesh_keys[key])
    # Preserve visuals, fixed tool links, and complete asset paths. Inertia must
    # come from the URDF rather than guessing density from display meshes.
    compiler.attrib.update(strippath='false', discardvisual='false', fusestatic='false', inertiafromgeom='false')
    compiler.attrib.pop('meshdir', None)
    try:
        spec = mujoco.MjSpec.from_string(ET.tostring(robot, encoding='unicode'))
        for mesh in spec.meshes:
            mesh.file = aliases[mesh.file]
        model = spec.compile()
        xml = ET.fromstring(spec.to_xml())
    except ValueError as exc:
        raise ValueError(f'Could not compile URDF {path}: {exc}. Check mesh files and valid mass/inertia on moving links.') from exc
    ordered = [model.joint(i).name for i in range(model.njnt)]
    if set(ordered) != set(definitions) or model.nq != len(definitions) or model.nv != len(definitions):
        raise ValueError('The compiled URDF does not match its rotary joints; check the robot tree and MuJoCo extensions.')
    leaves = [link for link in links if link not in parents]
    return xml, [(name, definitions[name]) for name in ordered], leaves


def validate_urdf_config(c):
    if not c['urdf']['path']:
        raise ValueError('urdf.path is required')
    vector(c['base_xyz_m'], 3, 'base_xyz_m')
    vector(c['base_rpy_deg'], 3, 'base_rpy_deg')
    vector(c['urdf']['tip_offset_m'], 3, 'urdf.tip_offset_m')
    joints = c['joints']
    if not joints or len({j['name'] for j in joints}) != len(joints):
        raise ValueError('Imported joint names must be nonempty and unique')
    for j in joints:
        if j['type'] not in ('revolute', 'continuous'):
            raise ValueError('Only rotary joints are supported by the imported robot controls')
        lo, hi = vector(j['limits_deg'], 2, j['name'] + '.limits_deg')
        if lo >= hi:
            raise ValueError('Joint lower limit must be less than upper limit')
        positive(j['kp_nm_rad'], 'kp_nm_rad')
        positive(j['kd_nm_s_rad'], 'kd_nm_s_rad', zero=True)
        motor = c['motors'][j['motor']]
        for key in ('rated_torque_nm', 'peak_torque_nm', 'rated_speed_rpm', 'no_load_speed_rpm'):
            positive(motor[key], key)
        if motor['rated_torque_nm'] > motor['peak_torque_nm'] or motor['rated_speed_rpm'] >= motor['no_load_speed_rpm']:
            raise ValueError('Inconsistent imported motor limits')


def prepare_urdf_config(path, package_roots=()):
    """Create a sibling control file once, preserving it on subsequent launches."""
    path = Path(path).expanduser().resolve()
    settings = path.with_suffix('.sim.yaml')
    roots = [str(Path(p).expanduser().resolve()) for p in package_roots]
    if settings.exists():
        c = load_config(settings)
        if c.get('model_type') != 'urdf':
            raise ValueError(f'{settings} already exists and is not a URDF simulation config.')
        c['urdf']['path'] = str(path)
        if roots:
            c['urdf']['package_roots'] = roots
    else:
        xml, joints, leaves = read_robot(path, roots)
        c = dict(model_type='urdf', name=xml.get('model', path.stem),
                 urdf=dict(path=str(path), package_roots=roots, tip_link=leaves[-1], tip_offset_m=[0., 0., 0.]),
                 base_xyz_m=[0., 0., 0.], base_rpy_deg=[0., 0., 0.], gravity_m_s2=[0., 0., -9.81],
                 motors={}, joints=[],
                 drive=dict(torque_derating=1., speed_envelope=False, gravity_compensation=True),
                 simulation=dict(step_s=.001, contacts=False, floor_z_m=-.05),
                 sweep=dict(a_deg=[], b_deg=[], duration_s=3., approach_s=2., settle_s=1., hold_s=1.,
                            tracking_tolerance_deg=2., jog_speed_deg_s=30.))
        for name, info in joints:
            # URDF only supplies an effort ceiling, not a continuous motor rating
            # or torque-speed curve. Use equal plot thresholds and no envelope.
            c['motors'][name] = dict(rated_torque_nm=info['effort'], peak_torque_nm=info['effort'],
                                     rated_speed_rpm=info['velocity']*30/np.pi*.5,
                                     no_load_speed_rpm=info['velocity']*30/np.pi,
                                     source='URDF effort/velocity limits; no continuous rating or torque-speed curve')
            c['joints'].append(dict(name=name, type=info['type'], motor=name,
                                    limits_deg=info['limits_deg'], kp_nm_rad=40., kd_nm_s_rad=4.))
            initial = float(np.clip(0, *info['limits_deg']))
            c['sweep']['a_deg'].append(initial)
            c['sweep']['b_deg'].append(initial)
    from .model import build_arm
    build_arm(c)  # Don't leave a settings file behind for an invalid import.
    serialized = '# Controls for an imported URDF. Angles are degrees; source URDF uses SI units.\n'
    serialized += '# Geometry/inertia live in the URDF. Reload YAML reimports it.\n'
    serialized += yaml.safe_dump(c, sort_keys=False)
    if not settings.exists() or load_config(settings) != c:
        temporary = settings.with_suffix('.yaml.tmp')
        temporary.write_text(serialized)
        temporary.replace(settings)
    return settings


def build_urdf_arm(c):
    from .model import Arm, element, quat, rpy
    source = c['urdf']
    root, joints, _ = read_robot(source['path'], source.get('package_roots', []))
    if [j['name'] for j in c['joints']] != [name for name, _ in joints]:
        raise ValueError('URDF joints changed. Rename/remove its .sim.yaml file and launch with --urdf again to regenerate controls.')
    for j, (_, info) in zip(c['joints'], joints):
        if j['type'] != info['type'] or (info['type'] == 'revolute' and
                (j['limits_deg'][0] < info['limits_deg'][0] - 1e-6 or j['limits_deg'][1] > info['limits_deg'][1] + 1e-6)):
            raise ValueError(f'Controls for {j["name"]} no longer match the URDF limits/type. Update or regenerate the .sim.yaml file.')
        if c['motors'][j['motor']]['peak_torque_nm'] > info['effort'] + 1e-9:
            raise ValueError(f'Configured peak torque for {j["name"]} exceeds its URDF effort limit.')
    compiler = root.find('compiler')
    compiler.set('autolimits', 'true')
    option = root.find('option')
    if option is None:
        option = element(root, 'option')
    option.attrib.update(timestep=str(c['simulation']['step_s']), gravity=' '.join(map(str, c['gravity_m_s2'])), integrator='implicitfast')
    world = root.find('worldbody')
    bodies = list(world.findall('body'))
    base = element(world, 'body', pos=c['base_xyz_m'], quat=quat(rpy(c['base_rpy_deg'])))
    for body in bodies:
        world.remove(body)
        base.append(body)
    if not c['simulation']['contacts']:
        for geom in root.findall('.//geom'):
            geom.attrib.update(contype='0', conaffinity='0')
    element(world, 'light', pos=[0, -1, 3], dir=[0, 1, -2])
    element(world, 'geom', type='plane', pos=[0, 0, c['simulation']['floor_z_m']], size=[2, 2, .05],
            rgba=[.18, .2, .23, 1], contype=int(c['simulation']['contacts']), conaffinity=int(c['simulation']['contacts']))
    tip_body = next((body for body in root.findall('.//body') if body.get('name') == source['tip_link']), None)
    if tip_body is None:
        raise ValueError(f'Tip link {source["tip_link"]!r} not found. Set urdf.tip_link in the .sim.yaml file.')
    element(tip_body, 'site', name='tip', pos=source['tip_offset_m'], size=.008, rgba=[1, .2, .2, 1])
    actuators = element(root, 'actuator')
    for j in c['joints']:
        peak = c['drive']['torque_derating'] * c['motors'][j['motor']]['peak_torque_nm']
        element(actuators, 'position', joint=j['name'], kp=j['kp_nm_rad'], kv=j['kd_nm_s_rad'],
                ctrlrange=np.deg2rad(j['limits_deg']), forcerange=[-peak, peak], gear=1)
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    return Arm(c, model, xml, np.array([]))
