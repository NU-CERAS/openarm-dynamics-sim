import csv
import subprocess
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

from arm_design.config import load_config
from arm_design.experiment import Experiment
from arm_design.model import build_arm
from arm_design.reporting import save_result, summary
from arm_design.simulation import control, run_headless, step
from arm_design.urdf import prepare_urdf_config, resolve_asset


def robot_file(tmp_path, count=2):
    """A chain with nonalphabetical joint names and a fixed tool frame."""
    path = tmp_path / 'robot.urdf'
    links = ['<link name="base"/>']
    joints = []
    for i in range(count):
        links.append(f'''<link name="link{i}">
          <inertial><origin xyz="0 0 .1"/><mass value=".5"/>
            <inertia ixx=".003" iyy=".003" izz=".001" ixy="0" ixz="0" iyz="0"/></inertial>
          <visual><origin xyz="0 0 .1"/><geometry><box size=".05 .05 .2"/></geometry></visual>
          <collision><origin xyz="0 0 .1"/><geometry><box size=".05 .05 .2"/></geometry></collision>
        </link>''')
        joints.append(f'''<joint name="joint_{count-i}" type="revolute">
          <parent link="{'base' if i == 0 else 'link'+str(i-1)}"/><child link="link{i}"/>
          <origin xyz="0 0 .2"/><axis xyz="0 0 1"/>
          <limit lower="-1.5" upper="1.5" effort="15" velocity="2"/>
          <dynamics damping=".1" friction=".02"/>
        </joint>''')
    path.write_text('<robot name="test">'+''.join(links)+''.join(reversed(joints))+f'''
      <link name="tool"/><joint name="mount" type="fixed">
        <parent link="link{count-1}"/><child link="tool"/><origin xyz="0 0 .2"/>
      </joint></robot>''')
    return path


def rewrite(path, edit):
    robot = ET.parse(path)
    edit(robot.getroot())
    robot.write(path)


@pytest.mark.parametrize('count', [1, 2, 8])
def test_import_controls_and_reports_adapt_to_joint_count(tmp_path, count):
    path = robot_file(tmp_path, count)
    original = path.read_bytes()
    settings = prepare_urdf_config(path)
    cfg = load_config(settings)
    cfg['sweep'].update(a_deg=[0.]*count, b_deg=[15.]*count, duration_s=.8,
                        approach_s=.1, settle_s=.1, hold_s=.2)
    arm, run = run_headless(cfg)
    assert (arm.model.nq, arm.model.nv, arm.model.nu) == (count, count, count)
    assert [j['name'] for j in cfg['joints']] == [f'joint_{i}' for i in range(count, 0, -1)]
    assert run.complete and summary(run)['tracking_ok']
    np.testing.assert_allclose(arm.model.body_mass.sum(), .5*count)
    np.testing.assert_allclose(arm.data().site('tip').xpos, [0, 0, .2*(count+1)], atol=1e-8)
    assert path.read_bytes() == original
    folder = save_result(arm, run, tmp_path / 'reports')
    assert mujoco.MjModel.from_xml_path(str(folder/'model.xml')).nu == count
    with (folder/'loads.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(run.rows)
    assert len(rows[0]) == 6 + 7*count
    assert (folder/'loads.png').stat().st_size > 1000
    assert len(summary(run)['joints']) == count


def test_saved_poses_gains_reload_and_geometry_guard(tmp_path):
    path = robot_file(tmp_path)
    settings = prepare_urdf_config(path)
    e = Experiment(settings, tmp_path/'reports')
    e.set_gains(0, 55, 5)
    e.set_jog_speed(10)
    e.set_target(0, 20)
    for _ in range(100):
        e.tick()
    e.save_pose('B')
    e.save_settings()
    saved = settings.read_text()
    prepare_urdf_config(path)
    assert settings.read_text() == saved
    again = Experiment(settings, tmp_path/'reports')
    assert again.cfg['joints'][0]['kp_nm_rad'] == 55
    np.testing.assert_allclose(again.cfg['sweep']['b_deg'], np.rad2deg(e.data.qpos))
    assert again.geometry_settings() is None
    with pytest.raises(ValueError, match='source URDF'):
        again.apply_geometry({})
    rewrite(path, lambda root: root.find("link[@name='link0']/inertial/mass").set('value', '1.5'))
    again.load()
    assert again.arm.model.body('link0').mass[0] == 1.5
    old = again.arm
    path.write_text('<invalid>')
    with pytest.raises(ValueError, match='Invalid URDF XML'):
        again.load()
    assert again.arm is old


def test_continuous_joint_window_and_effort_cap(tmp_path):
    path = robot_file(tmp_path, 1)
    rewrite(path, lambda root: root.find("joint[@name='joint_1']").set('type', 'continuous'))
    cfg = load_config(prepare_urdf_config(path))
    assert cfg['joints'][0]['limits_deg'] == [-180, 180]
    cfg['drive']['gravity_compensation'] = False
    arm = build_arm(cfg)
    assert not arm.model.jnt_limited[0]
    data = arm.data()
    control(arm, data, [1.])
    mujoco.mj_forward(arm.model, data)
    assert data.qfrc_actuator[0] == pytest.approx(15)
    assert step(arm, data, [1.])['cap'].shape == (1,)


def test_contacts_base_transform_and_tip_offset(tmp_path):
    cfg = load_config(prepare_urdf_config(robot_file(tmp_path, 1)))
    arm = build_arm(cfg)
    assert not arm.model.geom_contype.any()
    cfg['simulation']['contacts'] = True
    cfg['base_xyz_m'] = [.3, .2, 1.]
    cfg['base_rpy_deg'] = [0., 90., 0.]
    cfg['urdf']['tip_offset_m'] = [0., 0., .1]
    arm = build_arm(cfg)
    assert arm.model.geom_contype.any()
    np.testing.assert_allclose(arm.data().site('tip').xpos, [.8, .2, 1.], atol=1e-9)


def tetrahedron(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')


@pytest.mark.parametrize('scheme', ['relative', 'absolute', 'file', 'package', 'package_root', 'ros_env'])
def test_mesh_paths_compile_and_exports_reopen(tmp_path, monkeypatch, scheme):
    package = tmp_path/'robot_description'
    urdf_dir = package/'urdf'
    urdf_dir.mkdir(parents=True)
    path = robot_file(urdf_dir, 1)
    mesh = package/'meshes'/'arm.obj'
    tetrahedron(mesh)
    roots = []
    if scheme == 'relative':
        filename = '../meshes/arm.obj'
    elif scheme == 'absolute':
        filename = str(mesh)
    elif scheme == 'file':
        filename = mesh.as_uri()
    else:
        filename = 'package://robot_description/meshes/arm.obj'
        if scheme in ('package_root', 'ros_env'):
            external = tmp_path/'elsewhere'/'packages'
            external.mkdir(parents=True)
            package.rename(external/'robot_description')
            path = robot_file(tmp_path, 1)
            if scheme == 'package_root':
                roots = [external]
            else:
                monkeypatch.setenv('ROS_PACKAGE_PATH', str(external))
    def set_mesh(root):
        geometry = root.find("link[@name='link0']/visual/geometry")
        geometry.clear()
        ET.SubElement(geometry, 'mesh', filename=filename)
    rewrite(path, set_mesh)
    cfg = load_config(prepare_urdf_config(path, roots))
    arm = build_arm(cfg)
    assert arm.model.nmesh == 1
    output = tmp_path/'export.xml'
    output.write_text(arm.xml)
    monkeypatch.chdir(tmp_path.parent)
    assert mujoco.MjModel.from_xml_path(str(output)).nmesh == 1


def test_joint_schema_changes_are_not_silently_reused(tmp_path):
    path = robot_file(tmp_path, 1)
    settings = prepare_urdf_config(path)
    before = settings.read_bytes()
    rewrite(path, lambda root: root.find("joint[@name='joint_1']").set('name', 'renamed'))
    with pytest.raises(ValueError, match='joints changed'):
        prepare_urdf_config(path)
    assert settings.read_bytes() == before


def test_meshes_with_same_basename_keep_their_own_shape_and_scale(tmp_path):
    path = robot_file(tmp_path, 2)
    for i in range(2):
        mesh = tmp_path/str(i)/'part.obj'
        tetrahedron(mesh)
        if i == 1:
            mesh.write_text(mesh.read_text().replace('.1', '.2'))
    def edit(root):
        for i in range(2):
            geometry = root.find(f"link[@name='link{i}']/visual/geometry")
            geometry.clear()
            ET.SubElement(geometry, 'mesh', filename=f'{i}/part.obj', scale='2 2 2' if i else '1 1 1')
    rewrite(path, edit)
    arm = build_arm(load_config(prepare_urdf_config(path)))
    model = arm.model
    assert model.nmesh == 2
    sizes = []
    for i in range(2):
        start, count = model.mesh_vertadr[i], model.mesh_vertnum[i]
        vertices = model.mesh_vert[start:start+count]
        sizes.append(np.max(np.linalg.norm(vertices[:, None, :]-vertices[None, :, :], axis=-1)))
    assert sizes[1]/sizes[0] == pytest.approx(4.)


def test_file_uri_with_encoded_spaces_and_hash(tmp_path):
    mesh = tmp_path/'part #1.obj'
    tetrahedron(mesh)
    assert resolve_asset(mesh.as_uri(), tmp_path/'robot.urdf') == mesh.resolve()


def test_disconnected_cycle_is_rejected(tmp_path):
    path = robot_file(tmp_path, 1)
    def edit(root):
        for name, other in [('a', 'b'), ('b', 'a')]:
            ET.SubElement(root, 'link', name=name)
            joint = ET.SubElement(root, 'joint', name=name+'_joint', type='fixed')
            ET.SubElement(joint, 'parent', link=name)
            ET.SubElement(joint, 'child', link=other)
    rewrite(path, edit)
    with pytest.raises(ValueError, match='disconnected'):
        prepare_urdf_config(path)


@pytest.mark.parametrize('kind', ['prismatic', 'floating', 'planar'])
def test_unsupported_joints_rejected_before_settings_written(tmp_path, kind):
    path = robot_file(tmp_path, 1)
    rewrite(path, lambda root: root.find("joint[@name='joint_1']").set('type', kind))
    with pytest.raises(ValueError, match='unsupported type'):
        prepare_urdf_config(path)
    assert not path.with_suffix('.sim.yaml').exists()


@pytest.mark.parametrize('problem, match', [
    ('missing_mesh', 'Mesh not found'), ('dae', 'Unsupported mesh format'),
    ('mimic', 'mimic'), ('effort', 'effort'), ('inertia', 'mass/inertia'), ('xacro', 'Expand Xacro'),
])
def test_bad_imports_have_actionable_errors(tmp_path, problem, match):
    path = robot_file(tmp_path, 1)
    def edit(root):
        joint = root.find("joint[@name='joint_1']")
        if problem in ('missing_mesh', 'dae'):
            geometry = root.find("link[@name='link0']/visual/geometry")
            geometry.clear()
            filename = 'missing.obj' if problem == 'missing_mesh' else 'arm.dae'
            ET.SubElement(geometry, 'mesh', filename=filename)
            if problem == 'dae':
                (tmp_path/filename).write_text('<COLLADA/>')
        elif problem == 'mimic':
            ET.SubElement(joint, 'mimic', joint='other')
        elif problem == 'effort':
            joint.find('limit').set('effort', '0')
        elif problem == 'inertia':
            root.find("link[@name='link0']").remove(root.find("link[@name='link0']/inertial"))
        else:
            root.find("link[@name='link0']/inertial/mass").set('value', '${mass}')
    rewrite(path, edit)
    with pytest.raises(ValueError, match=match):
        prepare_urdf_config(path)
    assert not path.with_suffix('.sim.yaml').exists()


def test_cli_import_export_and_missing_file(tmp_path):
    path = robot_file(tmp_path, 1)
    output = tmp_path/'export'
    result = subprocess.run([sys.executable, '-m', 'arm_design.cli', 'export', '--urdf', str(path), '--out', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert mujoco.MjModel.from_xml_path(str(output/'arm.xml')).nu == 1
    result = subprocess.run([sys.executable, '-m', 'arm_design.cli', '--urdf', str(tmp_path/'missing.urdf')],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'URDF not found' in result.stderr and 'Traceback' not in result.stderr
