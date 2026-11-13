"""Integration checks for the replacement CAD arm with its real seventh link."""
from copy import deepcopy
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from arm_design.config import ROOT, load_config
from arm_design.model import build_arm
from arm_design.reporting import summary
from arm_design.simulation import run_headless, torque_limit


CONFIG = ROOT/'robots/moving_asm/urdf/moving_asm.sim.yaml'


def test_rs00_drives_and_housing_masses():
    cfg = load_config(CONFIG)
    arm = build_arm(cfg)
    assert (arm.model.nq, arm.model.nv, arm.model.nu) == (7, 7, 7)
    assert [j['motor'] for j in cfg['joints']] == ['03', '03', '06', '06', '00', '00', '00']
    assert [j['motor'] for j in load_config()['joints'][4:]] == ['00']*3
    for name in ('rs00', 'rs00_1', 'rs00_2'):
        assert arm.model.body(name).mass[0] == pytest.approx(.310)
    for name in ('rs06', 'rs06_1'):
        assert arm.model.body(name).mass[0] == pytest.approx(.621)
    np.testing.assert_allclose(torque_limit(cfg, np.zeros(7))[4:], [11.2]*3)
    np.testing.assert_allclose(torque_limit(cfg, np.zeros(7), rated=True)[4:], [4.]*3)
    speeds = np.zeros(7)
    speeds[4:] = 315*np.pi/30
    np.testing.assert_allclose(torque_limit(cfg, speeds)[4:], 0, atol=1e-12)


def test_j7_rotates_real_cad_link_on_fixed_housing():
    arm = build_arm(load_config(CONFIG))
    m, d = arm.model, arm.data()
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    housing = m.body('rs00_1').id
    tool = m.body('part_7').id
    j7 = m.joint('revolute_7').id
    assert m.body_parentid[tool] == housing
    assert m.body_jntnum[housing] == 0
    assert m.jnt_bodyid[j7] == tool
    assert m.body_mass[tool] == pytest.approx(.427991)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'end_effector') == -1
    assert m.nmesh == 10
    source = ET.parse(CONFIG.with_name('moving_asm.urdf')).getroot()
    cad_joint = source.find("joint[@name='revolute_7']")
    axis = np.fromstring(cad_joint.find('axis').get('xyz'), sep=' ')
    origin = np.fromstring(cad_joint.find('origin').get('xyz'), sep=' ')
    cad_inertia = source.find("link[@name='part_7']/inertial/inertia").attrib
    xx, yy, zz, xy, xz, yz = (float(cad_inertia[key]) for key in ('ixx','iyy','izz','ixy','ixz','iyz'))
    tensor = np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
    # MuJoCo's canonical XML round-trip writes six significant figures.
    np.testing.assert_allclose(sorted(m.body_inertia[tool]), np.linalg.eigvalsh(tensor), rtol=2e-5)
    fixed_position = d.xpos[:tool].copy()
    fixed_rotation = d.xmat[:tool].copy()
    housing_rotation = d.xmat[housing].reshape(3, 3).copy()
    initial_tool_rotation = d.xmat[tool].reshape(3, 3).copy()
    anchor = d.xpos[housing] + housing_rotation @ origin
    np.testing.assert_allclose(d.xanchor[j7], anchor, atol=1e-9)
    np.testing.assert_allclose(d.xaxis[j7], initial_tool_rotation @ axis, atol=1e-9)
    np.testing.assert_allclose(d.site('tip').xpos, d.xpos[tool], atol=1e-9)
    d.qpos[m.jnt_qposadr[j7]] = np.pi/2
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.xpos[:tool], fixed_position, atol=1e-10)
    np.testing.assert_allclose(d.xmat[:tool], fixed_rotation, atol=1e-10)
    relative_rotation = initial_tool_rotation.T @ d.xmat[tool].reshape(3, 3)
    np.testing.assert_allclose(relative_rotation, Rotation.from_rotvec(axis*np.pi/2).as_matrix(), atol=1e-9)


def test_seven_joint_sweep_tracks_with_rs00_caps():
    cfg = deepcopy(load_config(CONFIG))
    cfg['sweep']['a_deg'] = [0.]*7
    cfg['sweep']['b_deg'] = [10., -15., 15., 20., 10., -10., 60.]  # J7 limit is -56° to 70°
    arm, run = run_headless(cfg)
    result = summary(run)
    assert run.complete and result['tracking_ok'], result
    assert len(result['joints']) == 7
    assert result['joints'][-1]['joint'] == 'revolute_7'
    for row in run.rows:
        assert np.all(np.abs(row['torque']) <= row['cap']+1e-8)
    np.testing.assert_allclose(np.rad2deg(run.rows[-1]['q']), cfg['sweep']['b_deg'], atol=.1)
