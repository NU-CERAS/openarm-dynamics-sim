from copy import deepcopy
from pathlib import Path
import numpy as np
import mujoco
import pytest
import yaml
from arm_design.config import load_config, DEFAULT_CONFIG
from arm_design.experiment import Experiment
from arm_design.simulation import control


@pytest.fixture
def experiment(tmp_path):
    path=tmp_path/'arm.yaml'
    config=load_config(DEFAULT_CONFIG)
    # Pose-saving in the interactive viewer must not change test preconditions.
    config['sweep']['a_deg']=[0,-25,0,50,0,0,0]
    config['sweep']['b_deg']=[30,-55,20,85,15,10,-15]
    path.write_text(yaml.safe_dump(config))
    return Experiment(path,tmp_path/'results')


def finish_move(e):
    for _ in range(150000):
        e.tick()
        if not e.busy: return
    pytest.fail('Move never completed')


def test_plot_retains_ten_seconds_on_fast_displays(experiment):
    e=experiment
    for timestamp in np.arange(0,12.001,1/144):
        e.data.time=timestamp
        e.sample_history()
    assert 9.95 < e.history[-1][0]-e.history[0][0] <= 10
    assert len(e.history)<=501
    before=len(e.history)
    e.sample_history()  # A repeated frame with no simulation step adds no point.
    assert len(e.history)==before


def test_go_to_b_is_dynamic_and_go_to_a_returns(experiment):
    e=experiment
    q=e.data.qpos.copy()
    e.go_to('B')
    # Starting a Go-to must not teleport the actual robot.
    np.testing.assert_array_equal(e.data.qpos,q)
    e.tick()
    assert np.max(np.abs(e.data.qpos-q))<1e-3
    finish_move(e)
    np.testing.assert_allclose(np.rad2deg(e.data.qpos),e.cfg['sweep']['b_deg'],atol=.1)
    assert 'Reached B' in e.message
    e.go_to('A');finish_move(e)
    np.testing.assert_allclose(np.rad2deg(e.data.qpos),e.cfg['sweep']['a_deg'],atol=.1)
    assert e.last_report is None  # Go-to doesn't silently start a recorded sweep.


def test_save_pose_uses_actual_angles_and_preserves_other_edits(experiment):
    e=experiment
    e.target[0]=.8
    expected=np.rad2deg(e.data.qpos).tolist()
    external=load_config(e.path)
    external['links'][0]['length_mm']=110
    external['sweep']['jog_speed_deg_s']=25
    e.path.write_text(yaml.safe_dump(external))
    e.save_pose('B')
    saved=load_config(e.path)
    np.testing.assert_allclose(saved['sweep']['b_deg'],expected)
    assert saved['links'][0]['length_mm']==110
    assert saved['sweep']['jog_speed_deg_s']==25


def test_gains_change_actual_actuator_and_persist_independently(experiment):
    e=experiment
    initial=e.arm.model.actuator_gainprm[:,0].copy()
    e.set_gains(2,160,0)
    np.testing.assert_allclose(e.arm.model.actuator_gainprm[[0,1,3,4,5,6],0],initial[[0,1,3,4,5,6]])
    assert e.arm.model.actuator_gainprm[2,0]==160
    np.testing.assert_allclose(e.arm.model.actuator_biasprm[2,1:3],[-160,0])
    # Measured torque responds to the configured P gain before saturation.
    e.cfg['drive']['gravity_compensation']=False
    target=e.data.qpos.copy();target[2]+=.01
    control(e.arm,e.data,target)
    mujoco.mj_forward(e.arm.model,e.data)
    assert e.data.qfrc_actuator[2]==pytest.approx(1.6)
    e.set_duration(2.25);e.set_jog_speed(45)
    e.save_settings()
    saved=load_config(e.path)
    assert saved['joints'][2]['kp_nm_rad']==160 and saved['joints'][2]['kd_nm_s_rad']==0
    assert saved['sweep']['duration_s']==2.25 and saved['sweep']['jog_speed_deg_s']==45


def test_jog_remains_rate_limited_after_goto_and_stop(experiment):
    e=experiment;e.go_to('B');finish_move(e)
    before=e.command.copy()
    e.set_jog_speed(10)
    e.set_target(0,np.rad2deg(before[0])+10)
    np.testing.assert_array_equal(e.command,before)
    e.tick()
    assert np.rad2deg(e.command[0]-before[0])==pytest.approx(.01)
    e.stop();before=e.command.copy();e.set_target(0,np.rad2deg(before[0])+10)
    e.tick()
    assert np.rad2deg(e.command[0]-before[0])==pytest.approx(.01)


def test_motion_locks_settings_and_pause_stop_work(experiment):
    e=experiment;e.go_to('B')
    with pytest.raises(ValueError): e.save_pose('A')
    with pytest.raises(ValueError): e.set_gains(0,100,10)
    with pytest.raises(ValueError): e.start_sweep()
    with pytest.raises(ValueError): e.zero_arm()
    with pytest.raises(ValueError): e.randomize_joints()
    e.paused=True;t=e.data.time;e.tick();assert e.data.time==t
    e.stop();assert not e.busy
    np.testing.assert_allclose(e.command,e.data.qpos)


def test_zero_and_random_targets_use_jog_motion(experiment):
    e=experiment
    actual=e.data.qpos.copy();command=e.command.copy()
    saved=e.path.read_text()
    e.randomize_joints()
    limits=np.array([j['limits_deg'] for j in e.cfg['joints']])
    assert np.all(np.rad2deg(e.target)>=limits[:,0])
    assert np.all(np.rad2deg(e.target)<=limits[:,1])
    np.testing.assert_array_equal(e.data.qpos,actual)
    np.testing.assert_array_equal(e.command,command)
    e.zero_arm()
    np.testing.assert_array_equal(e.target,np.zeros(7))
    np.testing.assert_array_equal(e.data.qpos,actual)
    e.tick()
    max_step=np.deg2rad(e.cfg['sweep']['jog_speed_deg_s'])*e.arm.model.opt.timestep
    assert np.max(np.abs(e.command-command))<=max_step+1e-12
    assert e.path.read_text()==saved
    e.target[:]=.1
    e.cfg['joints'][0]['limits_deg']=[10,50]
    with pytest.raises(ValueError): e.zero_arm()
    np.testing.assert_allclose(e.target,.1)


def test_sweep_report_exports_live_gains(experiment):
    e=experiment;e.set_gains(0,145,14);e.start_sweep();finish_move(e)
    assert e.last_report.exists()
    path=e.last_report.parent
    assert load_config(path/'config.yaml')['joints'][0]['kp_nm_rad']==145
    m=mujoco.MjModel.from_xml_path(str(path/'model.xml'))
    assert m.actuator_gainprm[0,0]==145
    assert m.actuator_biasprm[0,2]==-14


def test_invalid_controls_and_reload_leave_state_intact(experiment):
    e=experiment
    for action,args in [(e.set_duration,(0,)),(e.set_jog_speed,(float('nan'),)),(e.set_gains,(0,-1,2))]:
        with pytest.raises(ValueError): action(*args)
    old=e.arm
    e.path.write_text('invalid: document')
    with pytest.raises(KeyError): e.load()
    assert e.arm is old


def test_geometry_rebuild_updates_physics_and_preserves_pose(experiment):
    e=experiment
    before=e.arm.structural_masses.copy()
    old_mass=e.arm.model.body_mass.sum()
    q=e.data.qpos.copy();tip=e.data.site('tip').xpos.copy()
    disk=e.path.read_text()
    edits=e.geometry_settings()
    edits['links'][0]['length_mm']*=2
    edits['links'][1]['mass_override_kg']=.7
    edits['tool_mass_kg']+=.4
    edits['payload_mass_kg']+=.7
    e.apply_geometry(edits)
    assert e.arm.structural_masses[0]>before[0]
    assert e.arm.structural_masses[1]==pytest.approx(.7)
    assert e.arm.model.body_mass.sum()-old_mass==pytest.approx(sum(e.arm.structural_masses-before)+1.1)
    assert np.linalg.norm(e.data.site('tip').xpos-tip)>.05
    np.testing.assert_array_equal(e.data.qpos,q)
    assert e.paused and not e.history
    assert e.path.read_text()==disk  # Apply is temporary.


def test_geometry_save_merges_and_invalid_edit_is_atomic(experiment):
    e=experiment
    external=load_config(e.path)
    external['sweep']['jog_speed_deg_s']=17
    external['joints'][0]['kp_nm_rad']=155
    e.path.write_text(yaml.safe_dump(external))
    edits=e.geometry_settings();edits['links'][3]['length_mm']=160
    e.apply_geometry(edits,save=True)
    saved=load_config(e.path)
    assert saved['links'][3]['length_mm']==160
    assert saved['sweep']['jog_speed_deg_s']==17
    assert saved['joints'][0]['kp_nm_rad']==155
    old=e.arm;disk=e.path.read_text()
    edits['tool_mass_kg']=-1
    with pytest.raises(ValueError): e.apply_geometry(edits,save=True)
    assert e.arm is old and e.path.read_text()==disk
    e.go_to('B')
    with pytest.raises(ValueError): e.apply_geometry(e.geometry_settings())


def test_actuator_mass_edits_update_dynamics_independently(experiment):
    e=experiment
    original_masses=e.arm.model.body_mass.copy()
    original_inertia=e.arm.model.body_inertia.copy()
    catalogs=deepcopy(e.cfg['motors'])
    edits=e.geometry_settings()
    edits['motor_mass_overrides_kg']['J2']=1.1
    edits['motor_mass_overrides_kg']['J3']=.9
    e.apply_geometry(edits,save=True)
    m=e.arm.model
    assert m.body('link1').mass[0]-original_masses[m.body('link1').id]==pytest.approx(1.1-.88)
    assert m.body('link2').mass[0]-original_masses[m.body('link2').id]==pytest.approx(.9-.621)
    assert not np.allclose(m.body_inertia,original_inertia)
    assert e.cfg['motors']==catalogs
    assert e.cfg['joints'][0].get('motor_mass_override_kg') is None
    assert load_config(e.path)['joints'][1]['motor_mass_override_kg']==1.1
    old=e.arm
    edits['motor_mass_overrides_kg']['J2']=0
    with pytest.raises(ValueError): e.apply_geometry(edits)
    assert e.arm is old
    edits['motor_mass_overrides_kg']={j['name']:None for j in e.cfg['joints']}
    e.apply_geometry(edits)
    np.testing.assert_allclose(e.arm.model.body_mass,original_masses)
    np.testing.assert_allclose(e.arm.model.body_inertia,original_inertia)
