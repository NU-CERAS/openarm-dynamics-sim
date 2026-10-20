from copy import deepcopy
from xml.etree import ElementTree as ET
import csv
import numpy as np
import mujoco
import pytest
from arm_design.config import ROOT, load_config, validate, save_sweep
from arm_design.model import build_arm, structure, rpy
from arm_design.simulation import run_headless, control, step, smooth_move
from arm_design.reporting import summary, save_result


def test_model_axes_frames_and_mass():
    c=load_config(); a=build_arm(c); m=a.model
    assert (m.nq,m.nv,m.nu)==(7,7,7)
    d=mujoco.MjData(m); mujoco.mj_forward(m,d)
    R=rpy(c['base_rpy_deg']); p=np.array(c['base_xyz_m'])
    for j,link in zip(c['joints'],c['links']):
        R=R@rpy(j['origin_rpy_deg'])
        np.testing.assert_allclose(d.xanchor[m.joint(j['name']).id],p,atol=1e-10)
        np.testing.assert_allclose(d.xaxis[m.joint(j['name']).id],R@j['axis'],atol=1e-10)
        p=p+R@np.array(link['direction'])*link['length_mm']/1000
    np.testing.assert_allclose(d.site('tip').xpos,p,atol=1e-10)
    expected=sum(a.structural_masses)+sum(c['motors'][j['motor']]['mass_kg'] for j in c['joints'][1:])+c['tool']['mass_kg']+c['payload']['mass_kg']
    moving=[m.body(f'link{i}').id for i in range(1,8)]
    assert sum(m.body_mass[moving])==pytest.approx(expected)
    cad=ET.parse(ROOT/'robots/moving_asm/urdf/moving_asm.urdf').getroot()
    cad_masses=[float(cad.find(f"link[@name='part_{i}']/inertial/mass").get('value')) for i in range(1,8)]
    np.testing.assert_allclose(a.structural_masses,cad_masses)


def test_hollow_tube_mass_and_inertia_follow_length():
    c=load_config(); l=c['links'][0]
    l['mass_override_kg']=None  # Exercise automatic mass estimation explicitly.
    m,com,I=structure(l)
    l['length_mm']*=2
    m2,com2,I2=structure(l)
    assert m2-l['bracket_mass_kg']==pytest.approx(2*(m-l['bracket_mass_kg']))
    assert I2[0,0]>I[0,0]*4
    l['mass_override_kg']=.3;l['com_override_m']=[0,0,.08]
    l['inertia_override_kg_m2']=[.001,.001,.0001,0,0,0]
    mo,co,Io=structure(l)
    assert mo==.3
    np.testing.assert_allclose(co,[0,0,.08]);np.testing.assert_allclose(Io,np.diag([.001,.001,.0001]))
    build_arm(c)


def test_gravity_matches_potential_gradient_and_payload_moment():
    c=load_config();c['payload']['mass_kg']=.5
    a=build_arm(c);d=a.data();m=a.model
    eps=1e-6
    def potential(q):
        t=mujoco.MjData(m);t.qpos[:]=q;mujoco.mj_forward(m,t)
        return -np.sum(m.body_mass*(t.xipos@np.array(c['gravity_m_s2'])))
    q=d.qpos.copy()
    derivative=[]
    for i in range(7):
        delta=np.eye(7)[i]*eps
        derivative.append((potential(q+delta)-potential(q-delta))/(2*eps))
    np.testing.assert_allclose(d.qfrc_bias,derivative,atol=1e-7)
    c0=deepcopy(c);c0['payload']['mass_kg']=0
    az=build_arm(c0);dz=az.data()
    # Added payload requires opposing its gravitational moment around each axis.
    payload=d.geom('payload').xpos
    force=np.array(c['gravity_m_s2'])*c['payload']['mass_kg']
    expected=[-np.dot(d.xaxis[i],np.cross(payload-d.xanchor[i],force)) for i in range(7)]
    np.testing.assert_allclose(d.qfrc_bias-dz.qfrc_bias,expected,atol=1e-8)


def test_gravity_compensation_is_inside_actuator_caps():
    c=load_config();c['payload']['mass_kg']=40
    # A user can save an upright zero pose, which has no gravity moment.
    c['sweep']['a_deg']=[0,-25,0,50,0,0,0]
    a=build_arm(c);d=a.data()
    caps=control(a,d,d.qpos.copy())
    mujoco.mj_forward(a.model,d)
    assert np.all(np.abs(d.qfrc_actuator)<=caps+1e-10)
    assert np.any(np.isclose(np.abs(d.qfrc_actuator),caps))
    np.testing.assert_array_equal(d.qfrc_applied,0)
    np.testing.assert_allclose(d.actuator_force,d.qfrc_actuator)


def test_default_sweep_and_timestep_convergence():
    c=load_config();a,r=run_headless(c);s=summary(r)
    assert r.complete and s['tracking_ok'] and s['joint_limits_ok']
    assert [x['phase'] for x in r.rows].count('sweep')==3000
    c['simulation']['step_s']/=2
    _,r2=run_headless(c)
    np.testing.assert_allclose([j['peak_torque_nm'] for j in s['joints']], [j['peak_torque_nm'] for j in summary(r2)['joints']],rtol=.01,atol=.01)
    # Crossing nominal torque is not required: it depends on motor assignment.


def test_underpowered_arm_cannot_silently_pass():
    c=load_config();c['drive']['torque_derating']=.01
    c['sweep']['a_deg']=[0,-25,0,50,0,0,0]
    _,r=run_headless(c)
    assert not summary(r)['tracking_ok']
    assert 'Could not reach A' in r.reason
    assert not any(x['phase']=='sweep' for x in r.rows)


def test_short_duration_exposes_tracking_error():
    c=load_config();c['sweep']['duration_s']=.1
    _,r=run_headless(c)
    assert r.complete
    assert not summary(r)['tracking_ok']
    assert max(j['saturated_percent'] for j in summary(r)['joints'])>0


def test_trajectory_is_smooth_and_duration_matters():
    a=np.zeros(7);b=np.ones(7)
    np.testing.assert_allclose(smooth_move(a,b,0,2),(a,a))
    np.testing.assert_allclose(smooth_move(a,b,2,2),(b,a))
    np.testing.assert_allclose(smooth_move(a,b,1,2)[1],smooth_move(a,b,2,4)[1]*2)


def test_save_and_export_are_replayable(tmp_path):
    import yaml
    c=load_config();p=tmp_path/'arm.yaml';p.write_text(yaml.safe_dump(c))
    c['sweep']['duration_s']=.5
    save_sweep(p,c['sweep']);loaded=load_config(p)
    assert loaded['sweep']['duration_s']==.5
    a,r=run_headless(loaded);folder=save_result(a,r,tmp_path)
    assert load_config(folder/'config.yaml')==loaded
    assert mujoco.MjModel.from_xml_path(str(folder/'model.xml')).nq==7
    with (folder/'loads.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==len(r.rows)
    assert float(rows[0]['J1_angle_deg'])==pytest.approx(loaded['sweep']['a_deg'][0])
    assert (folder/'loads.png').stat().st_size>1000


@pytest.mark.parametrize('field,value',[('length_mm',-1),('wall_mm',30),('mass_override_kg',0),('com_override_m',[float('nan'),0,0]),('inertia_override_kg_m2',[1,1,3,0,0,0])])
def test_bad_link_inputs_rejected(field,value):
    c=load_config();c['links'][0][field]=value
    with pytest.raises(ValueError): validate(c)
