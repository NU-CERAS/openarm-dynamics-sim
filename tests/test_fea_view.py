import mujoco
import numpy as np

from arm_design.fea_view import part_meshes, mesh_figure


def test_saved_mesh_transform_and_member_isolation(tmp_path):
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <asset><mesh name="tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>
      <worldbody><body name="link" pos="1 2 3" euler="0 0 90">
        <geom type="mesh" mesh="tetra" group="1" pos=".2 .3 .4" euler="90 0 0"/>
        <body name="fixed_child" pos=".5 0 0"><geom type="mesh" mesh="tetra" group="1"/></body>
        <body name="motor" pos="9 0 0"><geom type="mesh" mesh="tetra" group="1"/></body>
      </body></worldbody></mujoco>''')
    path = tmp_path/'model.mjb'
    mujoco.mj_saveModel(model,str(path),None)
    meshes = part_meshes(str(path),path.stat().st_mtime_ns,'link',('link','fixed_child'))
    assert [name for name,_,_ in meshes] == ['link','fixed_child']
    # Geometry's 90-degree X rotation, then local translation. Root world transform cancels.
    expected = np.array([[.2,.3,.4],[1.2,.3,.4],[.2,.3,1.4],[.2,-.7,.4]])
    for vertex in expected:
        assert np.min(np.linalg.norm(meshes[0][1]-vertex,axis=1)) < 1e-6
    assert np.max(meshes[1][1][:,0]) < 1.500001


def test_mesh_shows_mates_and_axes_without_load_overlays():
    meta = dict(parts=[],interfaces=[dict(part='link',kind='mate',label='link <- support'),dict(part='link',kind='mate',label='link <- loaded')])
    sample = dict(application_points_local_m=[[0,0,0],[.1,0,0]],loads_local=[[0]*6,[-10,0,0,0,0,-5]])
    figure = mesh_figure(meta,sample,'link')
    assert not figure.layout.updatemenus
    mates = [t for t in figure.data if t.type=='scatter3d' and t.mode=='markers+text']
    assert [t.text[0] for t in mates] == ['support','loaded']
    assert [t.x[0] for t in mates] == [0,.1]
    assert all(t.meta not in ('force','moment') for t in figure.data)
