"""Saved-model mesh, mounting points and local axes."""
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import plotly.graph_objects as go

AXIS_COLORS = ['#ff6868', '#65d98b', '#70aaff']


@lru_cache(maxsize=24)
def part_meshes(model_path, modified_ns, part, members):
    """MuJoCo's saved mesh vertices include its centering transform via geom_xmat.

    Transform the compiled visual geometry into the part root frame. Fixed merged
    members are included; neighboring actuator bodies are never included.
    """
    model = mujoco.MjModel.from_binary_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
    if root < 0:
        raise ValueError(f'{part} is absent from the saved model')
    rotation = data.xmat[root].reshape(3, 3)
    output = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if name not in members or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        # Imported CAD visuals use group 1; exclude any collision-only duplicate.
        if model.geom_group[g] != 1:
            continue
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = np.asarray(model.mesh_vert[start:start+count], float)
        world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        vertices = (world-data.xpos[root]) @ rotation
        start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        faces = model.mesh_face[start:start+count].copy()
        output.append((name, vertices, faces))
    return output


@lru_cache(maxsize=24)
def mate_points(model_path, modified_ns, interfaces):
    model = mujoco.MjModel.from_binary_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = []
    for body, child in interfaces:
        joint = int(model.body_jntadr[child])
        point = data.xanchor[joint] if model.body_jntnum[child] else data.xpos[child]
        points.append(data.xmat[body].reshape(3,3).T @ (point-data.xpos[body]))
    return points


def mesh_figure(meta, sample, part, run_path=None):
    figure = go.Figure()
    figure.update_layout(template='plotly_dark', paper_bgcolor='#101b29', plot_bgcolor='#101b29',
        title=f'{part} · mates & local XYZ', height=340,
        margin=dict(l=0,r=0,t=42,b=0), font=dict(color='#cbd8e5'), showlegend=False,
        uirevision=f'fea-mesh-{run_path}-{part}', scene=dict(aspectmode='data',
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'))
    body = next((p for p in meta.get('parts', []) if p['name'] == part), None)
    meshes, message = [], ''
    model = Path(run_path)/'model.mjb' if run_path else None
    if model and model.exists() and body:
        try:
            meshes = part_meshes(str(model), model.stat().st_mtime_ns, part, tuple(body['bodies']))
        except (ValueError, OSError) as exc:
            message = f'Mesh unavailable: {exc}'
    if not meshes:
        message = message or 'No saved visual mesh; mounting points shown only.'
    for name, vertices, faces in meshes:
        figure.add_mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],
            i=faces[:,0],j=faces[:,1],k=faces[:,2],color='#a9b9c9',opacity=.65,
            name=part, hovertemplate=f'{name}<br>X %{{x:.5f}} m<br>Y %{{y:.5f}} m<br>Z %{{z:.5f}} m<extra></extra>')
    mates = [(i,r) for i,r in enumerate(meta['interfaces']) if r.get('kind','mate') == 'mate'
             and (r.get('part') or r['label'].split(' <- ')[0]) == part]
    points = [np.asarray(sample['application_points_local_m'][i]) for i,_ in mates] if sample else []
    if not points and model and model.exists() and mates:
        try:
            points = mate_points(str(model),model.stat().st_mtime_ns,tuple((r['body_id'],r['child_id']) for _,r in mates))
        except (KeyError,ValueError,OSError) as exc:
            message = f'Mate positions unavailable: {exc}'
    all_points = [v for _,v,_ in meshes] + ([np.array(points)] if points else [])
    span = max(np.ptp(np.concatenate(all_points),axis=0).max(), .04) if all_points else .1
    length = .6*span

    def arrow(origin, vector, color, label, text=True, group='axes'):
        end = origin+vector
        figure.add_scatter3d(x=[origin[0],end[0]],y=[origin[1],end[1]],z=[origin[2],end[2]],
            mode='lines+text' if text else 'lines',text=['',label.split(' ')[0]] if text else None, meta=group,
            textfont=dict(color=color,size=11),line=dict(color=color,width=5),name=f'{part} <- {label}',
            hovertemplate=label+'<extra></extra>')
        figure.add_cone(x=[end[0]],y=[end[1]],z=[end[2]],u=[vector[0]],v=[vector[1]],w=[vector[2]],
            sizemode='absolute',sizeref=.05*span,anchor='tip',colorscale=[[0,color],[1,color]],
            showscale=False,meta=group,name=f'{part} <- {label}',hovertemplate=label+'<extra></extra>')

    for axis, color in enumerate(AXIS_COLORS):
        direction = np.eye(3)[axis]
        arrow(np.zeros(3), length*direction,color,'+'+'XYZ'[axis])
        figure.add_scatter3d(x=[0,-length*.45*direction[0]],y=[0,-length*.45*direction[1]],z=[0,-length*.45*direction[2]],
            mode='lines+text',text=['','−'+'XYZ'[axis]],line=dict(color=color,width=2,dash='dot'),
            textfont=dict(color=color,size=10),name=f'{part} <- −'+ 'XYZ'[axis])

    for n,((i,row), point) in enumerate(zip(mates,points)):
        label = row['label'].split(' <- ')[-1]
        figure.add_scatter3d(x=[point[0]],y=[point[1]],z=[point[2]],mode='markers+text',text=[label],
            textposition='bottom left' if n == 0 else 'top right',
            marker=dict(size=7,color='#ffcf70' if n==0 else '#25c9a3'),
            name=row['label'],hovertemplate=label+'<br>(%{x:.5f}, %{y:.5f}, %{z:.5f}) m<extra></extra>')
    if message:
        figure.add_annotation(text=message,x=.5,y=.02,xref='paper',yref='paper',showarrow=False)
    return figure
