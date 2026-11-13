"""Recorded-trial playback: forward kinematics only, never a new simulation."""
import base64
import gzip
import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .fea import METRICS


def visual_geoms(model):
    # CAD visuals are group 1; primitive-only models use their regular geoms.
    has_visuals = np.any(model.geom_group == 1)
    return [g for g in range(model.ngeom)
            if model.geom_rgba[g, 3] > 0 and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
            and (not has_visuals or model.geom_group[g] == 1)]


def packed(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode('ascii')


@lru_cache(maxsize=4)
def model_geometry(model_path, modified_ns):
    model = mujoco.MjModel.from_binary_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model,data)
    meshes, geoms = {}, []
    for g in visual_geoms(model):
        mesh_id = int(model.geom_dataid[g])
        kind = int(model.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_MESH and str(mesh_id) not in meshes:
            v, nv = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            f, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
            meshes[str(mesh_id)] = dict(vertices=packed(model.mesh_vert[v:v+nv], '<f4'),
                                       faces=packed(model.mesh_face[f:f+nf], '<u4'))
        b = int(model.geom_bodyid[g])
        R = data.xmat[b].reshape(3,3)
        local_pos = R.T@(data.geom_xpos[g]-data.xpos[b])
        local_quat = np.zeros(4)
        # Honor MuJoCo's compiled same-frame shortcuts, including CAD rotations
        # snapped to an inertial/body frame instead of the raw near-identity quat.
        mujoco.mju_mat2Quat(local_quat,(R.T@data.geom_xmat[g].reshape(3,3)).ravel())
        geoms.append(dict(body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b),
                          body_id=b,pos=local_pos.tolist(),quat=local_quat.tolist(),
                          rgba=model.geom_rgba[g].tolist(),kind=kind, mesh=mesh_id, size=model.geom_size[g].tolist()))
    bodies = [dict(parent=int(model.body_parentid[b]),pos=model.body_pos[b].tolist(),quat=model.body_quat[b].tolist())
              for b in range(model.nbody)]
    joints = [dict(body=int(model.jnt_bodyid[j]),kind=int(model.jnt_type[j]),axis=model.jnt_axis[j].tolist(),
                   pos=model.jnt_pos[j].tolist(),qpos=int(model.jnt_qposadr[j])) for j in range(model.njnt)]
    return dict(meshes=meshes, geoms=geoms,bodies=bodies,joints=joints,qpos0=model.qpos0.tolist())


def metric_value(load, metric):
    j = METRICS.index(metric)
    return float(load[j] if j < 6 else np.linalg.norm(load[:3] if j == 6 else load[3:]))


@lru_cache(maxsize=8)
def playback(path, trial_id, part, interface, metric, extreme):
    """Bound the animation payload, but include the exact recorded selected peak.

    The mesh uses compiled geom transforms (including mesh centering). Readouts
    and poses always come from the same recorded timestamp, without interpolation.
    """
    path = Path(path)
    folder = path/'trials'/str(int(trial_id))
    model_path = path/'model.mjb'
    if not model_path.exists() or not (folder/'trace.npz').exists():
        raise ValueError('This trial has no saved model or motion history to play.')
    meta = json.loads((path/'metadata.json').read_text())
    info = json.loads((folder/'trial.json').read_text())
    mates = [(i, r) for i, r in enumerate(meta['interfaces'])
             if r.get('kind', 'mate') == 'mate' and (r.get('part') or r['label'].split(' <- ')[0]) == part]
    if interface not in [i for i, _ in mates]:
        raise ValueError('Choose a connection on the selected link.')
    with np.load(folder/'trace.npz', allow_pickle=False) as trace:
        times, q, loads = trace['time'], trace['q'], trace['loads']
    if not len(times) or not all(np.isfinite(a).all() for a in (times, q, loads)):
        raise ValueError('This trial has no finite recorded motion to play.')
    if np.any(np.diff(times) < 0):
        raise ValueError('Recorded timestamps are not in chronological order.')
    score = lambda load: metric_value(load[interface], metric)
    select = (lambda x: -x) if metric in METRICS[:6] and extreme == 'minimum' else (
        abs if metric in METRICS[:6] and extreme == 'absolute' else lambda x: x)
    values = np.array([score(load) for load in loads])
    peak_index = max(range(len(times)), key=lambda k: select(values[k]))
    peak = dict(time_s=float(times[peak_index]), q_rad=q[peak_index], loads_local=loads[peak_index])
    cases_path = folder/'cases.json.gz'
    if cases_path.exists():
        with gzip.open(cases_path, 'rt') as stream:
            for case in json.load(stream):
                if select(score(case['loads_local'])) > select(score(peak['loads_local'])):
                    peak = case
    # About 60 visual frames/s, bounded for long trials. Include endpoints and peak.
    count = min(len(times), 900, max(2, int(np.ceil((times[-1]-times[0])*60))+1))
    indices = np.unique(np.linspace(0, len(times)-1, count).astype(int))
    frames = {float(times[k]): (q[k], loads[k]) for k in indices}
    frames[float(peak['time_s'])] = (np.asarray(peak['q_rad']), np.asarray(peak['loads_local']))
    model = mujoco.MjModel.from_binary_path(str(model_path))
    if q.shape[1] != model.nq:
        raise ValueError('Recorded joint coordinates do not match the saved model.')
    data, geoms = mujoco.MjData(model), visual_geoms(model)
    if not geoms:
        raise ValueError('The saved model has no visible robot geometry.')
    poses, displayed_loads = [], []
    ordered_times = sorted(frames)
    for t in ordered_times:
        pose, load = frames[t]
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        poses.append(np.concatenate((data.geom_xpos[geoms], data.geom_xmat[geoms]), axis=1).tolist())
        displayed_loads.append(load[[i for i, _ in mates]].tolist())
    members = next((p['bodies'] for p in meta.get('parts', []) if p['name'] == part), [part])
    kind = 'minimum' if metric in METRICS[:6] and extreme == 'minimum' else (
        'largest absolute' if metric in METRICS[:6] and extreme == 'absolute' else 'maximum')
    return dict(path=str(path), trial=int(trial_id), part=part, members=members,
                time=ordered_times, poses=poses, loads=displayed_loads,
                mates=[r['label'] for _, r in mates], metric=metric,
                selected_mate=[i for i, _ in mates].index(interface),
                peak_index=ordered_times.index(float(peak['time_s'])),
                peak_label=f'{kind} {metric.replace("_", " ")}: {score(peak["loads_local"]):+.6g}',
                full_history=bool(info.get('full_history')), accepted=bool(info.get('counted')),
                reason=info.get('reason', ''), source_samples=len(times))
