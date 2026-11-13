"""Optional additional test payload, fixed to the saved model's tip frame."""
import numpy as np


def default_payload_offset(cfg):
    """Place added mass 150 mm toward the grip on the right-arm drumstick."""
    return [0., .15, 0.] if cfg.get('urdf',{}).get('tip_link') == 'drumstick' else [0.,0.,0.]


def validate_payload(payload):
    from .config import positive, vector
    mass = payload['mass_kg']
    positive(mass, 'Additional payload mass', zero=True)
    vector(payload['com_offset_m'], 3, 'Payload COM offset')
    size = vector(payload['box_size_m'], 3, 'Payload box dimensions')
    if np.any(size <= 0):
        raise ValueError('Payload box dimensions must be positive.')
    inertia = payload.get('inertia_com_kg_m2')
    if inertia is not None:
        xx, yy, zz, xy, xz, yz = vector(inertia, 6, 'Payload inertia')
        eig = np.linalg.eigvalsh([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
        if eig[0] <= 0 or eig[2] > eig[0]+eig[1]+1e-12:
            raise ValueError('Payload inertia must be positive and satisfy the triangle inequality.')


def attach_payload(root, cfg):
    """Explicit inertia prevents CAD/compiler mass inference from double-counting."""
    p = cfg.get('analysis_payload')
    if not p or not p['mass_kg']:
        return
    from .model import element, box_inertia
    validate_payload(p)
    parent = next(body for body in root.iter('body') if body.find("site[@name='tip']") is not None)
    tip = parent.find("site[@name='tip']")
    # Both builders create an unrotated tip site; offsets/COM inertia use that frame.
    position = np.fromstring(tip.get('pos','0 0 0'),sep=' ') + np.asarray(p['com_offset_m'])
    body = element(parent,'body',name='analysis_payload',pos=position)
    inertia = p.get('inertia_com_kg_m2')
    if inertia is None:
        diagonal = np.diag(box_inertia(p['mass_kg'],p['box_size_m']))
        inertia = [*diagonal,0,0,0]
    element(body,'inertial',mass=p['mass_kg'],pos=[0,0,0],fullinertia=inertia)
    element(body,'geom',name='analysis_payload_box',type='box',size=np.asarray(p['box_size_m'])/2,
            group=1 if cfg.get('model_type') == 'urdf' else 0,rgba=[1,.65,.2,1],contype=int(cfg['simulation']['contacts']),
            conaffinity=int(cfg['simulation']['contacts']))
