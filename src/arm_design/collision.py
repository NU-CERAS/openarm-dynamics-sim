"""Self/floor collision checks on a contact-enabled copy of the arm; the physics never sees them."""
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


class CollisionChecker:
    """Meshes collide as convex hulls, so the check errs toward rejecting borderline poses."""
    def __init__(self, arm, max_overlap_mm=0.):
        root = ET.fromstring(arm.xml)
        for geom in root.iter('geom'):
            geom.attrib.update(contype='1', conaffinity='1')
        # Hulls and body collision filters are built at compile time, so recompile.
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        # Floating-point contacts at exactly touching hulls are not overlaps.
        self.tolerance = max(max_overlap_mm/1000, 1e-6)
        m = self.model
        self.names = ['floor' if b == 0 else m.body(b).name or f'body_{b}' for b in range(m.nbody)]
        limits = np.deg2rad([j['limits_deg'] for j in arm.cfg['joints']])
        # Pairs touching in the reference pose are by design (e.g. a housing on
        # the link it drives), like an allowed-collision matrix.
        self.ignored = set()
        self.ignored = set(self.contacts(np.clip(0, limits[:, 0], limits[:, 1])))

    def contacts(self, q):
        """{(body, body): deepest overlap in mm} for overlaps beyond the tolerance."""
        m, d = self.model, self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        found = {}
        c = d.contact
        for g1, g2, dist in zip(c.geom1, c.geom2, c.dist):
            if dist >= -self.tolerance:
                continue
            pair = tuple(sorted((self.names[m.geom_bodyid[g1]], self.names[m.geom_bodyid[g2]])))
            if pair not in self.ignored:
                found[pair] = max(found.get(pair, 0.), -dist*1000)
        return found

    def path_contacts(self, a, b, step_deg=2.):
        """Contacts along the planned A -> B path, every step_deg of the largest joint move."""
        # smooth_move only retimes a straight joint-space line, so sample its length evenly.
        a, b = np.deg2rad(a), np.deg2rad(b)
        n = max(2, int(np.ceil(np.rad2deg(np.max(np.abs(b-a), initial=0))/step_deg))+1)
        for s in np.linspace(0, 1, n):
            found = self.contacts(a + (b-a)*s)
            if found:
                return found
        return {}


def describe(found):
    return ', '.join(f'{a}-{b} {mm:.1f} mm' for (a, b), mm in sorted(found.items(), key=lambda x: -x[1]))
