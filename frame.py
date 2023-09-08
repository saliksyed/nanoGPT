import pyvista
import numpy as np
import random

BG_COLOR = "black"
pyvista.start_xvfb()
pl = pyvista.Plotter(off_screen=True, window_size=[256, 256])
pl.enable_lightkit()
pl.set_background(BG_COLOR)


actor = None


def render_polygon(N):
    global actor
    if actor:
        pl.remove_actor(actor)
    rng = np.random.default_rng()
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = rng.uniform(0.5, 1.5, N)
    coords = np.array([np.cos(angles), np.sin(angles)]) * radii
    points_2d = coords.T
    points_3d = np.pad(points_2d, [(0, 0), (0, 1)])
    face = [N + 1] + list(range(N)) + [0]
    poly = pyvista.PolyData(points_3d, faces=face)
    mesh = poly.extrude([0, 0, 1], capping=True)
    actor = pl.add_mesh(
        mesh,
        smooth_shading=False,
    )
    pl.camera.azimuth = random.random() * 360
    pl.camera.roll = random.random() * 360
    pl.camera.elevation = random.random() * 360
    return np.array(pl.screenshot(return_img=True))
