import pyvista
import numpy as np
import random
from skimage.transform import resize
from matplotlib import pyplot as plt
import time
NUM_FRAMES_PER_STEP = 1
N_PATCHES_PER_FRAME = 1  # patches at each depth: (1 + 4 + 16 + 64 + 256) 
BATCH_SIZE = 64
BG_COLOR = "black"


pyvista.start_xvfb()
pl = pyvista.Plotter(off_screen=True, window_size=[32,32])
pl.set_background(BG_COLOR)

def render_polygon(N, num_frames):
    pl.clear()
    rng = np.random.default_rng()
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = rng.uniform(0.5, 1.5, N)
    coords = np.array([np.cos(angles), np.sin(angles)]) * radii
    points_2d = coords.T 
    points_3d = np.pad(points_2d, [(0, 0), (0, 1)])
    face = [N + 1] + list(range(N)) + [0]
    poly = pyvista.PolyData(points_3d, faces=face)
    mesh = poly.extrude([0, 0, 1], capping=True)

    # pick a random start pos
    actor = pl.add_mesh(mesh, smooth_shading=False)
    pl.camera.azimuth = random.random()*360
    pl.camera.roll = random.random()*360
    pl.camera.elevation = random.random()*360
    sz = 5
    da = 0 #(1 if random.random() > 0.5 else -1) * sz
    dr =  (1 if random.random() > 0.5 else -1) * sz
    de =  0 #(1 if random.random() > 0.5 else -1) * sz
    curr_img = np.array(pl.screenshot(return_img=True))
    past_frames = [curr_img]
    for i in range(0, num_frames):
        pl.remove_actor(actor)
        pl.camera.azimuth += da
        pl.camera.roll += dr
        pl.camera.elevation += de
        actor = pl.add_mesh(mesh, smooth_shading=False)
        curr_img = np.array(pl.screenshot(return_img=True))
        past_frames.append(curr_img)
    return past_frames

if __name__ == '__main__':
    for i in range(0, 100):
        start = time.time()
        img, _, _ = render_polygon(5, 1)
        end = time.time()
        print(end - start)
