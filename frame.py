import pyvista
import numpy as np
import random
from skimage.transform import resize
from matplotlib import pyplot as plt
import time
NUM_FRAMES_PER_STEP = 1
N_PATCHES_PER_FRAME = 1  # patches at each depth: (1 + 4 + 16 + 64 + 256) 
BATCH_SIZE = 12
BG_COLOR = "black"


pyvista.start_xvfb()
pl = pyvista.Plotter(off_screen=True, window_size=[16, 16])
pl.set_background(BG_COLOR)

def patch_to_vector(patch):
    return (np.sum(patch, axis = 2) / np.array(255.0*3)).ravel()

def vector_to_patch(patch):
    return patch[0: 256].reshape((16, 16, 1))

def generate_tokens_from_frame(image):
    return patch_to_vector(resize(image, (16,16)))


def generate_frame_from_tokens(tokens):
    return tokens[0].reshape((16, 16))

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
    da = (1 if random.random() > 0.5 else -1) * sz
    dr =  (1 if random.random() > 0.5 else -1) * sz
    de =  (1 if random.random() > 0.5 else -1) * sz
    last_img = np.array(pl.screenshot(return_img=True))
    feature = []
    images = [last_img]
    for i in range(0, num_frames):
        pl.remove_actor(actor)
        tokens = [generate_tokens_from_frame(last_img)]
        feature += tokens
        pl.camera.azimuth += da
        pl.camera.roll += dr
        pl.camera.elevation += de
        actor = pl.add_mesh(mesh, smooth_shading=False)
        last_img = np.array(pl.screenshot(return_img=True))
        images.append(last_img)

    answer = generate_tokens_from_frame(last_img)
    return images, np.array(feature), answer

if __name__ == '__main__':
    for i in range(0, 100):
        start = time.time()
        img, _, _ = render_polygon(5, 1)
        end = time.time()
        print(end - start)
