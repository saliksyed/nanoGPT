import pyvista
from PIL import Image
import numpy as np
import random
from collections import defaultdict


def generate_tokens_from_frame(image):
    return None

def generate_frame_from_tokens(tokens):
    return None

def get_patch_from_frame(image, x, y):
    return None

def render_polygon(pl, N, num_frames):
    # pl = plotter
    # N = number of faces
    # num_frames = number of frames
    rng = np.random.default_rng()
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = rng.uniform(0.5, 1.5, N)
    coords = np.array([np.cos(angles), np.sin(angles)]) * radii
    points_2d = coords.T 
    points_3d = np.pad(points_2d, [(0, 0), (0, 1)])
    face = [N + 1] + list(range(N)) + [0]
    poly = pyvista.PolyData(points_3d, faces=face)
    mesh = poly.extrude([0, 0, 1], capping=True)
    actor = pl.add_mesh(mesh, smooth_shading=False)

    # pick a random start pos
    pl.camera.azimuth = random.random() * 180
    pl.camera.roll = random.random() * 180
    pl.camera.elevation = random.random() * 180
    sz = 1.0
    da = (random.random() - 0.5) * sz
    dr = (random.random() - 0.5) * sz
    de = (random.random() - 0.5) * sz
    last_img = pl.screenshot(return_img=True)
    feature = []
    data_points = []
    answers = []
    for i in range(0, num_frames):
        tokens = generate_tokens_from_frame(last_img)
        feature += tokens
        pl.camera.azimuth += da
        pl.camera.roll += dr
        pl.camera.elevation += de
        last_img = pl.screenshot(return_img=True)
    # each image frame generates 64 examples (one to predict each patch)
    for x in range(0, 8):
        for y in range(0, 8):
            answer = get_patch_from_frame(last_img, x, y)
            example = feature + [patch_id]
            data_points.append(example)
            answers.append(answer)
            patch_id += 1
    pl.remove_actor(actor)
    return data_points, answers
  
