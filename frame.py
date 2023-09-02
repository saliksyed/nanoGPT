import pyvista
import numpy as np
import random
from skimage.transform import resize

def patch_to_vector(patch, patch_id=0):
    vec = np.sum(patch, axis = 2) / np.array(255.0*3).ravel()
    return np.append(vec, patch_id)

def generate_tokens_from_frame(image):
    tokens = []
    # downsample 1 : 256x256 -> 16x16 patch
    tokens.append(patch_to_vector(resize(image, (16,16))))
    # downsample 2 : 4 128x128 -> 4 16x16 patch
    d2 = resize(image, (32, 32))
    for i in range(0, 2):
        for j in range(0, 2):
            crop = d2[i * 16: (i+1) * 16, j * 16: (j+1) * 16]
            tokens.append(patch_to_vector(crop))
    # downsample 3 : 16 64x64 -> 16 16x16 patch
    d3 = resize(image, (64, 64))
    for i in range(0, 4):
        for j in range(0, 4):
            crop = d3[i * 16: (i+1) * 16, j * 16: (j+1) * 16]
            tokens.append(patch_to_vector(crop))

    # downsample 4 : 64  32x32 -> 64 16x16 patch
    d4 = resize(image, (128, 128))
    for i in range(0, 8):
        for j in range(0, 8):
            crop = d4[i * 16: (i+1) * 16, j * 16: (j+1) * 16]
            tokens.append(patch_to_vector(crop))

    # downsample 5 : 256 16x16 -> 256 16x16 patch
    for i in range(0, 16):
        for j in range(0, 16):
            crop = image[i * 16: (i+1) * 16, j * 16: (j+1) * 16]
            tokens.append(patch_to_vector(crop))
    return tokens

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
    last_img = np.array(pl.screenshot(return_img=True))
    feature = []
    data_points = []
    answers = []
    for i in range(0, num_frames):
        tokens = generate_tokens_from_frame(last_img)
        feature += tokens
        pl.camera.azimuth += da
        pl.camera.roll += dr
        pl.camera.elevation += de
        last_img = np.array(pl.screenshot(return_img=True))
    # each image frame generates 64 examples (one to predict each patch)
    patch_id = 0
    for x in range(0, 8):
        for y in range(0, 8):
            answer = patch_to_vector(last_img[x*16: (x+1) * 16, y*16: (y+1)*16])
            example = feature + [patch_to_vector(np.zeros((16, 16, 3)), patch_id)]
            data_points.append(np.array(example))
            answers.append(answer)
            patch_id += 1
    pl.remove_actor(actor)
    return data_points, answers
  
if __name__ == '__main__':
    pl = pyvista.Plotter(off_screen=True, window_size=[256, 256])
    pl.set_background("red")
    x, y = render_polygon(pl, 3, 1)
    print(x[0].shape)
    print(y[0].shape)