import pyvista
import numpy as np
import random
from skimage.transform import resize
from matplotlib import pyplot as plt

NUM_FRAMES_PER_STEP = 10
N_PATCHES_PER_FRAME = 341  # patches at each depth: (1 + 4 + 16 + 64 + 256) 
BATCH_SIZE = 256
BG_COLOR = "black"

pl = pyvista.Plotter(off_screen=True, window_size=[256, 256])
pl.set_background(BG_COLOR)

def patch_to_vector(patch, patch_id=0):
    vec = np.sum(patch, axis = 2) / np.array(255.0*3).ravel()
    return np.append(vec, patch_id)

def vector_to_patch(patch):
    return patch[0: 256].reshape((16, 16, 1))

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
    patch_id = 0
    img = np.zeros((256, 256, 1))
    for x in range(0, 16):
        for y in range(0, 16):
            img[x * 16 : (x+1)*16, y*16 : (y+1)*16] = vector_to_patch(tokens[patch_id])
            patch_id +=1
    return img

def render_polygon(N, num_frames):
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
    data_points = []
    answers = []
    images = [last_img]
    for i in range(0, num_frames):
        pl.remove_actor(actor)
        tokens = generate_tokens_from_frame(last_img)
        feature += tokens
        pl.camera.azimuth += da
        pl.camera.roll += dr
        pl.camera.elevation += de
        actor = pl.add_mesh(mesh, smooth_shading=False)
        last_img = np.array(pl.screenshot(return_img=True))
        images.append(last_img)
    # each image frame generates 256 examples (one to predict each patch)
    patch_id = 0
    for x in range(0, 16):
        for y in range(0, 16):
            answer = patch_to_vector(last_img[x*16: (x+1) * 16, y*16: (y+1)*16])
            example = feature + [patch_to_vector(np.zeros((16, 16, 3)), patch_id)]
            data_points.append(np.array(example))
            answers.append(answer)
            patch_id += 1
    
    return images, data_points[:BATCH_SIZE], answers[:BATCH_SIZE]
  
if __name__ == '__main__':
    orig_img, x, y = render_polygon(3, 1)
    recon_img = generate_frame_from_tokens(y)
    
    image_list = [vector_to_patch(v) for v in x[0]]
    f, axarr = plt.subplots(4,4)
    axarr[0,0].imshow(image_list[5])
    axarr[0,1].imshow(image_list[6])
    axarr[0,2].imshow(image_list[7])
    axarr[0,3].imshow(image_list[8])

    axarr[1,0].imshow(image_list[9])
    axarr[1,1].imshow(image_list[10])
    axarr[1,2].imshow(image_list[11])
    axarr[1,3].imshow(image_list[12])

    axarr[2,0].imshow(image_list[13])
    axarr[2,1].imshow(image_list[14])
    axarr[2,2].imshow(image_list[15])
    axarr[2,3].imshow(image_list[16])

    axarr[3,0].imshow(image_list[17])
    axarr[3,1].imshow(image_list[18])
    axarr[3,2].imshow(image_list[19])
    axarr[3,3].imshow(image_list[20])
    plt.show()