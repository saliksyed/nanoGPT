import skimage
import flow_vis
import numpy as np
import matplotlib.pyplot as plt


for i in range(0, 10000):
    end = skimage.color.rgb2gray(skimage.io.imread(f"./data/example_{i}.png"))
    start = skimage.color.rgb2gray(skimage.io.imread(f"./data/example_{i}_delta.png"))
    flow = skimage.registration.optical_flow_ilk(start, end)
    flow_color = flow_vis.flow_to_color(np.transpose(flow), convert_to_bgr=False)
    skimage.io.imsave(f"./data/example_{i}_flow.png", flow_color)
