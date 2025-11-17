'''
Functions used to find the height map of the environment
'''

import mujoco
import cv2
import numpy as np
from constant import *


# get the height map of the environment
def get_height_map(model, data, camera, scene, context, window):
    
    # Render the scene
    viewport = mujoco.MjrRect(0, 0, WIDTH, HEIGHT)
    mujoco.mjr_render(viewport, scene, context)

    # Get the pixels
    rgb_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    depth_buffer = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    # vertically flip the image
    rgb_buffer_new = np.flipud(rgb_buffer)
    depth_buffer_new = np.flipud(depth_buffer)

    # Get the pixels
    mujoco.mjr_readPixels(rgb_buffer, depth_buffer, viewport, context)

    return depth_buffer_new, rgb_buffer_new


# Get the height map from the rgb buffer
def get_height_map_from_rgb(rgb_buffer):

    grayImage = cv2.cvtColor(rgb_buffer, cv2.COLOR_BGR2GRAY)
    heightMap = cv2.normalize(grayImage.astype('float32'), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return heightMap 
