import mujoco_viewer
import mujoco
import cv2
import numpy as np
from heightMap import *
from constant import *
import time



if __name__ == "__main__":

    # Paths to models
    patriq = "PATRIQ/patriq_backup.xml"
    example_quad = "models/google_barkour_v0/scene_mjx.xml"

    # Load the model and data
    model = mujoco.MjModel.from_xml_path(patriq)
    data = mujoco.MjData(model)
    # Initialize the MuJoCo viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Render
    viewer.render()


    while True:
        simstart = data.time

        # data.ctrl[0] = 0.1
        data.ctrl[1] = 0.001
        data.ctrl[2] = 0.001

        while (data.time - simstart < 1.0/60.0):
            mujoco.mj_step(model, data)


        viewer.render()
        time.sleep(0.1)

    viewer.close()