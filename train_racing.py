import gym
from gym import wrappers, spaces
import numpy as np
from constant import *
import mujoco

'''
GYM environment using the mujoco model
'''

# Paths to models
patriq = "PATRIQ/patriq.xml"
example_quad = "models/google_barkour_v0/scene_mjx.xml"


class quadrupedEnv(gym.Env):

    def __init__(self):

        # constants
        self.height = HEIGHT
        self.width = WIDTH
        

        # Load the model and data
        self.model = mujoco.MjModel.from_xml_path(patriq)
        self.data = mujoco.MjData(self.model)
        self.opt = mujoco.MjvOption()

        # Set the camera for the renderer
        self.camera_top = mujoco.MjvCamera()

        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "main_camera")
    
        if(camera_id < 0):
            # Set the camera for the viewer (onboard)
            self.camera = mujoco.MjvCamera()
            self.pov_working = False
        else:
            # Set the camera for the viewer (onboard)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = camera_id
            self.pov_working = True


        # Initialize the viewer
        mujoco.glfw.glfw.init()


        if(self.pov_working):
            # Create a window
            self.window = mujoco.glfw.glfw.create_window(self.width, self.height, "POV", None, None)
            # make the context current
            mujoco.glfw.glfw.make_context_current(self.window)
            mujoco.glfw.glfw.swap_interval(1)
            # Create a scene and context
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)


        self.window_top = mujoco.glfw.glfw.create_window(self.width, self.height, "Top-Down", None, None)
        mujoco.glfw.glfw.make_context_current(self.window_top)
        mujoco.glfw.glfw.swap_interval(1)

        self.scene_top = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context_top = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)


    
        # Define action space (3D vector)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), 
                                       high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        
        # Define observation space (RGB image)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.observation = np.zeros((self.height, self.width, 3))

    


    def reset(self):
        
        # Reset the model
        mujoco.glfw.glfw.terminate()

        # Load the model and data
        self.model = mujoco.MjModel.from_xml_path(patriq)
        self.data = mujoco.MjData(self.model)
        self.opt = mujoco.MjvOption()

        # Set the camera for the renderer
        self.camera_top = mujoco.MjvCamera()

        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "main_camera")
    
        if(camera_id < 0):
            # Set the camera for the viewer (onboard)
            self.camera = mujoco.MjvCamera()
            self.pov_working = False
        else:
            # Set the camera for the viewer (onboard)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = camera_id
            self.pov_working = True


        # Initialize the viewer
        mujoco.glfw.glfw.init()


        if(self.pov_working):
            # Create a window
            self.window = mujoco.glfw.glfw.create_window(self.width, self.height, "POV", None, None)
            # make the context current
            mujoco.glfw.glfw.make_context_current(self.window)
            mujoco.glfw.glfw.swap_interval(1)
            # Create a scene and context
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)


        self.window_top = mujoco.glfw.glfw.create_window(self.width, self.height, "Top-Down", None, None)
        mujoco.glfw.glfw.make_context_current(self.window_top)
        mujoco.glfw.glfw.swap_interval(1)

        self.scene_top = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context_top = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)


    # Used to view the environment
    def render(self):
        if(self.pov_working):
            # Render the scene
            viewport = mujoco.MjrRect(0, 0, self.width, self.height)

            # make the current window context 
            mujoco.glfw.glfw.make_context_current(self.window)
            mujoco.glfw.glfw.swap_interval(1)

            # update the scene & render
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.context)

            # Update the window
            mujoco.glfw.glfw.swap_buffers(self.window)
            mujoco.glfw.glfw.poll_events()

        # Render the scene
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        # make the current window context
        mujoco.glfw.glfw.make_context_current(self.window_top)
        mujoco.glfw.glfw.swap_interval(1)

        # update the scene & render
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.camera_top, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene_top)
        mujoco.mjr_render(viewport, self.scene_top, self.context_top)

        # Update the window
        mujoco.glfw.glfw.swap_buffers(self.window_top)
        mujoco.glfw.glfw.poll_events()


    # Used to do a step in the environment
    def step(self, action):
        



if __name__ == "__main__":
    env = quadrupedEnv()
    env.reset()
    env.render()