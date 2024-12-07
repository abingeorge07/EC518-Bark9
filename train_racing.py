import gym
from gym import wrappers, spaces
import numpy as np
from constant import *
import mujoco
from heightMap import *
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from spinup import sac_pytorch
from spinup.algos.pytorch.sac.core import MLPActorCritic

'''
GYM environment using the mujoco model
'''

# Paths to models
patriq = "PATRIQ/patriq.xml"
example_quad = "models/google_barkour_v0/scene_mjx.xml"


class quadrupedEnv(gym.Env):

    def __init__(self, model=None):

        # # Use gpu if available
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # model
        # self.model = model
        # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # self.midas_transform = midas_transforms.small_transform
        
        # # move model to device
        # self.model.to(self.device)

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
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)


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


    # function to render the window (pov)
    def windowView(self, model, data, opt, camera, scene, context, window):
        # Render the scene
        viewport = mujoco.MjrRect(0, 0, WIDTH, HEIGHT)

        # make the current window context 
        mujoco.glfw.glfw.make_context_current(window)
        mujoco.glfw.glfw.swap_interval(1)

        # update the scene & render
        mujoco.mjv_updateScene(model, data, opt, None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)

        # Update the window
        mujoco.glfw.glfw.swap_buffers(window)
        mujoco.glfw.glfw.poll_events()

    # Used to view the environment
    def render(self):
        if(self.pov_working):
 
            self.windowView(self.model, self.data, self.opt, self.camera, self.scene, self.context, self.window)
            
            # Get the height map 
            self.depth_mujoco, self.rgb_buffer = get_height_map(self.model, self.data, self.camera, self.scene, self.context, self.window)


            # image = cv2.imread("image.png")  # OpenCV loads as BGR

            # self.rgb_buffer = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.depth_image = get_height_map_from_rgb(self.rgb_buffer)
            

            

            # # Preprocess the image
            # input_batch = self.midas_transform(self.rgb_buffer)

            # # Move to device
            # input_batch = input_batch.to(self.device)


            # # Get the depth map
            # # Perform depth estimation
            # with torch.no_grad():
            #     prediction = model(input_batch)

            #     prediction = torch.nn.functional.interpolate(
            #         prediction.unsqueeze(1),
            #         size=[self.height, self.width],
            #         mode="bicubic",
            #         align_corners=False,
            #     ).squeeze()

            # depth_map = prediction.cpu().numpy()  # Remove batch dimension and convert to NumPy
            
            # self.depth_map = depth_map


        ## WINDOW 2 ##
        self.windowView(self.model, self.data, self.opt, self.camera_top, self.scene_top, self.context_top, self.window_top)

    # Used to do a step in the environment
    def step(self):
        mujoco.mj_step(self.model, self.data)


    # Start sim time
    def startSim(self):
        self.simStart = self.data.time

        return self.simStart
    
    # Set torques
    def setTorques(self, torques):
        self.data.ctrl[0:3] = torques

    
    # Close
    def close(self):
        mujoco.glfw.glfw.terminate()    

        



if __name__ == "__main__":

    # # Load the model
    # model_type = "MiDaS_small"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
    # model = torch.hub.load("intel-isl/MiDaS", model_type)



    env = quadrupedEnv()
    env_fn = lambda : quadrupedEnv()

    # Hyperparameters
    ac_kwargs = dict(hidden_sizes=[256, 256])

    # Train SAC
    sac_pytorch(
        env_fn=env_fn,
        actor_critic=None,  # Use default MLPActorCritic or provide a custom one
        ac_kwargs=ac_kwargs,
        seed=42,
        steps_per_epoch=4000,
        epochs=100,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        alpha=0.2,
        batch_size=100,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        num_test_episodes=10,
        max_ep_len=1000,
        save_freq=10,
    )


    # simstart = env.startSim()
    
    # while not mujoco.glfw.glfw.window_should_close(env.window_top):
    #     simstart = env.startSim()

    #     torques = [0.1, 0.001, 0.001]

    #     while (env.data.time - simstart < 10.0/60.0):
    #         env.step()

    #     env.render()
    #     if(env.pov_working):
    #         # Display the height map
    #         cv2.imshow("Height Map", env.depth_mujoco)
    #         # cv2.imshow("Depth Map", env.depth_map)

    #         if cv2.waitKey(1) == ord('q'):
    #             break
        
    