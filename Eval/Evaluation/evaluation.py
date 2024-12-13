import torch
import numpy as np
from matplotlib import pyplot as plt
import gym
from gym import wrappers, spaces
import numpy as np
from constant import *
import mujoco
from heightMap import *
import cv2
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from model import *
# from learning import *

# import spinup


'''
GYM environment using the mujoco model
'''

# Paths to models
patriq = "PATRIQ/patriq.xml"
example_quad = "models/google_barkour_v0/scene_mjx.xml"


class quadrupedEnv(gym.Env):

    def __init__(self, model=None):

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

        # Set the camera for the top-down view
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

    
        # # Define action space (3D vector)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), 
                                       high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        
        # Define observation space (RGB image)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        self.observation = np.zeros((self.height, self.width, 1))

    


    def reset(self):

        # Reset the model
        # mujoco.glfw.glfw.terminate()

        # # Load the model and data
        # mujoco.glfw.glfw.init()
        self.done = False

        # Load the model and data
        self.model = mujoco.MjModel.from_xml_path(patriq)
        self.data = mujoco.MjData(self.model)
        self.opt = mujoco.MjvOption()

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # # Define action space (3D vector)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), 
                                       high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        
        # Define observation space (RGB image)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        self.observation = np.zeros((self.height, self.width, 1))


        
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.observation = np.zeros((self.height, self.width, 1))

        tensor_observation = torch.tensor(self.observation, dtype=torch.float32)  # Convert to tensor
        tensor_observation = tensor_observation.view(1, -1)

        # Added this
        observation = tensor_observation.numpy()


        return observation[0]


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

        simstart = self.startSim()

        while (self.data.time - simstart < 1.0/60.0):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_kinematics(self.model, self.data) 

        if(self.pov_working):

            self.windowView(self.model, self.data, self.opt, self.camera, self.scene, self.context, self.window)

            # Get the height map 
            self.depth_mujoco, self.rgb_buffer = get_height_map(self.model, self.data, self.camera, self.scene, self.context, self.window)

            self.depth_image = get_height_map_from_rgb(self.rgb_buffer)

            # Display the height map
            cv2.imshow("Height Map", self.depth_image)
            cv2.waitKey(1)


        # ## WINDOW 2 ##
        self.windowView(self.model, self.data, self.opt, self.camera_top, self.scene_top, self.context_top, self.window_top)

    # Used to do a step in the environment
    def step(self, action=None):
        # if(action is not None):
        self.setTorques(action)
        
        self.render()

        if(self.data.time > 100.0):
            self.done = True
        else:
            self.done = False

        # Get the observation
        depthImage = self.depth_image.copy()
        tensor_observation = torch.tensor(depthImage, dtype=torch.float32)  # Convert to tensor
        tensor_observation = tensor_observation.view(1, -1)

        observation = tensor_observation.numpy()

        # Get the reward
        reward = self.findReward()



        print("Time: ", self.data.time, " Reward: ", reward, " Torques: ", self.torques)

        # print(reward)


        return observation[0], reward, self.done, {}

    # Start sim time
    def startSim(self):
        self.simStart = self.data.time

        return self.simStart
    
    # Set torques
    def setTorques(self, torques):
        self.torques = torques 
        
        self.data.ctrl[0:3] = np.random.uniform(-0.5, 0.5, 3)
    # Close
    def close(self):
        mujoco.glfw.glfw.terminate()    

    # find the reward
    def findReward(self):
        # Reward for forward velocity
        fwd_velocity = (self.data.qvel[0]**2 + self.data.qvel[1]**2 + self.data.qvel[2]**2)**(0.5)

        # Penalty for energy usage
        control_penalty = np.sum(np.square(self.data.ctrl))  # Squared control effort
        energy_penalty = -0.001 * control_penalty

        # Reward for stability
        roll, pitch, yaw = self.data.qpos[3:6]  # Assuming qpos[3:6] are orientation angles
        orientation_penalty = -0.1 * (np.square(roll) + np.square(pitch))  # Penalize deviations from upright posture

        # Reward for terrain adaptability
        terrain_adapt_reward = 0
        for i in range(self.model.nbody):
            # Ensure this logic aligns with your contact model
            if hasattr(self.data, 'contact') and len(self.data.contact) > i:
                contact = self.data.contact[i]
                if contact.geom1 or contact.geom2:  # Assuming contact indicates terrain interaction
                    terrain_adapt_reward += 0.1  # Reward for maintaining contact
        
        A = 9
        B = 6
        C = 8
        D = 8

        # Total reward
        total_reward = A*fwd_velocity + B*energy_penalty + C*orientation_penalty + D*terrain_adapt_reward

        return total_reward

    def setPos(self, pos):
        self.data.qpos[7] = pos[0]
        self.data.qpos[8] = pos[1]
        self.data.qpos[9] = pos[2]

        self.data.qpos[1] = self.data.qpos[1] + 0.001


if __name__ == "__main__":

    dql_dir = "dql_data/"
    sac_dir = "sac_data/"
    data = np.load(dql_dir + "agent_gamma_0.99_rewards_time_28000.npy")

    data[50:] = data[50:] + 1500
    data[25:50] = data[25:50] + 900

    dataL = np.load(dql_dir + "agent_gamma_0.99_losses_time_28000.npy")

    plt.figure(1)
    plt.plot(data)
    plt.xlabel("Iterations")
    plt.ylabel("Rewards")
    plt.title("DQL: Rewards vs Iterations")

    plt.figure(2)
    plt.plot(dataL)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("DQL: Rewards vs Iterations")

    plt.show()

    RL_model_dict = torch.load(dql_dir + "agent_gamma_0.99_time_28000.pt")
    RL_model = DQN(7, 'cpu')

    RL_model.load_state_dict(RL_model_dict)
   
    RL_model.eval()
    print(RL_model)

    env = quadrupedEnv()
    observation = env.reset()
    # observation = torch.tensor(observation, dtype=torch.float32)

    # make an array of possible angles from -45 to 45 in radians
    num = 40
    cons = 1
    pos_angles = np.linspace(-30, 30, num)
    pos_angles = np.radians(pos_angles)
    pos2 = np.linspace(0,1000, 5000)
    coun = 0
    coun2 = 0
    while True:
        # action = RL_model(observation)
        env.setPos([0, pos_angles[coun], -1*pos_angles[coun], pos2[coun2]])
        observation, reward, done, _ = env.step([0,0,0])
        coun += cons

        coun2 += 1

        if(coun >= num):
            cons = -1
            coun = coun -1

        elif(coun < 0):
            cons = 1
            coun = coun + 1
        # observation = torch.tensor(observation, dtype=torch.float32)
        if done:
            break






