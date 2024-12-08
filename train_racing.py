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

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

# import spinup


'''
GYM environment using the mujoco model
'''

# Paths to models
patriq = "PATRIQ/stairs.xml"
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

    
        # # Define action space (3D vector)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), 
                                       high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        
        # Define observation space (RGB image)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        self.observation = np.zeros((self.height, self.width, 1))

    


    def reset(self):

        # Reset the model
        mujoco.glfw.glfw.terminate()

        # Load the model and data
        mujoco.glfw.glfw.init()
        self.done = False

        # Load the model and data
        self.model = mujoco.MjModel.from_xml_path(patriq)
        self.data = mujoco.MjData(self.model)
        self.opt = mujoco.MjvOption()

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

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


        
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.observation = np.zeros((self.height, self.width, 1))

        tensor_observation = torch.tensor(self.observation, dtype=torch.float32)  # Convert to tensor
        tensor_observation = tensor_observation.view(1, -1)

        return tensor_observation


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
        if(action is not None):
            self.setTorques(action)
        
        self.render()

        if(self.data.time > 20.0):
            self.done = True
        else:
            self.done = False

        # Get the observation
        depthImage = self.depth_image.copy()
        tensor_observation = torch.tensor(depthImage, dtype=torch.float32)  # Convert to tensor
        tensor_observation = tensor_observation.view(1, -1)

        # Get the reward
        reward = self.findReward()



        print("Time: ", self.data.time, " Reward: ", reward, " Torques: ", self.torques)

        # print(reward)


        return tensor_observation, reward, self.done, {}

    # Start sim time
    def startSim(self):
        self.simStart = self.data.time

        return self.simStart
    
    # Set torques
    def setTorques(self, torques):
        self.torques = torques + 0.1*np.random.randn(3)
        self.data.ctrl[0:3] = self.torques

    
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
        
        A = 4
        B = 1
        C = 8
        D = 10

        # Total reward
        total_reward = A*fwd_velocity + B*energy_penalty + C*orientation_penalty + D*terrain_adapt_reward

        return 1/total_reward

        

def experiment(variant):
    expl_env = NormalizedBoxEnv(quadrupedEnv())
    eval_env = NormalizedBoxEnv(quadrupedEnv())
    obs_dim = expl_env.observation_space.low.size
 
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,

    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()





if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E4),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)



# if __name__ == "__main__":

# # # Load the model
# # model_type = "MiDaS_small"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
# # model = torch.hub.load("intel-isl/MiDaS", model_type)



#     env = quadrupedEnv()
    
#     while not mujoco.glfw.glfw.window_should_close(env.window_top):
#         simstart = env.startSim()

#         torques = [0.1, 0.001, 0.001]

#         while (env.data.time - simstart < 10.0/60.0):
#             env.step()

#         env.render()
#         if(env.pov_working):
#             # Display the height map
#             cv2.imshow("Height Map", env.depth_mujoco)
#             # cv2.imshow("Depth Map", env.depth_map)

#             if cv2.waitKey(1) == ord('q'):
#                 break
           
    