import gym
import mujoco
import mujoco_viewer
import numpy as np
from gym import spaces
# from mujoco_py import load_model_from_path, MjSim, MjViewer

class QuadrupedEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # self.sim = mujoco.MjSim(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = self.data.qpos.shape[0] + self.data.qvel.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        self.data.ctrl[:] = action
        # step the simulation
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        return obs, reward, done, info

    # def reset(self):
    #     self.sim.reset()
    #     return self._get_obs()

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer = None

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _compute_reward(self):
        # Define a custom reward function
        return 0

    def _is_done(self):
        # Define termination condition
        return False
