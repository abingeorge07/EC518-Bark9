import gym
import mujoco
import mujoco_viewer
import numpy as np
from gym import spaces

class QuadrupedEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        # Initialize step count
        self.step_count = 0

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = self.data.qpos.shape[0] + self.data.qvel.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        self.data.ctrl[:] = action
        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.data = mujoco.MjData(self.model)
        self.step_count = 0
        return self._get_obs()

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _compute_reward(self):
        # Reward for forward progress
        fwd_velocity = self.data.qvel[0]  # Assuming qvel[0] is forward velocity
        fwd_reward = fwd_velocity

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

        # Combine rewards
        total_reward = fwd_reward + energy_penalty + orientation_penalty + terrain_adapt_reward

        return total_reward

    def _is_done(self):
        # Check for body orientation thresholds
        roll, pitch = self.data.qpos[3:5]  # Assuming qpos[3:5] are roll and pitch
        if abs(roll) > np.pi / 4 or abs(pitch) > np.pi / 4:
            return True  # Terminate if roll or pitch angle exceeds 45 degrees

        # Check for out-of-bounds
        position = self.data.qpos[:3]  # Assuming qpos[:3] are the x, y, z position
        if position[2] < 0.2:  # Terminate if the body height drops below 0.2 meters
            return True

        # Maximum episode steps
        max_steps = 1000  # Define a reasonable maximum for your use case
        if self.step_count >= max_steps:
            return True

        # Increment step count
        self.step_count += 1

        return False
