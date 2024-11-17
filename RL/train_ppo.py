from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Initialize the environment
env = QuadrupedEnv("PATRIQ/patriq.xml")

# Check the environment
check_env(env)

# Train the PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_quadruped")
