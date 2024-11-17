from stable_baselines3 import PPO

from ENV import QuadrupedEnv

# Initialize the environment
env = QuadrupedEnv("PATRIQ/patriq.xml")

# Load the trained model
model = PPO.load("ppo_quadruped")

# Run the environment with the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
