import numpy as np

# Path to your MuJoCo XML file (update with the correct path to your model file)
xml_path = "path_to_your_quadruped_model.xml"

# Create the environment
env = QuadrupedEnv(xml_path)

# Reset the environment
obs = env.reset()
print("Initial Observation:", obs)

# Take a few steps with random actions
try:
    for _ in range(10):  # Run 10 steps
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, info = env.step(action)  # Step the environment
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        if done:
            print("Environment reached a termination condition.")
            break
    env.close()
except Exception as e:
    print(f"An error occurred during testing: {e}")
