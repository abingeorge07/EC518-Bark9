from ENV import QuadrupedEnv

# Initialize the environment with the path to your XML file
env = QuadrupedEnv("PATRIQ/patriq.xml")

# Test the environment
# obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Take random actions
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
