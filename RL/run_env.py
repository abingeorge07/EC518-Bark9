from ENV import QuadrupedEnv

# Initialize the environment with the path to your XML file
xml_path = "PATRIQ/patriq.xml"  # Update this to the correct path
env = QuadrupedEnv(xml_path)

try:
    # Reset the environment
    obs = env.reset()
    print("Initial Observation:", obs)

    # Take a few steps with random actions
    for _ in range(20):  # Run 10 steps
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, info = env.step(action)  # Step the environment

        # Render the environment
        env.render()  # Display the MuJoCo Viewer

        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        if done:
            print("Environment reached a termination condition.")
            break
finally:
    # Ensure the environment is properly closed
    env.close()
