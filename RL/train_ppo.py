from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from ENV import QuadrupedEnv

# Define the custom RenderCallback
class RenderCallback(BaseCallback):
    """
    Custom callback to render the environment during training.
    """
    def __init__(self, render_freq=1000):
        super(RenderCallback, self).__init__()
        self.render_freq = render_freq  # Frequency of rendering in timesteps

    def _on_step(self) -> bool:
        # Render the environment every `render_freq` steps
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True

# Initialize the custom environment
xml_path = "PATRIQ/patriq.xml"  # Update this path to the correct XML file
env = QuadrupedEnv(xml_path)

# Wrap the environment in a vectorized wrapper
vec_env = make_vec_env(lambda: env, n_envs=1)

# Initialize PPO model
model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu')

# Create the custom callback for rendering
render_callback = RenderCallback(render_freq=1000)

# Train the model with the rendering callback
model.learn(total_timesteps=50000, callback=render_callback)

# Save the trained model
model.save("ppo_quadruped")
print("Training completed and model saved!")
