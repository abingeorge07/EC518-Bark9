import numpy as np
import torch
import torch.optim as optim




# Learn
def learn(env,
          lr=1e-4,
          total_timesteps = 100,
          buffer_size = 50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=10,
          action_repeat=4,
          batch_size=32,
          learning_starts=600/10,
          gamma=0.99,
          target_network_update_freq=100,
          model_identifier='patriq'):
    
    model_identifiier = model_identifier+"_Gamma_"+str(gamma)
    episode_rewards = [0.0]
    training_losses = []
    actions = get_action_set()
    action_size = len(actions)
