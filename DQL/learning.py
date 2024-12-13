import numpy as np
import torch
import torch.optim as optim
from action import get_action_set, select_exploratory_action, select_greedy_action
from qlearning import perform_qlearning_step, update_target_net
from model import DQN
from replayBuffer import ReplayBuffer
from schedule import LinearSchedule
from utils import get_state, visualize_training
# initialize your carla env

def learn(env,
          lr=1e-4,
          total_timesteps = 100000,
          buffer_size = 50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=10,
          action_repeat=4,
          batch_size=32,
          learning_starts=1000,
          gamma=0.99,
          target_network_update_freq=100,
          save_freq=2000,
          model_identifier='agent'):
    """ Train a deep q-learning model.
    Parameters
    -------
    env: gym.Env
        environment to train on
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to take
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    action_repeat: int
        selection action on every n-th frame and repeat action for intermediate frames
    batch_size: int
        size of a batched sampled from replay buffer for training
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    model_identifier: string
        identifier of the agent
    """


    # action_repeat = i
    i = gamma
    model_identifier = 'agent_gamma_' + str(i)

    # model_identifier = 'agent_actionRepeat_' + str(i)
    episode_rewards = [0.0]
    training_losses = []
    actions = get_action_set()
    action_size = len(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    directory = 'experiment2/'

    # Build networks
    policy_net = DQN(action_size, device).to(device)
    target_net = DQN(action_size, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                initial_p=1.0,
                                final_p=exploration_final_eps)

    # Initialize environment and get first state
    obs = get_state(env.reset())

    # Iterate over the total number of time steps
    for t in range(total_timesteps):

        # Select action
        action_id = select_exploratory_action(obs, policy_net, action_size, exploration, t)
        env_action = actions[action_id]

        # Perform action fram_skip-times
        for f in range(action_repeat):
            new_obs, rew, done, _ = env.step(env_action)
            episode_rewards[-1] += rew
            if done:
                break

        # Store transition in the replay buffer.
        new_obs = get_state(new_obs)
        replay_buffer.add(obs, action_id, rew, new_obs, float(done))
        obs = new_obs

        if done:
            # Start new episode after previous episode has terminated
            print("timestep: " + str(t) + " \t reward: " + str(episode_rewards[-1]))
            obs = get_state(env.reset())
            episode_rewards.append(0.0)


        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            loss = perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device)
            training_losses.append(loss)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target_net(policy_net, target_net)

        if t > learning_starts and t % save_freq == 0:
            # Save the model
            torch.save(policy_net.state_dict(), directory + model_identifier+'_time_'+str(t)+'.pt')
            print("Model saved at timestep: " + str(t))
            np.save(directory + model_identifier+'_rewards_time_'+str(t)+'.npy', episode_rewards)
            np.save(directory + model_identifier+'_losses_time_'+str(t)+'.npy', training_losses)



    np.save(directory + model_identifier+'_rewards.npy', episode_rewards)
    np.save(directory + model_identifier+'_losses.npy', training_losses)

    # Save the trained policy network
    torch.save(policy_net.state_dict(), directory + model_identifier+'.pt')

    # Visualize the training loss and cumulative reward curves
    visualize_training(episode_rewards, training_losses, model_identifier)
