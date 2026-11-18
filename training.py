'''
script for training the DeepQLearningAgent (PyTorch) on snake, model version v17.1

Usage
-----
python training.py                       # full training run (as in the original repo)
python training.py --episodes 200 --log-frequency 50 --warmup-games 32
                                         # tiny run to check that everything works
'''
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
try:
    from tqdm import tqdm
except ImportError:          # tqdm is optional: fall back to a plain loop
    def tqdm(x, *args, **kwargs):
        return x

from agent import DeepQLearningAgent
from game_environment import SnakeNumpy
from utils import play_game2

# ---------------------------------------------------------------------------
# command line options, defaults reproduce the original training run
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train the DQN snake agent')
parser.add_argument('--episodes', type=int, default=2 * (10**5),
                    help='number of training iterations')
parser.add_argument('--log-frequency', type=int, default=500,
                    help='evaluate, sync target network and save model every n iterations')
parser.add_argument('--warmup-games', type=int, default=512,
                    help='parallel games used to fill the buffer before training')
args = parser.parse_args()

# some global variables
torch.manual_seed(42)   # replaces tf.random.set_seed(42) of the original code
version = 'v17.1'

# get training configurations
with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames']  # keep frames >= 2
    max_time_limit = m['max_time_limit']
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
    buffer_size = m['buffer_size']

# define no of episodes, logging frequency
episodes = args.episodes
log_frequency = args.log_frequency
games_eval = 8

# make sure the output folders exist
os.makedirs('model_logs', exist_ok=True)
os.makedirs('models/{:s}'.format(version), exist_ok=True)

# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions,
                           buffer_size=buffer_size, version=version)
agent.print_models()

# setup the epsilon range and decay rate for epsilon (epsilon greedy exploration)
# reward type 'current': every step is added to the buffer with its own reward
epsilon, epsilon_end = 1, 0.01
reward_type = 'current'
sample_actions = False
n_games_training = 8*16
decay = 0.97

# play some games initially (random policy, epsilon = 1) to fill the buffer
games = args.warmup_games
env = SnakeNumpy(board_size=board_size, frames=frames,
                 max_time_limit=max_time_limit, games=games,
                 frame_mode=True, obstacles=obstacles, version=version)
ct = time.time()
_ = play_game2(env, agent, n_actions, n_games=games, record=True,
               epsilon=epsilon, verbose=True, reset_seed=False,
               frame_mode=True, total_frames=games*64)
print('Playing {:d} frames took {:.2f}s'.format(games*64, time.time()-ct))

# environments for training and for evaluation
env = SnakeNumpy(board_size=board_size, frames=frames,
                 max_time_limit=max_time_limit, games=n_games_training,
                 frame_mode=True, obstacles=obstacles, version=version)
env2 = SnakeNumpy(board_size=board_size, frames=frames,
                  max_time_limit=max_time_limit, games=games_eval,
                  frame_mode=True, obstacles=obstacles, version=version)

# training loop
model_logs = {'iteration': [], 'reward_mean': [],
              'length_mean': [], 'games': [], 'loss': []}
for index in tqdm(range(episodes)):
    # make small changes to the buffer (play a few frames) and slowly train
    _, _, _ = play_game2(env, agent, n_actions, epsilon=epsilon,
                         n_games=n_games_training, record=True,
                         sample_actions=sample_actions, reward_type=reward_type,
                         frame_mode=True, total_frames=n_games_training,
                         stateful=True)
    # one gradient step on a random batch from the replay buffer
    loss = agent.train_agent(batch_size=64,
                             num_games=n_games_training, reward_clip=True)

    # check performance every once in a while
    if((index+1) % log_frequency == 0):
        # greedy evaluation (epsilon = -1 -> never random) on a few games
        current_rewards, current_lengths, current_games = \
            play_game2(env2, agent, n_actions, n_games=games_eval, epsilon=-1,
                       record=False, sample_actions=False, frame_mode=True,
                       total_frames=-1, total_games=games_eval)

        model_logs['iteration'].append(index+1)
        model_logs['reward_mean'].append(round(int(current_rewards)/current_games, 2))
        model_logs['length_mean'].append(round(int(current_lengths)/current_games, 2))
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'length_mean', 'games', 'loss']]\
          .to_csv('model_logs/{:s}.csv'.format(version), index=False)

        # copy weights to target network and save models
        agent.update_target_net()
        agent.save_model(file_path='models/{:s}'.format(version), iteration=(index+1))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
