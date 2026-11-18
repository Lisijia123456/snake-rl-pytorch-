# script to visualize how the trained agent plays a game (the final policy)
# useful to study different iterations
#
# Usage
# -----
# python game_visualization.py --iteration 163500            # mp4 (needs ffmpeg)
# python game_visualization.py --iteration 163500 --format gif   # gif (needs pillow only)

import argparse
import json
import os

from agent import DeepQLearningAgent
from game_environment import Snake
from utils import visualize_game

parser = argparse.ArgumentParser(description='Visualize games played by a trained agent')
parser.add_argument('--iteration', type=int, required=True,
                    help='which saved model (models/v17.1/model_<iteration>.pt) to use')
parser.add_argument('--games', type=int, default=5, help='number of games to record')
parser.add_argument('--format', choices=['mp4', 'gif'], default='mp4',
                    help='output format of the animation')
args = parser.parse_args()

# some global variables
version = 'v17.1'

with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames']  # keep frames >= 2
    obstacles = bool(m['obstacles'])

# longer time limit than in training so that good games can run to the end
max_time_limit = 398

# setup the environment (boards with random obstacles, see obstacles_board_generator.py)
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
            obstacles=obstacles, version=version)
s = env.reset()
n_actions = env.get_num_actions()

# setup the agent and load the trained weights
agent = DeepQLearningAgent(board_size=board_size, frames=frames,
                           n_actions=n_actions, buffer_size=10, version=version)
agent.load_model(file_path='models/{:s}'.format(version), iteration=args.iteration)

os.makedirs('images', exist_ok=True)
for i in range(args.games):
    visualize_game(env, agent,
                   path='images/game_visual_{:s}_{:d}_{:d}.{:s}'.format(
                       version, args.iteration, i, args.format),
                   debug=False, animate=True, fps=12)
