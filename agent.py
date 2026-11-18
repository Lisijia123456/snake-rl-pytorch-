"""
Agents for the Snake game.

This file is the PyTorch port of the original TensorFlow/Keras ``agent.py``
(https://github.com/DragonWarrior15/snake-rl). Only the pieces that are needed
for model v17.1 are kept: the ``Agent`` base class (replay buffer handling) and
``DeepQLearningAgent`` (a CNN based Deep Q-Network with a target network).

Main differences w.r.t. the TensorFlow version
----------------------------------------------
* Keras ``Model`` -> ``torch.nn.Sequential`` built from ``model_config/<version>.json``
* Keras expects boards as NHWC, PyTorch convolutions need NCHW. The environment
  and the replay buffer still use NHWC numpy arrays; the transposition happens
  in one place only (``_prepare_input``).
* ``model.compile(RMSprop, mean_huber_loss)`` -> ``torch.optim.RMSprop`` with the
  same hyper parameters as Keras + ``F.smooth_l1_loss`` (Huber, delta = 1)
* ``train_on_batch`` / ``predict_on_batch`` -> explicit forward / backward / step,
  and ``torch.no_grad()`` for everything that only needs predictions
* weights are stored as ``state_dict`` files (``.pt``) instead of ``.h5``
"""

import json
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import ReplayBufferNumpy


class Agent():
    """Base class for all agents (holds the replay buffer and common settings)

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as the env frame count
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, keep < 1 for convergence
    _use_target_net : bool
        If a target network is used to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Shape of a single state as stored by the environment (H, W, frames)
    _version : str
        model version string, used to find ``model_config/<version>.json``
    """

    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initialize the agent, see the class docstring for the arguments"""
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        # states coming from the environment are (board, board, frames)
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value"""
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset (re-create) the replay buffer, optionally with a new size"""
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size,
                                         self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the number of examples currently stored in the buffer"""
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step(s) to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board,
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the replay buffer to disk (pickle), file is tagged with iteration"""
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the replay buffer from disk, raises FileNotFoundError if missing"""
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)


class DeepQLearningAgent(Agent):
    """This agent learns the game via Deep Q learning (DQN).

    A convolutional network maps the last ``frames`` boards to one Q value per
    action. Training targets are computed with a separate, slowly updated
    target network (``_target_net``) which stabilises learning.

    Attributes
    ----------
    _device : torch.device
        GPU if available, otherwise CPU
    _model : torch.nn.Module
        The online network that is trained at every step
    _target_net : torch.nn.Module
        Copy of the online network, only synchronised from time to time
    _optimizer : torch.optim.Optimizer
        RMSprop, same settings as the Keras version (lr 5e-4, rho 0.9, eps 1e-7)
    """

    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Arguments are the same as for ``Agent``; additionally builds the
        online/target networks and the optimizer (``reset_models``)"""
        Agent.__init__(self, board_size=board_size, frames=frames,
                       buffer_size=buffer_size, gamma=gamma,
                       n_actions=n_actions, use_target_net=use_target_net,
                       version=version)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_models()

    # ------------------------------------------------------------------
    # model construction
    # ------------------------------------------------------------------
    def reset_models(self):
        """Create fresh online/target networks and a fresh optimizer"""
        self._model = self._agent_model().to(self._device)
        if(self._use_target_net):
            self._target_net = self._agent_model().to(self._device)
            self.update_target_net()
        # Keras RMSprop(0.0005) defaults: rho=0.9, epsilon=1e-7 (PyTorch calls rho
        # "alpha"; PyTorch default alpha=0.99 / eps=1e-8 would change the dynamics)
        self._optimizer = torch.optim.RMSprop(self._model.parameters(),
                                              lr=0.0005, alpha=0.9, eps=1e-7)

    @staticmethod
    def _init_weights(module):
        """Keras default initialisation: Glorot-uniform weights, zero biases"""
        if(isinstance(module, (nn.Conv2d, nn.Linear))):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _agent_model(self):
        """Build the Q network described in ``model_config/<version>.json``.

        The json only lists the hidden layers (Conv2D*, Flatten, Dense*); the
        input is (frames, board, board) and the output layer is a linear Dense
        layer with one unit per action, exactly like the Keras version.

        Returns
        -------
        model : torch.nn.Sequential
            Network mapping a (N, frames, H, W) tensor to (N, n_actions) Q values
        """
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            m = json.loads(f.read())

        modules = OrderedDict()
        in_channels = self._n_frames
        in_features = None   # known only after Flatten
        for name, cfg in m['model'].items():
            if('Conv2D' in name):
                kernel = tuple(cfg['kernel_size'])
                # Keras padding is 'valid' unless stated, PyTorch accepts 'same'
                # for stride 1 convolutions as well
                padding = cfg.get('padding', 'valid')
                modules[name] = nn.Conv2d(in_channels, cfg['filters'], kernel,
                                          padding=padding)
                in_channels = cfg['filters']
                if(cfg.get('activation') == 'relu'):
                    modules[name + '_relu'] = nn.ReLU()
            if('Flatten' in name):
                modules[name] = nn.Flatten()
                # infer the flattened size with a dummy forward pass
                with torch.no_grad():
                    dummy = torch.zeros(1, self._n_frames,
                                        self._board_size, self._board_size)
                    in_features = nn.Sequential(modules)(dummy).shape[1]
            if('Dense' in name):
                modules[name] = nn.Linear(in_features, cfg['units'])
                in_features = cfg['units']
                if(cfg.get('activation') == 'relu'):
                    modules[name + '_relu'] = nn.ReLU()
        # final layer, linear since the loss is a regression loss on Q values
        modules['action_values'] = nn.Linear(in_features, self._n_actions)

        model = nn.Sequential(modules)
        model.apply(self._init_weights)
        return model

    # ------------------------------------------------------------------
    # input handling and prediction
    # ------------------------------------------------------------------
    def _normalize_board(self, board):
        """Scale board values (0..4) to 0..1, returns a float32 copy"""
        return board.astype(np.float32)/4.0

    def _prepare_input(self, board):
        """Reshape, normalise and convert a numpy board to a network input

        Parameters
        ----------
        board : Numpy array
            A single state (H, W, frames) or a batch (N, H, W, frames)

        Returns
        -------
        board : torch.Tensor
            float32 tensor (N, frames, H, W) on the agent's device
        """
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board)
        # NHWC (environment / Keras layout) -> NCHW (PyTorch layout)
        board = np.ascontiguousarray(board.transpose(0, 3, 1, 2))
        return torch.from_numpy(board).to(self._device)

    def _get_model_outputs(self, board, model=None):
        """Get action values (Q values) as a numpy array, no gradients tracked

        Parameters
        ----------
        board : Numpy array
            The board state(s) for which to predict action values
        model : torch.nn.Module, optional
            Network to use, the online model (default) or the target network
        """
        board = self._prepare_input(board)
        if(model is None):
            model = self._model
        with torch.no_grad():
            model_outputs = model(board)
        return model_outputs.cpu().numpy()

    def move(self, board, legal_moves, value=None):
        """Greedy action: legal move with the largest Q value

        Parameters
        ----------
        board : Numpy array
            The board state(s) on which to calculate the best action
        legal_moves : Numpy array
            Binary array (N, n_actions), 1 where the action is allowed
        value : None, optional
            Kept for consistency with the interface used in utils.py
        """
        model_outputs = self._get_model_outputs(board, self._model)
        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)

    def get_action_proba(self, board, values=None):
        """Softmax over the Q values (only used when sampling actions)"""
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max does not change the softmax but avoids overflow
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1, 1))
        model_outputs = np.exp(model_outputs)
        return model_outputs/model_outputs.sum(axis=1).reshape((-1, 1))

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Sample a batch from the buffer and do one gradient descent step.

        For every sampled transition (s, a, r, s', done):

            target(s, a) = r + gamma * max_{a' legal} Q_target(s', a') * (1 - done)

        Only the Q value of the action that was taken is trained: for all other
        actions the target is set to the current prediction, so their error (and
        gradient) is exactly zero.

        Parameters
        ----------
        batch_size : int, optional
            Number of examples to sample from the buffer
        num_games : int, optional
            Not used here, kept for consistency with the training script
        reward_clip : bool, optional
            Clip rewards to -1/0/+1 with np.sign

        Returns
        -------
        loss : float
            Mean Huber loss of the batch
        """
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if(reward_clip):
            r = np.sign(r)

        # 1) bootstrap value from the target network (kept fixed for stability)
        net = self._target_net if self._use_target_net else self._model
        next_model_outputs = self._get_model_outputs(next_s, net)
        # only legal actions may be used for the max, illegal ones get -inf
        best_next = np.max(np.where(legal_moves == 1, next_model_outputs, -np.inf),
                           axis=1).reshape(-1, 1)
        # a terminal state may have no legal move (-inf); (1 - done) would turn
        # that into nan (-inf * 0), so replace non finite values with 0 first
        best_next = np.where(np.isfinite(best_next), best_next, 0.0)
        discounted_reward = r + self._gamma * best_next * (1 - done)

        # 2) build the regression target: only the taken action changes
        a = a.astype(np.float32)
        target = (1 - a) * self._get_model_outputs(s) + a * discounted_reward

        # 3) one optimisation step with the Huber loss (delta = 1, mean over
        #    all batch * action entries, same as the Keras mean_huber_loss)
        self._model.train()
        prediction = self._model(self._prepare_input(s))
        target = torch.from_numpy(target.astype(np.float32)).to(self._device)
        loss = F.smooth_l1_loss(prediction, target, beta=1.0)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def update_target_net(self):
        """Copy the online network weights to the target network.

        The target network is kept static between calls, this should not be
        done too frequently.
        """
        if(self._use_target_net):
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        """Utility to check whether online and target network are identical"""
        for (name, p), (_, q) in zip(self._model.state_dict().items(),
                                     self._target_net.state_dict().items()):
            print('{:s} match : {:d}'.format(name, int(torch.equal(p, q))))

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save_model(self, file_path='', iteration=None):
        """Save the network weights (``state_dict``) to ``model_XXXX.pt``
        (and ``model_XXXX_target.pt`` for the target network)"""
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(self._model.state_dict(),
                   "{}/model_{:04d}.pt".format(file_path, iteration))
        if(self._use_target_net):
            torch.save(self._target_net.state_dict(),
                       "{}/model_{:04d}_target.pt".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        """Load network weights saved by ``save_model``

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_state_dict(torch.load(
            "{}/model_{:04d}.pt".format(file_path, iteration),
            map_location=self._device))
        if(self._use_target_net):
            self._target_net.load_state_dict(torch.load(
                "{}/model_{:04d}_target.pt".format(file_path, iteration),
                map_location=self._device))

    def print_models(self):
        """Print the network architecture and the number of parameters"""
        print('Training Model')
        print(self._model)
        print('Trainable parameters : {:d}'.format(
            sum(p.numel() for p in self._model.parameters() if p.requires_grad)))
        if(self._use_target_net):
            print('Target Network')
            print(self._target_net)
