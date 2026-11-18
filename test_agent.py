'''
Quick sanity checks for the PyTorch DeepQLearningAgent (run from the repo root):

    python test_agent.py

Checks: network output shape, NHWC -> NCHW handling, one training step,
loss going down on a fixed batch, target network sync and save/load round trip.
'''
import os
import tempfile

import numpy as np
import torch

from agent import DeepQLearningAgent

board_size, frames, n_actions = 10, 2, 4
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions,
                           buffer_size=1000, version='v17.1')
agent.print_models()

# 1) output shape for a single state and for a batch (states are NHWC numpy arrays)
single = np.random.randint(0, 5, (board_size, board_size, frames)).astype(np.uint8)
batch = np.random.randint(0, 5, (7, board_size, board_size, frames)).astype(np.uint8)
assert agent._get_model_outputs(single).shape == (1, n_actions)
assert agent._get_model_outputs(batch).shape == (7, n_actions)

# 2) greedy move respects the legal move mask
legal = np.ones((7, n_actions), dtype=np.uint8)
legal[:, 0] = 0
assert (agent.move(batch, legal) != 0).all()

# 3) fill the buffer with random transitions, training on a fixed batch must reduce the loss
# reward and done are 1-D arrays (one value per transition), like the environment returns
agent.add_to_buffer(batch, np.random.randint(0, n_actions, 7), np.ones(7),
                    batch, np.zeros(7), np.ones((7, n_actions)))
agent.update_target_net()
losses = [agent.train_agent(batch_size=7, reward_clip=True) for _ in range(300)]
print('mean loss first/last 20 steps : {:.4f} / {:.4f}'.format(np.mean(losses[:20]), np.mean(losses[-20:])))
assert np.isfinite(losses).all()
# (targets move because the target network is frozen, so only compare the trend)
# compare averages of the first / last 20 steps, single values are noisy
assert np.mean(losses[-20:]) < np.mean(losses[:20])

# 4) target network is a copy only after update_target_net
agent.compare_weights()
agent.update_target_net()
for p, q in zip(agent._model.state_dict().values(), agent._target_net.state_dict().values()):
    assert torch.equal(p, q)

# 5) save / load round trip
with tempfile.TemporaryDirectory() as d:
    before = agent._get_model_outputs(batch)
    agent.save_model(file_path=d, iteration=1)
    assert os.path.exists(os.path.join(d, 'model_0001.pt'))
    agent.reset_models()
    agent.load_model(file_path=d, iteration=1)
    assert np.allclose(before, agent._get_model_outputs(batch), atol=1e-6)

print('all checks passed')
