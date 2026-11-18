'''
Plot the learning curves (mean reward, mean length, loss) of a training run.
Run it in PyCharm after training.py has written model_logs/v17.1.csv and
take a screenshot of the window that opens (also saved to images/).

    python plot_curves.py
'''
import os

import matplotlib.pyplot as plt

from utils import plot_logs

os.makedirs('images', exist_ok=True)
# plot_logs ends with plt.show(), so hook into it to save the figure first
_show = plt.show
def _save_and_show(*args, **kwargs):
    plt.savefig('images/learning_curves_v17.1.png', dpi=150)
    _show(*args, **kwargs)
plt.show = _save_and_show

plot_logs('model_logs/v17.1.csv', title='DQN snake v17.1 (PyTorch): learning curves')
