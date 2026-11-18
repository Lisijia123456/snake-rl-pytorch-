'''
One entry point for everything, run it from a terminal (Anaconda Prompt) with the
GA01 conda environment active. Nothing needs to be installed.

    python run.py check            # 1. sanity checks of the PyTorch agent (seconds)
    python run.py quick            # 2. tiny training run to test the whole pipeline
    python run.py train            # 3. full training (200 000 iterations, long)
    python run.py plot             # 4. learning curves  -> images/learning_curves_v17.1.png
    python run.py play             # 5. play with the latest checkpoint -> images/*.gif

Extra arguments are passed on to the script, e.g.
    python run.py train --episodes 50000
    python run.py play --iteration 2000 --games 3
'''
import glob
import os
import re
import subprocess
import sys

# always work from the folder of this file, the code uses relative paths
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

try:
    import torch  # noqa: F401
except ImportError:
    sys.exit('PyTorch was not found in this Python:\n  {}\n'
             'Open an Anaconda Prompt, activate the GA01 conda environment '
             '(conda activate <name of GA01 env>) and run this command again.'.format(sys.executable))


def latest_checkpoint():
    """Highest iteration number among models/v17.1/model_XXXX.pt"""
    its = [int(re.search(r'model_(\d+)\.pt$', f).group(1))
           for f in glob.glob(os.path.join('models', 'v17.1', 'model_*.pt'))
           if re.search(r'model_(\d+)\.pt$', f)]
    return max(its) if its else None


def run(script, *args):
    """Run a script of this project with the same Python that runs run.py"""
    cmd = [sys.executable, script] + list(args)
    print('>>>', ' '.join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('check', 'quick', 'train', 'plot', 'play'):
        print(__doc__)
        sys.exit(1)
    mode, extra = sys.argv[1], sys.argv[2:]
    if mode == 'check':
        run('test_agent.py')
    elif mode == 'quick':
        run('training.py', '--episodes', '200', '--log-frequency', '50',
            '--warmup-games', '32', *extra)
    elif mode == 'train':
        run('training.py', *extra)
    elif mode == 'plot':
        run('plot_curves.py')
    elif mode == 'play':
        if '--iteration' not in extra:
            it = latest_checkpoint()
            if it is None:
                sys.exit('No checkpoint found in models/v17.1/, run "python run.py quick" '
                         'or "python run.py train" first.')
            extra = ['--iteration', str(it)] + extra
        if '--format' not in extra:
            extra += ['--format', 'gif']
        run('game_visualization.py', *extra)


if __name__ == '__main__':
    main()
