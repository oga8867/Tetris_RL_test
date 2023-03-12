from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
from stable_baselines3 import A2C
from stable_baselines3 import DQN
import os

env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)

models_dir = "models/DQN_tetris"
logdir = "tetrislogs/DQN_tetris"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
    
#model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device='cuda')

TIMESTEPS = 100000

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
model.save(f"{models_dir}/TET_model_{TIMESTEPS}")
