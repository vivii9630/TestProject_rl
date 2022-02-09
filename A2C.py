import gym
import os
from stable_baselines3 import A2C
TOTAL_TIMESTEPS=100000

modeldirectory = "models/A2C"
logdir = "logs"

if not os.path.exists(modeldirectory):
    os.makedirs(modeldirectory)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('LunarLander-v2')


model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

for i in range(1,30):
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{modeldirectory}/{TOTAL_TIMESTEPS*i}")