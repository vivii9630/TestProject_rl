import gym
import os
from stable_baselines3 import PPO
TOTAL_TIMESTEPS=100000

modeldirectory = "models/PPO"
logdir = "logs"

if not os.path.exists(modeldirectory):
    os.makedirs(modeldirectory)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('LunarLander-v2')
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

for i in range(30):
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{modeldirectory}/{TOTAL_TIMESTEPS*i}")

"""max_episodes = 100
for ep in range(max_episodes):
    init_state = env.reset()
    st_reward = 0
    done= False
    step_count = 0
    while not done:
        next_state, reward, done, info = env.step(env.action_space.sample())
        env.render()
        init_state = next_state
        st_reward = reward
        step_count += 1
    print('Episode: {} , Step count: {} , Reward: {}'.format(ep, step_count, st_reward))
env.close()"""