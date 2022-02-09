import gym
from stable_baselines3 import PPO
env = gym.make('LunarLander-v2')
env.reset()

model_dir = "models/A2C"
model_path = f"{model_dir}/2200000.zip"

model = PPO.load(model_path, env=env)
max_episodes = 100
for ep in range(max_episodes):
    init_state = env.reset()
    st_reward = 0
    done= False
    step_count = 0
    while not done:
        action, _ = model.predict(init_state)
        next_state, reward, done, info = env.step(action)
        env.render()
        init_state = next_state
        st_reward = reward
        step_count += 1
    print('Episode: {} , Step count: {} , Reward: {}'.format(ep, step_count, st_reward))
env.close()