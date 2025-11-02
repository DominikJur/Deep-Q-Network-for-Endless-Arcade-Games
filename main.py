from src.dqn import *

import torch
import gymnasium as gym
import warnings
import numpy as np
import time
warnings.filterwarnings("ignore")

train = False
if train:
    env = gym.make("LunarLander-v3")

    replay_buffer = PrioritizedExperienceReplayBuffer(capacity=10000)
    q_network = train_ddqn(env, replay_buffer, num_episodes=2500, gamma=0.99)

    save_network(q_network, "lunar_lander_dqn.pth")

    env.close()

eval_env = gym.make("LunarLander-v3", render_mode="human")

q_network = load_network("lunar_lander_dqn.pth", eval_env.observation_space.shape[0], eval_env.action_space.n)
device = next(q_network.parameters()).device

input("Press Enter to watch the trained agent land...")

num_eval_episodes = 5
for ep in range(num_eval_episodes):
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0.0
    while not done:
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = int(q_network(state).argmax(dim=1).item())
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(0.01)  
    print(f"Render Eval episode {ep+1}: reward = {total_reward:.2f}")

eval_env.close()
