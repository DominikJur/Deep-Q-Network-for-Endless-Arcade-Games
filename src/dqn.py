import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import numpy as np

# choose device early so all tensors/networks can be moved there
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class PrioritizedExperienceReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=1e-6):
        self.memory = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon
        self.priorities = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)



        probabilities = np.array(self.priorities) ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.memory), size=batch_size, p=probabilities)
        
        weights = (len(self.memory) * probabilities) ** (self.beta)
        weights /= np.max(weights)
        weights = [weights[idx] for idx in indices]
        batch = [self.memory[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon
            
    def increment_beta(self):
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
    
    def __len__(self):
        return len(self.memory)

def create_network(input_dim, output_dim):
    network = DeepQNetwork(input_dim, output_dim)
    return network.to(device)

def load_network(path, input_dim, output_dim):
    network = create_network(input_dim, output_dim)
    # load with correct device mapping
    network.load_state_dict(torch.load(path, map_location=device))
    network.to(device)
    network.eval()
    return network

def save_network(network, path):
    torch.save(network.state_dict(), path)

def select_action(q_values, step, start, end, decay): # epsilon-greedy
    epsilon = end + (start - end) * math.exp(-step/decay)
    if random.random() < epsilon:
        return random.choice(range(len(q_values)))
    return torch.argmax(q_values).item()


def update_target_network(target_network, online_network, tau):
    # soft-update (in-place) of target network parameters from online network
    for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

def describe_episode(episode, episode_reward, steps):
    print(f"Episode {episode}: Total Reward: {episode_reward}, Steps: {steps}")


def train_ddqn(env, replay_buffer, num_episodes, gamma, batch_size=32, tau=0.01, learning_rate=1e-3):
    online_network = create_network(env.observation_space.shape[0], env.action_space.n)
    target_network = create_network(env.observation_space.shape[0], env.action_space.n)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()
    online_network.train()
    
    print("Training DDQN with Prioritized Experience Replay using device:", device)
    print("===============================================")
    
    replay_buffer.increment_beta()
    last_100_rewards = deque(maxlen=100)
    optimizer = optim.AdamW(online_network.parameters(), lr=learning_rate, weight_decay=1e-5)
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0
        while not done:
            # make sure the observation tensor is on the correct device
            q_values = online_network(torch.tensor(state, dtype=torch.float32, device=device))
            action = select_action(q_values, step, 0.9, 0.05, 500)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            
            if len(replay_buffer) < batch_size:
                state = next_state
                continue

            states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(batch_size)
            # move sampled data to the device
            states = torch.stack([torch.tensor(s, dtype=torch.float32, device=device) for s in states])
            next_states = torch.stack([torch.tensor(s, dtype=torch.float32, device=device) for s in next_states])
            actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.float32, device=device)
            
            q_values = online_network(states)
            current_q_values = q_values.gather(1, actions)
            
            with torch.no_grad():
                best_actions = online_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = target_network(next_states).gather(1, best_actions).squeeze(1)
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

            td_errors = target_q_values - current_q_values.squeeze(1)
            # move td errors to cpu before converting to numpy for the replay buffer
            replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

            # make sure importance-sampling weights are on the device
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            loss = torch.mean(weight_tensor * td_errors.pow(2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            update_target_network(target_network, online_network, tau)
            state = next_state
            step += 1
        
        describe_episode(episode, episode_reward, step)

        last_100_rewards.append(episode_reward)
        if episode % 100 == 0:
            save_network(online_network, f"lunar_lander_dqn_{episode}.pth")
            print(f"Average reward over last 100 episodes: {np.mean(last_100_rewards)}")
    return online_network