import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size, input_dim, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((max_size, input_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, input_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        return (
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.new_state_memory[batch],
            self.terminal_memory[batch]
        )

class Actor(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, n_actions, name="actor", checkpoint_dir="tmp/td3"):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.out = nn.Linear(fc2_dim, n_actions)
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))  # Output in [-1, 1]
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Critic(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, n_actions, name="critic", checkpoint_dir="tmp/td3"):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + n_actions, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.out = nn.Linear(fc2_dim, 1)
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class TD3Agent:
    def __init__(self, input_dim, n_actions, max_action, min_action,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 fc1=200, fc2=100, policy_noise=0.2, noise_clip=0.5, policy_delay=2):

        self.actor = Actor(input_dim, fc1, fc2, n_actions, 'actor').cuda()
        self.actor_target = Actor(input_dim, fc1, fc2, n_actions, 'target_actor').cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(input_dim, fc1, fc2, n_actions, 'critic1').cuda()
        self.critic_2 = Critic(input_dim, fc1, fc2, n_actions, 'critic2').cuda()
        self.critic_target_1 = Critic(input_dim, fc1, fc2, n_actions, 'target_critic1').cuda()
        self.critic_target_2 = Critic(input_dim, fc1, fc2, n_actions, 'target_critic2').cuda()
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.min_action = min_action
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

    def choose_action(self, observation, noise=0.1):
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).cuda()
        action = self.actor(state).cpu().detach().numpy()[0]
        if noise > 0:
            action += np.random.normal(0, noise, size=self.n_actions)
        return np.clip(action, self.min_action, self.max_action)

    def soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def update(self, replay_buffer, batch_size):
        if replay_buffer.mem_cntr < batch_size:
            return

        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample_buffer(batch_size)

        state = torch.tensor(state, dtype=torch.float32).cuda()
        action = torch.tensor(action, dtype=torch.float32).cuda()
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).cuda()

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(self.min_action, self.max_action)

            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = reward + self.gamma * (1 - done) * torch.min(target_Q1, target_Q2)

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic_1, self.critic_target_1)
            self.soft_update(self.critic_2, self.critic_target_2)

    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.critic_target_1.save_checkpoint()
        self.critic_target_2.save_checkpoint()
    
    def load_models(self):
        try:
            self.actor.load_checkpoint()
            self.actor_target.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.critic_target_1.load_checkpoint()
            self.critic_target_2.load_checkpoint()

            print("Successfully loaded models.")
        except:
            print("Failed to load models. Starting from scratch")