# Implementation of 2vs2 Soccer Environment with one trained striker
import mlagents
import mlagents_envs
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from mlagents_envs.base_env import ActionTuple
import time
import matplotlib.pyplot as plt
from collections import deque

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=336, hidden_dim=512, output_dim=3):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, input_dim=336, hidden_dim=512):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)

# Memory class to store transitions
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

# PPO Agent
class PPOAgent:
    def __init__(self, input_dim=336, hidden_dim=512, action_dim=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(input_dim, hidden_dim, action_dim).to(self.device)
        self.value = ValueNetwork(input_dim, hidden_dim).to(self.device)
        self.memory = PPOMemory()

        # Hyperparameters
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.c1 = 0.5
        self.c2 = 0.01
        self.batch_size = 64
        self.n_epochs = 10
        self.max_grad_norm = 0.5

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.learning_rate)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            logits = self.policy(state)
            value = self.value(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def store_transition(self, state, action, reward, value, log_prob, done):
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.rewards.append(reward)
        self.memory.values.append(value)
        self.memory.log_probs.append(log_prob)
        self.memory.dones.append(done)

    def compute_advantages(self):
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.memory.values, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.memory.dones, dtype=torch.float32).to(self.device)

        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self):
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)

        advantages, returns = self.compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_states = states[idx:idx + self.batch_size]
                batch_actions = actions[idx:idx + self.batch_size]
                batch_log_probs = old_log_probs[idx:idx + self.batch_size]
                batch_advantages = advantages[idx:idx + self.batch_size]
                batch_returns = returns[idx:idx + self.batch_size]

                # Policy loss
                logits = self.policy(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                values = self.value(batch_states).squeeze()
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()
        self.memory.clear()


def calculate_reward(decision_steps, terminal_steps):
    # Define your reward calculation logic here
    reward = 0
    if len(terminal_steps) > 0:
        reward = terminal_steps.reward[0]
    return reward

def train_single_striker(env, n_episodes=1000, max_steps_per_episode=2000):
    striker = PPOAgent()
    episode_rewards = []
    goals_scored = []
    moving_avg_reward = deque(maxlen=100)

    for episode in range(n_episodes):
        env.reset()
        episode_reward = 0
        episode_goals = 0
        steps = 0
        while steps < max_steps_per_episode:
            for behavior_name in list(env.behavior_specs.keys()):
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                if len(decision_steps) == 0:
                    continue

                n_agents = len(decision_steps)

                if "team=1" in behavior_name:
                    striker_obs = decision_steps.obs[0][0]
                    action, log_prob, value = striker.get_action(striker_obs)
                    reward = calculate_reward(decision_steps, terminal_steps)
                    done = len(terminal_steps) > 0
                    striker.store_transition(striker_obs, action, reward, value, log_prob, done)
                    episode_reward += reward
                    if len(terminal_steps) > 0 and np.any(terminal_steps.reward > 0):
                        episode_goals += 1
                    striker_action = action.reshape(1, 3)  # random actions for other 3 agents
                    other_actions = np.random.randint(0, 3, (n_agents - 1, 3), dtype=np.int32)
                    discrete_actions = np.vstack([striker_action, other_actions])
                else:
                    discrete_actions = np.random.randint(0, 3, (n_agents, 3), dtype=np.int32)
                continuous_actions = np.zeros((n_agents, 0))
                action_tuple = ActionTuple(continuous_actions, discrete_actions)
                env.set_actions(behavior_name, action_tuple)
            env.step()
            steps += 1

            if len(striker.memory.states) >= 2048:
                striker.update()

        episode_rewards.append(episode_reward)
        goals_scored.append(episode_goals)
        moving_avg_reward.append(episode_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Goals: {episode_goals}")
            plot_training_progress(episode_rewards, goals_scored)

    return episode_rewards, goals_scored

def plot_training_progress(rewards, goals):  # Plot the graphs
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(2, 1, 2)
    plt.plot(goals)
    plt.title('Goals Scored per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Goals')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize the Unity environment, need to modify with environment path in pc
    env = UnityEnvironment(file_name="/Users/phani/Desktop/DRL Project/ml-agents/training-envs-executables/SoccerTwos/SoccerTwos.app")  # Replace with your path
    try:
        rewards, goals = train_single_striker(env)
    finally:
        # Closing environment
        env.close()
