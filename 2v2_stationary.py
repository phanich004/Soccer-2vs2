import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import matplotlib.pyplot as plt


class A2CNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(A2CNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.actor = nn.Linear(hidden_size // 2, action_size)
        self.critic = nn.Linear(hidden_size // 2, 1)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value


class A2CAgent:
    def __init__(self, state_size, action_size, is_striker=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = A2CNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0003)
        self.gamma = 0.99
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.is_striker = is_striker
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), value.item(), log_prob.item()

    def train(self):
        if len(self.states) == 0:
            return 0

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)

        # Calculate returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R
            advantages[t] = R - values[t]

        # Get current predictions
        action_probs, current_values = self.network(states)
        dist = torch.distributions.Categorical(action_probs)
        entropy = dist.entropy().mean()

        # Calculate losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(current_values.squeeze(), returns)
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()

        return total_loss.item()


def calculate_reward(obs):
    ball_pos = obs[0:2]
    goal_pos = np.array([15.0, 0.0])
    team_goal_pos = np.array([-15.0, 0.0])

    # Check if ball entered opponent's goal
    goal_distance = np.linalg.norm(ball_pos - goal_pos)
    if goal_distance < 1.5:
        return 1.0

    # Check if ball entered team's goal
    team_goal_distance = np.linalg.norm(ball_pos - team_goal_pos)
    if team_goal_distance < 1.5:
        return -1.0

    return 0.0


def train_soccer_agents(env_path, num_episodes=5000):
    env = UnityEnvironment(file_name=env_path, worker_id=1, no_graphics=False)
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    state_size = spec.observation_specs[0].shape[0]
    action_size = 3

    striker = A2CAgent(state_size, action_size, is_striker=True)
    opponent = A2CAgent(state_size, action_size, is_striker=False)

    episode_rewards = []
    total_goals = 0

    try:
        for episode in range(num_episodes):
            env.reset()
            episode_reward = 0
            episode_goals = 0

            for step in range(1000):
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                if len(decision_steps) == 0:
                    continue

                actions = np.zeros((len(decision_steps), action_size), dtype=np.int32)

                for index, agent_id in enumerate(decision_steps.agent_id):
                    obs = decision_steps.obs[0][index]
                    if agent_id % 2 == 0:
                        action, value, log_prob = striker.select_action(obs)
                        striker.states.append(obs)
                        striker.actions.append(action)
                        striker.values.append(value)
                        striker.log_probs.append(log_prob)
                    else:
                        action = np.random.randint(0, action_size)
                    actions[index, action] = 1

                action_tuple = ActionTuple(discrete=actions)
                env.set_actions(behavior_name, action_tuple)
                env.step()

                next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)

                for index, agent_id in enumerate(decision_steps.agent_id):
                    if agent_id % 2 == 0:
                        obs = decision_steps.obs[0][index]
                        reward = calculate_reward(obs)
                        if reward > 0:
                            episode_goals += 1
                            total_goals += 1
                        striker.rewards.append(reward)
                        episode_reward += reward

            loss = striker.train()
            episode_rewards.append(episode_reward)

            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Goals: {episode_goals}, "
                  f"Total Goals: {total_goals}, "
                  f"Loss: {loss:.4f}")

            if (episode + 1) % 100 == 0:
                torch.save(striker.network.state_dict(), f"saved_models/striker_model_episode_{episode + 1}.pth")

    finally:
        torch.save(striker.network.state_dict(), "saved_models/striker_final.pth")
        env.close()
    return episode_rewards


if __name__ == "__main__":
    env_path = r"C:\Users\shrav\ml-agents\training-envs-executables\SoccerTwos\SoccerTwos.app"
    try:
        rewards = train_soccer_agents(env_path)

        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title('Soccer Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.savefig('training_progress.png')
        plt.show()

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()