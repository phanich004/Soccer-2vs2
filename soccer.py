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


class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(PPONetwork, self).__init__()
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

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)


class GoalTracker:
    def __init__(self):
        self.goals = 0
        self.total_goals = 0
        self.episode_goals = 0

    def check_goal(self, obs):
        ball_pos = obs[0:2]
        goal_pos = np.array([15.0, 0.0])
        goal_distance = np.linalg.norm(ball_pos - goal_pos)
        if goal_distance < 1.5:
            self.goals += 1
            self.episode_goals += 1
            self.total_goals += 1
            return True
        return False

    def reset_episode(self):
        self.episode_goals = 0
        self.goals = 0


class PPOAgent:
    def __init__(self, state_size, action_size, is_striker=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        self.memory = PPOMemory()
        self.is_striker = is_striker

        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.ppo_epochs = 10
        self.batch_size = 64
        self.max_grad_norm = 0.5

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), value.item(), log_prob.item()

    def train(self):
        if len(self.memory.states) == 0:
            return 0

        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.memory.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.memory.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(self.memory.values)).to(self.device)
        log_probs = torch.FloatTensor(np.array(self.memory.log_probs)).to(self.device)
        dones = torch.FloatTensor(np.array(self.memory.dones)).to(self.device)

        # Calculate GAE
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.ppo_epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_indices = slice(idx, min(idx + self.batch_size, len(states)))

                action_probs, current_values = self.network(states[batch_indices])
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions[batch_indices])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - log_probs[batch_indices])
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[batch_indices]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(current_values.squeeze(), returns[batch_indices])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        self.memory.clear()
        return total_loss / (len(states) // self.batch_size)


def calculate_reward(obs, is_striker):
    ball_pos = obs[10:12]
    agent_pos = obs[2:4]
    ball_velocity = obs[4:61]
    goal_pos = np.array([15.0, 0.0])

    reward = 0
    # Striker reward
    if is_striker:
        ball_distance = np.linalg.norm(agent_pos - ball_pos)
        if ball_distance < 1.0:
            reward += 2.0
        else:
            reward -= ball_distance * 0.02

        goal_distance = np.linalg.norm(ball_pos - goal_pos)
        if goal_distance < 1.5:
            reward += 100.0
        elif goal_distance < 5.0:
            reward += (5.0 - goal_distance) * 2.0

        if ball_velocity[0] > 0:
            reward += ball_velocity[0] * 3.0

    # Goalie reward
    else:
        ball_distance = np.linalg.norm(agent_pos - ball_pos)
        reward -= ball_distance * 0.1

        goal_distance = np.linalg.norm(ball_pos - goal_pos)
        reward -= goal_distance * 0.5

    return reward


def train_soccer_agents(env_path, num_episodes=3):
    env = UnityEnvironment(file_name=env_path, worker_id=4, no_graphics=False)
    env.reset()

    # Get all behavior names
    behavior_names = list(env.behavior_specs.keys())
    print(f"Available behaviors: {behavior_names}")  # Debug line to see available behaviors

    # Get first behavior name and specs
    behavior_name = behavior_names[0]
    spec = env.behavior_specs[behavior_name]
    state_size = spec.observation_specs[0].shape[0]
    action_size = 3

    striker = PPOAgent(state_size, action_size, is_striker=True)
    goalie = PPOAgent(state_size, action_size, is_striker=False)
    goal_tracker = GoalTracker()
    episode_rewards = []

    try:
        for episode in range(num_episodes):
            env.reset()
            goal_tracker.reset_episode()
            episode_reward = 0
            episode_loss = 0

            for step in range(1000):
                # Handle all behaviors in each step
                for behavior_name in behavior_names:
                    decision_steps, terminal_steps = env.get_steps(behavior_name)

                    if len(decision_steps) == 0:
                        continue

                    n_agents = len(decision_steps)
                    actions = np.zeros((n_agents, action_size), dtype=np.int32)

                    for index in range(n_agents):
                        obs = decision_steps.obs[0][index]

                        if "team=1" in behavior_name:
                            if index % 2 == 0:  # Striker
                                action, value, log_prob = striker.select_action(obs)
                                actions[index, action] = 1
                                reward = calculate_reward(obs, is_striker=True)
                                striker.memory.add(obs, action, reward, value, log_prob, goal_tracker.check_goal(obs))
                            else:  # Goalie
                                action, value, log_prob = goalie.select_action(obs)
                                actions[index, action] = 1
                                reward = calculate_reward(obs, is_striker=False)
                                goalie.memory.add(obs, action, reward, value, log_prob, goal_tracker.check_goal(obs))
                            episode_reward += reward
                        else:  # Opposing team
                            action = np.random.randint(0, action_size)
                            actions[index, action] = 1

                    env.set_actions(behavior_name, ActionTuple(discrete=actions))

                env.step()

            striker_loss = striker.train()
            goalie_loss = goalie.train()
            episode_loss = (striker_loss + goalie_loss) / 2

            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Goals: {goal_tracker.episode_goals}, "
                  f"Total Goals: {goal_tracker.total_goals}, "
                  f"Avg Loss: {episode_loss:.4f}")

            if (episode + 1) % 100 == 0:
                os.makedirs("saved_models", exist_ok=True)
                torch.save(striker.network.state_dict(), f"saved_models/striker_model_episode_{episode + 1}.pth")
                torch.save(goalie.network.state_dict(), f"saved_models/goalie_model_episode_{episode + 1}.pth")

    finally:
        os.makedirs("saved_models", exist_ok=True)
        torch.save(striker.network.state_dict(), "saved_models/striker_final.pth")
        torch.save(goalie.network.state_dict(), "saved_models/goalie_final.pth")
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