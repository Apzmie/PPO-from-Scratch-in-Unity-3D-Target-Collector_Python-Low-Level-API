from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(x).squeeze(-1)
        return mean, std, value
        

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.returns = []
        self.advantages = []

    def add(self, state, action, reward, next_state, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.__init__()
        
        
class Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam

    def select_action(self, state, train=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.actor_critic(state)
            if train:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            else:
                action = mean
                log_prob = None
        
        action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0).cpu().numpy(), log_prob, value
        
    def compute_gaes(self, buffer):
        with torch.no_grad():
            rewards = torch.FloatTensor(np.array(buffer.rewards))
            values = torch.FloatTensor(np.array(buffer.values))
            dones = torch.FloatTensor(np.array(buffer.dones))
            
            last_next_state = torch.FloatTensor(buffer.next_states[-1]).unsqueeze(0)
            _, _, last_value = self.actor_critic(last_next_state)
            
            gaes = [0] * len(rewards)
            last_gae = 0
            next_value = last_value.item()
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gaes[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
                next_value = values[t]
                last_gae = gaes[t]
             
            gaes = torch.FloatTensor(np.array(gaes))
            buffer.advantages = gaes.view(-1)
            buffer.returns = gaes.view(-1) + values.view(-1)

    def update(self, buffers, epochs=10, batch_size=128):
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        
        for buf in buffers:
            all_states.extend(buf.states)
            all_actions.extend(buf.actions)
            all_old_log_probs.extend(buf.log_probs)
            all_returns.append(buf.returns)
            all_advantages.append(buf.advantages)
            
        states = torch.FloatTensor(np.array(all_states))
        actions = torch.FloatTensor(np.array(all_actions))
        old_log_probs = torch.FloatTensor(np.array(all_old_log_probs))
        returns = torch.cat(all_returns).view(-1)
        advantages = torch.cat(all_advantages).view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                mean, std, values = self.actor_critic(states[idx])
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions[idx]).sum(-1)
                entropy = dist.entropy().sum(-1)
                
                ratios = torch.exp(log_probs - old_log_probs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.view(-1), returns[idx].view(-1))
                entropy_loss = -entropy.mean()
                loss = policy_loss + 0.5 * value_loss + 0.001 * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
        for buf in buffers:
            buf.clear()
            
        return policy_loss.item(), value_loss.item(), entropy_loss.item()


if __name__ == "__main__":
    channel1 = EngineConfigurationChannel()
    channel1.set_configuration_parameters(time_scale=10.0)
    channel2 = EngineConfigurationChannel()
    channel2.set_configuration_parameters(time_scale=10.0)
    env = UnityEnvironment(file_name="Build.x86_64", side_channels=[channel1], no_graphics=True, worker_id=0)
    test_env = UnityEnvironment(file_name="Build.x86_64", side_channels=[channel2], no_graphics=True, worker_id=1)
    env.reset()
    test_env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    t_behavior_name = list(test_env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    state_dim = spec.observation_specs[0].shape[0]
    action_dim = spec.action_spec.continuous_size
    agent = Agent(state_dim, action_dim)
    #agent.actor_critic.load_state_dict(torch.load("saved_model.pth"))
    buffer = RolloutBuffer()
    writer = SummaryWriter(log_dir="a")
    
    target_transitions = 128    # (* num_agents) per one update period
    test_interval = 10
    test_max_step = 1500

    update_count = 0
    best_test_reward = -float('inf')
    agent_buffers = {}
    collecting = {}
    
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            for agent_id in agent_ids:
                if agent_id not in agent_buffers:
                    agent_buffers[agent_id] = RolloutBuffer()
                    collecting[agent_id] = True
                
            states = [decision_steps[aid].obs[0] for aid in agent_ids]
            actions_list = []
            log_probs_list = []
            values_list = []
        
            for state in states:
                action, log_prob, value = agent.select_action(state)
                actions_list.append(action)
                log_probs_list.append(log_prob)
                values_list.append(value)

            actions_array = np.array(actions_list, dtype=np.float32)
            actions_tuple = ActionTuple(continuous=actions_array)
            env.set_actions(behavior_name, actions_tuple)
            
        env.step()
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)

        for i, agent_id in enumerate(agent_ids):
            state = states[i]
            action = actions_list[i]
            log_prob = log_probs_list[i]
            value = values_list[i]

            if agent_id in terminal_steps:
                reward = terminal_steps[agent_id].reward
                done = 1.0
                next_state = np.zeros_like(state)
            elif agent_id in next_decision_steps:
                reward = next_decision_steps[agent_id].reward
                done = 0.0
                next_state = next_decision_steps[agent_id].obs[0]
            else:
                continue              
            
            if collecting[agent_id]:
                agent_buffers[agent_id].add(state, action, reward, next_state, log_prob.item(), value.item(), done)                
                if len(agent_buffers[agent_id]) >= target_transitions or done:
                    collecting[agent_id] = False
            
        if all(not collecting[aid] for aid in agent_buffers):
            for buf in agent_buffers.values():
                agent.compute_gaes(buf)
            
            policy_loss, value_loss, entropy_loss = agent.update(list(agent_buffers.values()))
            update_count += 1
                
            for aid in agent_buffers:
                collecting[aid] = True             
                
            if update_count % test_interval == 0:
                print(f"Update Count {update_count}")
                test_env.reset()
                test_reward = 0
                max_step_count = 0
                test_done = False
                while not test_done:
                    t_decision_steps, t_terminal_steps = test_env.get_steps(t_behavior_name)
                    
                    t_agent_ids = t_decision_steps.agent_id
                    t_states = [t_decision_steps[aid].obs[0] for aid in t_agent_ids]
                    t_actions_list = []
                    
                    for t_state in t_states:
                        t_action, _, _ = agent.select_action(t_state, train=False)
                        t_actions_list.append(t_action)

                    t_actions_array = np.array(t_actions_list, dtype=np.float32)
                    t_actions_tuple = ActionTuple(continuous=np.array(t_actions_array))
                    test_env.set_actions(t_behavior_name, t_actions_tuple)

                    test_env.step()
                    max_step_count += 1
                    if max_step_count >= test_max_step:
                        test_done = True
                    
                    t_next_decision_steps, t_terminal_steps = test_env.get_steps(t_behavior_name)
                    
                    for t_agent_id in t_decision_steps.agent_id:
                        if t_agent_id in t_terminal_steps:
                            test_reward += t_terminal_steps[t_agent_id].reward
                            test_done = True
                        elif t_agent_id in t_next_decision_steps:
                            test_reward += t_next_decision_steps[t_agent_id].reward
                        else:
                            continue
                
                writer.add_scalar("Plot/1_Policy_Loss", policy_loss, update_count)
                writer.add_scalar("Plot/2_Value_Loss", value_loss, update_count)
                writer.add_scalar("Plot/3_Entropy_Loss", entropy_loss, update_count)
                writer.add_scalar("Plot/4_Test_Reward", test_reward, update_count)
                print(f"{test_reward:.4f}")
                
                if test_reward > best_test_reward:
                    best_test_reward = test_reward
                    torch.save(agent.actor_critic.state_dict(), "best_model.pth")
                    print(f"[Test] Model saved as 'best_model.pth' at new best reward {best_test_reward:.4f}")
