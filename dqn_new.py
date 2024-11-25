import os
import random
import time
from dataclasses import dataclass
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# 定义 Sample 数据结构
Sample = namedtuple("Sample", ["observations", "next_observations", "actions", "rewards", "dones"])

class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space):
        self.buffer_size = buffer_size
        self.obs_buf = np.zeros((buffer_size, *observation_space.shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((buffer_size, *observation_space.shape), dtype=np.float32)
        self.actions_buf = np.zeros((buffer_size, *action_space.shape), dtype=np.int64)
        self.rewards_buf = np.zeros(buffer_size, dtype=np.float32)
        self.dones_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.dones_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return Sample(
            observations=torch.tensor(self.obs_buf[indices], dtype=torch.float32),
            next_observations=torch.tensor(self.next_obs_buf[indices], dtype=torch.float32),
            actions=torch.tensor(self.actions_buf[indices], dtype=torch.int64),
            rewards=torch.tensor(self.rewards_buf[indices], dtype=torch.float32),
            dones=torch.tensor(self.dones_buf[indices], dtype=torch.float32),
        )

    def __len__(self):
        return self.size



if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.make(args.env_id)

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
    )

    start_time = time.time()

    episodic_return = 0
    episodic_length = 0

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.tensor(obs).unsqueeze(0).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()[0]

        next_obs, reward, done, info = envs.step(actions)
        episodic_return += reward
        episodic_length += 1

        rb.add(obs, next_obs, actions, reward, done)
        obs = next_obs

        if done == True:
            # 计算回合的奖励和长度
            obs = envs.reset()
            print(f"global_step={global_step}, episodic_return={episodic_return}")
            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
            writer.add_scalar("charts/episodic_length", episodic_length, global_step)
            episodic_return = 0
            episodic_length = 0

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max = target_network(data.next_observations).max(dim=1)[0]
                td_target = data.rewards + args.gamma * target_max * (1 - data.dones)
            current_q = q_network(data.observations).gather(1, data.actions.unsqueeze(-1)).squeeze(-1)
            loss = F.mse_loss(td_target, current_q)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", current_q.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update target network
        if global_step % args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                )

    envs.close()
    writer.close()