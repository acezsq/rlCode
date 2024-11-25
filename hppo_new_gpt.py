import os
import random
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    num_steps: int = 512
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class HybridAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape[0]).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Discrete Actor
        self.actor_discrete = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape[0]).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space.n), std=0.01), #
        )
        # Continuous Actor
        self.actor_continuous = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape[0]).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space.shape[0]), std=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(envs.action_space.shape[0]))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action_discrete=None, action_continuous=None):
        logits_dis  = self.actor_discrete(x)
        dist_dis  = Categorical(logits=logits_dis)

        mean_con  = self.actor_continuous(x)
        std_con = torch.exp(self.log_std)
        dist_con  = torch.distributions.Normal(mean_con, std_con)

        if action_discrete is None:
            action_discrete = dist_dis.sample()
        if action_continuous is None:
            action_continuous = dist_con.sample()

        logp_discrete = dist_dis.log_prob(action_discrete)
        entropy_discrete = dist_dis.entropy()

        action_continuous_selected = action_continuous[action_discrete]
        logp_continuous = dist_con.log_prob(action_continuous).sum(dim=-1)
        logp_continuous_selected = logp_continuous[action_discrete]
        entropy_continuous = dist_con.entropy().sum(dim=-1)

        value = self.critic(x)
        return (
            action_discrete,
            action_continuous_selected,
            logp_discrete,
            logp_continuous_selected,
            entropy_discrete,
            entropy_continuous,
            value,
        )


if __name__ == "__main__":
    args = Args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.make(args.env_id)
    agent = HybridAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, envs.observation_space.shape[0])).to(device)
    actions_discrete = torch.zeros((args.num_steps,)).to(device)
    actions_continuous = torch.zeros((args.num_steps,) + envs.action_space.shape).to(device)
    logprobs_discrete = torch.zeros((args.num_steps,)).to(device)
    logprobs_continuous = torch.zeros((args.num_steps,)).to(device)
    rewards = torch.zeros((args.num_steps,)).to(device)
    dones = torch.zeros((args.num_steps,)).to(device)
    values = torch.zeros((args.num_steps,)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action_discrete, action_continuous, logprob_discrete, logprob_continuous, _, _, value = agent.get_action_and_value(next_obs)
                actions_discrete[step] = action_discrete
                actions_continuous[step] = action_continuous
                logprobs_discrete[step] = logprob_discrete
                logprobs_continuous[step] = logprob_continuous
                values[step] = value

            next_obs, reward, done, info = envs.step(action_discrete.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(np.array(done)).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs_discrete = logprobs_discrete.reshape(-1)
        b_logprobs_continuous = logprobs_continuous.reshape(-1)
        b_actions_discrete = actions_discrete.reshape(-1)
        b_actions_continuous = actions_continuous.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, _, newlogprob_discrete, newlogprob_continuous, entropy_discrete, entropy_continuous, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions_discrete.long()[mb_inds], b_actions_continuous[mb_inds]
                )

                logratio_discrete = newlogprob_discrete - b_logprobs_discrete[mb_inds]
                logratio_continuous = newlogprob_continuous - b_logprobs_continuous[mb_inds]

                ratio_discrete = logratio_discrete.exp()
                ratio_continuous = logratio_continuous.exp()

                clip_adv_discrete = torch.clamp(ratio_discrete, 1 - args.clip_coef, 1 + args.clip_coef) * b_advantages[mb_inds]
                clip_adv_continuous = torch.clamp(ratio_continuous, 1 - args.clip_coef, 1 + args.clip_coef) * b_advantages[mb_inds]

                pg_loss_discrete = torch.max(-b_advantages[mb_inds] * ratio_discrete, -clip_adv_discrete).mean()
                pg_loss_continuous = torch.max(-b_advantages[mb_inds] * ratio_continuous, -clip_adv_continuous).mean()

                pg_loss = pg_loss_discrete + pg_loss_continuous

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * v_loss_unclipped.mean()

                entropy_loss = entropy_discrete.mean() + entropy_continuous.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
