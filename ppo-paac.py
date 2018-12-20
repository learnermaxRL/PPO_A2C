import fire
from tqdm import tqdm
import pybullet_envs
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as U
from paac import PAACRunner
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, activation=nn.Tanh, clip_range=0.2):
        super().__init__()
        self.clip_range = clip_range
        layers_list = []
        layers_list.append(nn.Linear(num_inputs, 64))
        layers_list.append(activation())
        layers_list.append(nn.Linear(64, 64))
        layers_list.append(activation())
        layers_list.append(nn.Linear(64, num_outputs))

        self.layers = nn.Sequential(*layers_list)
        # self.layers.weight.data.normal_(std=0.01)
        self.log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.opt = torch.optim.Adam(self.parameters())

    def forward(self, x):
        mean = self.layers(x)
        log_std = self.log_std.expand_as(mean)
        return torch.stack((mean, log_std), dim=0)

    def create_dist(self, states):
        mean, log_std = self.forward(states)
        return Normal(loc=mean, scale=log_std.exp())

    def act(self, x):
        dist = self.create_dist(x)
        action = dist.sample()
        return U.to_np(action)

    def train(self, advantages, states, actions, num_epochs=10):
        with torch.no_grad():
            dist = self.create_dist(states)
            old_log_prob = dist.log_prob(actions).sum(-1)

        for _ in range(num_epochs):
            dist = self.create_dist(states)
            new_log_prob = dist.log_prob(actions).sum(-1)

            prob_ratio = (new_log_prob - old_log_prob).exp()
            clip_ratio = prob_ratio.clamp(
                min=1 - self.clip_range, max=1 + self.clip_range
            )
            surrogate = prob_ratio * advantages
            clip_surrogate = clip_ratio * advantages
            losses = torch.min(surrogate, clip_surrogate)
            assert len(losses.shape) == 1
            loss = -losses.mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


class Baseline(nn.Module):
    def __init__(self, num_inputs, activation=nn.Tanh):
        super().__init__()
        layers_list = []
        layers_list.append(nn.Linear(num_inputs, 64))
        layers_list.append(activation())
        layers_list.append(nn.Linear(64, 64))
        layers_list.append(activation())
        layers_list.append(nn.Linear(64, 1))

        self.layers = nn.Sequential(*layers_list)

        self.opt = torch.optim.Adam(self.parameters())

    def forward(self, x):
        return self.layers(x)

    def train(self, states, targets, bs=64, num_epochs=10, shuffle=True):
        ds = TensorDataset(states, targets)
        dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)

        for _ in range(num_epochs):
            for x, y in dl:
                preds = self.forward(x).squeeze()
                loss = F.mse_loss(input=preds, target=y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


def main(
    env="HumanoidBulletEnv-v0", num_envs=16, max_num_steps=1e6, log_dir="./logs", horizon=2048, num_workers=None, log_freq=10
):
    """
    Runs the training agent for a specified number of steps.

    Parameters
    ----------
    env: string
        The name of the env.
    num_envs: int
        Number of envs to use.
    max_num_steps: int
        How many steps to train on.
    log_dir: string
        Directory to save the tensorboard summary.
    horizon: int
        Number of steps per batch. (Default is 2048)
    num_workers: int
        Number of parallel processes. (Default is None, which will use all cores available)
    log_freq: int
        Number of steps between every log. (Default is 10)
    """
    print ("i"*80)
    env = [gym.make(env) for _ in range(num_envs)]
    print ("created all envs")
    env = PAACRunner(env, num_workers=num_workers)

    state_normalizer = U.MeanStdFilter(num_features=env.observation_space.shape[0])
    policy = Policy(
        num_inputs=env.observation_space.shape[0], num_outputs=env.action_space.shape[0]
    )
    baseline = Baseline(num_inputs=env.observation_space.shape[0])

    writer = SummaryWriter(log_dir=log_dir)

    for i_step in tqdm(range(0, int(max_num_steps), num_envs)):
        states, actions, rewards, returns, dones = [], [], [], [], []
        state = env.reset()

        # This collects a trajectory
        for _ in range(horizon):
            state = state_normalizer.normalize(state)
            state = U.to_tensor(state)
            action = policy.act(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state

        # Calculates the state_value for every state for the trajectory
        states = torch.stack(states)
        st = states.reshape((-1, *states.shape[2:]))
        states_value_t = baseline(st).detach().reshape(states.shape[:2])

        # Calculate the return
        ret = U.discounted_sum_rewards(
            rewards=np.array(rewards),
            dones=np.array(dones),
            last_state_value_t=states_value_t[-1],
        )
        ret = U.to_tensor(ret)

        # Calculate the advatange
        assert len(ret.shape) == 2
        assert len(states_value_t.shape) == 2
        adv = ret - states_value_t

        actions = U.to_tensor(U.to_np(actions))
        # Concatenate num_samples with num_envs
        states = states.reshape((-1, *states.shape[2:]))
        actions = actions.reshape((-1, *actions.shape[2:]))
        ret = ret.reshape((-1, *ret.shape[2:]))
        adv = adv.reshape((-1, *adv.shape[2:]))

        # Train the policy net
        policy.train(advantages=adv, states=states, actions=actions, num_epochs=5)
        # Train the critic net
        baseline.train(states=states, targets=ret, num_epochs=5)

        # Update the normalizer
        state_normalizer.update()

        # Display logs
        if i_step % log_freq == 0:
            rew_mean = np.mean(env.rewards[-10:])
            tqdm.write(str(rew_mean))
            writer.add_scalar("reward", scalar_value=rew_mean, global_step=i_step)


if __name__ == "__main__":
    fire.Fire(main)
