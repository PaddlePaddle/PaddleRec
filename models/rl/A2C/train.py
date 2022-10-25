# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import gym
import numpy as np

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.distribution import Categorical
import paddle.distributed as dist

BASE_RPC_ENV_NAME = "RPC_ENV"
RPC_TRAINER_NAME = "RPC_TRAINER"

paddle.device.set_device("cpu")
device = paddle.get_device()
env = gym.make("CartPole-v0")


def rpc_env_name(rank=None):
    if rank is None:
        return "{}-{}".format(BASE_RPC_ENV_NAME, dist.get_rank())
    else:
        return "{}-{}".format(BASE_RPC_ENV_NAME, rank)


def rpc_step(action):
    global env
    ob, reward, done, info = env.step(action)
    if done:
        ob = env.reset()
    return ob, reward, done, info


def rpc_reset():
    global env
    return env.reset()


def rpc_reset_task():
    global env
    env.reset_task()


class RpcEnv:
    def __init__(self):
        self.remote_names = []
        for i in range(dist.get_world_size()):
            if i != dist.get_rank():
                self.remote_names.append(rpc_env_name(i))

    def step(self, actions):
        futs = []
        for name, action in zip(self.remote_names, actions):
            futs.append(dist.rpc.rpc_async(name, rpc_step, args=(action, )))
        results = [fut.wait() for fut in futs]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        futs = []
        for name in self.remote_names:
            futs.append(dist.rpc.rpc_async(name, rpc_reset))
        results = [fut.wait() for fut in futs]
        return np.stack(results)


# code from: https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/reinforcement_learning/Advantage_Actor_Critic/Advantage_Actor_Critic.html
class ActorCritic(nn.Layer):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()
        nn.initializer.set_global_initializer(
            nn.initializer.XavierNormal(), nn.initializer.Constant(value=0.0))

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, 1))

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(axis=1), )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


@paddle.no_grad()
def test_env(model, vis=False):
    model.eval()
    state = env.reset()
    if vis:
        env.render()
    done = False
    total_reward = 0
    while not done:
        state = paddle.to_tensor(
            state, dtype="float32", place=device).unsqueeze(0)
        dist, _ = model(state)
        actions = dist.sample([1])
        next_state, reward, done, _ = env.step(actions.cpu().numpy()[0][0])
        state = next_state
        if vis:
            env.render()
        total_reward += reward
    model.train()
    return total_reward


def train(args):
    rpc_env = RpcEnv()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = ActorCritic(state_size, action_size, args.hidden_size)
    lr = args.lr
    optimizer = optim.Adam(args.lr, parameters=model.parameters())
    max_frames = 20000
    frame_idx = 0
    state = rpc_env.reset()
    while frame_idx < max_frames:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(args.num_steps):
            # env.render()
            state = paddle.to_tensor(state, dtype="float32", place=device)
            dist, value = model(state)

            actions = dist.sample([1]).squeeze(0)
            next_state, reward, done, _ = rpc_env.step(actions.cpu().numpy())
            actions = paddle.reshape(actions, [-1, 1])
            log_prob = dist.log_prob(actions)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(
                paddle.to_tensor(
                    reward, dtype="float32", place=device).unsqueeze(1))
            masks.append(
                paddle.to_tensor(
                    1 - done, dtype="float32", place=device).unsqueeze(1))

            state = next_state
            frame_idx += 1
            if frame_idx % 100 == 0:
                test_reward = np.mean([test_env(model) for _ in range(2)])
                print("frame idx: {}, test reward: {}".format(frame_idx,
                                                              test_reward))
        next_state = paddle.to_tensor(
            next_state, dtype="float32", place=device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = paddle.concat(log_probs)
        returns = paddle.concat(returns).detach()
        values = paddle.concat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        if frame_idx % 2000 == 0:
            lr = 0.92 * lr
            optimizer.set_lr(lr)
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        # if frame_idx % 1000 == 0:
        #     print("frame idx: {}, loss: {}".format(frame_idx, loss.numpy()))

    env.close()


def main(args):
    if dist.get_rank() == 0:
        dist.rpc.init_rpc(RPC_TRAINER_NAME)
        train(args)
    else:
        global env
        env = gym.make("CartPole-v0")
        env.reset()
        dist.rpc.init_rpc(rpc_env_name())
    dist.rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A2C")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    print(args)
    main(args)
