#!/usr/bin/env python3

# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque

GRID_SIZE = 10.0


class MoveEnv:
    def __init__(self):
        self.x = 0.0
        self.n_observations = 1
        self.n_actions = 2

    def reset(self):
        self.x = 0.0
        return [self.x]

    def step(self, action):
        if action == 0:
            self.x -= 1.0
        elif action == 1:
            self.x += 1.0

        done = abs(self.x) > GRID_SIZE * 0.5

        reward = 0.0
        if self.x > GRID_SIZE * 0.5:
            reward = 1.0
        elif self.x < -GRID_SIZE * 0.5:
            reward = -1.0

        return ([self.x], reward, done, None)


# define policy network
class policy_net(nn.Module):
    # nS: state space size, nH: n. of neurons in hidden layer, nA: size action
    # space
    def __init__(self, nS, nH, nA):
        super(policy_net, self).__init__()
        self.h = nn.Linear(nS, nH)
        self.out = nn.Linear(nH, nA)

    # define forward pass with one hidden layer with ReLU activation and sofmax
    # after output layer
    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.softmax(self.out(x), dim=1)
        return x


# create environment
env = MoveEnv()

f = open("data.csv", "w")
f.write("episode,reward\n")
f.close()

# instantiate the policy
policy = policy_net(env.n_observations, 20, env.n_actions)
# create an optimizer
optimizer = torch.optim.Adam(policy.parameters())
# initialize gamma and stats
gamma = 0.99
n_episode = 1
returns = deque(maxlen=100)
render_rate = 1  # render every render_rate episodes
while True:
    rewards = []
    actions = []
    states = []
    # reset environment
    state = env.reset()
    n_steps = 0
    while True:
        # calculate probabilities of taking each action
        probs = policy(torch.tensor(state).unsqueeze(0).float())
        # sample an action from that set of probs
        sampler = Categorical(probs)
        action = sampler.sample()

        # use that action in the environment
        new_state, reward, done, info = env.step(action.item())
        # store state, action and reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = new_state

        n_steps += 1

        if done:
            break

    # preprocess rewards
    rewards = np.array(rewards)
    # calculate rewards to go for less variance
    # R = torch.tensor([np.sum(rewards[i:]*(gamma**np.array(range(i, len(rewards))))) for i in range(len(rewards))])
    # or uncomment following line for normal rewards
    R = torch.sum(torch.tensor(rewards))

    # preprocess states and actions
    states = torch.tensor(states).float()
    actions = torch.tensor(actions)

    # calculate gradient
    probs = policy(states)
    sampler = Categorical(probs)
    log_probs = -sampler.log_prob(actions)   # "-" because it was built to work with gradient descent, but we are using gradient ascent
    pseudo_loss = torch.sum(log_probs * R)  # loss that when differentiated with autograd gives the gradient of J(θ)
    # update policy weights
    optimizer.zero_grad()
    pseudo_loss.backward()
    optimizer.step()

    # calculate average return and print it out
    returns.append(np.sum(rewards))
    print("Episode: {:6d}\tReturn: {:6.2f} n_steps: {}".format(n_episode, R, n_steps))
    n_episode += 1
    f = open("data.csv", "a")
    f.write("{},{}\n".format(n_episode, R))
    f.close()

# close environment
env.close()
