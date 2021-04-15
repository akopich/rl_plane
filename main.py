from itertools import count

import gym
import logging

import torch.nn as nn
import torch.nn.functional as F
import random
import torch as T
import numpy as np
from joblib.externals.loky import set_loky_pickler
from torch import memory_format

from DiveStrategy import DiveStrategy, RandomDiveDiveStrategy
from QLearning import QLearning
from RandomStrategy import RandomStrategy
from bomber_env.Memory import ReplayMemory, MergedMemory, Memory, STATE_WIDTH
from bomber_env.Transition import Transition
from bomber_env.TransitionHistory import TransitionHistory
from bomber_env.envs.bomber_env import Action
from play import play, Strategy
from joblib import Parallel, delayed, wrap_non_picklable_objects
from numpy.random import uniform

T.set_default_tensor_type(T.FloatTensor)


logging.root.setLevel(logging.INFO)


def net2strat(net: nn.Module) -> Strategy:
    def strat(obs):
        return Action(net(obs).argmax().item())
    return strat


BATCH_SIZE = 1024
HIDDEN_N = 20

net = nn.Sequential(nn.Linear(STATE_WIDTH, HIDDEN_N),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_N, HIDDEN_N),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_N, HIDDEN_N),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_N, HIDDEN_N),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_N, HIDDEN_N),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_N, len(Action))
                    )

env = gym.make('bomber-v0')


def init_memory_buffer(n):
    memoryPositive = ReplayMemory(n, lambda tr: tr.reward > 0)
    memoryNegative = ReplayMemory(n, lambda tr: tr.reward < 0)
    memoryZero = ReplayMemory(n, lambda tr: tr.reward == 0)
    return MergedMemory([memoryNegative, memoryPositive, memoryZero])


def sample_games(iters, memory, strat) -> Memory:
    for i in range(iters):
        play(env, strat(), memory, log=False)

    return memory


def average_score(strat: Strategy, memory=None, log=False) -> float:
    def play_once(i):
        return play(env, strat, log=log, memory=memory)

    return np.array(Parallel(n_jobs=20)(delayed(play_once)(i) for i in range(1000))).mean()


# random_games_n = 10000
# train = init_memory_buffer(10000)
# test = init_memory_buffer(10000)
# random_strat = lambda: RandomDiveDiveStrategy(uniform(low=0., high=15000.), -np.pi/2)
# sample_games(random_games_n, train, random_strat)
# sample_games(random_games_n, test, random_strat)
#
# random_strat = lambda: DiveStrategy(uniform(low=0., high=15000.), -np.pi/2)
# sample_games(random_games_n, train, random_strat)
# sample_games(random_games_n, test, random_strat)
#

import dill
# with open('memory.pkl', 'wb') as output:
#     dill.dump((train, test), output, dill.HIGHEST_PROTOCOL)
with open('memory.pkl', 'rb') as input:
    train, test = dill.load(input)


qlearning = QLearning(net, lr=1e-3, gamma=0.99999)

train_iters_n = 200000+1
for j in range(train_iters_n):
    qlearning.optimize(train.sample(BATCH_SIZE))
    if j % 2000 == 0:
        logging.info(f"ITER {j}")

        net.eval()
        logging.info(f"Train vs. test loss: {qlearning.get_loss(train.get())} vs. {qlearning.get_loss(test.get())}")
        score = average_score(net2strat(net))
        logging.info(score)

        if score > 0.95:
            break

        sample_games(2000, train, lambda: net2strat(net))
        sample_games(2000, test, lambda: net2strat(net))
        net.train()




average_score(net2strat(net), log=True)
