from itertools import count

import gym
import logging

import torch.nn as nn
import torch.nn.functional as F
import random
import torch as T
import numpy as np

from QLearning import QLearning
from RandomStrategy import RandomStrategy
from bomber_env.Memory import ReplayMemory, MergedMemory, Memory
from bomber_env.Transition import Transition
from bomber_env.TransitionHistory import TransitionHistory
from play import play, Strategy
from joblib import Parallel, delayed

T.set_default_tensor_type(T.FloatTensor)


logging.root.setLevel(logging.DEBUG)


def net2strat(net: nn.Module) -> Strategy:
    return lambda obs: net(obs).argmax()


BATCH_SIZE = 1024
HIDDEN_N = 10

net = nn.Sequential(nn.Linear(3, HIDDEN_N),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_N, HIDDEN_N),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_N, HIDDEN_N),
                    nn.ReLU(),
                    nn.Dropout(p=0.05),
                    nn.Linear(HIDDEN_N, HIDDEN_N),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_N, HIDDEN_N),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_N, 2)
                    )

env = gym.make('bomber-v0')


def sample_random_games(iters) -> Memory:
    memoryPositive = ReplayMemory(10000, lambda tr: tr.reward > 0)
    memoryNegative = ReplayMemory(10000, lambda tr: tr.reward < 0)
    memoryZero = ReplayMemory(10000, lambda tr: tr.reward == 0)
    memory = MergedMemory([memoryNegative, memoryPositive, memoryZero])

    for i in range(iters):
        strat = RandomStrategy()
        play(env, strat, memory)

    return memory


def average_score(strat: Strategy) -> float:
    def play_once(i):
        return play(env, strat, log=False)

    return np.array(Parallel(n_jobs=20)(delayed(play_once)(i) for i in range(1000))).mean()


random_games_n = 10000
train = sample_random_games(random_games_n)
test = sample_random_games(random_games_n)


qlearning = QLearning(net, 1e-2, 0.99999)

train_iters_n = 4000+1
for j in range(train_iters_n):
    qlearning.optimize(train.sample(BATCH_SIZE))
    if j % 1000 == 0:
        logging.info(f"ITER {j}")

        net.eval()
        logging.info(f"Train vs. test loss: {qlearning.get_loss(train.get())} vs. {qlearning.get_loss(test.get())}")
        logging.info(average_score(net2strat(net)))
        net.train()



