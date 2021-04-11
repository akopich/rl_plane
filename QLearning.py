import torch.nn as nn
import torch as T

from bomber_env.TransitionHistory import TransitionHistory
import logging
import torch.nn.functional as F


class QLearning:
    def __init__(self, net: nn.Module, lr: float, gamma: float = 0.99999999):
        self.net = net
        self.optimizer = T.optim.Adam(net.parameters(), lr=lr)
        self.gamma = gamma

    def get_loss(self, history: TransitionHistory):
        state, reward, action, has_next, next_state = history

        state_action_Q = self.net(state).gather(1, action)
        next_state_Q = T.zeros(len(history))
        next_state_Q[has_next] = self.net(next_state[has_next]).max(1)[0].detach()
        expected_state_action_Q = (next_state_Q * self.gamma) + reward

        return F.mse_loss(state_action_Q, expected_state_action_Q.unsqueeze(1))

    def optimize(self, history: TransitionHistory):
        loss = self.get_loss(history)
        logging.debug(f"Q-learning train loss: {loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

