import sys
import math
import numpy

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
import pdb

class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return action_scores


class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space, lr):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space).float()
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

    def select_action(self, state):
        state = torch.from_numpy(numpy.array(state)).float()
        probs = self.model(Variable(state).float().cuda())  
        #action = probs.multinomial(1).data
        action = torch.argmax(probs).data
        prob = probs[action].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()

        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R).cuda())).sum() - (0.0001*entropies[i].cuda()).sum()
        loss = loss / len(rewards)
        
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()