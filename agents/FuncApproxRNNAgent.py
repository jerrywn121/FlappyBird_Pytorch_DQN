from __future__ import print_function
import os
import sys
sys.path.append('../game')
sys.path.append('../utils')

from collections import defaultdict, deque
import json
import random
import gym
from TemplateAgent import FlappyBirdAgent
from FlappyBirdGame import FlappyBirdDNN
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


class ExperienceReplay(object):
    ''' Experience Replay technique, which samples a minibatch of past observations. '''

    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def memorize(self, observation):
        '''
        Adds a new observation to the memory.
        Args:
            onservation (tuple): A new observation.
        '''
        self.memory.append(observation)

    def getBatch(self, batch_size):

        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.RNN(7, 64, num_layers=4, batch_first=True, nonlinearity="relu")
        self.linear = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        assert x.size(1) == 30
        _, h = self.lstm(x)
        return self.linear(h[-1]).squeeze(0)


class HParams():
    ''' A class storing hyperparameters of the model. '''

    def __init__(self, lr, seed=0, batch_size=32):
        self.lr = lr
        self.seed = seed
        self.batch_size = batch_size


class FuncApproxRNNAgent(FlappyBirdAgent):
    def __init__(self, actions=[0, 1]):
        super().__init__(actions)
        self.env = FlappyBirdDNN(gym.make('FlappyBird-v0'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = Net().to(self.device)
        self.target_net = Net().to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.criterion = torch.nn.MSELoss()
        self.actionEncoding = torch.eye(2, device=self.device).unsqueeze(1)

    def qValues(self, state):
        input = torch.Tensor(state).to(self.device)
        return self.net(input)

    def act(self, state, test=False, rand=False):
        def randomAct():
            return random.sample([0, 1], k=1)[0]

        if not test and random.random() < self.epsilon:
            return randomAct()

        qValues = self.net(state)
        if qValues[0] < qValues[1]:
            return 1
        elif qValues[0] > qValues[1]:
            return 0
        else:
            return randomAct()

    def train(self, epsilon, lr, numItersEval, seed=0):
        discount = 0.95
        self.epsilon = epsilon
        self.initialEpsilon = epsilon
        self.discount = discount
        self.numItersEval = numItersEval

        self.hparams = HParams(lr=lr, seed=seed)
        if self.hparams.seed != 0:
            torch.manual_seed(self.hparams.seed)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        self.experienceReplay = ExperienceReplay(50000)

        self.env.seed(random.randint(0, 100))
        self.net.train()

        reward = 0
        done = False
        maxScore = 0
        maxReward = 0
        counter = 0
        i = 0
        j = 0
        d = 0
        while True:
            self.net.train()
            score = 0
            totalReward = 0
            ob = self.env.reset()
            # gameIter = []
            state = self.env.getGameState()
            state = torch.tensor(state)[None].float()
            s = torch.stack([state]*30, dim=1)
            k = 0
            while True:
                k += 1
                i += 1
                counter += 1
                if i < 5000:
                    if k % 18 == 0:
                        action = 0
                    else:
                        action = 1
                else:
                    action = self.act(s.to(self.device))
                nextState, reward, done, _ = self.env.step(action)
                nextState = torch.tensor(nextState)[None, None].float()
#                self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                if reward >= 1:
                    score += 1
                # ---
                reward = torch.Tensor([reward]).to(self.device)
                survived = 1 - torch.Tensor([done]).to(self.device)
                action = self.actionEncoding[action]
                new_s = torch.cat([s[:, 1:], nextState], dim=1)
                self.experienceReplay.memorize((s, action, reward, new_s, survived))

                s = new_s
                if counter >= 5000:
                    batch = self.experienceReplay.getBatch(self.hparams.batch_size)
                    if batch:
                        d += 1
                        self.updateWeights(batch)
                        if d % 200 == 0:
                            self.target_net.load_state_dict(self.net.state_dict())

                if i % 1000 == 0:
                    print(f"Iter: {i}, numgame: {j}")
                if i == 1e7:
                    break
                if (i + 1) % 1e4 == 0:
                    output = self.test(self.numItersEval)
                    self.saveOutput(output, i + 1)
                    self.saveModel()
                # ---

                if done:
                    break

            if score > maxScore:
                maxScore = score
            if totalReward > maxReward:
                maxReward = totalReward

            j += 1

        self.env.close()
        print("Max Score: ", maxScore)
        print("Max Reward: ", maxReward)
        print()

    def test(self, numIters):
        self.epsilon = 0
        self.env.seed(0)
        self.net.eval()

        reward = 0
        done = False
        maxScore = 0
        maxReward = 0
        output = defaultdict(int)
        self.net.eval()
        with torch.no_grad():
            for i in range(numIters):
                score = 0
                totalReward = 0
                ob = self.env.reset()
                state = self.env.getGameState()
                state = torch.tensor(state)[None].float().to(self.device)
                s = torch.stack([state]*30, dim=1)

                while True:
                    action = self.act(s, test=True)
                    state, reward, done, _ = self.env.step(action)
    #                    self.env.render()  # Uncomment it to display graphics.
                    state = torch.tensor(state)[None, None].float().to(self.device)
                    s = torch.cat([s[:, 1:], state], dim=1)
                    totalReward += reward
                    if reward >= 1:
                        score += 1
                    if done:
                        break

                output[score] += 1
                if score > maxScore:
                    maxScore = score
                if totalReward > maxReward:
                    maxReward = totalReward

        self.env.close()
        print("Max Score: ", maxScore)
        print("Max Reward: ", maxReward)
        print()
        return output

    def updateWeights(self, batch):
        self.net.train()
        stateBatch, actionBatch, rewardBatch, nextStateBatch, survivedBatch = batch
        stateBatch = torch.cat(stateBatch).to(self.device)
        actionBatch = torch.cat(actionBatch)
        rewardBatch = torch.cat(rewardBatch)
        nextStateBatch = torch.cat(nextStateBatch).to(self.device)
        survivedBatch = torch.cat(survivedBatch)

        currQValuesBatch = self.net(stateBatch)
        currQValuesBatch = torch.sum(currQValuesBatch * actionBatch, dim=1)
        with torch.no_grad():
            nextQValuesBatch = self.target_net(nextStateBatch)
        targetQValuesBatch = rewardBatch + self.discount * survivedBatch * \
                             torch.max(nextQValuesBatch, dim=1).values

        loss = self.criterion(currQValuesBatch, targetQValuesBatch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def saveOutput(self, output, iter):
        if not os.path.isdir('scores'):
            os.mkdir('scores')
        with open('./scores/scores_{}.json'.format(iter), 'w') as fp:
            json.dump(output, fp)

    def saveModel(self):
        torch.save(self.net.state_dict(), "model.params")

    def loadModel(self):
        self.net = Net()
        self.net.load_state_dict(torch.load("model.params"))
        self.net = self.net.to(self.device)
