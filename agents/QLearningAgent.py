import os
import sys
sys.path.append('../game')
sys.path.append('../utils')

from collections import defaultdict
import json
import random

import gym
from TemplateAgent import FlappyBirdAgent
from FlappyBirdGame import FlappyBirdNormal

import warnings
warnings.filterwarnings('ignore')


class QLearningAgent(FlappyBirdAgent):
    def __init__(self, actions=[0, 1], rounding=None):
        super().__init__(actions)
        self.qValues = defaultdict(float)
        self.env = FlappyBirdNormal(gym.make('FlappyBird-v0'), rounding=rounding)

    def act(self, state, test=False):
        '''
        Returns the next action for the current state.
        Args:
            state (str): The current state.
        Returns:
            int: 0 or 1.
        '''
        def randomAct():
            return random.sample([0, 1], k=1)[0]

        if not test and random.random() < self.epsilon:
            return randomAct()

        qValues = [self.qValues.get((state, action), 0) for action in self.actions]

        if qValues[0] < qValues[1]:
            return 1
        elif qValues[0] > qValues[1]:
            return 0
        else:
            return randomAct()

    def saveQValues(self):
        toSave = {key[0] + ' action ' + str(key[1]): self.qValues[key] for key in self.qValues}
        with open('qValues.json', 'w') as fp:
            json.dump(toSave, fp)

    def loadQValues(self):
        def parseKey(key):
            state = key[:-9]
            action = int(key[-1])
            return (state, action)

        with open('qValues.json') as fp:
            toLoad = json.load(fp)
            self.qValues = {parseKey(key): toLoad[key] for key in toLoad}

    def train(self, epsilon, eta, numItersEval):
        discount = 0.95
        self.epsilon = epsilon
        self.discount = discount
        self.eta = eta
        self.numItersEval = numItersEval
        self.env.seed(random.randint(0, 100))

        done = False
        maxScore = 0
        maxReward = 0

        # for i in range(numIters):
        i = 0
        j = 0
        while True:
            score = 0
            totalReward = 0
            ob = self.env.reset()
            state = self.env.getGameState()
            k = 0
            while True:
                i += 1
                k += 1
                if i < 5000:
                    if k % 18 == 0:
                        action = 0
                    else:
                        action = 1
                else:
                    action = self.act(state)
                nextState, reward, done, _ = self.env.step(action)
                self.updateQ(state, action, reward, nextState)
                state = nextState
#                self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                if reward >= 1:
                    score += 1
                if i % 1000 == 0:
                    print(f"Iter: {i}, numgame: {j}")
                if i == 1e7:
                    break
                if (i + 1) % 1e4 == 0:
                    output = self.test(self.numItersEval)
                    self.saveOutput(output, i + 1)
                    self.saveQValues()
                if done:
                    break

            if i == 1e7:
                break

            if score > maxScore:
                maxScore = score
            if totalReward > maxReward:
                maxReward = totalReward
            j += 1

        self.env.close()
        print("Max Score Train: ", maxScore)
        print("Max Reward Train: ", maxReward)
        print()

    def test(self, numIters):
        self.epsilon = 0
        self.env.seed(0)

        done = False
        maxScore = 0
        maxReward = 0
        output = defaultdict(int)

        for i in range(numIters):
            score = 0
            totalReward = 0
            ob = self.env.reset()
            state = self.env.getGameState()

            while True:
                action = self.act(state, test=True)
                state, reward, done, _ = self.env.step(action)
#                self.env.render()  # Uncomment it to display graphics.
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
        print("Max Score Test: ", maxScore)
        print("Max Reward Test: ", maxReward)
        print()
        return output

    def updateQ(self, state, action, reward, nextState):
        nextQValues = [self.qValues.get((nextState, nextAction), 0) for nextAction in self.actions]
        nextValue = max(nextQValues)
        self.qValues[(state, action)] = (1 - self.eta) * self.qValues.get((state, action), 0) \
                                        + self.eta * (reward + self.discount * nextValue)

    def saveOutput(self, output, iter):
        '''
        Saves the scores.
        Args:
            output (dict): A set of scores.
            iter (int): Current iteration.
        '''
        if not os.path.isdir('scores'):
            os.mkdir('scores')
        with open('./scores/scores_{}.json'.format(iter), 'w') as fp:
            json.dump(output, fp)
