import sys

sys.path.append('game')
sys.path.append('utils')
sys.path.append('agents')

import argparse
from QLearningAgent import QLearningAgent
from FuncApproxCNNAgent import FuncApproxCNNAgent
from FuncApproxDNNAgent import FuncApproxDNNAgent
from FuncApproxRNNAgent import FuncApproxRNNAgent
import warnings

warnings.filterwarnings('ignore')

agent_options = ['QLearning', 'FuncApproxDNN', 'FuncApproxRNN', 'FuncApproxCNN']


def parseArgs():
    parser = argparse.ArgumentParser(description='An AI Agent for Flappy Bird.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--algo', type=str, default='QLearning',
                        help='Learning algorithm.', choices=agent_options)
    parser.add_argument('--rounding', type=int, default=None,
                        help='Level of discretization (used only for Q-learning)')
    parser.add_argument('--numTestIters', type=int, default=100,
                        help='Number of testing iterations.')
    parser.add_argument('--epsilon', type=float, default=0.,
                        help='Epsilon-greedy policy.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--past_frame', type=int, default=4,
                        help='number of input Frames for CNN')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for PyTorch.')
    args = parser.parse_known_args()[0]
    return args


def main():
    args = parseArgs()

    if args.algo == 'QLearning':
        agent = QLearningAgent(actions=[0, 1], rounding=args.rounding)
        agent.train(epsilon=args.epsilon, eta=args.lr,
                    numItersEval=args.numTestIters)
        agent.saveQValues()

    elif args.algo == 'FuncApproxDNN':
        agent = FuncApproxDNNAgent()
        agent.train(epsilon=args.epsilon, lr=args.lr,
                    numItersEval=args.numTestIters, seed=args.seed)
        agent.saveModel()

    elif args.algo == 'FuncApproxCNN':
        agent = FuncApproxCNNAgent(past_frame=args.past_frame)
        agent.train(epsilon=args.epsilon, lr=args.lr,
                    numItersEval=args.numTestIters, seed=args.seed)
        agent.saveModel()

    elif args.algo == 'FuncApproxRNN':
        agent = FuncApproxRNNAgent()
        agent.train(epsilon=args.epsilon, lr=args.lr,
                    numItersEval=args.numTestIters, seed=args.seed)
        agent.saveModel()


if __name__ == '__main__':
    main()
