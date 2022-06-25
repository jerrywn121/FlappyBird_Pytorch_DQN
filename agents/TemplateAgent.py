class FlappyBirdAgent(object):
    def __init__(self, actions):
        self.actions = actions

    def act(self, state):
        raise NotImplementedError

    def train(self, numIters):
        raise NotImplementedError

    def test(self, numIters):
        raise NotImplementedError

    def saveOutput(self):
        ''' Saves the scores. '''
        raise NotImplementedError
