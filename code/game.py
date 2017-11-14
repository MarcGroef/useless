import numpy as np
import random
class Game:
    
    def __init__(self, statesize):
        self.stateSize = statesize
        self.state = np.ones(statesize)
        self.negRewardForActingModifier = 0
        self.randomFlipChance = 0.5
        
    def getState(self):
        return self.state
    
    def getReward(self, action):
        reward = (np.sum(self.state) + np.sum(action) * self.negRewardForActingModifier) * -1
        if np.sum(self.state) == 0:
            reward += 100
        if np.min(self.state - action) >= 0 and np.sum(action) != 0: #hit
            reward += 2
        if np.min(self.state - action) < 0 and np.sum(action) != 0: #miss
            reward -= 2
        return reward

    
    def update(self, action):
        self.state[action == 1] = 0
 #       self.state[action == 1] *= -1

        #self.state -= action
        ##self.state = np.absolute(self.state)
        #self.state[self.state < 0] *= -1
        reward = self.getReward(self.state)
        if random.random() < self.randomFlipChance:
            self.state[random.randint(0, self.stateSize - 1)] = 1
        return self.state, reward
        
        


if __name__ == '__main__':
    game = Game(5)