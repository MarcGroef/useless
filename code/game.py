import numpy as np

class Game:
    
    def __init__(self, statesize):
        self.stateSize = statesize
        self.state = np.array([statesize],dtype='float')
        
        
        


if __name__ == '__main__':
    game = Game(5)