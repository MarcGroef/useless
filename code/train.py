import numpy as np

from game import Game
from agent import Agent


if __name__ == '__main__':
    game = Game(5)
    state = game.getState()
    agent = Agent(state)
    avg = 0
    for epoch in range (1000000):
        old_state = np.copy(state)
        action, Q, is_explore = agent.act(state)
        state, reward = game.update(action)
        agent.update(state, reward)
        avg += reward
        
        if epoch % 1000 == 0:
            mark = ' '
            if is_explore:
                mark = '*'
            print Q, ",\t", avg / 1000, mark
            avg = 0
        