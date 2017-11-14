from sarsa import Sarsa
import numpy as np

class Agent():
    
    def __init__(self, state):
        self.old_state = state
        self.current_state = state
        self.old_action = np.zeros(len(state))
        self.action_performed = np.zeros(len(state))
        
        self.model = Sarsa(len(state), 0.1, 0.002, 0.8)
    
    def act(self, state):
        self.action_performed, Q, is_explore = self.model.chooseAction(state)
        return self.action_performed, Q, is_explore
    
    def update(self, new_state, reward, isFinal = False):
        self.model.update(self.old_state, self.old_action, new_state, self.action_performed, reward, isFinal)
        self.old_state = np.copy(self.current_state)
        self.current_state = np.copy(new_state)
        self.old_action = np.copy(self.action_performed)
        
        