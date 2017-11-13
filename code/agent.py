from sarsa import Sarsa

class Agent():
    
    def __init__(self, state):
        self.old_state = state
        self.current_state = state
        self.old_action = np.zeros(len(state))
        self.action_performed = np.zeros(len(state))
        
        self.model = Sarsa(len(state), 0.2, 0.002, 0.999)
    
    def act(self, state):
        self.action_performed = self.model.chooseAction(state)[0]
        return self.action_performed
    
    def update(new_state, reward, isFinal = False):
        self.model.update(self.old_state, self.old_action, new_state, self.action_performed, reward, isFinal)
        self.old_state = self.current_state
        self.current_state = new_state
        self.old_action = self.action_performed
        
        