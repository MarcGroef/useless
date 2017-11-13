import numpy as np
from MLP2 import MLP
import sys
import matplotlib.pyplot as plt  ##sudo apt-get install python-matplotlib
#from sklearn.neural_network import MLPClassifier

##https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-175.pdf
class Sarsa():
  a_max = None
  a_min = None
  a_delta = None



  def __init__(self, state_size, random_chance = 1, learningRate = 0.001, discount = 0.999):
    self.state_size = state_size
    self.action_size = state_size
    self.mlp = MLP(self.action_size + self.state_size,1, [50, 10, 100,100,20], 1)
    self.learningRate = learningRate
    self.discount = discount
    self.random_chance = random_chance


  def resetBrainBuffers(self):
    self.mlp.resetBuffers()

  def getBrain(self):
    return self.mlp.getBrain()

  def setBrain(self, brain):
    self.mlp.setBrain(brain)   

 
  def getQ(self, state, action):
    
    mlpvec = np.concatenate([state, action])
    return self.mlp.process(mlpvec)

  def updateQ(self, action, state, targetOut):
    self.mlp.train(np.concatenate([state, action]), targetOut, self.learningRate, 0)

  def chooseAction(self, s):
    zero = np.zeros([self.state_size])
    best_action = zeros
    best_q = self.getQ(s, best_action)
    for act in range(self.state_size):
        action = zero
        action[act] = 1
        q = self.getQ(s, action)
        if (q > best_q)
            best_action = action
            best_q = q
    return [best_action, best_q]

  def update(self, old_state, old_action, new_state, action_performed, reward, isFinalState = False):
    learningRate = 1
    old_Q = self.getQ(old_state, old_action)
    new_Q = self.getQ(new_state, action_performed)
    if isFinalState:
       diff = learningRate * (reward - old_Q)
    else:
       diff = learningRate * (reward + self.discount * new_Q - old_Q)
    
    target = new_Q + diff
    self.updateQ(old_action, old_state, target)

    
   
