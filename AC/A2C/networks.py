import os 
import keras 
#import tensorflow.python.keras as keras 
from keras.layers import Dense 
#from tensorflow.python.keras.layers import Dense 

class ActorCritic(keras.Model): 
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims = 512, name = 'actor_critic', ckpoint_dir = 'tmp/actor_critic'): 
        super().__init__()
        self.fc1_dims = fc1_dims 
        self.fc2_dims = fc2_dims 
        self.n_actions = n_actions 
        self.model_name = name 
        self.checkpointdir = ckpoint_dir
        self.checkpoint_file = os.path.join(self.checkpointdir, name +'_ac')
        self.fc1 = Dense(self.fc1_dims, activation = 'relu')
        self.fc2 = Dense(self.fc2_dims, activation = 'relu')
        self.v = Dense(1, activation=None) #Value 
        self.pi = Dense(n_actions, activation = 'softmax') #Policy function 
    
    def call(self, state): 
        value = self.fc1(state)
        value = self.fc2(value)
        v = self.v(value)  #For critic -> Value 
        pi  = self.pi(value) #For actor -> action 
        return v, pi 
