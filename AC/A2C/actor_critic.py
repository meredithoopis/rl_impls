import tensorflow as tf 
import keras 
import numpy as np 
from keras.optimizers import Adam
#from tensorflow.python.keras.optimizers import Adam 
import tensorflow_probability as tfp 
from networks import ActorCritic

class Agent: 
    def __init__(self, alpha = 0.0003, gamma = 0.99, n_actions=2):
        self.gamma = gamma  
        self.n_actions = n_actions 
        self.action = None 
        self.action_space = [i for i in range(self.n_actions)]
        self.actor_critic = ActorCritic(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate = alpha))

    def choose_action(self, observation): 
        #if isinstance(observation, list) and any(isinstance(i, list) for i in observation):
        #    max_len = max(len(i) for i in observation)
         #   observation = [i + [0]*(max_len-len(i)) if len(i) < max_len else i for i in observation]
        #observation = np.array(observation)

        #if observation.ndim == 1:
          #  observation = np.reshape(observation, (1, -1))

        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state) #Given a state, return softmax probs of the next action 
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.action = action 
        return action.numpy()[0] 

    def save_model(self): 
        print("..Saving the model...")
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def learn(self,state, reward, state_, done): #state_ is next state 
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape: 
            state_value, probs = self.actor_critic(state)  #for critic, actor 
            state_value_, _ = self.actor_critic(state_) #next state 
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)
            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value #Temporal difference 
            critic_loss = delta ** 2 
            actor_loss = -log_prob * delta
            total_loss = critic_loss + actor_loss 
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))


