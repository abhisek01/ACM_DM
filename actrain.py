from actor_model import Actor
from critic_model import Critic
import tensorflow as tf
from collections import deque
import numpy as np
import random
import tensorflow.keras.backend as K
from dialogue_config import rule_requests, agent_actions


class Actor_Critic():
    def __init__(self, state_dim, constant):
        # Hyperparameters
        self.REPLAY_MEMORY_SIZE = 50000
        self.MINIMUM_REPLAY_MEMORY = 1000
        self.MINIBATCH_SIZE = 32
        self.EPSILON = 1
        self.EPSILON_DECAY = 0.999
        self.MIN_EPSILON = 0.1
        self.DISCOUNT = 0.7
        self.possible_actions = agent_actions
        #EPISODES = 1_00_000
        #ENV_NAME = 'CartPole-v1'
        #VISUALIZATION = False

        # creating own session to use across all the Keras/Tensorflow models we are using
        sess = tf.Session()
        K.set_session(sess)
        # Environment details
        #env = gym.make(ENV_NAME).unwrapped

        self.action_dim = len(agent_actions)
        self.observation_dim = state_dim

        # Actor model to take actions 
        # state -> action
        self.actor = Actor(sess, self.action_dim, self.observation_dim)
        # Critic model to evaluate the action taken by the actor
        # state + action -> Expected reward to be achieved by taking action in the state.
        self.critic = Critic(sess, self.action_dim, self.observation_dim)

        # Replay memory to store experiences of the model with the environment
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        sess.run(tf.initialize_all_variables())


    def train_actor_critic(self):
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        X_states = []
        X_actions = []
        y = []
        for sample in minibatch:
            cur_state, cur_action, reward, next_state, done = sample
            next_actions = self.actor.model.predict(np.expand_dims(next_state, axis=0))
            if done:
                # If episode ends means we have lost the game so we give -ve reward
                # Q(st, at) = -reward
                reward = -reward
            else:
                # Q(st, at) = reward + DISCOUNT * Q(s(t+1), a(t+1))
                next_reward = self.critic.model.predict([np.expand_dims(next_state, axis=0), next_actions])[0][0]
                reward = reward + self.DISCOUNT * next_reward

            X_states.append(cur_state)
            X_actions.append(cur_action)
            y.append(reward)

        X_states = np.array(X_states)
        X_actions = np.array(X_actions)
        X = [X_states, X_actions]
        y = np.array(y)
        y = np.expand_dims(y, axis=1)
        # Train critic model
      
        self.critic.model.fit(X, y, batch_size=self.MINIBATCH_SIZE, verbose = 0)

        # Get the actions for the cur_states from the minibatch.
        # We are doing this because now actor may have learnt more optimal actions for given states
        # as Actor is constantly learning and we are picking the states from the previous experiences.
        X_actions_new = []
        for sample in minibatch:
            X_actions_new.append(self.actor.model.predict(np.expand_dims(sample[0], axis=0))[0])
        X_actions_new = np.array(X_actions_new)

        # grad(J(actor_weights)) = sum[ grad(log(pi(at | st, actor_weights)) * grad(critic_output, action_input), actor_weights) ]
        critic_gradients_val = self.critic.get_critic_gradients(X_states, X_actions)
        self.actor.train(critic_gradients_val, X_states)

    def act(self,cur_state):
        if np.random.uniform(0, 1) < self.EPSILON:
            if self.EPSILON > self.MIN_EPSILON and len(self.replay_memory) >= self.MINIMUM_REPLAY_MEMORY:
                self.EPSILON  = self.EPSILON - 0.000001
                self.EPSILON = max(self.EPSILON, self.MIN_EPSILON)
            # Taking random action (Exploration)
            action = [0] * self.action_dim
            action[np.random.randint(0, self.action_dim)] = 1
            action = np.array(action, dtype=np.float32)
        else:
            # Taking optimal action (Exploitation)
            print('**')
            action = self.actor.model.predict(np.expand_dims(cur_state, axis=0))[0]
            print('Actionn:',action)
        index = np.argmax(action)
        AC = self._map_index_to_action(index)
        return index,AC,action 

    def remember(self,cur_state, action, reward, next_state, done):
            # Add experience to replay memory
        self.replay_memory.append((cur_state, action, reward, next_state, done))


    def _map_action_to_index(self, response):

        for (i, action) in enumerate(self.possible_actions):
            if response == action:
                return i

            """
            Maps an action to an index from possible actions.

            Parameters:
                response (dict)

            Returns:
                int
            """

        
            #raise ValueError('Response: {} not found in possible actions'.format(response))
            
    def _map_index_to_action(self, index):


        for (i, action) in enumerate(self.possible_actions):
            if i == index:
                return action
        raise ValueError('Index: {} not in range of possible actions'.format(index))
        """
        Maps an index to an action in possible actions.

        Parameters:
            index (int)

        Returns:
            dict
        """
        

    def empty_memory(self):
        """Empties the memory and resets the memory index."""

        self.replay_memory = []
        
    def reset(self):
        """Resets the rule-based variables."""

        self.rule_current_slot_index = 0
        self.rule_phase = 'not done'








        
