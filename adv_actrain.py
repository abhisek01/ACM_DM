from collections import deque
from advantage_actor import Actor
from advantage_critic import Critic
import numpy as np
import random
import tensorflow as tf
from dialogue_config import rule_requests, agent_actions


class AdvantageACM():
    def __init__(self,state_size,constants):
# Hyperparameters
        
        self.DISCOUNT = 0.75
        self.min_memory_size = 1000
        self.EPSILON = 1
        self.MINIMUM_EPSILON = 0.01
        self.MINIBATCH_SIZE = 32
        self.possible_actions = agent_actions
        self.rule_request_set = rule_requests

        # Environment details
        self.action_dim = len(agent_actions)
        self.observation_dim = state_size
        self.memory = []
        self.memory_index = 0
        self.max_memory_size = 50000

        # creating own session to use across all the Keras/Tensorflow models we are using
        sess = tf.Session()


        # Actor model to take actions
        # state -> action
        self.actor = Actor(sess, self.action_dim,self.observation_dim)
        # Critic model to evaluate the acion taken by the actor
        # state -> value of state V(s_t)
        self.critic = Critic(sess, self.observation_dim)

        sess.run(tf.initialize_all_variables())

    def train_advantage_actor_critic(self):
        minibatch = random.sample(self.memory, self.MINIBATCH_SIZE)
        X = []
        y = []
        advantages = np.zeros(shape=(self.MINIBATCH_SIZE, self.action_dim))
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state, done = sample
            if done:
                # If last state then advatage A(s, a) = reward_t - V(s_t)
                advantages[index][action] = reward - self.critic.model.predict(np.expand_dims(cur_state, axis=0))[0][0]
            else:
                # If not last state the advantage A(s_t, a_t) = reward_t + DISCOUNT * V(s_(t+1)) - V(s_t)
                print('Checking:  ',next_state-cur_state)
                next_reward = self.critic.model.predict(np.expand_dims(next_state, axis=0))[0][0]
                print('Next Reward: ',next_reward)
                tt = self.critic.model.predict(np.expand_dims(cur_state, axis=0))[0][0]
                print('Current Reward:',tt)
                advantages[index][action] = reward + self.DISCOUNT * next_reward - self.critic.model.predict(np.expand_dims(cur_state, axis=0))[0][0]
                # Updating reward to trian state value fuction V(s_t)
                reward = reward + self.DISCOUNT * next_reward
                print('Reward: ',reward)
            X.append(cur_state)
            y.append(reward)
        X = np.array(X)
        y = np.array(y)
        y = np.expand_dims(y, axis=1)
        # Training Actor and Critic
        self.actor.train(X, advantages)
        self.critic.model.fit(X, y, batch_size=self.MINIBATCH_SIZE, verbose=0)

    def act(self,cur_state,warmup):


        action = np.zeros(shape=(self.action_dim))
        if(warmup==1):
            return self._rule_action()
        else:

            if(np.random.uniform(0, 1) < self.EPSILON):
                self.EPSILON = self.EPSILON - 0.0001
                # Taking random actions (Exploration)

                action[np.random.randint(0, self.action_dim)] = 1
            else:
                # Taking optimal action suggested by the actor (Exploitation)
                print('***')
                action = self.actor.model.predict(np.expand_dims(cur_state, axis=0))
            index= np.argmax(action)
            ACT = self._map_index_to_action(index)
            return index,ACT

    def remember(self,cur_state, action, reward, next_state, done):

            # Recording experience to train the actor and critic
        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = (cur_state, action, reward, next_state, done)
        self.memory_index = (self.memory_index + 1) % self.max_memory_size


        

    def save(self):
        self.actor.model.save_weights("advantage.h5")

    def load(self):
        self.actor.model.load_weights("advantage.h5")

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

        self.memory = []

    def reset(self):
        """Resets the rule-based variables."""

        self.rule_current_slot_index = 0
        self.rule_phase = 'not done'

    def _rule_action(self):
        """
        Returns a rule-based policy action.

        Selects the next action of a simple rule-based policy.

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself

        """

        if self.rule_current_slot_index < len(self.rule_request_set):
            slot = self.rule_request_set[self.rule_current_slot_index]
            self.rule_current_slot_index += 1
            rule_response = {'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}}
        elif self.rule_phase == 'not done':
            rule_response = {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
            self.rule_phase = 'done'
        elif self.rule_phase == 'done':
            rule_response = {'intent': 'done', 'inform_slots': {}, 'request_slots': {}}
        else:
            raise Exception('Should not have reached this clause')

        index = self._map_action_to_index(rule_response)
        return index, rule_response


    def is_memory_full(self):
        """Returns true if the memory is full."""

        return len(self.memory) == self.max_memory_size


            
