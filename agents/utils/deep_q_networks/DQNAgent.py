
#
# Deep Q-Network Agent Class for various Deep Q-learning agents.
#
# Copyright Michal Dvořák 2024
# Copyright Ondřej Vadinský 2024
# Copyright Jan Štipl 2024
# Released under GNU GPLv3
#

import math
import random

import numpy as np
import torch

from agents.Agent import Agent
from agents.utils.deep_q_networks import eligibility_traces
from agents.utils.deep_q_networks.dqn_config import DQNConfig
from agents.utils.deep_q_networks.neural_net import get_criterion, NeuralNet, get_optimizer
from agents.utils.deep_q_networks.replay_memory import ReplayMemory, Transition
from agents.utils.epsilon_decay import EpsilonDecayMixin
from agents.utils.observation_encoder import encode_observations_n_hot


class DQNAgent(Agent, EpsilonDecayMixin):
    REWARD_DIVIDER = DQNConfig["reward_divider"]
    STATE_FOR_Q_VALUES_SAVING = DQNConfig["state_for_Q_values_saving"]

    TRACES_METHODS = [
        "replacing",
        "accumulating",
        "dutch"
    ]

    def __init__(self,
                 refm,
                 disc_rate,
                 learning_rate,
                 gamma,
                 batch_size,
                 min_epsilon,
                 episodes_till_min_decay,
                 neural_size_l1,
                 neural_size_l2,
                 neural_size_l3,
                 use_rmsprop,
                 history_length=0,
                 Lambda=0.0,
                 eligibility_strategy=0):
        """
        :param refm:
        :param disc_rate:
        :param learning_rate: Step size during training via gradient descent
        :param gamma: Discount factor for future rewards
        :param batch_size: Number of samples selected from replay memory for training
        :param min_epsilon: Minimum value of epsilon for exploration-exploitation trade-off
        :param episodes_till_min_decay: Steps over which epsilon decays to min_epsilon
        :param neural_size_l1: Size of the first hidden layer
        :param neural_size_l2: Size of the second hidden layer
        :param neural_size_l3: Size of the third hidden layer (0 to omit)
        :param use_rmsprop: 1 for RMSProp optimizer, 0 for ADAM
        :param history_length: Length of observation history input to the neural network
        :param Lambda: λ parameter for eligibility traces
        :param eligibility_strategy: Strategy for eligibility traces (0: replacement, 1: accumulation, 2: Dutch)
        """
        Agent.__init__(self, refm, disc_rate)
        EpsilonDecayMixin.__init__(self, min_epsilon, episodes_till_min_decay)
        self.optimizer = None
        self.neural_size_l1 = int(neural_size_l1)
        self.neural_size_l2 = int(neural_size_l2)
        self.neural_size_l3 = int(neural_size_l3)
        self.use_rmsprop = int(use_rmsprop)
        self.history_len = int(history_length)
        self.policy_net = None
        self.memory = None
        self.ref_machine = refm
        self.num_states = refm.getNumObs()  # assuming that states = observations
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells = refm.getNumObsCells()
        self.state_vec_size = self.obs_cells * self.obs_symbols
        self.neural_input_size = self.state_vec_size * (self.history_len + 1)
        self.gamma = float(gamma)

        self.learning_rate = float(learning_rate)
        self.batch_size = math.floor(batch_size)

        self.Lambda = Lambda
        self.eligibility = torch.zeros((self.batch_size, self.num_actions))
        strat_int = int(eligibility_strategy)
        self.eligibility_strategy_index = strat_int
        self.eligibility_strategy = self.TRACES_METHODS[strat_int]
        self.uses_eligibility = self.Lambda > 0.0

        if self.uses_eligibility:
            self.criterion = get_criterion(reduction='none')
        else:
            self.criterion = get_criterion(reduction='mean')

        self.cached_states_raw = []
        self.states_history = torch.zeros((self.history_len, self.state_vec_size))
        self.prev_action = None
        self.current_state_with_history = None
        self.steps_done = 0

        self.state_for_saving = self.STATE_FOR_Q_VALUES_SAVING
        if len(self.state_for_saving) < self.neural_input_size:
            for i in range(self.neural_input_size - len(self.STATE_FOR_Q_VALUES_SAVING)):
                self.state_for_saving.append(0)
        if len(self.state_for_saving) > self.neural_input_size:
            self.state_for_saving = self.state_for_saving[:self.neural_input_size]

        # Used for logging
        self.last_losses_log = None
        self.q_values_arr_log = None
        self.actions_taken_log = None
        self.rewards_given_log = None

    def reset(self):
        EpsilonDecayMixin.reset(self)
        torch.set_num_threads(1)
        self.memory = ReplayMemory(10000)
        self.policy_net = NeuralNet(self.neural_input_size, self.num_actions,
                                    [self.neural_size_l1, self.neural_size_l2, self.neural_size_l3])

        self.optimizer = get_optimizer(
            self.policy_net,
            learning_rate=self.learning_rate,
            use_rmsprop=self.use_rmsprop == 1)

        self.cached_states_raw = []
        self.states_history = torch.zeros((self.history_len, self.state_vec_size))
        self.current_state_with_history = None

        self.prev_action = None
        self.steps_done = 0
        self.eligibility = torch.zeros((self.batch_size, self.num_actions))

        self.reset_values_for_logs()

    def perceive(self, observations, reward):
        new_state_vec = self.observations_to_vec(observations)
        new_state_with_history_tensor = self.add_history_to_observation(new_state_vec)
        new_state_with_history_unsqueezed = new_state_with_history_tensor.unsqueeze(0)
        # Add to replay memory
        if (self.current_state_with_history is not None) and (self.prev_action is not None):
            self.memory.push(
                self.current_state_with_history,
                self.prev_action,
                new_state_with_history_unsqueezed,
                torch.tensor(reward / self.REWARD_DIVIDER, dtype=torch.float32).unsqueeze(0)
            )

        if self.logging_enabled:
            self.rewards_given_log.append(reward / self.REWARD_DIVIDER)

        self.learn_from_experience()

        opt_action = self.get_action(new_state_with_history_tensor)

        self.prev_action = torch.tensor(opt_action).unsqueeze(0).unsqueeze(0)
        self.cached_state_raw = observations
        self.current_state_with_history = new_state_with_history_unsqueezed

        self.save_prev_values(opt_action, new_state_vec, new_state_with_history_unsqueezed)

        return opt_action

    def get_logs( self ) -> dict:

        return {
            "losses": self.last_losses_log,
            "rewards": self.rewards_given_log,
            "actions": self.actions_taken_log,
            "q_values": self.q_values_arr_log
        }

    def save_prev_values(self, action, current_state, new_state_with_history_unsqueezed):
        self.prev_action = torch.tensor(action).unsqueeze(0).unsqueeze(0)
        self.current_state_with_history = new_state_with_history_unsqueezed
        prev_state_history = self.states_history.detach().clone()

        if self.history_len <= 0: return

        for i in range(self.state_vec_size):
            self.states_history[0][i] = current_state[i]

        if self.history_len < 1: return

        for history_index in range(self.history_len - 1):
            self.states_history[history_index + 1] = prev_state_history[history_index]

    def observations_to_vec(self, observations):
        encoded = encode_observations_n_hot(observations, self.obs_cells, self.obs_symbols)
        return torch.tensor(encoded)

    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, random action will be taken or
          otherwise the best policy action will be taken.
        """
        is_random = random.random() < self.epsilon
        legal_actions = [action for action in range(self.num_actions)]
        action = None
        if is_random:
            action = random.choice(legal_actions)
        else:
            action = self.compute_action_from_Q_value(state)

        if self.logging_enabled:
            self.actions_taken_log.append(action)

        return action

    def compute_action_from_Q_value(self, state):
        with torch.no_grad():
            action_values = self.policy_net.forward(state).tolist()

            if self.logging_enabled:
                self.append_q_values(action_values, state)

            policy = np.argmax(action_values)
            return policy

    def add_history_to_observation(self, observation_vec):
        if len(observation_vec) != self.state_vec_size:
            raise Exception("Observation is not in count as observation cells.")

        state_vec = torch.zeros(self.neural_input_size, dtype=torch.float32)

        for prev_state_index in range(self.history_len):
            prev_state = self.states_history[prev_state_index]
            arr_start_point = prev_state_index * self.state_vec_size
            for i in range(self.state_vec_size):
                state_vec[i + arr_start_point] = prev_state[i]

        for i in range(self.state_vec_size):
            index_after_prev_states = self.history_len * self.state_vec_size
            index = index_after_prev_states + i
            state_vec[index] = observation_vec[i]
        return state_vec

    def get_learning_batches(self):
        """
        Samples batches of self.batch_size size of states, actions, rewards
        and next state values.
        :return: tuple with values as (state_batch, action_batch, reward_batch, next_states)
        """

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        return state_batch, action_batch, reward_batch, next_states

    def learn_from_experience(self):
        """
        Function that should handle learning of agent's neural net.
        """
        raise NotImplementedError()

    def set_logging( self, logging_enabled: bool ):
        Agent.set_logging(self, logging_enabled)

        if self.logging_enabled:
            self.reset_values_for_logs()

    def reset_values_for_logs(self):
        self.last_losses_log = list()
        self.actions_taken_log = list()
        self.rewards_given_log = list()
        self.q_values_arr_log = [[] for i in range(self.num_actions)]

    def append_q_values(self, q_values, state):
        if not self.logging_enabled:
            return
        for i, state_ref_i in enumerate(self.state_for_saving):
            state_i = int(state[i].item())
            if state_ref_i is not state_i:
                return
        for i, q_value in enumerate(q_values):
            self.q_values_arr_log[i].append(q_value)

    def reset_trace(self):
        return torch.zeros((self.batch_size, self.num_actions))

    def update_trace(self, action_vals):
        return getattr(eligibility_traces, self.eligibility_strategy)(self.learning_rate, action_vals, self.eligibility)

    def update_eligibility(self, action_batch, q_val_error):
        if self.steps_done == 0:
            self.eligibility = self.reset_trace()
        self.eligibility = self.update_trace(action_batch.squeeze())
        el_vals = self.eligibility.gather(1, action_batch)
        q_val_error *= el_vals
        self.eligibility *= self.gamma * self.Lambda
        return q_val_error

