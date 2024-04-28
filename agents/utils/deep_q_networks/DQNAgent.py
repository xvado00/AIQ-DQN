import math
import random
from collections import Counter

import numpy as np
import torch

from agents.Agent import Agent
from agents.utils.deep_q_networks import eligibility_traces
from agents.utils.deep_q_networks.dqn_config import DQNConfig
from agents.utils.deep_q_networks.neural_net import get_criterion, NeuralNet, get_optimizer
from agents.utils.deep_q_networks.plotting_tools import PlottingTools, PlotBuilder, SubplotBuilder, PlotType
from agents.utils.deep_q_networks.replay_memory import ReplayMemory, Transition
from agents.utils.deep_q_networks.string_hashing import generate_md5
from agents.utils.observation_encoder import encode_observations_n_hot


class DQNAgent(Agent):
    START_EPSILON = DQNConfig["start_epsilon"]
    MIN_EPSILON = DQNConfig["min_epsilon"]
    REWARD_DIVIDER = DQNConfig["reward_divider"]
    PLOT_Q_VALUES = DQNConfig["plot_Q_values"]
    PLOT_LOSS = DQNConfig["plot_rewards"]
    PLOT_ACTIONS_TAKEN = DQNConfig["plot_actions_taken"]
    PLOT_REWARDS = DQNConfig["plot_rewards"]
    PLOT_PROBABILITY = DQNConfig["plot_probability"]
    STATE_FOR_Q_VALUES_SAVING = DQNConfig["state_for_Q_values_saving"]

    TRACES_METHODS = [
        "replacing",
        "accumulating",
        "dutch"
    ]

    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, epsilon_decay_length, neural_size_l1,
                 neural_size_l2, neural_size_l3, use_rmsprop, history_length=0, Lambda=0, eligibility_strategy=0):
        Agent.__init__(self, refm, disc_rate)
        self.optimizer = None
        self.neural_size_l1 = neural_size_l1
        self.neural_size_l2 = neural_size_l2
        self.neural_size_l3 = neural_size_l3
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

        # Epsilon
        self.starting_epsilon = epsilon if epsilon > 0 else self.START_EPSILON
        self.epsilon = self.starting_epsilon
        self.has_epsilon_decay = epsilon_decay_length > 0
        self.episodes_till_min_decay = epsilon_decay_length

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


        self.last_losses = list()
        self.q_values_arr = [[] for i in range(self.num_actions)]
        self.actions_taken = list()
        self.rewards_given = list()

        self.plotting_tools = PlottingTools()
        self.epsilon_linear_decay = 0 if self.episodes_till_min_decay == 0 \
            else (self.starting_epsilon - self.MIN_EPSILON) / self.episodes_till_min_decay

    def reset(self):
        self.memory = ReplayMemory(10000)
        self.policy_net = NeuralNet(self.neural_input_size, self.num_actions,
                                    [self.neural_size_l1, self.neural_size_l2, self.neural_size_l3])

        self.optimizer = get_optimizer(
            self.policy_net,
            learning_rate=self.learning_rate,
            use_rmsprop=self.use_rmsprop == 1)

        self.cached_states_raw = []
        self.states_history = torch.zeros((self.history_len, self.state_vec_size))

        self.prev_action = None
        self.steps_done = 0
        self.epsilon = self.starting_epsilon
        self.eligibility = torch.zeros((self.batch_size, self.num_actions))
        self.reset_values_for_plots()

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
        self.rewards_given.append(reward / self.REWARD_DIVIDER)

        self.learn_from_experience()

        opt_action = self.get_action(new_state_with_history_tensor)

        self.prev_action = torch.tensor(opt_action).unsqueeze(0).unsqueeze(0)
        self.cached_state_raw = observations
        self.current_state_with_history = new_state_with_history_unsqueezed

        self.save_prev_values(opt_action, new_state_vec, new_state_with_history_unsqueezed)

        return opt_action

    def get_logs( self ) -> dict:

        return {
            "losses": self.last_losses,
            "rewards": self.rewards_given,
            "actions": self.actions_taken,
            "q_values": self.q_values_arr
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

    def episode_ended(self, stratum, program):
        if len(self.q_values_arr[0]) < 15:
            return
        if random.random() > self.PLOT_PROBABILITY:
            return

        if self.PLOT_LOSS and len(self.last_losses) > 0:
            self.plotting_tools.plot_array(y=self.last_losses, title=self.get_plot_title(stratum, program, "Loss"))

        if self.PLOT_Q_VALUES and len(self.q_values_arr[0]) > 0:
            self.plotting_tools.plot_multiple_array(self.q_values_arr,
                                                    title=self.get_plot_title(stratum, program, "Q-values"))

        if self.PLOT_REWARDS and len(self.rewards_given) > 0:
            self.plotting_tools.plot_array(y=self.rewards_given,
                                           title=self.get_plot_title(stratum, program, "Rewards"), type="o")

        if self.PLOT_ACTIONS_TAKEN and len(self.actions_taken) > 0:
            self.plotting_tools.plot_array(y=self.actions_taken,
                                           title=self.get_plot_title(stratum, program, "Actions taken"), type="o")

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
            self.actions_taken.append(action)

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

    def decrement_epsilon(self):
        """
        Decrements epsilon by calculated step until it hits self.MIN_EPSILON
        """
        if not self.has_epsilon_decay:
            return
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon -= self.epsilon_linear_decay
        else:
            self.epsilon = self.MIN_EPSILON

    def learn_from_experience(self):
        """
        Function that should handle learning of agent's neural net.
        """
        raise NotImplementedError()

    def reset_values_for_plots(self):
        self.last_losses = list()
        self.actions_taken = list()
        self.rewards_given = list()
        self.q_values_arr = [[] for i in range(self.num_actions)]

    def append_q_values(self, q_values, state):
        if not self.logging_enabled:
            return
        for i, state_ref_i in enumerate(self.state_for_saving):
            state_i = int(state[i].item())
            if state_ref_i is not state_i:
                return
        for i, q_value in enumerate(q_values):
            self.q_values_arr[i].append(q_value)

    def get_plot_title(self, stratum, program, title):
        return "%s for: S%s\n%s" % (title, stratum, program)

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

    def on_program_end(self, stratum, program):
        agent = self.__str__().split("(")[0]
        hashed_program = generate_md5(program)
        filename = f"plots/{hashed_program}_{agent}.png"
        plot_builder = PlotBuilder(f"{agent} on program: {program} in stratum: {stratum}", filename)

        loss = self.last_losses
        if self.PLOT_LOSS and loss and len(loss) > 0:
            plot_builder.add_sub_plot(SubplotBuilder().called("Losses").has_data(loss).build())

        rewards = self.rewards_given
        if rewards and len(rewards) > 0:
            plot_builder.add_sub_plot(SubplotBuilder().called("Rewards").has_data(rewards).typeof(PlotType.Dots)
                                      .build())

        actions = self.actions_taken
        most_freq_action = 0
        if self.PLOT_ACTIONS_TAKEN and actions and len(actions) > 0:
            counter = Counter(actions)
            most_freq_action = counter.most_common(1)[0][0]
            plot_builder.add_sub_plot(SubplotBuilder().called("Taken actions").has_data(actions).typeof(PlotType.Dots)
                                      .build())

        q_values = self.q_values_arr
        if self.PLOT_Q_VALUES and q_values and len(q_values) > 0 and len(q_values[most_freq_action]) > 100:
            plot_builder.add_sub_plot(
                SubplotBuilder()
                .called(f"Q values of action {most_freq_action}")
                .has_data(q_values[most_freq_action])
                .build()
            )
            plot_builder.build()
