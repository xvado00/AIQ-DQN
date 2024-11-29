
#
# Dual Network Deep Q-learning agent
# with policy and target deep Q-network, replay memory, epsilon decay and (experimental) eligibility traces.
# Human-level control through deep reinforcement learning
#   by Volodymyr Mnih et al, 2015
#
# Deep Eligibility Traces
#   by Karush Suri, 2021
#
# Copyright Michal Dvořák 2024
# Copyright Ondřej Vadinský 2024
# Copyright Jan Štipl 2024
# Released under GNU GPLv3
#

import torch

from agents.utils.deep_q_networks.DQNAgent import DQNAgent
from agents.utils.deep_q_networks.neural_net import NeuralNet, get_optimizer


class DualDQ_l(DQNAgent):
    """
    Configuration based on Mnih et al, 2015
    DualDQ_l(...,
        learning_rate=0.0003,
        gamma=0.99,
        batch_size=32,
        min_epsilon=0.01,
        epsilon_decay_length=2000,
        neural_size_l1=64,
        neural_size_l2=512,
        neural_size_l3=0,
        use_rmsprop=1,
        history_len=2,
        tau=1.0,
        update_interval_length=200,
        Lambda=0.0,
        eligibility_strategy=0)
    DualDQ_l,0.0003,0.99,32,2000,0.01,64,512,0,1,2,1.0,200,0.0,0
    """

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
                 history_len,
                 tau,
                 update_interval_length,
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
        :param history_len: Length of observation history input to the neural network
        :param tau: Controls the proportion of weights copied from the policy network to the target network.
                    A value of `tau < 1` results in slower updates, allowing the target network to gradually approximate
                    the policy network. A value of 1 results in a full copy of the policy network, as used in the original
                    Deep Q-Learning paper by Mnih et al. (2015).
        :param update_interval_length: Number of training steps that must pass before copying the weights from
                                       the policy network to the target network.
        :param Lambda: λ parameter for eligibility traces (0.0: turns off)
        :param eligibility_strategy: Strategy for eligibility traces (0: replacement, 1: accumulation, 2: Dutch)
        """
        DQNAgent.__init__(self,
                          refm=refm,
                          disc_rate=disc_rate,
                          learning_rate=learning_rate,
                          gamma=gamma,
                          batch_size=batch_size,
                          min_epsilon=min_epsilon,
                          episodes_till_min_decay=episodes_till_min_decay,
                          neural_size_l1=neural_size_l1,
                          neural_size_l2=neural_size_l2,
                          neural_size_l3=neural_size_l3,
                          use_rmsprop=use_rmsprop,
                          history_length=history_len,
                          Lambda=Lambda,
                          eligibility_strategy=eligibility_strategy)
        self.update_interval_length = int(update_interval_length)
        self.tau = float(tau)
        self.target_net = None

    def reset(self):
        DQNAgent.reset(self)
        self.target_net = NeuralNet(self.neural_input_size, self.num_actions, [self.neural_size_l1, self.neural_size_l2,
                                                                               self.neural_size_l3])
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn_from_experience(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch = self.get_learning_batches()

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        q_next_values = None
        with torch.no_grad():
            target_next_state_results = self.target_net(next_state_batch)
            max_next_state_q = target_next_state_results.max(1)[0]
            q_next_values = reward_batch + self.gamma * max_next_state_q

        if self.uses_eligibility:
            loss = self.criterion(q_values, q_next_values.unsqueeze(1))
            loss = self.update_eligibility(action_batch, loss)
            loss = loss.mean()
        else:
            loss = self.criterion(q_values, q_next_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        if self.logging_enabled:
            self.last_losses_log.append(loss.detach().item())

        self.decay_epsilon_linear()
        self.steps_done += 1

        if self.steps_done % self.update_interval_length == 0:
            self.copy_network_weights()

    def copy_network_weights(self):
        policy_net_state_dict = self.policy_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()

        policy_net_keys = set(policy_net_state_dict.keys())
        target_net_keys = set(target_net_state_dict.keys())

        if policy_net_keys != target_net_keys:
            raise ValueError("Policy and target network state dictionaries have different keys.")

        for key in target_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau \
                                         + target_net_state_dict[key] * (1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def __str__(self):
        return "DualDQ_l(" \
                + str(self.learning_rate) + "," \
                + str(self.gamma) + "," \
                + str(self.batch_size) + "," \
                + str(self.min_epsilon) + "," \
                + str(self.episodes_till_min_decay) + "," \
                + str(self.neural_size_l1) + "," \
                + str(self.neural_size_l2) + "," \
                + str(self.neural_size_l3) + "," \
                + str(self.use_rmsprop) + "," \
                + str(self.history_len) + "," \
                + str(self.tau) + "," \
                + str(self.update_interval_length) + "," \
                + str(self.Lambda) + "," \
                + str(self.eligibility_strategy_index) + \
                ")"
