import torch

from agents.utils.deep_q_networks.DQNAgent import DQNAgent
from agents.utils.deep_q_networks.neural_net import NeuralNet, get_optimizer


class DualDQ_l(DQNAgent):
    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, epsilon_decay_length, min_epsilon,
    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size,
                 min_epsilon, episodes_till_min_decay,  # Epsilon decay vars
                 neural_size_l1, neural_size_l2, neural_size_l3, use_rmsprop, history_len,
                 tau, update_interval_length, Lambda=0.0, eligibility_strategy=0):
        DQNAgent.__init__(self, refm, disc_rate, learning_rate, gamma, batch_size,
                          episodes_till_min_decay, min_epsilon,  # Epsilon decay vars
                          neural_size_l1, neural_size_l2, neural_size_l3, use_rmsprop, history_len,
                          Lambda, eligibility_strategy)
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

        self.last_losses.append(loss.detach().item())

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
                + str(self.episodes_till_min_decay) + "," \
                + str(self.min_epsilon) + "," \
                + str(self.neural_size_l1) + "," \
                + str(self.neural_size_l2) + "," \
                + str(self.neural_size_l3) + "," \
                + str(self.use_rmsprop) + "," \
                + str(self.history_len) + "," \
                + str(self.tau) + "," \
                + str(self.update_interval_length) + "," \
                + str(self.Lambda) + "," \
                + str(self.eligibility_strategy_index) + "," \
                ")"
