import torch

from agents.utils.deep_q_networks.DQNAgent import DQNAgent
from agents.utils.deep_q_networks.neural_net import NeuralNet, get_optimizer


class DualDQ_l(DQNAgent):
    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, epsilon_decay_length, neural_size_l1,
                 neural_size_l2, neural_size_l3, use_rmsprop, history_len, tau, update_interval_length, Lambda=0,
                 eligibility_strategy=0):
        DQNAgent.__init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, epsilon_decay_length,
                              neural_size_l1, neural_size_l2, neural_size_l3, use_rmsprop, history_len, Lambda, eligibility_strategy)
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

        self.decrement_epsilon()
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
        if self.eligibility_strategy is not None:
            return "DualDQ_l(%.4f,%.2f,%d,%.3f,%d,%d,%d,%d,%d,%d,%.3f,%d,%.3f,%d)" % (
                self.learning_rate,
                self.gamma,
                self.batch_size,
                self.starting_epsilon,
                self.episodes_till_min_decay,
                self.neural_size_l1,
                self.neural_size_l2,
                self.neural_size_l3,
                self.use_rmsprop,
                self.history_len,
                self.tau,
                self.update_interval_length,
                self.Lambda,
                self.eligibility_strategy_index
            )

        return "DualDQ_l(%.4f,%.2f,%d,%.3f,%d,%d,%d,%d,%d,%d,%.3f,%d)" % (
            self.learning_rate,
            self.gamma,
            self.batch_size,
            self.starting_epsilon,
            self.episodes_till_min_decay,
            self.neural_size_l1,
            self.neural_size_l2,
            self.neural_size_l3,
            self.use_rmsprop,
            self.history_len,
            self.tau,
            self.update_interval_length
        )