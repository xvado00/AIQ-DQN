import torch

from agents.utils/deep_q_networks.DQNAgent import DQNAgent
from agents.utils/deep_q_networks.neural_net import NeuralNet, get_optimizer


class DualDQ_l(DQNAgent):
    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, epsilon_decay_length, neural_size_l1,
                 neural_size_l2, neural_size_l3, use_rmsprop, tau, update_interval_length, Lambda=0,
                 eligibility_strategy=0):
        DQNAgent.__init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, epsilon_decay_length,
                              neural_size_l1, neural_size_l2, neural_size_l3, use_rmsprop, Lambda, eligibility_strategy)
        self.update_interval_length = update_interval_length
        self.tau = tau

    def reset(self):
        DQNAgent.reset(self)
        self.policy_net = NeuralNet(self.neural_input_size, self.num_actions, self.neural_size_l1, self.neural_size_l2,
                                    self.neural_size_l3)
        self.optimizer = get_optimizer(
            self.policy_net,
            learning_rate=self.learning_rate,
            use_rmsprop=self.use_rmsprop == 1)

    def learn_from_experience(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from Replay memory
        state_batch, action_batch, reward_batch, next_state_batch = self.get_learning_batches()

        # Compute Q value
        # The model computes Q(s_t), then we select the columns of actions taken.
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Selecting their best reward with max(1)[0].
        q_next_values = None
        with torch.no_grad():
            q_next_values = reward_batch + self.gamma * self.target_net(next_state_batch).max(1)[0]

        # Compute loss
        loss = self.criterion(q_values, q_next_values.unsqueeze(1))

        self.update_eligibility(action_batch, loss)

        loss = loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # Store loss
        self.last_losses.append(loss.item())

        # Update epsilon
        self.decrement_epsilon()
        self.steps_done += 1

        # Update target network
        if self.steps_done % self.update_interval_length == 0:
            self.copy_network_weights()

    def copy_network_weights(self):
        target_net_state_dict = self.policy_net.state_dict()
        policy_net_state_dict = self.target_net.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau \
                                         + target_net_state_dict[key] * (1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

    def __str__(self):
        if self.eligibility_strategy is not None:
            return "DualDQ_l(%.4f,%.2f,%d,%.3f,%d,%d,%d,%d,%d,%.3f,%d,%.3f,%d)" % (
                self.learning_rate,
                self.gamma,
                self.batch_size,
                self.starting_epsilon,
                self.episodes_till_min_decay,
                self.neural_size_l1,
                self.neural_size_l2,
                self.neural_size_l3,
                self.use_rmsprop,
                self.tau,
                self.update_interval_length,
                self.Lambda,
                self.eligibility_strategy_index
            )

        return "DualDQ_l(%.4f,%.2f,%d,%.3f,%d,%d,%d,%d,%d,%.3f,%d)" % (
            self.learning_rate,
            self.gamma,
            self.batch_size,
            self.starting_epsilon,
            self.episodes_till_min_decay,
            self.neural_size_l1,
            self.neural_size_l2,
            self.neural_size_l3,
            self.use_rmsprop,
            self.tau,
            self.update_interval_length
        )