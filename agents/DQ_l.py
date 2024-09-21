import torch

from agents.utils.deep_q_networks.DQNAgent import DQNAgent


class DQ_l(DQNAgent):
    def learn_from_experience(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch = self.get_learning_batches()

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        q_next_values = None
        with torch.no_grad():
            q_next_values = reward_batch + self.gamma * self.policy_net(next_state_batch).max(1)[0]

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

    def __str__(self):
        return "DQ_l(" \
                + str(self.learning_rate) + "," \
                + str(self.gamma) + "," \
                + str(self.batch_size) + "," \
                + str(self.starting_epsilon) + "," \
                + str(self.episodes_till_min_decay) + "," \
                + str(self.neural_size_l1) + "," \
                + str(self.neural_size_l2) + "," \
                + str(self.neural_size_l3) + "," \
                + str(self.use_rmsprop) + "," \
                + str(self.history_len) + "," \
                + str(self.Lambda) + "," \
                + str(self.eligibility_strategy_index) + \
                ")"
