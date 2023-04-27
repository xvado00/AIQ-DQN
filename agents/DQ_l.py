import torch

from agents.utils/deep_q_networks.DQNAgent import DQNAgent


class DQ_l(DQNAgent):
    def learn_from_experience(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from Replay memory
        state_batch, action_batch, reward_batch, next_state_batch = self.get_learning_batches()

        # Compute Q value
        # The model computes Q(s_t), then we select the columns of actions taken.
        q_values = self.target_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states with Bellman equation.
        # Selecting their best reward with max(1)[0].
        q_next_values = None
        with torch.no_grad():
            q_next_values = reward_batch + self.gamma * self.target_net(next_state_batch).max(1)[0]

        # Compute loss between neural net's Q value of action taken and result of bellman equation for next state.
        loss = self.criterion(q_values, q_next_values.unsqueeze(1))

        self.update_eligibility(action_batch, loss)

        loss = loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.target_net.parameters(), 1)
        self.optimizer.step()

        # Store loss
        self.last_losses.append(loss.detach().item())

        # Update epsilon
        self.decrement_epsilon()
        self.steps_done += 1

    def __str__(self):
        if self.eligibility_strategy is not None:
            return "DQ_l(%.4f,%.2f,%d,%.3f,%d,%d,%d,%d,%d,%.3f,%d)" % (
                self.learning_rate,
                self.gamma,
                self.batch_size,
                self.starting_epsilon,
                self.episodes_till_min_decay,
                self.neural_size_l1,
                self.neural_size_l2,
                self.neural_size_l3,
                self.use_rmsprop,
                self.Lambda,
                self.eligibility_strategy_index
            )

        return "DQ_l(%.4f,%.2f,%d,%.3f,%d,%d,%d,%d,%d)" % (
            self.learning_rate,
            self.gamma,
            self.batch_size,
            self.starting_epsilon,
            self.episodes_till_min_decay,
            self.neural_size_l1,
            self.neural_size_l2,
            self.neural_size_l3,
            self.use_rmsprop
        )
