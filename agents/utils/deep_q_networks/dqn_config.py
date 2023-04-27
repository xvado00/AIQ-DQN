DQNConfig = {
    # If agent uses liner decay epsilon will start at this value.
    "start_epsilon": 1,
    # If agent uses liner decay epsilon will end at this value.
    "min_epsilon": 0.01,
    # Value that received reward will be divided with.
    "reward_divider": 100,
    # Will be Loss plotted?
    "plot_loss": False,
    # Will be Q values plotted?
    "plot_Q_values": False,
    # State for which the Q values will be saved?
    "state_for_Q_values_saving": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    # Will be rewards plotted?
    "plot_rewards": False,
    # Will be actions taken plotted?
    "plot_actions_taken": False,
    # Probability that anything will be plotted after episode ends.
    #   Allows not to be overwhelmed by plots
    "plot_probability": 1.0,
}
