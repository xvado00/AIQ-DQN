
#
# Configuration function for Deep Q-Network agents.
#
# Copyright Michal Dvořák 2024
# Released under GNU GPLv3
#

DQNConfig = {
    # Value that received reward will be divided with.
    "reward_divider": 100,
    # State for which the Q values will be saved?
    "state_for_Q_values_saving": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
}
