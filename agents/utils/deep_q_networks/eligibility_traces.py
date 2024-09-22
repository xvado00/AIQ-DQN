
#
# Eligibility Traces functions for Deep Q-Network agents.
#
# Copyright Michal Dvořák 2024
# Released under GNU GPLv3
#

def replacing(learning_rate, actions, trace):
    trace = trace.scatter_(1, actions.unsqueeze(1), 1)
    return trace


def accumulating(learning_rate, actions, trace):
    trace = trace.scatter_(1, actions.unsqueeze(1), 1, reduce='add')
    return trace


def dutch(learning_rate, actions, trace):
    trace = trace.scatter_(1, actions.unsqueeze(1), 1 - learning_rate, reduce='multiply')
    trace = trace.scatter_(1, actions.unsqueeze(1), 1, reduce='add')
    return trace
