

DEEP Q-NETWORKS AGENTS MODULE FOR THE AIQ TEST README
=====================================================


This is a module for the Python v3 AIQ Test (https://github.com/xvado00/AIQ)
that adds implementation of selected Deep Q-Networks Agents:

- DQ_l.py  Deep Q-Learning agent based on:
Playing Atari with Deep Reinforcement Learning
by Volodymyr Mnih et al, 2013.

- DualDQ_l.py  Dual Network Deep Q-Learning agent based on:
Human-level Control through Deep Reinforcement Learning
by Volodymyr Mnih et al, 2015.

The module extends the original implementation that was part of a thesis:

Evaluating Deep Q-Learning Agents using an Algorithmic IQ Test
by Michal Dvořák, Prague University of Economics and Business, 2024.

This work was funded by the Internal Grant Agency
of Prague University of Economics and Business (F4/41/2023).

The code is released under the GNU GPLv3. See Licence.txt file.


Known Issues
------------

Eligibility traces implementation based on:
https://github.com/karush17/Deep-Eligibility-Traces
by Karush Suri, 2021
is included, however, it should be considered experimental.


Installation
------------

0) Install PyTorch.
1) Get the AIQ test from: https://github.com/xvado00/AIQ
2) Copy the contents of agents directory into the AIQ/agents.
3) Edit AIQ/agents/__init__.py to list the agents "DQ_l" and "DualDQ_l".


Usage
-----

Hyperparametrs for the DQ_l agent (and their defaults) are as follows:

- alpha -- Deep Q-network optimizer learning rate (0.0003),
- gamma -- future rewards discount factor (0.99),
- BS -- number of samples selected from replay memory for training (32),
- epsilon -- (final) exploration value (0.01),
- EDL -- number of interactions with environment over which
		epsilon=1 is linearly decayed to final epsilon value;
		if set to 0, constant epsilon is used (2000),
- hidden1 -- number of units in the 1st hidden layer (64),
- hidden2 -- number of units in the 2nd hidden layer (512),
- hidden3 -- number of units in the 3rd hidden layer;
		if set to 0, the 3rd layer is not used (0),
- O -- use Adam (0) or RMSProp (1) optimizer in the Deep Q-network (1),
- K -- length of observation history; if set to 0, only the current
		observation is used (2),
- Lambda -- eligibility trace decay rate;
		if set to 0, eligibility traces are disabled (0),
- ETS -- use replacing (0), accumulating (1), or Dutch (2)
		strategy for eligibility traces (0).

This resutls in the following agent string for the AIQ test:

DQ_l,0.0003,0.99,32,0.01,2000,64,512,0,1,2,0,0

Hyperparametrs for the DualDQ_l agent (and their defaults) are as follows:

- alpha -- Deep Q-network optimizer learning rate (0.0003),
- gamma -- future rewards discount factor (0.99),
- BS -- number of samples selected from replay memory for training (32),
- epsilon -- (final) exploration value (0.01),
- EDL -- number of interactions with environment over which
		epsilon=1 is linearly decayed to final epsilon value;
		if set to 0, constant epsilon is used (2000),
- hidden1 -- number of units in the 1st hidden layer (64),
- hidden2 -- number of units in the 2nd hidden layer (512),
- hidden3 -- number of units in the 3rd hidden layer;
		if set to 0, the 3rd layer is not used (0),
- O -- use Adam (0) or RMSProp (1) optimizer in the Deep Q-network (1),
- K -- length of observation history; if set to 0, only the current
		observation is used (2),
- tau -- proportion of weights copied from the policy Deep Q-network
		to the target Deep Q-network (1.0),
- TNUL -- number of interactions with environment that must pass
		before copying weights from the policy to the target Deep Q-network (200),
- Lambda -- eligibility trace decay rate;
		if set to 0, eligibility traces are disabled (0),
- ETS -- use replacing (0), accumulating (1), or Dutch (2)
		strategy for eligibility traces (0).

This resutls in the following agent string for the AIQ test:

DualDQ_l,0.0003,0.99,32,0.01,2000,64,512,0,1,2,1.0,200,0,0


Refer to the AIQ Test Readme.txt for the test parameters and how to run it.

