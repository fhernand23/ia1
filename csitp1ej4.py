# curso sistemas inteligentes - tp 1 - actividad 4
import gym
import itertools
import matplotlib
import matplotlib.style
import numpy as np
import pandas as pd
import sys
from gymEnvBottonTop import BottomTopEnv

from collections import defaultdict
from gym.envs.toy_text import discrete
import plotting
from enum import Enum

matplotlib.style.use('ggplot')

# Create environment.
class Action(Enum):
    REST = 'Rest'
    CLIMB = 'Climb'
    DOWN = 'Down'

class Position(Enum):
    BOTTOM = 'Bottom'
    MIDDLE = 'Middle'
    TOP = 'Top'

class BottomTopEnv():
    """
    BottomTop environment.
    You are an agent on an 3 state network with actions / rewards / next state:
    - BOTTOM (REST 0.1 BOTTOM; CLIMB 0 TOP)
    - MIDDLE (CLIMB 0.3 TOP; DOWN 1 BOTTOM)
    - TOP (REST 0.1 TOP; DOWN 0.2 MIDDLE)
    """
    def __init__(self):
        # shape es la estructura del ambiente
        self.shape = defaultdict();

        self.shape[Position.BOTTOM] = [[Action.REST,0.1,Position.BOTTOM],[Action.CLIMB,0,Position.TOP]]
        self.shape[Position.MIDDLE] = [[Action.CLIMB,0.3,Position.TOP],[Action.DOWN,1,Position.BOTTOM]]
        self.shape[Position.TOP] = [[Action.REST,0.1,Position.TOP],[Action.DOWN,0.2,Position.MIDDLE]]

        # initial position
        self.state = Position.BOTTOM

    def getPossibleAction(self, current):
        return self.shape[current]

    def reset(self, current):
        # initial position
        self.state = Position.BOTTOM

        return self.state

    def step(self, current):
        # initial position
        self.state = Position.BOTTOM

        return self.state


# $\epsilon$-random policy.
def agentEpsilonRandomPolicy(Q, epsilon, num_actions):
    """
    Agente con política epsilon-random con Q-function y epsilon.

    Returns a function that takes the state as an input and returns the probabilities
    for each action in the form of a numpy array of length of the action space(set of possible actions).
    """
    def policyFunction(state):
        Action_probabilities = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction

# Make the $\epsilon$-greedy policy.
def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Returns a function that takes the state as an input and returns the probabilities
    for each action in the form of a numpy array of length of the action space(set of possible actions).
    """
    def policyFunction(state):
        Action_probabilities = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction

# Q-Learning Model
def qLearning(env, num_episodes, discount_factor=1.0,
              alpha=0.6, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving following an epsilon-greedy policy
    """

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    # Q = defaultdict(lambda: np.zeros(env.action_space.n))
    env_action_space_n = 2
    Q = defaultdict(lambda: np.zeros(env_action_space_n))

    # Keeps track of useful statistics
    # stats = plotting.EpisodeStats(
    #     episode_lengths=np.zeros(num_episodes),
    #    episode_rewards=np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env_action_space_n)

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            # done is True if episode terminated
            if done:
                break

            state = next_state

    return Q, stats

# Train the model.ç
print("training model")
Q, stats = qLearning(BottomTopEnv, 100)



