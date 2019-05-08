import itertools
import matplotlib.style
import numpy as np
import pandas as pd
from gymEnvNChain import NChainEnv
from matplotlib import pyplot as plt
from collections import defaultdict, namedtuple

matplotlib.style.use('ggplot')

# variable used for statistics
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

# Create gym environment.
env = NChainEnv()

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

# Build Q-Learning Model
def qLearning(env, num_episodes, discount_factor=1.0,
              alpha=0.6, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving following an epsilon-greedy policy
    """

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    print("env.action_space.n %s" % env.action_space.n)

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)

    # For every episode
    for ith_episode in range(num_episodes):
        print("Episode %s" % ith_episode)
        print(dict(Q))

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():
            print("---------------")

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)
            print("state " + str(state) + " action " + str(action))

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(action)
            print("next state " + str(next_state) + " reward " + str(reward))

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            print("next best_next_action " + str(best_next_action))
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            print("td_target " + str(td_target))
            td_delta = td_target - Q[state][action]
            print("td_delta " + str(td_delta))
            Q[state][action] += alpha * td_delta
            print("Q[state][action] " + str(Q[state][action]))

            # episode terminated if env return Done or after 50 movements
            if (done or t == 50):
                break

            state = next_state

    return Q, stats

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3

# Train the model
print("training model")
Q, stats = qLearning(env, 100, discount_factor=0.6)

# plot important statistics
print("plot stats")
plot_episode_stats(stats)

