
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import wrappers


# function will do q_learning but convert the continuous infinite state vector
# ... into discrete bins and then use tabular approach (WHERE WE STORE ALL Q
# ... VALUES FOR ALL POSSIBLE STATES)

LEARNING_RATE = 1e-2


def build_state(features):
    """ Given a list of integers, it will convert into an int. For example ([1,2,3,4]) -> 1234. It is used to convert
    a state space of four floating numbers into an integer of 0000->9999 given the bins of the elements. """
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    """ Given a list of bins, it will return the index of the bins list that the given value belongs to. E.g. if we have
    bins = [-2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4] and value = 0.1, then we return 5 as this is the bin we are in.
    """
    return np.digitize(x=[value], bins=bins)[0]


class FeatureTransformer:
    """ A user class which is effectively a customised sklearn transforer. We create 10 bins for each of the four elements
    within the state vector using the constructor. We then transform the state to an integer using the functions above. So
    the state vector will range from 0000->9999"""
    def __init__(self):
        # Note, to get these bin bounds you would run the simulation of cart-pole many times and plot the histogram. You could
        # ... also change the bind width s.t the probability of being in each bin is the same. But these bins are ok...
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)  # (-inf, inf) (I did not check that these were good values)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)  # (-inf, inf) (I did not check that these were good values)

    def transform(self, observation):
        """"""
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return build_state([
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velocity_bins),
        ])


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer

        num_states = 10**env.observation_space.shape[0]  # number of possible states
        num_actions = env.action_space.n                 # number of possible actions
        self.Q = np.random.uniform(low=1, high=1, size=(num_states, num_actions))  # q is dependent on both state and s

    def predict(self, s):
        # convert our state vector into an integer, and return the Q value for the given state. As we are indexing q for
        # ... only the state, we are outputting Q(s,:), so for all possible actions. (Useful for arg max later)
        x = self.feature_transformer.transform(s)
        return self.Q[x]

    def update(self, s, a, G):
        # convert state to integer
        x = self.feature_transformer.transform(s)
        # perform gradient descent to update the Q values for the given state action pair
        self.Q[x, a] += LEARNING_RATE * (G - self.Q[x, a])

    def sample_action(self, s, eps):
        # epsilon greedy strategy
        # if random then sample a random action from env action space. Else, we use the predict function to get the Q (
        # ... expected future reward) from the given state and choose the action with the best Q.
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)


def play_one(model, eps, gamma):
    # initialise the game
    observation = model.env.reset()
    print("Starting State: {}".format(observation))
    done = False
    totalreward = 0
    iters = 0
    # play the game
    while not done:
        action = model.sample_action(observation, eps)   # use episilon greedy to get action (one that maximises Q, probably)
        prev_observation = observation                   # save the previous state
        observation, reward, done, info = model.env.step(action)   # take the action and get resulting information
        totalreward += reward    # keep track of total reward
        print("Step: {}, Action: {}, Next State: {}, Reward: {}, Is done? {}".format(iters, action, observation, reward, done))
        # if done, then reward is -300
        if done:
            reward = -300
        # update the model. Expected future reward is equal to the next reward plus the maximum reward from the next state (
        # ... its recursive). We then update the model using this G.
        G = reward + gamma*np.max(model.predict(observation))
        model.update(prev_observation, action, G)
        iters += 1
    return totalreward


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.9

    N = 1
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

