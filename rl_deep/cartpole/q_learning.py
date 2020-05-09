import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


class SGDRegressor:
    """ SGD Regressor. This is stil a linear model!"""
    def __init__(self, D):
        # D is the dimension of the feature vector (i.e. how many elements in our feature vector!). This constructor
        # ... initialises a random weight vector, normalised by the sqrt so we have z = (x-mu)/sigma (recall mu = 0)
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1   # learning rate.

    def partial_fit(self, X, Y):
        # X is a feature vector of 400x1. Y is the expected return from our state (so G)
        # so our weight vector gets updated using
        # w <- w + lr(G - Q(s,a))
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        # Using the weight vector of this class, we apply a linear model to the X values. This gives our prediction of
        # ... Q(s,a). Note X is a feature vector at this stage. This return is a float, which gives the expected return
        # ... for taking a specific action a from state s.
        return X.dot(self.w)



class FeatureTransformer:
    """ Transform the state vector into ... using ..."""
    def __init__(self):
        # In mountain-car, we could sample 10k observation samples using the following:
        # ... observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        # ... but using these state samples for cart-pole are poor, b/c you get velocities --> infinity
        # ... and we would never sample them!
        # so we sample 20000 samples of random lists of 4x1 ranging between -1 to 1. These are meant to
        # ... represet the state vector. Recall State is [cp, cv, pa, pv]
        # # with cp = Cart Position, cv = Cart Velocity, pa = Pole Angle, pv = Pole Velocity
        observation_examples = np.random.random((20000, 4))*2 - 1
        # Next create a standard scaler object and fit the observations, so calculating the mean and stdev
        # ... to scale later to a z = (x-mu)/sigma.
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=100)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=100))
        ])
        # Feature examples is a 20000 by 400 array. 20000 is the number of random states we have created, and
        # ... for each of those we create a feature vector of 400 (each of the RBF samplers). Remember, each RBF sampler
        # ... has its own exemplar (centre c), so each will calculate the radial basis. Note, we have converted a state of
        # ... 4x1 to a state of 400x1.
        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))
        self.dimensions = feature_examples.shape[1]    # number of dimensions for our feature vector
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        # see feature example above. this creates the scaled feature vector.
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


# Holds one SGDRegressor for each action
class Model:
    """ Does some stuff. """
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            # create a model for all of the actions. For cart-pole, this is 2 (L or R)
            model = SGDRegressor(feature_transformer.dimensions)
            # So we have a list of two models, both of which are a SGDRegressor object
            self.models.append(model)

    def predict(self, s):
        # To predict (i.e. get Q(s,:) for the given state, we transform our state vector to a feature vector
        # ... using the feature transform. Then, for each of the actions available, we get the Q(S,a) for it
        # ... using the SGDRegressor predict function. This applied a linear model using the weights to give
        # ... us the expected return for taking action a from state s. Hence, result is an array of 2x1, with
        # ... expected return for each of the possible action (L or R).
        X = self.feature_transformer.transform(np.atleast_2d(s))
        result = np.stack([m.predict(X) for m in self.models]).T
        return result

    def update(self, s, a, G):
        # First thing is to transform the state from a 4x1 state vector to a 400x1 feature vector.
        X = self.feature_transformer.transform(np.atleast_2d(s))
        # For the action that was made,
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        # epsilon greedy strategy. If random, choose a sample from the env action space. If not, we predict
        # ... Q(s,:) from current state s for all possible actions and take the largest one (that is, take
        # ... the action with the largest Q and hence the largest future reward).
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def play_one(env, model, eps, gamma):
    # Reset the environment
    observation = env.reset()

    # Set all variables
    done = False
    totalreward = 0
    iters = 0

    # While game is not finished, sample an action and play the game
    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        # if we are done, then add the massive negative reward
        if done:
            reward = -200

        # update the model for each action played. Do this by first predicting the value
        # ... of the next state Q(S,:) for all actions a. We then use the maximum of these
        # .. future values to calculate G, and subsequently update Q(s,a) for the given
        # ... state action pair we have just played.
        next = model.predict(observation)
        # print(next.shape)
        assert(next.shape == (1, env.action_space.n))
        # we update G using G = r = gamma * argmax(Q(s,:)). So G is the future return from
        # ... the current state.
        G = reward + gamma * np.max(next)
        model.update(prev_observation, action, G)

        if reward == 1: # if we changed the reward to -200
            totalreward += reward
            iters += 1

    return totalreward


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


def main():
    # Initialise environment, feature transformer and the model
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.99

    # Setup variables
    N = 500
    totalrewards = np.empty(N)
    costs = np.empty(N)

    # Run the game and get the rewards
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(env, model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)


if __name__ == '__main__':
    main()


