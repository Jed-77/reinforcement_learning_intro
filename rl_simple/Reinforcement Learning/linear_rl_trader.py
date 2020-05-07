import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler


def get_data():
    """ Retrieve stock data for AAPL (Apple), MSI (Motorola) and SBUX (Starbucks), between
    Feb 2013 to Feb 2018.

    :returns T (time) x 3 (number of stocks) closing stock prices (0 = AAPL, 1 = MSI, 2 = SBUX)
    :rtype pandas data frame
    """
    data_url = 'C:\\Users\\jscanlon\\Documents\\FY20\\RL\\aapl_msi_sbux.csv'
    df = pd.read_csv(data_url)
    return df.values


def get_scaler(env):
    """ We use a random policy to calculate a state vector at each time step. For each time step in the data, we
    take a random action (from list of available actions) and get the new state vector back. We continue until
    we have a state vector for all time steps [n1, n2, n3, p1, p2, p3, C]. Next, we use sklearn to get the mean and
    std, to be used later for standardizing when calculating z = (x-mu)/sigma, which is important as our values
    n, p, C will have different scales, so this is important for linear/nn models. We return the scaler object
    , which has the functionality to return standardized array of state vectors, one for each time step. We do it
    for every time step to get enough data points to reflect the true mu and sigma of the state vector! Therefore
    all we really need this function for is the mu and sigma, and functionality to calculate z, at a later stage.
    We are trying to get a 'feel' for the range in the data and what the distribution of values will look like.

    :param env: MultiStockEnv object for the current environment, so we can take action and get state vector
    :type env: MultiStockEnv object
    :returns An array of state vectors, one for each time step in the environment
    :rtype StandardScaler object
    """
    states = []
    # for all time steps in the data
    for _ in range(env.n_step):
        # get a random action from the available actions
        action = np.random.choice(env.action_space)
        # make the action and step through time, get the new state vector, reward, done and info
        state, reward, done, info = env.step(action)
        states.append(state)
        # add the state vector to the empty list
        if done:
            break
    # states is an array of state vectors for each of the time steps
    scaler = StandardScaler()
    scaler.fit(states)
    # The standard scaler converts the state vector elements into z scores (z=(x-mu)/sigma) to standardise our elements
    return scaler


def maybe_make_dir(directory):
    """ Checks to see if the directory exists, and if it doesn't it creates that directory.

    :param directory: folder location
    :type directory: string
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


class LinearModel:
    """ A linear regression model. This works very similar to one pass of a forward-feed
    NN model but without the activation function to introduce non-linearity. It can make
    predictions using the model, and perform stochastic gradient descent to improve.
    """
    def __init__(self, input_dim, n_action):
        """ Constructor Method.

        :param input_dim: the size of the state vector [n1, n2, n3, p1, p2, p3, C] (so its 7)
        :type input_dim: integer
        :param n_action: the number of valid actions allowed at each state (3**3 = 27)
        :type n_action: integer
        """
        # Weight is a 27x7 array, reflecting the weights of the linear model matrix. (Each element in state
        # (7) maps to each of the possible actions (27). W is initialised to N(0,1) distribution
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        # Bias is a length 27 vector, reflecting one for each possible. The bias is ...
        # ... initialised at (0,0,0,0,0,0, ..., 0)
        self.b = np.zeros(n_action)

        # momentum terms and empty losses list (containing the mse for the sgd)
        self.vW = 0
        self.vb = 0
        self.losses = []

    def predict(self, X):
        """ Make a prediction of future expected return for each of the available actions given the
        current state using the standard feed-forward NN logic (a dot product with the weights and
        adding on the bias vector). This means we can see for each action, its expected return and
        compare them! VERY COOL! And note, because the future expected reward is dependent on both
        action and state, it is Q(s,a)

        :param X: The STANDARDIZED (z values) state vector of the environment [n1, n2, n3, p1, p2, p3, C]
        :type X: an array of 7x1 elements
        :return A vector with 27 elements which represent the expected future reward for each action
        :rtype list with 27 elements
        """
        # this means we have standardized is, as scaler returns 2-d shape
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        """ Updates the weight and bias vectors used in the linear modelling function (self.predict(X)). It uses
        the current state X (which has been standardized using z values) and the current expected return prediction
        to descend down the cost function.

        :param X: The STANDARDIZED state vector of the environment [n1, n2, n3, p1, p2, p3, C]
        :type X: list with 7 elements
        :param Y: The current Q predictions (expected future reward given state s) for each of the available actions
        :type Y: list with 27 elements
        :param learning_rate: Learning rate of the stochastic gradient descent
        :type learning_rate: float
        :param momentum: Momentum of the stochastic gradient descent
        :type momentum: float
        """
        # this means we have standardized is, as scaler returns 2-d shape
        assert(len(X.shape) == 2)

        # the loss values are 2-D, normally we would divide by N only, but now we divide by N x K
        # this returns the number of values in the target vector (i think accounting for multi-dimensional)
        # we are using average squared error so need to divide by num_values
        num_values = np.prod(Y.shape)

        # do one step of gradient descent, we multiply by 2 to get the exact gradient (its a differentiation!)
        # (not adjusting the learning rate) i.e. d/dx (x^2) --> 2x

        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        # add the mse to the losses so we can track how mse changes through iteration
        mse = np.mean((Yhat - Y)**2)
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)


class MultiStockEnv:
    """ A 3-stock trading environment. It manages the current stock price, current stock amounts, cash available,
    can implement a trading action and step through time.
    State: vector of size 7 (n_stock * 2 + 1) [n1, n2, n3, p1, p2, p3, C]
      - # shares of stock 1 owned
      - # shares of stock 2 owned
      - # shares of stock 3 owned
      - price of stock 1 (using daily close price)
      - price of stock 2
      - price of stock 3
      - cash owned (can be used to purchase more stocks)
    Action: categorical variable with 27 (3^3) possibilities [0,0,0], [0,0,1]....
      - for each stock, you can:
      - 0 = sell
      - 1 = hold
      - 2 = buy
    """

    def __init__(self, data, initial_investment=20000):
        """ Constructor Method.

        :param data: Historical prices for the 3 assets
        :type data: T x 3 array
        :param initial_investment: Amount of cash we start with
        :type initial_investment: float
        """
        # Initialise variables
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # Action space is list of integers 1 -> 3^n
        # Action list is a nested list of possible action permutations:
        #   [[0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], ... ]
        #   so [0,1,2] = SELL AAPL, HOLD MSI, BUY SBUX
        self.action_space = np.arange(3 ** self.n_stock)
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # State is [n1, n2, n3, s1, s2, s3, C] so size is 3*2 + 1
        self.state_dim = self.n_stock * 2 + 1

        # Initialise the other variables
        self.reset()

    def reset(self):
        """ Reset Method.

        :returns The state vector of the environment (n1, n2, n3, p1, p2, p3, C)
        :rtype list with num_stock * 2 + 1 elements
        """
        self.cur_step = 0   # this is the current time step of the environment
        self.stock_owned = np.zeros(self.n_stock)     # this is the amount of the stock owned, initialised to 0
        self.stock_price = self.stock_price_history[self.cur_step]   # stock prices at time t=cur_step (for all 3 stocks)
        self.cash_in_hand = self.initial_investment     # the amount of cash initially equals to investment
        return self._get_obs()

    def step(self, action):
        """ Step Method.

        :param action: Integer between 1 and number of possible actions for each stock (3^3=27)
        :type action: integer

        :returns current state vector, reward value (change in portfolio value), done (have we run out of data?), information (current portfolio value)
        :rtype list with 7 elements, float, boolean, dictionary
        """
        # if action is not an integer between 1 -> 26, assertion error
        assert action in self.action_space
        # previous portfolio value is equal to portfolio value before action
        prev_val = self._get_val()
        # go to the next day and update the stock price
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]
        # perform the trade using the action int (which updates stock owned and cash in hand)
        self._trade(action)
        # get the new value after taking the action
        cur_val = self._get_val()
        # reward is the increase in portfolio value
        reward = cur_val - prev_val
        # this step checks to see if we have run out of data
        done = self.cur_step == self.n_step - 1
        # store the current portfolio value for the time step
        info = {'cur_val': cur_val}
        # conform to the Gym API
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """ Gets the current state vector of the environment. This is essentially creating a
        feature vector for the state.

        :returns The state vector of the environment (n1, n2, n3, p1, p2, p3, C)
        :rtype list with self.state_dim elements
        """
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2 * self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        """ Gets the value of the current portfolio, combining position and cash value.

        :returns Value of the portfolio dot product (price*stock) + cash
        :rtype float
        """
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        """ Execute the action for each stock.

        :param action: Integer between 1 and (3^3=27), so belongs to self.action_space
        :type action: integer
        """
        # We index the action we want to execute, so [0,1,2] = SELL AAPL, HOLD MSI, BUY SBUX
        action_vec = self.action_list[action]

        # determine which stocks to buy or sell
        sell_index = []  # stores index of stocks we want to sell
        buy_index = []  # stores index of stocks we want to buy
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # If the sell index is not empty, then sell any stocks we want to sell, and buy any stocks we want to buy.
        if sell_index:
            # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
            # here, i is the stock we wish to sell
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            # NOTE: when buying, we will loop through each stock and buy one share at a time until we run out of cash
            can_buy = True
            while can_buy:
                for i in buy_index:
                    # if we can afford one stock, buy it and reduce cash
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1  # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False


class DQNAgent:
    """ The agent is responsible for taking previous information, and taking actions that
    will lead to future rewards.
    """
    def __init__(self, state_size, action_size):
        """ Constructor with size of state and action, hyper-parameters and an object of linear model"""
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # as we will decay, need lower bound
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        """ Uses epsilon greedy strategy to get the action for a given state. It will choose a random
        action with prob = epsilon. It will chose the greedy action with prob = 1- epsilon. The greedy
        action is calculated by predicting the expected future return for each action (using prediction)
        and then calculating the action with the maximum expected return (argmax Q)

        :param state: the state vector (normalised) (n1, n2, n3, p1, p2, p3, C)
        :type state: list with 7 elements
        :returns the action to take, which is an integer from env.action_space
        :rtype integer
        """
        # if random number is less than epsilon, we explore
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        # otherwise, we predict the future expected value using the model.predict function
        #   and returns the action with largest expected future return
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action with largest Q value

    def train(self, state, action, reward, next_state, done):
        """ Uses information on the previous state, the action taken (defined by epsilon greedy, the
        reward and the next state to train (i.e. update) our linear model so that it better represents
        Q(s,:), that is the expected return for each action given a state s. We will use stochastic
        gradient descent (sgd), and in order to do so we need to get the cost (error) function. This is
        defined as E=(target-y_hat)^2, where y_hat is our estimation of the future value. The target is
        equal to the reward if a terminal state, and equal to r + gamma*max(Q(s',a')) if not. Note, this estimate
        is specific to BOTH an action and state (Q(s,a)), but our target vector is for ALL actions available
        at state s, so we need to update the Q(s,:) for the given action, Q(s,a). We then use sgd to update
        the weights and biases of the linear model.

        :param state: the state vector (normalised) (n1, n2, n3, p1, p2, p3, C)
        :type state: list with 7 elements
        :param action: Integer between 1 and number of possible actions for each stock (3^3=27)
        :type action: integer
        :param reward: change in value of the portfolio
        :type reward: float
        :param next_state: the state vector (normalised) (n1, n2, n3, p1, p2, p3, C)
        :type next_state: list with 7 elements
        :param done: have we used all the data?
        :type done: boolean
        """
        # create the target Q(s,a), depending on whether the state is terminal
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        # target is the expected future reward for taking action a given we are in state s, by using the
        # formula before. (taking the maximum q(s,a) for predicted next state). Therefore, we want to update
        # the full target Q(S,:) for all actions a by updating the entry for the action to the updated target.
        target_full = self.model.predict(state)
        target_full[0, action] = target
        # this way the error for all other actions are 0! as the target is the same as the current prediction
        # ... so these weights will not be updated.

        # Run one training step
        # sgd will update the weights, bias, momentum and mse to minimise the cost function (y - y_hat)
        self.model.sgd(state, target_full)
        # if the epsilon is greater than defined minimum amount, decay it.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """ Loads the weights and biases, with the correct name"""
        self.model.load_weights(name)

    def save(self, name):
        """ Save the weights and biases"""
        self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
    """ Play one episode (all data) either training or testing. We scale the state vector
    to standardize it (z=(x-mu)/sigma), and go through each time step, by selecting an action
    according to an epsilon greedy strategy, gets the reward and next state. If we are training,
    we update the model weights and biases using sgd. We return the portfolio value."""
    # get the current state and transform to get z values
    # note: after transforming, states are already 1xD so no assertion errors
    state = env.reset()
    state = scaler.transform([state])
    done = False
    info = {}

    while not done:
        action = agent.act(state)  # action is calculated using epsilon greedy strategy
        next_state, reward, done, info = env.step(action)   # do the action! and get the next state plus reward and info
        next_state = scaler.transform([next_state])  # make sure t standardize the next state too
        # if we are training the model, then update the parameters. If not then we are using the parameters 'as is' on our test set
        if is_train == 'train':
            # training the data gets the target Q(s,a) for givens tate and action, we then add it to the full Q(s,:) (i.e.
            # Q for all actions given a state s (which is what we care about for decision making!). We then do sgd and decay
            # the epsilon
            agent.train(state, action, reward, next_state, done)
        # we move state and repeat the process of selecting an action given the new state, getting the reward, and training if need be!
        state = next_state
    # the return is the info dictionary which contains the portfolio value over all time steps modeled
    return info['cur_val']


# ------------- RUN SCRIPT
if __name__ == '__main__':
    # variables
    num_episodes = 100
    initial_investment = 20000
    #batch_size = 32

    # config
    models_folder = 'C:\\Users\\jscanlon\\Documents\\FY20\\RL\\linear_rl_trader_models'
    rewards_folder = 'C:\\Users\\jscanlon\\Documents\\FY20\\RL\\linear_rl_trader_rewards'
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)
    mode = 'test'

    # prepare data
    data = get_data()
    n_timesteps, n_stocks = data.shape
    n_train = n_timesteps // 2
    train_data = data[:n_train]
    test_data = data[n_train:]

    # prepare environment, agent and scaler objects
    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # portfolio value
    portfolio_value = []

    # if we are testing, then load the previous scaler and trained weights
    if mode == 'test':
        # load previous scaler
        with open(models_folder+'\\scaler.pkl', 'rb') as f:
            pickle.load(f)
        # remake the environment with test data
        env = MultiStockEnv(test_data, initial_investment)
        agent.epsilon = 0.01  # make epsilon small as we want little exploration (but not 0 otherwise its deterministic)
        agent.load(models_folder+'\\linear.npz')   # load the trained weights

    # play the game num_episodes times
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, 'train')
        dt = datetime.now()-t0
        portfolio_value.append(val)
        print("episode num: {}, portfolio value: {}, duration: {}".format((e + 1) / num_episodes, val, dt))

    # save the weights when we are done training
    if mode == 'train':
        # save the dqn
        agent.save(models_folder+'\\linear.npz')
        # save the scaler
        with open(models_folder+'\\scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    # plot losses
    plt.plot(agent.model.losses)
    plt.show()

    # save portfolio value for each episode
    np.save(f'{rewards_folder}/{mode}.npy', portfolio_value)


