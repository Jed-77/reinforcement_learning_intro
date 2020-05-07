import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers


# this code will use a simple linear model to find a good model using random search


def get_action(s, w):
    """ Apply a simple linear model between the state vector s and our model weights. If
    the result is bigger than 1, we go right (action=1), else we go left (action=0).

    :param s: state vector, of the form [cp, cv, pa, pv]
    :type s: array of 4x1 elements
    :param w: weight vector for my simple linear model
    :type w: array of 4x1 elements
    :return action for the cartpole (1=right, 0=left)
    :rtype integer
    """
    return 1 if s.dot(w) > 0 else 0


def play_one_episode(env, params):
    """ Plays one episode of the cart-pole game. This works by creating a fresh environment with
    starting state. Whilst the pole is upright, we get the action to take using the weight vector
    'params' which is used in the linear model. Note we play the whole game using this weight
    vector, and the different state vector s that gets fed into it results in different actions.
    We return the time-steps taken for the pole to topple.

    :param env: environment object, using gym cart-pole
    :type env: gym object
    :param params: weight vector used in the linear model
    :type params: array of 4x1 elements
    :return number of time-steps for the episode to finish
    :rtype integer
    """
    # Create a fresh environment and starting state. Note the starting state is itself a random
    # ... variable so not repeatable for same set of actions
    observation = env.reset()
    done = False
    t = 0
    # while game has not finished, take another action according to the get_action function.
    while not done and t < 1000:
        #env.render()
        t += 1
        action = get_action(observation, params)  # use the weight vector provided to get action
        observation, reward, done, info = env.step(action)  # take the action
        if done:
            break
    return t


def play_multiple_episodes(env, T, params):
    """ Play T episodes, using a given environment and a specific weight vector for the linear model. Note,
    we are using the same weight vector but will get different results. This is because the starting state
    vector is a random variable so will change each time we play, thus truly testing how 'good' the linear
    model is at playing the game - hence we take average length of the game to get the measure of the
    weight vector params we passed. """
    # initialise
    episode_lengths = np.empty(T)
    # for the number of loops required, play the game and output the episode length
    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)
    # Calculate the mean and output
    avg_length = episode_lengths.mean()
    return avg_length


def random_search(env, num_seaches):
    """ Given our environment, we test n = num_searches random weight vectors in our linear model to see which
     one gives us the best average length of keeping the cartpole upright. We assign a random weight
     vector, play multiple episodes with it (randomness due to different start vector), then we add the length
     to the list. If the average length is better than all previous ones, it becomes the best. We output the
     best episode length and the 'best' weight vector."""
    episode_lengths = []
    best = 0
    params = None
    for t in range(num_seaches):
        # make a random weight vector uniformly sampled between -1 and 1
        new_params = np.random.random(4)*2 - 1
        avg_length = play_multiple_episodes(env, 100, new_params)
        episode_lengths.append(avg_length)
        print("Iteration {}: Average Length {}".format(t, avg_length))
        if avg_length > best:
            params = new_params
            best = avg_length
    return episode_lengths, params


if __name__ == '__main__':
    # Make the environment
    env = gym.make('CartPole-v0')
    # Do the random search on the environment
    episode_lengths, params = random_search(env, 100)
    print(params)
    # Plot the progress
    plt.plot(episode_lengths)
    plt.show()

    # play the final set of 100 episodes and get video
    env = wrappers.Monitor(env, 'C:\\Personal\\Deep Reinforcement Learning\\cartpole\\videos\\random-search', force=True)
    avg_len = play_multiple_episodes(env, 10, params)
    print("\nAvg Len Final: {}".format(avg_len))

