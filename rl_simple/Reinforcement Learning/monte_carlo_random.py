import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# note: this is policy evaluation (i.e. play a game using a given policy)


def random_action(a):
    p = np.random.random()
    if p < 0.5:
        return a
    else:
        tmp = list(ALL_POSSIBLE_ACTIONS)
        tmp.remove(a)
        return np.random.choice(tmp)


def play_game(grid, policy):
    # returns a list of states and corresponding returns

    # reset game to start at random position
    # we need to do this, because our current deterministic policy would
    # ... never end up at certain states, but we want to measure their reward
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    # play the game
    s = grid.current_state()
    #print("Starting State for the Game is: {}".format(s))
    # initialise the reward to 0
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = policy[s]    # make an action according the policy and state s
        #print("Taking action: {}".format(a))
        # here we let the wind possible choose another action
        a = random_action(a)
        r = grid.move(a)   # get the reward from the move
        #print("Reward from move: {}".format(r))
        s = grid.current_state()    # return the current state
        #print("New State: {}".format(s))
        states_and_rewards.append((s, r))    # add the state and value to the list
        #print("Updated states and reward: {}".format(states_and_rewards))

    # calculate the returns by working backwards from the terminal state
    G = 0
    states_and_returns = []
    first = True
    for s, r in reversed(states_and_rewards):
        #print("State, Immediate Reward: {},{}".format(s, r))
        if first:
            first = False
        else:
            #print("State, Future Reward: {},{}".format(s, G))
            states_and_returns.append((s, G))
        G = r + GAMMA*G
    states_and_returns.reverse()
    #print(states_and_returns)
    return states_and_returns


if __name__ == '__main__':
    # create standard_grid
    grid = standard_grid()

    # print rewards
    print("Rewards:")
    print_values(grid.rewards, grid)

    # state-> action is the policy
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',
    }

    # initialize V(s) and returns
    V = {}
    returns = {}  # dictionary of state -> list of returns we've received
    states = grid.all_states()
    # states is all possible states you can reach (so everything apart from the wall)
    for s in states:
        # if the state has an associated action (i.e. is not terminal!)
        # ... then initialise a return list for the state
        if s in grid.actions:
            returns[s] = []
        else:
            # terminal state or state we can't otherwise get to
            V[s] = 0

    # repeat
    for t in range(100):
        print('\n')
        print(t)
        # generate an episode using pi
        states_and_returns = play_game(grid, policy)
        # get a list of states and associated G values (expected future reward for this value)
        seen_states = set()  # get unique states
        for s, G in states_and_returns:
            # for all states and expected future rewards,
            # check if we have already seen s
            # called "first-visit" MC policy evaluation
            if s not in seen_states:
                returns[s].append(G)  # add the G to the returns for the chosen state
                print("returns:{}".format(returns))
                V[s] = np.mean(returns[s])  # update the mean value for the state
                print("v(s):{}".format(V))
                seen_states.add(s)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
