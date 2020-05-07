import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# THIS IS FOR POLICY EVALUATION, NOT OPTIMISATION


def random_action(a, eps=0.1):
    # choose the given action a with probability 0.5
    # choose an action != a with prob 0.5/3
    p = np.random.random()
    if p < (1-eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)



def play_game(grid, policy):
    # returns a list of states and corresponding returns
    print("\n Playing Game with Policy: ")
    print_policy(policy, grid)
    # reset game to start at random position
    # we need to do this, because our current deterministic policy would
    # ... never end up at certain states, but we want to measure their reward
    s = (2, 0)
    grid.set_state(s)

    # play the game
    print("\nStarting State for the Game is: {}".format(s))

    # each triple is s(t), a(t), r(t)
    # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
    states_and_rewards = [(s, 0)]
    num_steps = 0

    while not grid.game_over():
        # play the game
        print("\nState at move {} : {}".format(num_steps+1, s))
        # play until the game finishes
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        print("Action at move {} : {}".format(num_steps + 1, a))
        num_steps += 1
        s = grid.current_state()
        states_and_rewards.append((s, r))
    return states_and_rewards


if __name__ == '__main__':
    # create standard_grid
    grid = standard_grid()

    # print rewards
    print("\nRewards:")
    print_values(grid.rewards, grid)

    # policy
    print("\nPolicy:")
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    print_policy(policy, grid)

    # initialize V(s) and returns
    V = {}
    states = grid.all_states()
    # states is all possible states you can reach (so everything apart from the wall)
    for s in states:
            V[s] = 0

    for it in range(100):
        print('\n\n\n\n Iteration: {}'.format(it))
        states_and_rewards = play_game(grid, policy)
        print('\n State and Reward: {}'.format(states_and_rewards))
        for t in range(len(states_and_rewards)-1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t+1]
            V[s] = V[s] + ALPHA*(r + GAMMA*V[s2] - V[s])
        print(V)

    print("\nvalues:")
    print_values(V, grid)

    print("\npolicy:")
    print_policy(policy, grid)
