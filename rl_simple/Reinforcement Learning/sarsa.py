import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_es import max_dict
from td0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


if __name__ == '__main__':
    # create standard_grid
    grid = negative_grid()

    # print rewards
    print("\nRewards:")
    print_values(grid.rewards, grid)

    # initialise Q(s,a)
    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
    # print rewards
    print("\nQ Initialisation: {}".format(Q))

    # LETS KEEP TRACK OF HOW MANY TIMES Q[S] HAS BEEN UPDATED - FOR LEARNING RATE DECAY
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0


    #repeat until convergence
    t = 1.0
    deltas = []
    for it in range(10000):
        if it % 100 == 0:
            t += 10e-3   # we are decreasing epsilon through time, for less exploration
        if it & 2000 == 0:
            print("Iteration {}".format(it))
        # inside the loop we play the game
        a = max_dict(Q[s])[0]
        a = random_action(a, eps=0.5/t)
        biggest_change = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()
            # we need next action values for the update
            a2 = max_dict(Q[s2])[0]
            a2 = random_action(a2, eps=0.5 / t)
            # we will update Q AS! we experience the episode (on each move)
            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005
            old_qsa = Q[s][a]
            Q[s][a] = Q[s][a] + alpha * (r + GAMMA*Q[s2][a2]-Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa-Q[s][a]))
            # update the counts
            update_counts[s] = update_counts.get(s, 0) + 1
            # next state becomes this state
            s = s2
            a = a2
        deltas.append(biggest_change)
    plt.plot(deltas)
    plt.show()

    # determine the policy from Q*
    # find V* from Q*
    # determine the policy from Q*
    # find V* from Q*
    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # what's the proportion of time we spend updating each part of Q?
    print("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)

    print("\nvalues:")
    print_values(V, grid)

    print("\npolicy:")
    print_policy(policy, grid)
