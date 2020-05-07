import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# this is still a deterministic environment
# so all p(s',r|s,a) = 1 or 0

if __name__ == '__main__':
    # this grid gives you a reward of -0.1 on each step (non-terminal)
    # so this will encourage the shorter path to goal
    grid = negative_grid()

    # print rewards
    print("\nGrid World Rewards: ")
    print_values(grid.rewards, grid)

    # create a random policy (i.e. a random action for each accessible non-terminal state)
    # state -> action
    # this will randomly be updated as we learn
    # INVALID MOVES ALLOWED AT THIS POINT!!
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    print("\nInitial Policy - Random!: ")
    print_policy(policy, grid)

    # initial V(s) randomly
    # we know v(s) for terminal states = 0, so do this
    # otherwise we set random number. You could set it all to 0
    V = {}
    states = grid.all_states()
    for s in states:
        # if state is a accessible, non-terminal state, randomise
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0
    print("\nInitial Value Function - Random!: ")
    print_values(V, grid)

    # repeat the policy iteration until convergence
    # that is, when the policy does not change for a given iteration
    for k in range(100):
        print("\n")
        # policy evaluation step
        # find the value function (expected future reward) for the current policy
        while True:
            biggest_change = 0
            # for each of the possible states (accessible and non-terminal)
            for s in states:
                old_v = V[s]
                # V[s] only has value if its not a terminal state
                # for the states in our policy (i.e. we can make action)
                if s in policy:
                    a = policy[s]      # get the action
                    grid.set_state(s)  # set the position of the robot
                    r = grid.move(a)   # make the move, and get the reward
                    V[s] = r + GAMMA * V[grid.current_state()]  # use bellman to update the value function iteratively
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))   # keep track of the biggest change in value
            # when the value function has converged, break the loop
            # we now have the value function for the given policy (remember we are looping through policies)
            if biggest_change < SMALL_ENOUGH:
                break
        # we now have the V(S) for the current policy!
        print("\n Value Function for {} iteration".format(k))
        print_values(V, grid)
        print('\n')

        # policy improvement step
        policy_converged = True
        # for each of the possible states (accessible and non-terminal)
        for s in states:
            # for the states in our policy (i.e. we can make an action)
            if s in policy:
                print('State: {}, Action: {}'.format(s, policy[s]))
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                # loop through all possible actions to find the best current action
                # by taking the one with the maximum value
                for a in ALL_POSSIBLE_ACTIONS:
                    print('Action: {}'.format(a))
                    if a == policy[s]:
                        p = 0.5
                    else:
                        p = 0.5/3
                    grid.set_state(s)  # set robot position
                    r = grid.move(a)   # make the move and get the reward
                    print('Reward: {}'.format(r))
                    v = p*(r + GAMMA * V[grid.current_state()])  # update the value
                    # find the maximum value
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                print('Optimal Policy is: {}'.format(new_a))
                if new_a != old_a:
                    policy_converged = False
        if policy_converged:
            break

    print("\nUpdated Values: ")
    print_values(V, grid)

    print("\nUpdated Values: ")
    print_policy(policy, grid)
