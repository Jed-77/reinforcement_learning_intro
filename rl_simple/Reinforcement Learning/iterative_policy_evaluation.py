import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

SMALL_ENOUGH = 10e-4  # threshold for convergence of our value function V(S)

def print_values(V, g):
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, g):
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end="")
        print("")


if __name__ == '__main__':
    # iterative policy evaluaton
    # given a policy, lets find the value function V(S)
    # which is the expected future return given we are in state S
    # there are two sources of randomness:
    # p(a/s) - deciding on the action to take, as we are doing random this is a uniform distribution so p=1/n
    # p(s',r | s,a) - probability of the next state and reward given the current state and action...
    # ... our coding is deterministic (if you go left you go there and recieve that reward with prob=1) so we don't have to worry
    grid = standard_grid()

    # states will be positions (i,j)
    # only one position at a given point in time
    # this returns all possible states
    states = grid.all_states()


    # UNIFORM RANDOM ACTIONS
    # initialise our value function V(S)=0 for every possible s
    # so expected future return for every state is 0, and we start iterating to converge
    V = {}
    for s in states:
        V[s] = 0
    gamma = 1.0  # discount factor
    # we are going to iteratively update our v(s) until convergence
    # iteration using bellman equation
    # we look at every possible board position s, and every possible action a
    k = 0
    while True:
        k += 1
        print("\n\nIteration: {}".format(k))
        # keep track of biggest change between iterations of bellman eq
        biggest_change = 0
        # for every possible board position (which is the state)
        for s in states:
            print("\n State: {}".format(s))
            # set an old value function to the current V (expected future reward of s)
            old_v = V[s]
            print("Old Value: {}".format(old_v))
            # V(S) only has value for non-terminal states (if terminal, the expected future return and hence value is 0)
            if s in grid.actions:
                # we will accumulate our answer
                new_v = 0
                # probability of doing the action is uniform, so 1 divided by the number of available actions for this state (e.g. if R,U then 1/2)
                p_a = 1.0/len(grid.actions[s])
                # for each of the possible actions for the state
                for a in grid.actions[s]:
                    # we update the position of the robot to s (starting position for our action)
                    grid.set_state(s)
                    # move the robot, using the action, which returns the reward
                    r = grid.move(a)
                    # use bellmans equation to update the new_v
                    # bellmans equations is dependent on v(s') which is the value of the next state, given the action
                    # so this works as for each loop, we set the game to the correct state and make the move
                    # so the value of the current state is effectively v(s') and enables us to calculate v(s)
                    # so are calculating the expected future return for each possible action from our current state
                    # if they had different probabilities, it would be applied at p_a
                    x1, x2 = grid.current_state()
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                    print("Action: {} has probability {}, return of {} and additional value {} for new state({}, {})".format(a, p_a, r, (r + gamma * V[grid.current_state()]), x1, x2))
                # update the value function accordingly
                # for a given state, adds the expected future return for the given state
                V[s] = new_v
                print("New Value: {}".format(new_v))
                # create the biggest change, as largest difference in the value functions (expected future return) for the entire iteration
                # its the max change for all states
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
                print(biggest_change)
        # if we have converged, break the loop
        if biggest_change < SMALL_ENOUGH:
            break
    print("values for uniformly random actions:")
    print_values(V, grid)
    print("\n\n")

    # FIXED POLICY
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

    # initialize V(s) = 0
    V = {}
    for s in states:
        V[s] = 0

    # let's see how V(s) changes as we get further away from the reward
    gamma = 0.9  # discount factor

    # repeat until convergence
    while True:
        biggest_change = 0
        # for each of the possible states on the board
        for s in states:
            old_v = V[s]
            # V(s) only has value if it's not a terminal state
            # we are not using random, its deterministic movement policy
            # so the probability is 1 and we don't have to loop over the action
            if s in policy:
                # get the action from the current position
                a = policy[s]
                # move the robot to that position
                grid.set_state(s)
                # move the robot
                r = grid.move(a)
                # iteratively update the bellmans equation
                V[s] = r + gamma * V[grid.current_state()]
                # calculate the biggest change for the iteration
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        # if we have converged, break
        if biggest_change < SMALL_ENOUGH:
            break
    print("values for fixed policy:")
    print_values(V, grid)