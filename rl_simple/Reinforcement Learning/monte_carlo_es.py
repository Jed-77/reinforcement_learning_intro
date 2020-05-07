import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# note: this is policy evaluation (i.e. play a game using a given policy)
# es = EXPLORING STARTS


def play_game(grid, policy):
    # returns a list of states and corresponding returns
    print("\n Playing Game with Policy: ")
    print_policy(policy, grid)
    # reset game to start at random position
    # we need to do this, because our current deterministic policy would
    # ... never end up at certain states, but we want to measure their reward
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    # play the game
    s = grid.current_state()
    print("\nStarting State for the Game is: {}".format(s))
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)
    print("Starting Action for the Game is {}".format(a))

    # each triple is s(t), a(t), r(t)
    # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
    states_actions_rewards = [(s, a, 0)]
    seen_states = set()
    seen_states.add(grid.current_state())
    num_steps = 0

    while True:
        # play the game
        print("\nState at move {} : {}".format(num_steps+1, s))
        print("Action at move {} : {}".format(num_steps+1, a))
        # play until the game finishes
        r = grid.move(a)
        num_steps += 1
        s = grid.current_state()

        if s in seen_states:
            # hack so that we don't end up in an infinitely long episode
            # bumping into the wall repeatedly
            # if num_steps == 1 -> bumped into a wall and haven't moved anywhere
            #   reward = -10
            # else:
            #   reward = falls off by 1 / num_steps
            reward = -10. / num_steps
            states_actions_rewards.append((s, None, reward))
            break
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            # THE FIRST MOVE IS RANDOM, BUT PAST THIS ITS ACCORDING TO THE POLICY
            # THIS NEEDS TO BE THE CASE OTHERWISE WE WOULD NEVER REACH CERTAIN STATES USING
            # OUR DETERMINISTIC POLICY
            a = policy[s]
            states_actions_rewards.append((s, a, r))
        seen_states.add(s)
    print("\nState Action Reward: {}".format(states_actions_rewards))

    # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    states_actions_returns.reverse()  # we want it to be in order of state visited
    print("\nState Action Return (G): {}".format(states_actions_returns))
    return states_actions_returns



def max_dict(d):
    # return the argmax (key) and the max value from a dictionary
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


if __name__ == '__main__':
    # create standard_grid
    grid = negative_grid()

    # print rewards
    print("\nRewards:")
    print_values(grid.rewards, grid)

    # random initial policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # initialize Q(s,a) and returns
    Q = {}
    returns = {}  # dictionary of state -> list of returns we've received
    states = grid.all_states()
    # states is all possible states you can reach (so everything apart from the wall)
    for s in states:
        # if not a terminal state
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0  # initialise so argmax works
                returns[(s, a)] = []
        else:
            # terminal state or state can't go to
            pass
    # so q contains every possible action state pair, with a value
    print("\nQ Initialisation: {}".format(Q))
    # Q HOLDS THE SAMPLE MEAN, SO OUR VALUE!
    print("\nReturns Initialistion: {}".format(returns))
    # RETURN HOLDS THE FULL LIST SO WE CAN AVERAGE IT

    # repeat until convergence
    deltas = []
    for t in range(10):
        print('\n\n\n\n Iteration: {}'.format(t))
        if t % 1000 == 0:
            print(t)

        # generate an episode using pi
        biggest_change = 0
        states_actions_returns = play_game(grid, policy)
        seen_state_action_pairs = set()
        print("\n")
        for s, a, G in states_actions_returns:
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                print("Updating for {}".format((s, a)))
                old_q = Q[s][a]
                returns[sa].append(G)
                print("Updating Returns: {}".format(returns))
                Q[s][a] = np.mean(returns[sa])
                print("Updating Q: {}".format(Q))
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
            deltas.append(biggest_change)

        # update the policy
        for s in policy.keys():
            print("Updating the policy for State S")
            print("Value Function: {}".format(Q[s]))
            policy[s] = max_dict(Q[s])[0]
            print("Maximum Value: {}".format(policy[s]))

    plt.plot(deltas)
    plt.show()

    # find V
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print("\nvalues:")
    print_values(V, grid)

    print("\npolicy:")
    print_policy(policy, grid)
