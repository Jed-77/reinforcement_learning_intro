import numpy as np
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]
        self.rewards = 0
        self.actions = 0

    def set(self, rewards, actions):
        # showing the position on the board, it shows the reward
        # rewards should be a dict of (i, j): r(row,col): reward
        # for each position on the board, it shows valid moves
        # actions should be a dict of (i, j): r(row,col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        # update the position of the robot
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        # get the current position of the robot
        return self.i, self.j

    def is_terminal(self, s):
        # true if terminal, false if not
        # as the action dictionary contains only moves for non-terminal squares,
        # if we are not in one of these squares, it is terminal
        return s not in self.actions

    def move(self, action):
        # this functions takes, current state, does a move, updates the state, and gets a reward
        # (s, a) -> (s', r)
        # check if move is legal first given our current position (using action dictionary)
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        # return the reward (if any) by using the updated position
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        # these are the opposite of what U/D/L/R normally do
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        # raise an exception if we arrive somewhere we shouldn't be
        assert(self.current_state() in self.all_states())

    def game_over(self):
        # returns true if game over, false if not
        # actions holds all non-terminal cells, so if we aren't in that the game has finished
        return (self.i, self.j) not in self.actions

    def all_states(self):
        # returns all possible states (both terminal and non-terminal)
        # terminal is from rewards, non-terminal is from actions
        # excludes the wall as we can't get here!
        return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state
    # the grid looks like this
    # x means you can't go there
    # s means start position
    # number means reward at that state
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    return g


def negative_grid(step_cost=-0.1):
    # in this game we want to try to minimize the number of moves, so we will penalize every move.
    # instead of random movements, we penalise every move with the step cost. (-0.1 for each move)
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
    })
    return g

