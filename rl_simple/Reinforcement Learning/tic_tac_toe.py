
import numpy as np

LENGTH = 3


class Agent:
    def __init__(self, eps=0.1, alpha=0.5):
        # this class will be coded as epsilon greedy
        self.eps = eps       # probability of choosing a random action instead of greedy
        self.alpha = alpha   # learning rate
        self.verbose = False
        self.state_history = []
        self.V = None
        self.sym = None

    def setV(self, V):
        self.V = V

    def set_sym(self, sym):
        self.sym = sym

    def set_verbose(self, verb):
        self.verbose = verb

    def reset_history(self):
        self.state_history = []

    def take_action(self, env):
        # choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        best_state = None
        if r < self.eps:
            # taking a random move as rand is less than epsilon
            if self.verbose:
                print("Taking a random action")
            # Get list of all possible moves using is empty method for the current environment
            possible_moves = []
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        possible_moves.append((i, j))
            # use random strategy
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves
        else:
            # taking the greedy move as rand is greater than epsilon
            # Get list of all possible moves and evaluate
            pos2value = {}  # for debugging
            next_move = None
            best_value = -1
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        # this is the eligible moves
                        # what does the board look like if i play this move
                        env.board[i, j] = self.sym   # add the players symbol to the board
                        state = env.get_state()      # get the hash of the board state given i play this move
                        env.board[i, j] = 0          # don't forget to change it back!
                        pos2value[(i, j)] = self.V[state]
                        # if the value of the state for the chosen action is better than the current best value
                        # recall state here is the hash number from 1->3^9 so we need to initalise V such that it has
                        # value for given index
                        if self.V[state] > best_value:
                            best_value = self.V[state]  # update our best value
                            best_state = state          # update our best state after the move
                            next_move = (i, j)          # update position for the next move

            # if verbose, draw the board w/ the values
            if self.verbose:
                print("Taking a greedy action")
                for i in range(LENGTH):
                    print("------------------")
                    for j in range(LENGTH):
                        if env.is_empty(i, j):
                            # print the value
                            print(" %.2f|" % pos2value[(i, j)], end="")
                        else:
                            print("  ", end="")
                        if env.board[i, j] == env.x:
                            print("x  |", end="")
                        elif env.board[i, j] == env.o:
                            print("o  |", end="")
                        else:
                            print("   |", end="")
                        print("")
                    print("------------------")

            # make the move
            env.board[next_move[0], next_move[1]] = self.sym

    def update_state_history(self, s):
        # needs to be updated every iteration
        # so cannot happen in take_action as this only happens once every other iteration for each player
        # note s = env.get_state() so s is hashed function for the game state
        self.state_history.append(s)

    def update(self, env):
        # remember V = Value Function
        # we want to backtrack over the states so that:
        # V(prev_state) = V(prev_state) + alpha*(V(next_state)-V(prev_state))
        # where V(next_state) = reward if its in the most current state
        # we only want to do this at the end of an episode (end of a game!)
        # REALLY IMPORTANT TO UNDERSTAND!
        reward = env.reward(self.sym)  # if the sym is the winner, you get 1, if not you get 0
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha*(target - self.V[prev])
            self.V[prev] = value
            target = value
        self.reset_history()


class Environment:
    def __init__(self):
        self.board = np.zeros((LENGTH, LENGTH))
        self.x = -1  # represents an x on the board, player 1
        self.o = 1   # represents an o on the board, player 2
        self.winner = None
        self.ended = False
        self.num_states = 3**(LENGTH*LENGTH)

    def is_empty(self, i, j):
        # returns true if position on the board is empty, false if not
        return self.board[i, j] == 0

    def reward(self, sym):
        # sym is the symbol for the player
        # reward = 0 if game isn't over
        if not self.game_over():
            return 0
        # if game is finished, winner will be self.x or self.o and you return 1 if winner is argument or 0 if not
        return 1 if self.winner == sym else 0

    def get_state(self):
        # returns the current state of the board, represented as an int
        # from 0 to [S]-1 where S = set of all states
        # S = 3^(BOARD SIZE), as each cell can be one of 3 values (x,o,empty)
        # ignore that some states are possible
        # so we are getting an integer represented by base-3 number!
        # each number refers to the position on board
        # if first cell is empty, will have 0
        # if third cell is x/o will have 1/2
        # return the base 3 number
        k = 0   # used to scale the multiplier up
        h = 0   # return hash value
        v = 0   # initialise v
        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                h += (LENGTH**k) * v
                k += 1
        return h

    def game_over(self, force_recalculate=False):
        # returns true if game over (a player has won or it's a draw)
        # otherwise returns false
        # also sets 'winner' instance variable and 'ended' instance variable
        if not force_recalculate and self.ended:
            return self.ended

        # check rows
        # for each of the rows and for each player, take the sum of the board
        # if the sum is equal to the player * 3 (i.e. -3 for x, 3 for o) then its a win
        # if its a win, update the winner to the player, the ended to True
        # then return True as the game has finished
        for i in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[i].sum() == player * LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        # check columns
        # do the same thing
        for j in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[:, j].sum() == player * LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        # check diagonals
        # do the same thing
        for player in (self.x, self.o):
            # top-left -> bottom-right diagonal
            if self.board.trace() == player * LENGTH:
                self.winner = player
                self.ended = True
                return True
            # top-right -> bottom-left diagonal
            if np.fliplr(self.board).trace() == player * LENGTH:
                self.winner = player
                self.ended = True
                return True

        # check if draw
        # if there are no empty cells, and we reach this stage it must be a draw
        # as no winning has returned a value
        if np.all((self.board == 0) == False):
            # winner stays None
            self.winner = None
            self.ended = True
            return True

        # if we reach this stage the game is not finished so winner is None and return false
        # game is not over
        self.winner = None
        return False

    def is_draw(self):
        # game is a draw if ended is true and the winner is None
        return self.ended and self.winner is None

    # Example board
    # -------------
    # | x |   |   |
    # -------------
    # |   |   |   |
    # -------------
    # |   |   | o |
    # -------------
    def draw_board(self):
        for i in range(LENGTH):
            print("-------------")
            for j in range(LENGTH):
                print("  ", end="")
                if self.board[i, j] == self.x:
                    print("x ", end="")
                elif self.board[i,j] == self.o:
                    print("o ", end="")
                else:
                    print("  ", end="")
            print("")
        print("-------------")


class Human:
    def __init__(self):
        pass

    def set_symbol(self, sym):
        self.sym = sym

    def take_action(self, env):
        while True:
            # break if we make a legal move
            move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i, j):
                env.board[i,j] = self.sym
                break

    def update(self, env):
        pass

    def update_state_history(self, s):
        pass


# recursive function that will return all
# possible states (as ints) and who the corresponding winner is for those states (if any)
# (i, j) refers to the next cell on the board to permute (we need to try -1, 0, 1)
# impossible games are ignored, i.e. 3x's and 3o's in a row simultaneously
# since that will never happen in a real game
def get_state_hash_and_winner(env, i=0, j=0):
      results = []
      for v in (0, env.x, env.o):
            env.board[i,j] = v # if empty board it should already be 0
            if j == 2:
            # j goes back to 0, increase i, unless i = 2, then we are done
                if i == 2:
                    # the board is full, collect results and return
                    state = env.get_state()
                    ended = env.game_over(force_recalculate=True)
                    winner = env.winner
                    results.append((state, winner, ended))
                else:
                    results += get_state_hash_and_winner(env, i + 1, 0)
            else:
                # increment j, i stays the same
                results += get_state_hash_and_winner(env, i, j + 1)
      return results

# play all possible games
# need to also store if game is over or not
# because we are going to initialize those values to 0.5
# NOTE: THIS IS SLOW because MANY possible games lead to the same outcome / state
# def get_state_hash_and_winner(env, turn='x'):
#   results = []

#   state = env.get_state()
#   # board_before = env.board.copy()
#   ended = env.game_over(force_recalculate=True)
#   winner = env.winner
#   results.append((state, winner, ended))

#   # DEBUG
#   # if ended:
#   #   if winner is not None and env.win_type.startswith('col'):
#   #     env.draw_board()
#   #     print "Winner:", 'x' if winner == -1 else 'o', env.win_type
#   #     print "\n\n"
#   #     assert(np.all(board_before == env.board))

#   if not ended:
#     if turn == 'x':
#       sym = env.x
#       next_sym = 'o'
#     else:
#       sym = env.o
#       next_sym = 'x'

#     for i in xrange(LENGTH):
#       for j in xrange(LENGTH):
#         if env.is_empty(i, j):
#           env.board[i,j] = sym
#           results += get_state_hash_and_winner(env, next_sym)
#           env.board[i,j] = 0 # reset it
#   return results


def initialV_x(env, state_winner_triples):
      # initialize state values as follows
      # if x wins, V(s) = 1
      # if x loses or draw, V(s) = 0
      # otherwise, V(s) = 0.5
      V = np.zeros(env.num_states)
      for state, winner, ended in state_winner_triples:
          if ended:
              if winner == env.x:
                  v = 1
              else:
                v = 0
          else:
            v = 0.5
          V[state] = v
      return V


def initialV_o(env, state_winner_triples):
    # initialize state values as follows
    # if x wins, V(s) = 1
    # if x loses or draw, V(s) = 0
    # otherwise, V(s) = 0.5
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V



def play_game(p1, p2, env, draw=0):
    # loops until the game is over
    current_player = None
    while not env.game_over():
        # alternate between players
        # p1 always starts first
        if current_player == p1:
              current_player = p2
        else:
              current_player = p1

        # draw the board before the user who wants to see it makes a move
        if draw:
          if draw == 1 and current_player == p1:
                env.draw_board()
          if draw == 2 and current_player == p2:
                env.draw_board()

        # current player makes a move
        current_player.take_action(env)

        # update state histories
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

        if draw:
            env.draw_board()

        # do the value function update
        p1.update(env)
        p2.update(env)


if __name__ == '__main__':
      # train the agent
      p1 = Agent()
      p2 = Agent()

      # set initial V for p1 and p2
      env = Environment()
      state_winner_triples = get_state_hash_and_winner(env)


      Vx = initialV_x(env, state_winner_triples)
      p1.setV(Vx)
      Vo = initialV_o(env, state_winner_triples)
      p2.setV(Vo)

      # give each player their symbol
      p1.set_sym(env.x)
      p2.set_sym(env.o)

      T = 10000
      for t in range(T):
            if t % 200 == 0:
                  print(t)
            play_game(p1, p2, Environment())

      # play human vs. agent
      # do you think the agent learned to play the game well?
      human = Human()
      human.set_symbol(env.o)
      while True:
            p1.set_verbose(True)
            play_game(p1, human, Environment(), draw=2)
            # I made the agent player 1 because I wanted to see if it would
            # select the center as its starting move. If you want the agent
            # to go second you can switch the human and AI.
            answer = input("Play again? [Y/n]: ")
            if answer and answer.lower()[0] == 'n':
                  break