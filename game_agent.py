"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
from isolation import Board


INFINITY = float('inf')


def number_of_legal_moves(game, move):
    """Get the number of the list of possible moves for an L-shaped motion
    (like a knight in chess)
    """
    if move == Board.NOT_MOVED:
        return game.get_blank_spaces()

    r, c = move
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]

    valid_moves = [(r + dr, c + dc) for dr, dc in directions
                   if game.move_is_legal((r + dr, c + dc))]

    return len(valid_moves)


def deep_moves(game, player):
    """Get the difference between all possbile moves from the current legal moves
    """
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    own_num, opp_num = 0, 0
    own_num = sum([number_of_legal_moves(game, move) for move in own_moves])
    opp_num = sum([number_of_legal_moves(game, move) for move in opp_moves])
    return float(own_num - opp_num)


def manhattan_distance(game, player):
    """Get the standard Manhattan distance. In here, the distance between
    two players should be divided by 3 for better heuristic, because
    the agent moves 3 sqaures at a time like a knight in chess."""

    (x1, y1) = game.get_player_location(player)
    (x2, y2) = game.get_player_location(game.get_opponent(player))

    D = abs(x1 - x2) + abs(y1 - y2)
    D = D/3

    return D


def manhattan_distance_with_deep_moves(game, player):
    """Get the Manhattan distance with all possbile moves. `deep_moves()`
    function can be considered the mininum cost function for moving from
    one space to an adjancnt space.
    """
    cost  = deep_moves(game, player)
    distance = manhattan_distance(game, player)
    return cost * distance


def forecast_manhattan_distance(game, move, player):
    """Get the Manhattan distance at new game generated the given move.
    """
    new_game = game.forecast_move(move)
    return manhattan_distance(new_game, player)


def deep_distance(game, player):
    """
    Get the difference of Manhattan distance between all possbile moves
    from the current legal moves.
    """
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    own_dis, opp_dis = 0, 0
    own_dis = sum([forecast_manhattan_distance(game, move, player) for move in own_moves])
    opp_dis = sum([forecast_manhattan_distance(game, move, player) for move in opp_moves])
    return float(own_dis - opp_dis)


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return -INFINITY

    if game.is_winner(player):
        return INFINITY

    return manhattan_distance_with_deep_moves(game, player)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return -INFINITY

    if game.is_winner(player):
        return INFINITY

    return deep_moves(game, player)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return -INFINITY

    if game.is_winner(player):
        return INFINITY

    return deep_distance(game, player)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # TODO: finish this function!

        # Check the time limit
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Return (-1, -1) if there are no legam moves
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        # Find the moves with the highest score
        def max_value(game, depth):

            # Check the time limit
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Evaluate the score value if the depth is terminal state
            if depth == 0:
                return self.score(game, self)

            # Initialize with the lowest score
            v = -INFINITY

            # Find the higheset score from the list of possible moves
            for m in game.get_legal_moves():
                # Maximize the minimum value of next new borad
                v = max(v, min_value(game.forecast_move(m), depth-1))
            return v

        # Find the moves with the lowest score
        def min_value(game, depth):

            # Check the time limit
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Evaluate the score value if the depth is in terminal state
            if depth == 0:
                return self.score(game, self)

            # Initialize with the highest score
            v = INFINITY

            # Find the lowest score from the list of possible moves
            for m in game.get_legal_moves():
                # Minimize the maximum value of next new board
                v = min(v, max_value(game.forecast_move(m), depth-1))
            return v

        # Initialize the best score
        best_score = -INFINITY

        # Initialize the best move
        best_move = (-1, -1)

        # Find the best move from the list of possible moves
        for m in legal_moves:
            v = min_value(game.forecast_move(m), depth-1)
            if v > best_score:
                best_score = v
                best_move = m
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move
        best_move = (-1, -1)

        # Initialize the bset score
        best_score = -INFINITY

        # Get the score and move at the current depth with alphabeta search
        def depth_limited_search(game, depth):
            score = self.score(game, self)
            move = self.alphabeta(game, depth)
            return score, move

        try:
            # Run repeatedly with increasing depth limits until the best moves found
            depth = 1;
            while True:
                # Check the time limit
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()

                # Update the best core and best move
                best_score, best_move = depth_limited_search(game, depth)

                # Increase current depth
                depth += 1

                # Break the loop if the best core is the lowest value or the
                # highest value
                if best_score == INFINITY or best_score == -INFINITY:
                    break

        except SearchTimeout:
            pass

        # Return the final best move
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # Check the time limit
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Return (-1, -1) if there are no legam moves
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        # Find the move with the highest score
        def max_value(game, depth, alpha, beta):
            # Check the time limit
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Evaluate the score value if the depth is in terminal state
            if depth == 0:
                return self.score(game, self)

            # Initialize the score with the lowest score
            v = -INFINITY

            # Find the best alpha
            for m in game.get_legal_moves():
                # Maximize the minimum value of next new board
                v = max(v, min_value(game.forecast_move(m), depth-1, alpha, beta))
                if v >= beta:
                    return v

                # Update the current alpha
                alpha = max(alpha, v)
            return v

        # Find the move with the lowest score
        def min_value(game, depth, alpha, beta):
            # Check the time limit
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Evaluate the score value if the depth is in terminal state
            if depth == 0:
                return self.score(game, self)

            # Initialize the score with the highest score
            v = INFINITY

            # Find the best beta
            for m in game.get_legal_moves():
                # Minimize the maximum value of next new board
                v = min(v, max_value(game.forecast_move(m), depth-1, alpha, beta))
                if v <= alpha:
                    return v

                # Update the current beta
                beta = min(beta, v)
            return v

        # Initialize the best core with lowest score
        best_score = -INFINITY

        # Initialize beta with the highest score
        beta = INFINITY

        # Initialize the best move
        best_move = (-1, -1)

        # Find the best move from the list of possible moves with alpha beta
        # search algorithm
        for m in game.get_legal_moves():
            v = min_value(game.forecast_move(m), depth-1, best_score, beta)
            if v > best_score:
                best_score = v
                best_move = m
        return best_move
