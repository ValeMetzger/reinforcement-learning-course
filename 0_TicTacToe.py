import random
import pickle # To save and load the Q-table

# --- Reinforcement Learning Parameters ---
Q_TABLE = {} # Stores Q-values: {state_string: {action: q_value}}
LEARNING_RATE = 0.1 # Alpha (α)
DISCOUNT_FACTOR = 0.9 # Gamma (γ)
EXPLORATION_RATE = 1.0 # Epsilon (ε) - starts high for exploration
EXPLORATION_DECAY_RATE = 0.001 # How much epsilon decreases after each episode
MIN_EXPLORATION_RATE = 0.01 # Minimum epsilon value

# --- Constants for Rewards ---
REWARD_WIN = 1
REWARD_LOSS = -1
REWARD_DRAW = 0.5
REWARD_ILLEGAL_MOVE = -10 # Discourage trying to place where not allowed
REWARD_MOVE_ONGOING = -0.1 # Small penalty for each move to encourage speed


game_state = ["_" for i in range(0, 9)]


def print_board():
    print(game_state[0:3])
    print(game_state[3:6])
    print(game_state[6:9])


def update_board(position, symbol):
    # Task: Check if the position is valid (0-8)
    if position < 0 or position > 8:
        print("Invalid position. Please choose a position between 0 and 8.")
        return
    # Task: Check if the position is empty
    if game_state[position] != "_":
        print("Position already taken. Please choose another position.")
        return
    # otherwise return with a invalid message to the user
    game_state[position] = symbol


def game_finished():
    # Write logic to decide if game is finished
    winning_combinations = [
        [0, 1, 2], [3, 4, 5],
        [6, 7, 8], [0, 3, 6],
        [1, 4, 7], [2, 5, 8],
    ]

    for combination in winning_combinations:
        if game_state[combination[0]] == game_state[combination[1]] == game_state[combination[2]] == "x":
            print("Player x wins!")
            return True
        elif game_state[combination[0]] == game_state[combination[1]] == game_state[combination[2]] == "o":
            print("Computer o wins!")
            return True
        elif "_" not in game_state:
            print("It's a draw!")
            return True

    return False

# Hashes the current state of the board to a string for the Q-table
def get_state_string(board):
    return "".join(board)

def initialize_q_values(state_string):
    if state_string not in Q_TABLE:
        Q_TABLE[state_string] = {i: 0.0 for i in range(9) if game_state[i] == "_"}  # Initialize only for available moves



if __name__ == "__main__":
    print("Welcome to TicTacToe!")
    print("You can put your 'x' at the following positions:")
    print('[0,1,2]\n[3,4,5]\n[6,7,8]')

    print("Current board:")
    print_board()
    while not game_finished():
        i = int(input("Where do you want to put your 'x'? (0-8)"))
        update_board(i, "x")
        # Task: implement the opponents move
        
        print_board()
