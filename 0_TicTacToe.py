import random
import numpy as np

class RewardSystem:
    def __init__(self):
        self.reward_config = {
            'win': 1,
            'lose': -1,
            'draw': 0.25,
            'valid_move': 0.1,
            'invalid_move': -10,    # Penalty for invalid move
        }
    
    def calculate_reward(self, state, action, next_state, done, winner, valid_move):
        """Calculate reward based on state transition"""
        reward = 0
        
        # Game outcome rewards
        if done:
            if winner == 'agent':
                reward += self.reward_config['win']
            elif winner == 'opponent':
                reward += self.reward_config['lose']
            else:
                reward += self.reward_config['draw']
        
        # Action validity
        if valid_move:
            reward += self.reward_config['valid_move']
        else:
            reward += self.reward_config['invalid_move']
        
        return reward


# Legacy functions for compatibility
game_state = ["_" for i in range(0, 9)]

def print_board():
    print(game_state[0:3])
    print(game_state[3:6])
    print(game_state[6:9])

def update_board(position, symbol):
    if position < 0 or position > 8:
        print("Invalid position. Please choose a position between 0 and 8.")
        return False
    if game_state[position] != "_":
        print("Position already taken. Please choose another position.")
        return False
    game_state[position] = symbol
    return True

def game_finished():
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]

    for combination in winning_combinations:
        if game_state[combination[0]] == game_state[combination[1]] == game_state[combination[2]] == "x":
            print("Player x wins!")
            return True
        elif game_state[combination[0]] == game_state[combination[1]] == game_state[combination[2]] == "o":
            print("Player o wins!")
            return True
    
    if "_" not in game_state:
        print("It's a draw!")
        return True
    
    return False

