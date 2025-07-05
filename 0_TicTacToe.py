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
    
    
class TicTacToeEnvironment:
    def __init__(self):
        self.reset()
        self.reward_system = RewardSystem()
    
    def reset(self):
        """Reset the game state"""
        self.game_state = ["_" for i in range(9)]
        return self.get_state()
    
    def get_state(self):
        """Convert game state to numerical representation"""
        state = []
        for cell in self.game_state:
            if cell == "_":
                state.append(0)
            elif cell == "x":  # human player
                state.append(-1)
            elif cell == "o":  # agent
                state.append(1)
        return np.array(state)
    
    def is_valid_move(self, position):
        """Check if move is valid"""
        return 0 <= position <= 8 and self.game_state[position] == "_"
    
    def make_move(self, position, symbol):
        """Make a move on the board"""
        if self.is_valid_move(position):
            self.game_state[position] = symbol
            return True
        return False
    
    def check_winner(self):
        """Check for winner and return result"""
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]               # diagonals
        ]
        
        for combination in winning_combinations:
            if (self.game_state[combination[0]] == 
                self.game_state[combination[1]] == 
                self.game_state[combination[2]] == "o"):
                return "agent"
            elif (self.game_state[combination[0]] == 
                  self.game_state[combination[1]] == 
                  self.game_state[combination[2]] == "x"):
                return "opponent"
        
        if "_" not in self.game_state:
            return "draw"
        
        return None
    
    def step(self, action):
        """Execute one step in the environment"""
        state = self.get_state()
        
        # Agent makes move with "o"
        valid_move = self.make_move(action, "o")
        winner = self.check_winner()
        done = winner is not None
        
        # If game not done, opponent (human) makes move
        if not done:
            opponent_action = self.get_random_opponent_move()
            if opponent_action is not None:
                self.make_move(opponent_action, "x")
                winner = self.check_winner()
                done = winner is not None
        
        next_state = self.get_state()
        reward = self.reward_system.calculate_reward(
            state, action, next_state, done, winner, valid_move
        )
        
        return next_state, reward, done, {'winner': winner, 'valid_move': valid_move}
    
    def get_random_opponent_move(self):
        """Simple random opponent strategy"""
        available_moves = [i for i in range(9) if self.game_state[i] == "_"]
        return random.choice(available_moves) if available_moves else None
    
    def print_board(self):
        """Print the current board state"""
        print(self.game_state[0:3])
        print(self.game_state[3:6])
        print(self.game_state[6:9])


class AgentTrainer:
    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
        self.training_stats = {
            'episodes': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_reward': 0
        }
    
    def train_agent(self, episodes=1000):
        """Train the agent before game starts"""
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Update Q-table
                self.agent.update_q_table(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
            
            # Update statistics
            self._update_stats(episode_reward, info)
            
            # Log progress every 100 episodes
            if episode % 100 == 0:
                self._log_progress(episode)
        
        print("Training completed!")
        self._log_final_stats()
    
    def _update_stats(self, reward, info):
        self.training_stats['episodes'] += 1
        self.training_stats['total_reward'] += reward
        
        winner = info.get('winner')
        if winner == 'agent':
            self.training_stats['wins'] += 1
        elif winner == 'opponent':
            self.training_stats['losses'] += 1
        else:
            self.training_stats['draws'] += 1
    
    def _log_progress(self, episode):
        if self.training_stats['episodes'] > 0:
            win_rate = self.training_stats['wins'] / self.training_stats['episodes']
            avg_reward = self.training_stats['total_reward'] / self.training_stats['episodes']
            print(f"Episode {episode}: Win Rate: {win_rate:.2%}, Avg Reward: {avg_reward:.2f}")
    
    def _log_final_stats(self):
        total = self.training_stats['episodes']
        print(f"\nFinal Training Stats:")
        print(f"Total Episodes: {total}")
        print(f"Wins: {self.training_stats['wins']} ({self.training_stats['wins']/total:.2%})")
        print(f"Losses: {self.training_stats['losses']} ({self.training_stats['losses']/total:.2%})")
        print(f"Draws: {self.training_stats['draws']} ({self.training_stats['draws']/total:.2%})")

class SimpleAgent:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.q_table = {}
    
    def get_state_key(self, state):
        """Convert state array to string key for Q-table"""
        return str(state.tolist())
    
    def select_action(self, state):
        """Select action using epsilon-greedy strategy"""
        state_key = self.get_state_key(state)
        
        # Get available actions (empty positions)
        available_actions = [i for i in range(9) if state[i] == 0]
        
        if not available_actions:
            return 0
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon or state_key not in self.q_table:
            return random.choice(available_actions)
        
        # Choose best action from Q-table
        q_values = self.q_table[state_key]
        best_action = max(available_actions, key=lambda x: q_values[x])
        return best_action
    
    def update_q_table(self, state, action, reward, next_state, done, alpha=0.1, gamma=0.9):
        """Update Q-table using Q-learning"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * 9
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * 9
        
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + gamma * max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] += alpha * (target - self.q_table[state_key][action])


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


if __name__ == "__main__":
    # Training phase
    print("=== TRAINING PHASE ===")
    env = TicTacToeEnvironment()
    agent = SimpleAgent(epsilon=0.3)
    trainer = AgentTrainer(agent, env)
    
    # Train the agent - episodes can be adjusted
    print("Training the agent...")
    trainer.train_agent(episodes=5000)
    
    # Reduce exploration for gameplay
    agent.epsilon = 0.1
    
    print("\n=== GAME PHASE ===")
    print("Welcome to TicTacToe!")
    print("You can put your 'x' at the following positions:")
    print('[0,1,2]\n[3,4,5]\n[6,7,8]')
    
    # Reset for human game
    env.reset()
    print("Current board:")
    env.print_board()
    
    while True:
        winner = env.check_winner()
        if winner:
            if winner == "agent":
                print("Computer (o) wins!")
            elif winner == "opponent":
                print("You (x) win!")
            else:
                print("It's a draw!")
            break
        
        # Human move
        try:
            position = int(input("Where do you want to put your 'x'? (0-8): "))
            if env.make_move(position, "x"):
                env.print_board()
                
                winner = env.check_winner()
                if winner:
                    continue
                
                # Agent move
                state = env.get_state()
                agent_action = agent.select_action(state)
                if env.make_move(agent_action, "o"):
                    print(f"Computer plays 'o' at position {agent_action}")
                    env.print_board()
            else:
                print("Invalid move! Try again.")
        except ValueError:
            print("Please enter a valid number between 0 and 8.")