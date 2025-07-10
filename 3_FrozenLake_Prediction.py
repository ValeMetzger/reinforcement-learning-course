import random
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="ansi")
random.seed(0)
np.random.seed(0)

print("## Frozen Lake ##")
print("Start state:")
state = env.reset(seed=0)
print(env.render())
all_states = [s for s in range(env.observation_space.n)]
print("All states:", all_states)

Q_values = np.zeros((env.observation_space.n, env.action_space.n))
N_counts = np.zeros((env.observation_space.n, env.action_space.n))
Q_values_td = np.zeros((env.observation_space.n, env.action_space.n))


def play_episode():
    state, _ = env.reset()
    done = False
    episode = []  # Store (state, action, reward) tuples
    
    while not done:
        action = random.randint(0, 3)
        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    
    return episode


def play_greedy_episodes(num_episodes=100):
    """Play episodes using greedy policy based on current Q-values"""
    total_rewards = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Greedy action selection with tie-breaking
            q_values_for_state = Q_values[state]
            # Add small random noise to break ties
            action = np.argmax(q_values_for_state + np.random.random(4) * 1e-6)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
        total_rewards += episode_reward
    
    return total_rewards / num_episodes
    


def update_q_values_mc(episode):
    global Q_values, N_counts
    
    # Calculate returns (G) for each step
    G = 0
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G += reward  # No discounting (gamma = 1)
        
        # Every-visit MC: update for every occurrence of (s,a)
        N_counts[state, action] += 1
        
        # Q(s,a) = Q(s,a) + 1/N(s,a) * (G - Q(s,a))
        Q_values[state, action] += (1.0 / N_counts[state, action]) * (G - Q_values[state, action])


alpha = 0.01

def update_q_values_td(episode):
    """Update Q-values using TD(0) prediction"""
    global Q_values_td
    
    # TD updates happen during the episode, step by step
    for t in range(len(episode)):
        state, action, reward = episode[t]
        
        if t < len(episode) - 1:  # Not the terminal step
            next_state, next_action, _ = episode[t + 1]
            # TD(0) update: Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
            # Since gamma = 1 and we use the next action from the episode
            td_target = reward + Q_values_td[next_state, next_action]
        else:  # Terminal step
            # No next state, so target is just the reward
            td_target = reward
        
        td_error = td_target - Q_values_td[state, action]
        Q_values_td[state, action] += alpha * td_error


def play_greedy_episodes_td(num_episodes=100):
    """Play episodes using greedy policy based on current TD Q-values"""
    total_rewards = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Greedy action selection with tie-breaking
            q_values_for_state = Q_values_td[state]
            action = np.argmax(q_values_for_state + np.random.random(4) * 1e-6)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
        total_rewards += episode_reward
    
    return total_rewards / num_episodes


def main():
    successful_episodes = 0
    episode_count = 0
    
    print("=== MONTE CARLO PREDICTION ===")
    
    while successful_episodes < 100:
        episode = play_episode()
        episode_count += 1
        
        # Calculate total reward for this episode
        total_reward = sum([reward for _, _, reward in episode])
        
        if total_reward > 0:  # Successful episode
            successful_episodes += 1
            
            # Task 1: Update Q-values using MC
            update_q_values_mc(episode)
            
            print(f"\nMC - Successful episode {successful_episodes}:")
            print("Current Q-values:")
            print(Q_values)
            
            # Task 2: Play 100 episodes with greedy policy
            avg_reward = play_greedy_episodes(100)
            print(f"MC - Average reward over 100 greedy episodes: {avg_reward:.3f}")
    
    # Reset for TD learning
    print("\n=== TD PREDICTION ===")
    successful_episodes = 0
    episode_count = 0
    
    while successful_episodes < 100:
        episode = play_episode()
        episode_count += 1
        
        # Calculate total reward for this episode
        total_reward = sum([reward for _, _, reward in episode])
        
        if total_reward > 0:  # Successful episode
            successful_episodes += 1
            
            # Task 3: Update Q-values using TD
            update_q_values_td(episode)
            
            print(f"\nTD - Successful episode {successful_episodes}:")
            print("Current Q-values:")
            print(Q_values_td)
            
            # Task 3: Play 100 episodes with greedy policy using TD Q-values
            avg_reward = play_greedy_episodes_td(100)
            print(f"TD - Average reward over 100 greedy episodes: {avg_reward:.3f}")

main()