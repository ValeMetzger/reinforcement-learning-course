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


def main():
    successful_episodes = 0
    episode_count = 0
    
    while successful_episodes < 100:
        episode = play_episode()
        episode_count += 1
        
        # Calculate total reward for this episode
        total_reward = sum([reward for _, _, reward in episode])
        
        if total_reward > 0:  # Successful episode
            successful_episodes += 1
            
            # Task 1: Update Q-values using MC
            update_q_values_mc(episode)
            
            print(f"\nSuccessful episode {successful_episodes}:")
            print("Current Q-values:")
            print(Q_values)
            
            # Debug: Show non-zero Q-values
            non_zero_q = np.where(Q_values != 0)
            if len(non_zero_q[0]) > 0:
                print("Non-zero Q-values:")
                for i, j in zip(non_zero_q[0], non_zero_q[1]):
                    print(f"  Q({i},{j}) = {Q_values[i,j]:.4f}, visits: {N_counts[i,j]}")
            
            # Task 2: Play 100 episodes with greedy policy
            avg_reward = play_greedy_episodes(100)
            print(f"Average reward over 100 greedy episodes: {avg_reward:.3f}")

main()