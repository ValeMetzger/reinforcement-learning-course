import gymnasium as gym
import random
import pandas as pd
import numpy as np

# Initialize environments
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
env_8x8 = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8", render_mode="ansi")

random.seed(42)
np.random.seed(42)

print("## Frozen Lake ##")
print("Starting training...")

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

# Task 1: 4x4 Environment (Random Policy)
print("\n=== Task 1: 4x4 Random Policy ===")
goal_reached = False
episode_count = 0
successful_path = []
max_attempts = 1000

while not goal_reached and episode_count < max_attempts:
    episode_count += 1
    if episode_count % 100 == 0:
        print(f"Episode {episode_count}...")
    
    episode_done = False
    reward = 0
    state, _ = env.reset(seed=42)
    current_episode_path = []

    step_count = 0
    while not episode_done and step_count < 100:
        action = random.randint(0, 3)
        current_episode_path.append((state, action))
        state, reward, terminated, truncated, _ = env.step(action)
        episode_done = terminated or truncated
        step_count += 1
        
        if episode_done and reward > 0:
            goal_reached = True
            successful_path = current_episode_path
            break

if goal_reached:
    print(f"Success! Goal reached after {episode_count} episodes.")
    print(f"Actions taken: {len(successful_path)}")
else:
    print(f"Failed to find solution in {max_attempts} attempts")

def create_policy_from_path(path):
    """Create a policy dictionary from successful path"""
    policy = {}
    for state, action in path:
        policy[state] = action
    return policy

optimal_policy = create_policy_from_path(successful_path) if successful_path else {}

# Task 2: 8x8 Environment (Random Policy)
print("\n=== Task 2: 8x8 Random Policy ===")
goal_reached_8x8 = False
episode_count_8x8 = 0
successful_path_8x8 = []
max_attempts_8x8 = 5000

while not goal_reached_8x8 and episode_count_8x8 < max_attempts_8x8:
    episode_count_8x8 += 1
    if episode_count_8x8 % 500 == 0:
        print(f"Episode {episode_count_8x8}...")
    
    episode_done = False
    reward = 0
    state, _ = env_8x8.reset(seed=42)
    current_episode_path = []

    step_count = 0
    while not episode_done and step_count < 200:
        action = random.randint(0, 3)
        current_episode_path.append((state, action))
        state, reward, terminated, truncated, _ = env_8x8.step(action)
        episode_done = terminated or truncated
        step_count += 1
        
        if episode_done and reward > 0:
            goal_reached_8x8 = True
            successful_path_8x8 = current_episode_path
            break

if goal_reached_8x8:
    print(f"8x8 Success! Goal reached after {episode_count_8x8} episodes.")
    print(f"Actions taken: {len(successful_path_8x8)}")
else:
    print(f"8x8 Failed to find solution in {max_attempts_8x8} attempts")

optimal_policy_8x8 = create_policy_from_path(successful_path_8x8) if successful_path_8x8 else {}

# Task 3: Test learned policy in slippery environment
print("\n=== Task 3: Testing Deterministic Policy in Slippery Environment ===")

if not optimal_policy:
    print("No policy learned from Task 1, skipping slippery test")
    successful_episodes = 0
    num_test_episodes = 10
else:
    env_slippery = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
    
    num_test_episodes = 50
    successful_episodes = 0
    total_steps_slippery = 0

    for episode in range(num_test_episodes):
        state, _ = env_slippery.reset(seed=42 + episode)
        episode_done = False
        steps_slippery = 0
        max_episode_steps = 100
        
        while not episode_done and steps_slippery < max_episode_steps:
            if state in optimal_policy:
                action = optimal_policy[state]
            else:
                action = random.randint(0, 3)
            
            steps_slippery += 1
            state, reward, terminated, truncated, _ = env_slippery.step(action)
            episode_done = terminated or truncated
            
            if episode_done and reward > 0:
                successful_episodes += 1
                total_steps_slippery += steps_slippery
                break

    avg_steps_slippery = total_steps_slippery / successful_episodes if successful_episodes > 0 else 0

print(f"Slippery Environment Results:")
print(f"Success rate: {successful_episodes}/{num_test_episodes} ({successful_episodes/num_test_episodes:.2%})")
print(f"Average steps (successful): {avg_steps_slippery:.1f}")

# Task 4: Q-Learning ONLY for Slippery Environment
print("\n=== Task 4: Q-Learning for Slippery Environment ===")

env_slippery = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

# Q-Learning parameters
alpha = 0.1
gamma = 0.95
epsilon = 0.1
num_episodes = 2000

# Initialize Q-table
Q = np.zeros((env_slippery.observation_space.n, env_slippery.action_space.n))
episode_rewards = []

print(f"Training Q-Learning for {num_episodes} episodes...")

for episode in range(num_episodes):
    state, _ = env_slippery.reset(seed=42 + episode)
    episode_reward = 0
    episode_length = 0
    episode_done = False
    
    while not episode_done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])
        
        # Take action
        next_state, reward, terminated, truncated, _ = env_slippery.step(action)
        episode_done = terminated or truncated
        
        # Q-learning update
        best_next_action = np.argmax(Q[next_state])
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * Q[next_state, best_next_action] - Q[state, action]
        )
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        if episode_length > 100:
            break
    
    episode_rewards.append(episode_reward)
    
    # Print progress
    if (episode + 1) % 500 == 0:
        recent_success_rate = np.mean(episode_rewards[-100:])
        print(f"Episode {episode + 1}: Success rate (last 100): {recent_success_rate:.2%}")

# Test Q-Learning policy
print("\n=== Testing Q-Learning Policy ===")
def create_q_policy(Q_table):
    policy = {}
    for state in range(Q_table.shape[0]):
        policy[state] = np.argmax(Q_table[state])
    return policy

q_learned_policy = create_q_policy(Q)

num_test_episodes_q = 100
q_successful_episodes = 0
q_total_steps = 0

for episode in range(num_test_episodes_q):
    state, _ = env_slippery.reset(seed=1000 + episode)
    episode_done = False
    steps = 0
    
    while not episode_done:
        action = q_learned_policy[state]
        state, reward, terminated, truncated, _ = env_slippery.step(action)
        episode_done = terminated or truncated
        steps += 1
        
        if episode_done and reward > 0:
            q_successful_episodes += 1
            q_total_steps += steps
            break
        elif steps > 100:
            break

q_success_rate = q_successful_episodes / num_test_episodes_q
q_avg_steps = q_total_steps / q_successful_episodes if q_successful_episodes > 0 else 0

print(f"\nQ-Learning Results:")
print(f"Success rate: {q_successful_episodes}/{num_test_episodes_q} ({q_success_rate:.2%})")
print(f"Average steps (successful): {q_avg_steps:.1f}")

# Final Results Summary
print(f"\n=== Final Results Summary ===")
results_data = {
    'Task': ['4x4 Random', '8x8 Random', '4x4 Slippery (Det. Policy)', '4x4 Slippery (Q-Learning)'],
    'Episodes_to_Learn': [episode_count, episode_count_8x8, 1, num_episodes],
    'Success_Rate': [
        '100%' if goal_reached else '0%',
        '100%' if goal_reached_8x8 else '0%',
        f'{successful_episodes/num_test_episodes:.1%}',
        f'{q_success_rate:.1%}'
    ],
    'Avg_Steps': [
        len(successful_path) if successful_path else 0,
        len(successful_path_8x8) if successful_path_8x8 else 0,
        f'{avg_steps_slippery:.1f}',
        f'{q_avg_steps:.1f}'
    ]
}

results_df = pd.DataFrame(results_data)
print(results_df.to_string(index=False))

print("\n=== Analysis ===")
print("1. Random policy works for deterministic environments but is inefficient")
print("2. Deterministic policy fails in slippery environment due to stochasticity")
print("3. Q-Learning learns to handle stochastic transitions effectively")