import gymnasium as gym
import random
from rich.console import Console
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
discount = 1

random.seed(0)

actions = range(0, env.unwrapped.action_space.n)
state_size = env.unwrapped.nrow * env.unwrapped.ncol
states = range(0, state_size)

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
state = env.reset(seed=0)

# here is an example how to access the transitions in the MDP
tp_matrix = env.unwrapped.P
state=0
action=1
print(f"Transition probabilities from state {state} with action {action}:")
for p, s_next, reward, _ in tp_matrix[state][action]:
	print("Probability", p)
	print("Next state", s_next)
	print("Reward", reward)

# here you can see the whole matrix in pretty print
#c = Console()
#c.print(env.unwrapped.P)

# your solution comes here
def policy_evaluation(policy, theta=0.0001):
    """
    Evaluate a policy using iterative policy evaluation
    
    Args:
        policy: dict mapping state -> action probabilities
        theta: convergence threshold
    
    Returns:
        V: state value function
    """
    # Step 1: Initialize value function
    V = np.zeros(state_size)
    
    iteration = 0
    while True:
        # Step 2: Initialize delta for convergence check
        delta = 0
        
        # Step 3: Loop through all states
        for s in states:
            # Store old value
            v = V[s]
            
            # Step 4: Calculate new value using Bellman equation
            new_value = 0
            for action in actions:
                # Get action probability from policy
                action_prob = policy[s][action]
                
                # Calculate expected value for this action
                action_value = 0
                for prob, next_state, reward, done in tp_matrix[s][action]:
                    action_value += prob * (reward + discount * V[next_state])
                
                new_value += action_prob * action_value
            
            # Update value function
            V[s] = new_value
            
            # Step 5: Update delta for convergence check
            delta = max(delta, abs(v - V[s]))
        
        iteration += 1
        print(f"Iteration {iteration}, Delta: {delta:.6f}")
        
        # Step 6: Check convergence
        if delta < theta:
            break
    return V

# Policy improvement testing

def create_random_policy():
    """Create a random policy where each action has equal probability"""
    policy = {}
    num_actions = len(actions)
    for s in states:
        policy[s] = {a: 1.0/num_actions for a in actions}
    return policy

# Run tests
print("=" * 50)
print("Testing Policy Evaluation")
print("=" * 50)

# Test policy
print("\n1. Random Policy:")
random_policy = create_random_policy()
V_random = policy_evaluation(random_policy)
print(f"Value function:\n{V_random.reshape(4, 4)}")