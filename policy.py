import numpy as np

def argmax_rand(arr):
    """
    Returns the index of the maximum value in arr. If there are multiple maximum values, one is chosen randomly.

    Args:
        arr: The array to find the maximum value in
    """
    max_val = np.max(arr)
    max_indices = np.where(arr == max_val)[0]
    return np.random.choice(max_indices)

def equiprobable_policy(env):
    def policy(observation):
        action = np.random.choice(env.action_space.n)
        return (action, 1 / env.action_space.n)
    return policy

def greedy_epsilon_policy(env, Q, epsilon, break_ties_randomly=True):
    def policy(observation):
        argmax_ = np.argmax
        if break_ties_randomly: # use custom argmax function if we want to break ties randomly
            argmax_ = argmax_rand
        action_probs = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        max_action = argmax_(Q[observation]) # breaks ties randomly
        action_probs[max_action] += 1 - epsilon 
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return (action, action_probs[action])
    return policy

def greedy_policy(env, Q, break_ties_randomly=True):
    return greedy_epsilon_policy(env, Q, 0, break_ties_randomly)