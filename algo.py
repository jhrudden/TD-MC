from gymnasium import Env
from tqdm import trange
from typing import Callable
import numpy as np

def td0(env: Env, policy: Callable, alpha: float, gamma: float, n_episodes: int, initial_V: np.ndarray = None):
    # TODO: Should these be initialized to 0 or random? Besides the terminal states
    if initial_V is not None:
        V = initial_V
    else:
        V = np.zeros(env.observation_space.n)
    V_estimates = np.zeros((n_episodes, env.observation_space.n))
    for i in trange(n_episodes):
        # Save the current value function estimates
        V_estimates[i] = V.copy()
        
        state, _ = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            action, _ = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
    return V, V_estimates