from gymnasium import Env
from tqdm import trange
from typing import Callable, Optional
import numpy as np

def generate_episode(env: Env, policy: Callable):
    episode = []
    state, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action, _ = policy(state)
        next_state, reward, done, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode

def mc_prediction(env: Env, policy: Callable, gamma: float, n_episodes: int, initial_V: np.ndarray = None, alpha: Optional[float] = None, verbose: bool = False):
    """
    Every visit Monte Carlo prediction algorithm for estimating the value function of a policy using n_episodes.

    Args:
        env: The environment to use
        policy: The policy to evaluate
        gamma: The discount factor
        n_episodes: The number of episodes to use
        initial_V: The initial value function estimates
        alpha: The step size to use. If None, alpha will be set to 1 / N(s), where N(s) is the number of times state s has been visited.
        verbose: Whether to print the progress of the algorithm
    """
    if initial_V is not None:
        V = initial_V.copy()
    else:
        V = np.zeros(env.observation_space.n)

    C = np.zeros(env.observation_space.n)

    V_estimates = np.zeros((n_episodes, env.observation_space.n))
    
    # use range instead of trange to avoid tqdm overhead when verbose is False
    if verbose:
        range_ = trange(n_episodes)
    else:
        range_ = range(n_episodes)
    for i in range_:
        # Save the current value function estimates
        V_estimates[i] = V.copy()
        episode = generate_episode(env, policy)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            C[state] += 1
            if alpha is None:
                alpha = 1 / C[episode[t][0]]
            V[state] += alpha * (G - V[state])
    return V, V_estimates


def td0(env: Env, policy: Callable, alpha: float, gamma: float, n_episodes: int, initial_V: np.ndarray = None, verbose: bool = False):
    """
    Temporal Difference Learning algorithm for estimating the value function of a policy using n_episodes.

    Args:
        env: The environment to use
        policy: The policy to evaluate
        alpha: The step size to use
        gamma: The discount factor
        n_episodes: The number of episodes to use
        initial_V: The initial value function estimates
        verbose: Whether to print the progress of the algorithm
    """
    # TODO: Should these be initialized to 0 or random? Besides the terminal states
    if initial_V is not None:
        V = initial_V.copy()
    else:
        V = np.zeros(env.observation_space.n)
    V_estimates = np.zeros((n_episodes, env.observation_space.n))

    # use range instead of trange to avoid tqdm overhead when verbose is False
    if verbose:
        range_ = trange(n_episodes)
    else:
        range_ = range(n_episodes)
    for i in range_:
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